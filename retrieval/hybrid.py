"""
Hybrid retrieval: dense vector search (Qdrant) + sparse BM25,
merged via Reciprocal Rank Fusion (RRF).

Uses Qdrant in-memory mode — no Docker, no cloud account, no config.
Embeddings via sentence-transformers (BAAI/bge-large-en-v1.5),
a model fine-tuned for retrieval that works well on financial text.

Architecture:
  - Qdrant collection per company, created fresh each pipeline run
  - Dense vectors: 1024-dim BAAI/bge-large-en-v1.5
  - Sparse: BM25 via rank_bm25 (run client-side, results merged via RRF)
  - RRF merge: standard k=60 constant (Cormack et al. 2009)

Fallback: if qdrant-client or sentence-transformers are not installed,
automatically falls back to the pure BM25 implementation so the
pipeline never breaks.
"""
import re
import math
import uuid
from collections import defaultdict
from config import BM25_TOP_K, FINAL_TOP_K

# ── Optional imports — graceful fallback if not installed ─────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue, SearchRequest,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# ── Embedding model ───────────────────────────────────────────────────────────
# BAAI/bge-large-en-v1.5: 1024-dim, strong on retrieval tasks, ~1.3GB download
# Falls back to all-MiniLM-L6-v2 (90MB, 384-dim) if memory is tight
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM   = 1024
COLLECTION_NAME = "investment_research"

_embedder = None


def _get_embedder():
    """Load embedding model once, reuse across calls."""
    global _embedder
    if _embedder is None:
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        print("  (First run downloads ~1.3GB — subsequent runs use cache)")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("  Embedding model loaded.")
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    model = _get_embedder()
    # BGE models work best with a query instruction prefix
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.tolist()


# ── BM25 helpers ──────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', text.lower())


def _tfidf_scores(query_tokens: list[str], chunks: list[dict]) -> list[float]:
    """Lightweight TF-IDF fallback when BM25 is unavailable."""
    N  = len(chunks)
    df = defaultdict(int)
    for chunk in chunks:
        for t in set(_tokenize(chunk["text"])):
            df[t] += 1
    scores = []
    for chunk in chunks:
        tf_map = defaultdict(int)
        tokens = _tokenize(chunk["text"])
        for t in tokens:
            tf_map[t] += 1
        score = sum(
            (tf_map[qt] / (len(tokens) + 1)) *
            math.log((N + 1) / (df.get(qt, 0) + 1))
            for qt in query_tokens
        )
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """RRF merge — Cormack, Clarke & Buettcher 2009."""
    rrf = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            rrf[idx] += 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ── Main retriever class ──────────────────────────────────────────────────────
class HybridRetriever:
    """
    Hybrid retriever backed by Qdrant (dense) + BM25 (sparse).

    Falls back to pure BM25/TF-IDF if Qdrant or sentence-transformers
    are not installed — the pipeline never breaks.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks   = chunks
        self.use_qdrant = QDRANT_AVAILABLE and ST_AVAILABLE and len(chunks) > 0

        if self.use_qdrant:
            self._build_qdrant_index()
        else:
            self._build_bm25_index()

    # ── Index builders ────────────────────────────────────────────────────────

    def _build_qdrant_index(self):
        """Build Qdrant in-memory collection and upsert all chunks."""
        print(f"  Building Qdrant vector index ({len(self.chunks)} chunks)...")

        self.client = QdrantClient(":memory:")

        # Create collection with cosine similarity
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        # Embed all chunk texts in batches
        texts      = [c["text"] for c in self.chunks]
        embeddings = _embed(texts)

        # Upsert points with full metadata as payload
        points = []
        for i, (chunk, vector) in enumerate(zip(self.chunks, embeddings)):
            payload = {k: v for k, v in chunk.items() if k != "text"}
            payload["text"]        = chunk["text"]
            payload["chunk_index"] = i
            points.append(PointStruct(
                id     = str(uuid.uuid4()),
                vector = vector,
                payload = payload,
            ))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name = COLLECTION_NAME,
                points          = points[i:i + batch_size],
            )

        # Also build BM25 for the sparse leg of hybrid search
        self._build_bm25_index(silent=True)

        print(f"  Qdrant index built: {len(self.chunks)} vectors @ {EMBEDDING_DIM}d")
        print(f"  Embedding model: {EMBEDDING_MODEL}")

    def _build_bm25_index(self, silent: bool = False):
        """Build BM25 index (always built — used as sparse leg or sole retriever)."""
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        if BM25_AVAILABLE:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
        self.tokenized_chunks = tokenized
        if not silent:
            mode = "BM25" if BM25_AVAILABLE else "TF-IDF"
            print(f"  Built {mode} index over {len(self.chunks)} chunks.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        filter_doc_types: list[str] = None,
        filter_after_date: str = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks most relevant to the query.
        Uses Qdrant dense search + BM25 sparse search merged via RRF.
        Falls back to BM25/TF-IDF only if Qdrant unavailable.
        """
        if not self.chunks:
            return []

        if self.use_qdrant:
            return self._qdrant_hybrid_retrieve(
                query, top_k, filter_doc_types, filter_after_date
            )
        else:
            return self._bm25_retrieve(
                query, top_k, filter_doc_types, filter_after_date
            )

    def _qdrant_hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        filter_doc_types: list[str],
        filter_after_date: str,
    ) -> list[dict]:
        """Dense Qdrant search + BM25 sparse, merged via RRF."""

        # Build Qdrant filter if needed
        qdrant_filter = None
        if filter_doc_types:
            # Simple doc_type filter
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=filter_doc_types[0])
                )]
            ) if len(filter_doc_types) == 1 else None
            # Note: multi-value filter requires should[] — keep simple for now

        # ── Dense leg: Qdrant vector search ──────────────────────────────
        query_vector = _embed([query])[0]
        dense_results = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query           = query_vector,
            limit           = BM25_TOP_K,
            query_filter    = qdrant_filter,
            with_payload    = True,
        ).points

        # Map Qdrant results → local chunk indices
        dense_indices = []
        qdrant_scores = {}
        for hit in dense_results:
            chunk_idx = hit.payload.get("chunk_index")
            if chunk_idx is not None:
                # Apply date filter post-retrieval if needed
                if filter_after_date:
                    chunk_date = self.chunks[chunk_idx].get("date", "")
                    if chunk_date and chunk_date < filter_after_date:
                        continue
                dense_indices.append(chunk_idx)
                qdrant_scores[chunk_idx] = hit.score

        # ── Sparse leg: BM25 ──────────────────────────────────────────────
        query_tokens  = _tokenize(query)
        bm25_indices  = self._run_bm25(
            query_tokens, filter_doc_types, filter_after_date
        )

        # ── RRF merge ─────────────────────────────────────────────────────
        merged = reciprocal_rank_fusion([dense_indices, bm25_indices])

        results = []
        for local_idx, rrf_score in merged[:top_k]:
            if local_idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[local_idx])
            chunk["retrieval_score"]  = round(rrf_score, 6)
            chunk["dense_score"]      = round(qdrant_scores.get(local_idx, 0.0), 4)
            chunk["retrieval_method"] = "qdrant_hybrid"
            results.append(chunk)

        return results

    def _bm25_retrieve(
        self,
        query: str,
        top_k: int,
        filter_doc_types: list[str],
        filter_after_date: str,
    ) -> list[dict]:
        """Pure BM25/TF-IDF retrieval — fallback when Qdrant unavailable."""
        query_tokens = _tokenize(query)

        # Apply filters
        if filter_doc_types or filter_after_date:
            active = [
                (i, c) for i, c in enumerate(self.chunks)
                if (filter_doc_types is None or c.get("doc_type") in filter_doc_types)
                and (filter_after_date is None or c.get("date", "") >= filter_after_date)
            ]
            indices  = [i for i, _ in active]
            a_chunks = [c for _, c in active]
        else:
            indices  = list(range(len(self.chunks)))
            a_chunks = self.chunks

        if not a_chunks:
            return []

        if BM25_AVAILABLE:
            bm25    = BM25Okapi([_tokenize(c["text"]) for c in a_chunks])
            scores  = bm25.get_scores(query_tokens)
        else:
            scores  = _tfidf_scores(query_tokens, a_chunks)

        ranked = sorted(range(len(a_chunks)), key=lambda i: scores[i], reverse=True)

        results = []
        for local_idx in ranked[:top_k]:
            chunk = dict(a_chunks[local_idx])
            chunk["retrieval_score"]  = round(float(scores[local_idx]), 6)
            chunk["retrieval_method"] = "bm25_fallback"
            results.append(chunk)

        return results

    def _run_bm25(
        self,
        query_tokens: list[str],
        filter_doc_types: list[str],
        filter_after_date: str,
    ) -> list[int]:
        """Run BM25 and return ranked list of chunk indices."""
        if not self.chunks:
            return []

        if filter_doc_types or filter_after_date:
            active_indices = [
                i for i, c in enumerate(self.chunks)
                if (filter_doc_types is None or c.get("doc_type") in filter_doc_types)
                and (filter_after_date is None or c.get("date", "") >= filter_after_date)
            ]
            active_chunks = [self.chunks[i] for i in active_indices]
        else:
            active_indices = list(range(len(self.chunks)))
            active_chunks  = self.chunks

        if not active_chunks:
            return []

        if BM25_AVAILABLE:
            bm25   = BM25Okapi([_tokenize(c["text"]) for c in active_chunks])
            scores = bm25.get_scores(query_tokens)
        else:
            scores = _tfidf_scores(query_tokens, active_chunks)

        ranked_local = sorted(
            range(len(active_chunks)),
            key=lambda i: scores[i],
            reverse=True
        )[:BM25_TOP_K]

        return [active_indices[i] for i in ranked_local]

    def retrieve_multi(
        self,
        queries: list[str],
        top_k_per_query: int = 6,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Run multiple queries and merge results, deduplicating by chunk_id.
        Used by agents that have several sub-questions.
        """
        seen_ids   = set()
        all_results = []

        for query in queries:
            results = self.retrieve(query, top_k=top_k_per_query)
            for chunk in results:
                cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
                if deduplicate and cid in seen_ids:
                    continue
                seen_ids.add(cid)
                all_results.append(chunk)

        all_results.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)
        return all_results[:FINAL_TOP_K * 2]
