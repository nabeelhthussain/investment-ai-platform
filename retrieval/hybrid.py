"""
Hybrid retrieval: BM25 sparse search + keyword/TF-IDF dense search,
merged via Reciprocal Rank Fusion (RRF).

No vector database required — operates on the in-memory chunk corpus.
For hundreds of companies, this layer would be replaced with Qdrant +
dense embeddings, but for a two-company demo the quality is comparable.
"""
import re
import math
from collections import defaultdict

from rank_bm25 import BM25Okapi
from config import BM25_TOP_K, FINAL_TOP_K


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _tfidf_scores(query_tokens: list[str], chunks: list[dict]) -> list[float]:
    """
    Lightweight TF-IDF score as a proxy for dense retrieval.
    Computes term overlap weighted by inverse document frequency.
    """
    # Build IDF
    N = len(chunks)
    df = defaultdict(int)
    for chunk in chunks:
        tokens = set(_tokenize(chunk["text"]))
        for t in tokens:
            df[t] += 1

    scores = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk["text"])
        tf_map = defaultdict(int)
        for t in chunk_tokens:
            tf_map[t] += 1

        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            idf = math.log((N + 1) / (df.get(qt, 0) + 1))
            score += (tf / (len(chunk_tokens) + 1)) * idf

        scores.append(score)

    return scores


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists using RRF.
    k=60 is the standard constant from Cormack et al. 2009.
    Returns list of (index, rrf_score) sorted by score descending.
    """
    rrf_scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Retriever over an in-memory chunk corpus.
    Initialized once per company, reused across all agent queries.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._build_index()

    def _build_index(self):
        """Build BM25 index over all chunks."""
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.tokenized_chunks = tokenized
        print(f"  Built retrieval index over {len(self.chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        filter_doc_types: list[str] = None,
        filter_after_date: str = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks most relevant to the query.

        Args:
            query: Natural language query.
            top_k: Number of chunks to return.
            filter_doc_types: If set, only return chunks from these doc types.
            filter_after_date: If set (YYYY-MM-DD), only return chunks after this date.

        Returns:
            List of chunk dicts, sorted by relevance, each with a `retrieval_score` field.
        """
        if not self.chunks:
            return []

        query_tokens = _tokenize(query)

        # Apply pre-filter if needed
        if filter_doc_types or filter_after_date:
            indices = [
                i for i, c in enumerate(self.chunks)
                if (filter_doc_types is None or c.get("doc_type") in filter_doc_types)
                and (filter_after_date is None or c.get("date", "") >= filter_after_date)
            ]
            active_chunks = [self.chunks[i] for i in indices]
            active_indices = indices
        else:
            active_chunks = self.chunks
            active_indices = list(range(len(self.chunks)))

        if not active_chunks:
            return []

        # BM25 ranking
        tokenized_active = [_tokenize(c["text"]) for c in active_chunks]
        bm25_active = BM25Okapi(tokenized_active)
        bm25_scores = bm25_active.get_scores(query_tokens)
        bm25_ranking = sorted(range(len(active_chunks)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]

        # TF-IDF ranking (dense proxy)
        tfidf_scores = _tfidf_scores(query_tokens, active_chunks)
        tfidf_ranking = sorted(range(len(active_chunks)), key=lambda i: tfidf_scores[i], reverse=True)[:BM25_TOP_K]

        # RRF merge
        merged = reciprocal_rank_fusion([bm25_ranking, tfidf_ranking])

        # Return top_k chunks with scores
        results = []
        for local_idx, score in merged[:top_k]:
            chunk = dict(active_chunks[local_idx])
            chunk["retrieval_score"] = round(score, 6)
            results.append(chunk)

        return results

    def retrieve_multi(
        self,
        queries: list[str],
        top_k_per_query: int = 6,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Run multiple queries and merge results, deduplicating by chunk_id.
        Useful for agents that have several sub-questions.
        """
        seen_ids = set()
        all_results = []

        for query in queries:
            results = self.retrieve(query, top_k=top_k_per_query)
            for chunk in results:
                cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
                if deduplicate and cid in seen_ids:
                    continue
                seen_ids.add(cid)
                all_results.append(chunk)

        # Re-sort by retrieval score
        all_results.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)
        return all_results[:FINAL_TOP_K * 2]  # Return more since multi-query
