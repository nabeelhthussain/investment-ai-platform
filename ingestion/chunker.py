"""
Document chunker.

Strategy: section-aware chunking. We first try to split on natural section
boundaries (headings, Item numbers in SEC filings), then fall back to
fixed-size word-count windows with overlap.

Each chunk carries full provenance metadata so every retrieval result
is immediately traceable to a source.
"""
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


# SEC filing section patterns
SEC_SECTION_RE = re.compile(
    r'^(ITEM\s+\d+[A-Z]?\.?\s+[A-Z][A-Z\s,;]+|'
    r'PART\s+[IVX]+\.?\s+[A-Z][A-Z\s,;]+)',
    re.MULTILINE
)

# General heading patterns (markdown-style and all-caps)
HEADING_RE = re.compile(
    r'^(#{1,4}\s+.+|[A-Z][A-Z\s\-:]{10,60})$',
    re.MULTILINE
)


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-count windows."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + size]
        chunks.append(" ".join(chunk_words))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 100]


def _section_split(text: str) -> list[tuple[str, str]]:
    """
    Split text on section boundaries. Returns list of (section_title, section_text).
    """
    # Try SEC Item pattern first
    splits = SEC_SECTION_RE.split(text)
    if len(splits) > 3:
        sections = []
        i = 0
        while i < len(splits):
            if SEC_SECTION_RE.match(splits[i].strip()):
                title = splits[i].strip()
                body = splits[i + 1] if i + 1 < len(splits) else ""
                sections.append((title, body))
                i += 2
            else:
                if splits[i].strip():
                    sections.append(("Introduction", splits[i]))
                i += 1
        return sections

    # Fall back to general heading split
    splits = HEADING_RE.split(text)
    if len(splits) > 3:
        sections = []
        i = 0
        while i < len(splits):
            if HEADING_RE.match(splits[i].strip()):
                title = splits[i].strip()
                body = splits[i + 1] if i + 1 < len(splits) else ""
                sections.append((title, body))
                i += 2
            else:
                if splits[i].strip():
                    sections.append(("Document", splits[i]))
                i += 1
        return sections

    # No structure detected
    return [("Document", text)]


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a document dict into retrievable pieces.
    Each chunk inherits all metadata from the parent doc.

    Returns list of chunk dicts, each with:
      - text: chunk content
      - chunk_id: unique identifier
      - section: section title if detected
      - All parent metadata (ticker, doc_type, date, source, etc.)
    """
    text = doc.get("text", "")
    if not text or len(text.strip()) < 100:
        return []

    # Skip failed fetches
    if doc.get("fetch_failed"):
        return []

    base_meta = {k: v for k, v in doc.items() if k not in ("text",)}
    sections = _section_split(text)

    chunks = []
    chunk_index = 0

    for section_title, section_body in sections:
        if not section_body.strip():
            continue

        word_count = len(section_body.split())

        if word_count <= chunk_size:
            # Section fits in one chunk
            if len(section_body.strip()) > 100:
                chunk_id = f"{doc['ticker']}_{doc['doc_type']}_{doc.get('date','nd')}_{chunk_index:04d}"
                chunks.append({
                    **base_meta,
                    "text": section_body.strip(),
                    "section": section_title,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "word_count": word_count,
                })
                chunk_index += 1
        else:
            # Section too long — split into windows
            windows = _word_chunks(section_body, chunk_size, overlap)
            for window in windows:
                chunk_id = f"{doc['ticker']}_{doc['doc_type']}_{doc.get('date','nd')}_{chunk_index:04d}"
                chunks.append({
                    **base_meta,
                    "text": window,
                    "section": section_title,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "word_count": len(window.split()),
                })
                chunk_index += 1

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """Chunk all documents in a corpus."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  Chunked: {doc['doc_type']} ({doc.get('date','?')}) → {len(chunks)} chunks")
    return all_chunks
