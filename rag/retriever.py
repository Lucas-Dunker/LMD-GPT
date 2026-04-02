"""Retrieve relevant chunks from the vector store."""
from typing import Optional

from embeddings.store import VectorStore
from config import TOP_K_RESULTS

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
) -> list[dict]:
    return get_store().query(query, top_k=top_k, source_filter=source_filter)


def format_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        source = meta.get("source", "unknown")
        score = hit.get("score", 0)
        label = {
            "obsidian": f"Note: {meta.get('file', '')}",
            "discord":  f"Discord — {meta.get('guild', '')} / #{meta.get('channel', '')}",
            "gdocs":    f"Doc: {meta.get('file', '')}",
            "gcal":     f"Calendar: {meta.get('summary', '')} on {meta.get('start', '')}",
        }.get(source, source)
        parts.append(f"[{i}] (source: {source}, relevance: {score:.2f}) {label}\n{hit['text']}")
    return "\n\n".join(parts)
