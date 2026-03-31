"""RAG generation chain — retrieve context then stream from Ollama."""
from typing import Generator, Optional

import ollama

from rag.retriever import retrieve, format_context
from config import INFERENCE_MODEL, TOP_K_RESULTS

_SYSTEM = """\
You are a personal AI assistant with deep knowledge of a specific person — their \
thoughts, schedule, projects, relationships, and communication style — drawn from \
their private notes, calendar, documents, and messages. Answer questions about this \
person accurately and specifically using the context below. Refer to the person in \
first person (I/me) as if you are them. If the context doesn't contain enough \
information, say so clearly rather than guessing.\
"""


def _build_messages(
    query: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    if context:
        user_msg = f"Context from my personal data:\n\n{context}\n\nQuestion: {query}"
    else:
        user_msg = query
    return [
        {"role": "system", "content": _SYSTEM},
        *history,
        {"role": "user", "content": user_msg},
    ]


def stream_chat(
    query: str,
    top_k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> Generator[str, None, None]:
    """Stream a RAG response token by token."""
    hits = retrieve(query, top_k=top_k, source_filter=source_filter)
    context = format_context(hits)
    messages = _build_messages(query, context, history or [])

    for chunk in ollama.chat(model=INFERENCE_MODEL, messages=messages, stream=True):
        yield chunk.message.content or ""
