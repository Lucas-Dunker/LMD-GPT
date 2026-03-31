"""Split documents into overlapping word-count chunks."""
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks by word count, respecting paragraph boundaries.
    Falls back to hard word-count splits for very long paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_words: list[str] = []

    def flush():
        if current_words:
            chunks.append(" ".join(current_words))

    for para in paragraphs:
        para_words = para.split()

        # If a single paragraph exceeds chunk_size, hard-split it
        if len(para_words) > chunk_size:
            flush()
            current_words = []
            for i in range(0, len(para_words), chunk_size - overlap):
                slice_ = para_words[i: i + chunk_size]
                chunks.append(" ".join(slice_))
            # Carry overlap into next chunk
            current_words = para_words[-overlap:] if overlap else []
            continue

        if len(current_words) + len(para_words) > chunk_size:
            flush()
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(para_words)

    flush()
    return chunks


def chunk_document(doc: dict, **kwargs) -> list[dict]:
    """Chunk a document dict, carrying metadata through to each chunk."""
    return [
        {
            "text": chunk,
            "metadata": {**doc["metadata"], "chunk_index": i},
        }
        for i, chunk in enumerate(chunk_text(doc["text"], **kwargs))
    ]
