"""ChromaDB vector store — embed, persist, and query documents."""
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
import ollama

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K_RESULTS
from embeddings.chunker import chunk_document

_COLLECTION_NAME = "lmd_gpt"
_BATCH_SIZE = 50  # embed + upsert N chunks at a time


def _embed(text: str) -> list[float]:
    return ollama.embed(model=EMBEDDING_MODEL, input=text).embeddings[0]


def _doc_id(text: str, metadata: dict) -> str:
    key = f"{metadata.get('source','')}|{metadata.get('file', metadata.get('message_id',''))}|{metadata.get('chunk_index','')}|{text[:120]}"
    return hashlib.md5(key.encode()).hexdigest()


def _stringify_meta(meta: dict) -> dict:
    """Chroma requires all metadata values to be str/int/float/bool."""
    result = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif isinstance(v, list):
            result[k] = ", ".join(str(i) for i in v)
        else:
            result[k] = str(v)
    return result


class VectorStore:
    def __init__(self, persist_dir: Path = VECTORSTORE_DIR):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(
        self,
        docs: list[dict],
        chunk: bool = True,
        source: Optional[str] = None,
    ) -> int:
        """Embed and upsert documents. Returns number of chunks stored."""
        all_chunks: list[dict] = []
        for doc in docs:
            if source:
                doc = {**doc, "metadata": {**doc["metadata"], "source": source}}
            all_chunks.extend(chunk_document(doc) if chunk else [doc])

        if not all_chunks:
            return 0

        total = 0
        for i in range(0, len(all_chunks), _BATCH_SIZE):
            batch = all_chunks[i: i + _BATCH_SIZE]
            ids, texts, metas, embeddings = [], [], [], []
            for c in batch:
                text = c["text"]
                meta = _stringify_meta(c["metadata"])
                ids.append(_doc_id(text, meta))
                texts.append(text)
                metas.append(meta)
                embeddings.append(_embed(text))

            self._col.upsert(
                ids=ids,
                documents=texts,
                metadatas=metas,
                embeddings=embeddings,
            )
            total += len(batch)
            print(f"  stored {total}/{len(all_chunks)} chunks...", end="\r")

        print()
        return total

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Return the top_k most similar chunks to query_text."""
        q_embedding = _embed(query_text)
        where = {"source": source_filter} if source_filter else None

        results = self._col.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {"text": doc, "metadata": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
