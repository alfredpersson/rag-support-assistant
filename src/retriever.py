from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH

_model: SentenceTransformer | None = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except ValueError:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found. "
                f"Run 'uv run python src/ingest.py' before starting the API."
            )
    return _collection


def retrieve(query: str, top_k: int = 5) -> List[dict]:
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "article_title": meta.get("article_title", ""),
            "article_id": meta.get("article_id", ""),
            "distance": dist,
        })

    return chunks
