import os
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "wix_kb"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

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
        _collection = client.get_collection(COLLECTION_NAME)
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
