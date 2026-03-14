from typing import List

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """Rerank chunks using a cross-encoder. Returns top_k chunks sorted by score descending."""
    if not chunks:
        return []

    reranker = _get_reranker()

    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs).tolist()

    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = score

    ranked = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)
    return ranked[:top_k]
