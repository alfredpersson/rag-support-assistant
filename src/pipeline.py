from src.rate_limit import check_and_increment
from src.query_rewriter import maybe_rewrite
from src.retriever import retrieve
from src.reranker import rerank
from src.generator import generate

RELEVANCE_SCORE_THRESHOLD = 0.0  # cross-encoder scores below this → off-topic


def run(question: str) -> dict:
    check_and_increment()

    rewritten = maybe_rewrite(question)

    candidates = retrieve(rewritten, top_k=20)
    chunks = rerank(rewritten, candidates, top_k=5)

    if not chunks or chunks[0]["reranker_score"] < RELEVANCE_SCORE_THRESHOLD:
        return {
            "answer": "I couldn't find relevant information to answer your question. Please check the Wix Help Center directly.",
            "sources": [],
            "chunks_used": [],
        }

    answer, sources = generate(question, chunks)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks,
    }
