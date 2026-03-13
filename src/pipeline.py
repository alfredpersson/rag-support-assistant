from langfuse import observe

from src.classifier import classify, QueryCategory
from src.rate_limit import check_and_increment
from src.query_rewriter import maybe_rewrite
from src.retriever import retrieve
from src.reranker import rerank
from src.generator import generate, generate_followup, generate_high_stakes

RELEVANCE_SCORE_THRESHOLD = 2.0

_SUGGESTIONS = (
    "Here are some things I can help you with:\n"
    "- Connecting or managing a custom domain\n"
    "- Setting up or editing your Wix site\n"
    "- Managing products in your online store\n"
    "- Billing, plans, and upgrades\n"
    "- Wix apps and integrations"
)

_NONSENSE_RESPONSE = (
    "I didn't quite catch that — it doesn't look like a question I can help with.\n\n"
    + _SUGGESTIONS
)

_IRRELEVANT_RESPONSE = (
    "That topic is outside what I can help with — I'm a Wix support assistant "
    "and can only answer questions about Wix products and features.\n\n" + _SUGGESTIONS
)

_OUT_OF_SCOPE_RESPONSE = (
    "That's outside what I'm able to handle "
    "directly — it may require access to your account or involve a topic managed "
    "by our specialist teams.\n\n"
    "Would you like me to connect you with a support agent who can help?"
)


@observe(name="wix-pipeline")
def run(question: str) -> dict:
    check_and_increment()

    classification = classify(question)
    category = classification.category

    # ── Category 2: Nonsense ───────────────────────────────────
    if category == QueryCategory.NONSENSE:
        return {
            "answer": _NONSENSE_RESPONSE,
            "sources": [],
            "chunks_used": [],
            "routing": "nonsense",
        }

    # ── Category 3: Irrelevant ─────────────────────────────────
    if category == QueryCategory.IRRELEVANT:
        return {
            "answer": _IRRELEVANT_RESPONSE,
            "sources": [],
            "chunks_used": [],
            "routing": "irrelevant",
        }

    # ── Category 4: Out of scope ───────────────────────────────
    if category == QueryCategory.OUT_OF_SCOPE:
        return {
            "answer": _OUT_OF_SCOPE_RESPONSE,
            "sources": [],
            "chunks_used": [],
            "routing": "out_of_scope",
        }

    # ── Categories 1 & 5: retrieve first ──────────────────────
    rewritten = maybe_rewrite(question)
    candidates = retrieve(rewritten, top_k=20)
    chunks = rerank(rewritten, candidates, top_k=5)
    has_results = (
        bool(chunks) and chunks[0]["reranker_score"] >= RELEVANCE_SCORE_THRESHOLD
    )

    # ── Category 5: High-stakes ────────────────────────────────
    if category == QueryCategory.HIGH_STAKES:
        answer, sources = generate_high_stakes(question, chunks if has_results else [])
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": chunks,
            "routing": "high_stakes",
        }

    # ── Category 1: Answerable ─────────────────────────────────
    if not has_results:
        followup = generate_followup(question)
        return {
            "answer": followup,
            "sources": [],
            "chunks_used": chunks,
            "routing": "followup",
        }

    answer, sources, routing_key = generate(question, chunks)
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks,
        "routing": routing_key,
    }
