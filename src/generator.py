"""
Generate answers from retrieved chunks and handle specialist response paths.
All LLM calls use pydantic-ai with structured output models.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

from src.config import LLM_TIMEOUT
from src.prompts import get_prompt, generation_context

load_dotenv()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Minimum reranker score for a source to be eligible for linking.
# Cross-encoder scores below this are considered too uncertain to cite.
# Must be > RELEVANCE_SCORE_THRESHOLD (pipeline.py, currently 2.0) to keep
# the low_confidence path reachable.
CONFIDENCE_THRESHOLD = 5.0



def _fallback(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text().strip()


# ── Pydantic output models ─────────────────────────────────────


class GeneratorOutput(BaseModel):
    answer: str


class SelfCritiqueOutput(BaseModel):
    assessment: Literal["FULLY_ANSWERED", "PARTIALLY_ANSWERED", "CANNOT_ANSWER"]
    reasoning: str


class FollowUpOutput(BaseModel):
    follow_up_question: str


class HighStakesOutput(BaseModel):
    emotional_acknowledgment: str
    context_summary: Optional[str]
    retention_offer: Optional[str]
    closing: str


# ── Lazy agent state ───────────────────────────────────────────

_generator_agent: Optional[Agent] = None
_self_critique_agent: Optional[Agent] = None
_followup_agent: Optional[Agent] = None
_high_stakes_agent: Optional[Agent] = None

_generator_prompt = None
_self_critique_prompt = None
_followup_prompt = None
_high_stakes_prompt = None


def _get_generator_agent() -> Agent:
    global _generator_agent, _generator_prompt
    if _generator_agent is None:
        _generator_prompt, text = get_prompt(
            "wix-generator", _fallback("generator.txt")
        )
        _generator_agent = Agent(
            "openai:gpt-4o-mini", output_type=GeneratorOutput, system_prompt=text
        )
    return _generator_agent


def _get_self_critique_agent() -> Agent:
    global _self_critique_agent, _self_critique_prompt
    if _self_critique_agent is None:
        _self_critique_prompt, text = get_prompt(
            "wix-self-critique", _fallback("self_critique.txt")
        )
        _self_critique_agent = Agent(
            "openai:gpt-4o-mini", output_type=SelfCritiqueOutput, system_prompt=text
        )
    return _self_critique_agent


def _get_followup_agent() -> Agent:
    global _followup_agent, _followup_prompt
    if _followup_agent is None:
        _followup_prompt, text = get_prompt("wix-followup", _fallback("followup.txt"))
        _followup_agent = Agent(
            "openai:gpt-4o-mini", output_type=FollowUpOutput, system_prompt=text
        )
    return _followup_agent


def _get_high_stakes_agent() -> Agent:
    global _high_stakes_agent, _high_stakes_prompt
    if _high_stakes_agent is None:
        _high_stakes_prompt, text = get_prompt(
            "wix-high-stakes", _fallback("high_stakes.txt")
        )
        _high_stakes_agent = Agent(
            "openai:gpt-4o-mini", output_type=HighStakesOutput, system_prompt=text
        )
    return _high_stakes_agent


# ── Helpers ────────────────────────────────────────────────────


def _build_context(chunks: List[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (Source: {c['article_title']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )


def _dedupe_chunks_by_source(chunks: List[dict]) -> List[dict]:
    """One entry per article_title, keeping the highest-scoring chunk (chunks arrive pre-sorted)."""
    seen: set = set()
    result: List[dict] = []
    for c in chunks:
        title = c.get("article_title", "")
        if title and title not in seen:
            seen.add(title)
            result.append(c)
    return result


# ── Public API ─────────────────────────────────────────────────


async def generate(query: str, chunks: List[dict]) -> Tuple[str, List[str], str]:
    """
    Generate an answer, run self-critique, and decide which sources to cite.
    Returns (answer, sources, routing_key).

    routing_key values:
      "answered"      – normal answered response
      "cannot_answer" – model could not answer; frontend shows connect-agent button
      "low_confidence"– retrieval scores too low; no links, ask to clarify
    """
    context_blocks = _build_context(chunks)

    # ── 1. Generate answer ─────────────────────────────────────
    with generation_context("wix-generator", _generator_prompt):
        gen_result = await _get_generator_agent().run(
            query,
            instructions=f"Context:\n{context_blocks}",
            model_settings={"timeout": LLM_TIMEOUT},
        )
    answer = gen_result.output.answer

    # ── 2. Deduplicate chunks by source ────────────────────────
    deduped = _dedupe_chunks_by_source(chunks)

    # ── 3. Confidence gate ─────────────────────────────────────
    top_score = deduped[0]["reranker_score"] if deduped else 0.0
    if top_score < CONFIDENCE_THRESHOLD:
        logging.getLogger(__name__).info("Low confidence (top_score=%.2f); no sources", top_score)
        return answer, [], "low_confidence"

    # ── 4. Self-critique ───────────────────────────────────────
    critique_message = (
        f"User question: {query}\n\nContext:\n{context_blocks}\n\nAnswer:\n{answer}"
    )
    with generation_context("wix-self-critique", _self_critique_prompt):
        critique_result = await _get_self_critique_agent().run(
            critique_message, model_settings={"timeout": LLM_TIMEOUT}
        )
    assessment = critique_result.output.assessment
    logging.getLogger(__name__).info(
        "assessment=%s reasoning=%r", assessment, critique_result.output.reasoning
    )

    # ── 5. Source link logic ───────────────────────────────────
    confident_sources = [
        c
        for c in deduped
        if c.get("article_title") and c["reranker_score"] >= CONFIDENCE_THRESHOLD
    ]

    if assessment == "CANNOT_ANSWER":
        cannot_answer_msg = (
            "I wasn't able to find a clear answer to your question in our help content.\n\n"
            "Would you like me to connect you with a support agent who can help?"
        )
        return cannot_answer_msg, [], "cannot_answer"

    if assessment == "PARTIALLY_ANSWERED":
        # 1 link – top confident source, as a starting point to fill the gap
        sources = [confident_sources[0]["article_title"]] if confident_sources else []
        return answer, sources, "partially_answered"

    # FULLY_ANSWERED — up to 2 source links ("read more" for the user)
    sources = [c["article_title"] for c in confident_sources[:2]]
    return answer, sources, "answered"


async def generate_followup(query: str) -> str:
    """Return a clarifying follow-up question when retrieval found nothing useful."""
    with generation_context("wix-followup", _followup_prompt):
        result = await _get_followup_agent().run(
            f"User question: {query}", model_settings={"timeout": LLM_TIMEOUT}
        )
    followup = result.output.follow_up_question
    logging.getLogger(__name__).info("followup=%r", followup)
    return followup


async def generate_high_stakes(query: str, chunks: List[dict]) -> Tuple[str, List[str]]:
    """Empathetic response for cancellations, disputes, and complaints."""
    context_blocks = _build_context(chunks) if chunks else ""
    with generation_context("wix-high-stakes", _high_stakes_prompt):
        result = await _get_high_stakes_agent().run(
            query,
            instructions=(
                f"Relevant context chunks:\n{context_blocks}"
                if context_blocks
                else "No relevant context found."
            ),
            model_settings={"timeout": LLM_TIMEOUT},
        )
    out = result.output

    parts = [out.emotional_acknowledgment]
    if out.retention_offer:
        parts.append(out.retention_offer)
    if out.context_summary:
        parts.append(out.context_summary)
    parts.append(out.closing)

    answer = "\n\n".join(parts)
    logging.getLogger(__name__).info(
        "retention_offer=%s context_summary=%s", bool(out.retention_offer), bool(out.context_summary)
    )

    return answer, []  # high-stakes never cites sources
