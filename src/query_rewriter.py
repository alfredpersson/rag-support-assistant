"""
Rewrite short queries (<8 words) into fuller, retrieval-optimised questions.
Uses pydantic-ai with a structured output model.
"""

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

from src.config import LLM_TIMEOUT
from src.prompts import get_prompt, generation_context

load_dotenv()

_FALLBACK = (Path(__file__).parent.parent / "prompts" / "query_rewriter.txt").read_text().strip()


class RewriteResult(BaseModel):
    rewritten_query: str


_agent: Optional[Agent] = None
_prompt_client = None


def _get_agent() -> Agent:
    global _agent, _prompt_client
    if _agent is None:
        _prompt_client, prompt_text = get_prompt("wix-query-rewriter", _FALLBACK)
        _agent = Agent(
            "openai:gpt-4o-mini",
            output_type=RewriteResult,
            system_prompt=prompt_text,
        )
    return _agent


async def maybe_rewrite(query: str) -> str:
    """Rewrite query if it is fewer than 8 words, otherwise return as-is."""
    if len(query.split()) >= 8:
        return query

    logger = logging.getLogger(__name__)
    logger.info("Rewriting short query: %r", query)
    agent = _get_agent()
    with generation_context("wix-query-rewriter", _prompt_client):
        result = await agent.run(query, model_settings={"timeout": LLM_TIMEOUT})
    rewritten = result.output.rewritten_query
    logger.info("Rewritten to: %r", rewritten)
    return rewritten
