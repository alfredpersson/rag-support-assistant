"""
Classify an incoming user query into one of five routing categories before
any retrieval happens.

Categories
----------
1  ANSWERABLE   – within the assistant's scope; proceed to retrieval even if ambiguous
2  NONSENSE     – no discernible intent (random text, gibberish, empty meaning)
3  IRRELEVANT   – clear query but off-topic or about another company/product
4  OUT_OF_SCOPE – clearly Wix-related but beyond the assistant's knowledge domain
5  HIGH_STAKES  – cancellation, deletion, billing dispute, or a real user complaint
"""

from enum import IntEnum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

from src.prompts import get_prompt, generation_context

load_dotenv()

_FALLBACK = (Path(__file__).parent.parent / "prompts" / "classifier.txt").read_text().strip()


class QueryCategory(IntEnum):
    ANSWERABLE = 1
    NONSENSE = 2
    IRRELEVANT = 3
    OUT_OF_SCOPE = 4
    HIGH_STAKES = 5


class ClassificationResult(BaseModel):
    category: QueryCategory
    reasoning: str


_agent: Optional[Agent] = None
_prompt_client = None


def _get_agent() -> Agent:
    global _agent, _prompt_client
    if _agent is None:
        _prompt_client, prompt_text = get_prompt("wix-classifier", _FALLBACK)
        _agent = Agent(
            "openai:gpt-4o-mini",
            output_type=ClassificationResult,
            system_prompt=prompt_text,
        )
    return _agent


def classify(query: str) -> ClassificationResult:
    agent = _get_agent()
    with generation_context("wix-classifier", _prompt_client):
        result = agent.run_sync(query)
    classification = result.output
    print(
        f"[classifier] category={classification.category.name} "
        f"reasoning={classification.reasoning!r}"
    )
    return classification

