"""
Prompt management via Langfuse.

Usage in each LLM module:
    from src.prompts import get_prompt, generation_context

    _prompt_client, prompt_text = get_prompt("wix-classifier", FALLBACK_TEXT)

    # Wrap each agent.run_sync() call with a named generation observation so
    # Langfuse links the prompt to that observation:
    with generation_context("wix-classifier", _prompt_client):
        result = agent.run_sync(user_message)
"""

from typing import Tuple

from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from pydantic_ai import Agent

load_dotenv()

# Initialise the Langfuse client (and its OTel exporter) eagerly so the
# exporter is registered before Agent.instrument_all() hooks into OTel.
# Without this ordering, pydantic-ai spans are emitted to OTel but never
# forwarded to Langfuse.
_lf: Langfuse = get_client()

Agent.instrument_all()


def _get_lf() -> Langfuse:
    return _lf


def get_prompt(name: str, fallback: str) -> Tuple:
    """
    Fetch the production-labelled prompt from Langfuse.
    Returns (prompt_client, compiled_text).
    Falls back to (None, fallback) if Langfuse is unreachable.
    """
    try:
        client = _get_lf().get_prompt(name, label="production", fallback=fallback)
        return client, client.compile()
    except Exception as exc:
        print(
            f"[prompts] could not fetch {name!r} from Langfuse ({exc}); using fallback"
        )
        return None, fallback


def generation_context(name: str, prompt_client):
    """
    Context manager that creates a Langfuse generation observation linked to
    the given prompt. Wrap each agent.run_sync() call with this so Langfuse
    counts it as an observation of that prompt version.

    Usage:
        with generation_context("wix-classifier", _prompt_client):
            result = agent.run_sync(query)
    """
    return _get_lf().start_as_current_generation(name=name, prompt=prompt_client)
