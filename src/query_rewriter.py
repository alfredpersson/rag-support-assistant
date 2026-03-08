import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

REWRITE_PROMPT = (
    "You are a search query optimizer for a Wix Help Center assistant. "
    "Rewrite the user's short query into a fuller, retrieval-optimised question "
    "that includes relevant Wix context. Output only the rewritten query, nothing else."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def maybe_rewrite(query: str) -> str:
    """Rewrite query if it is fewer than 8 words, otherwise return as-is."""
    if len(query.split()) >= 8:
        return query

    print(f"[query_rewriter] Rewriting short query: {query!r}")
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=60,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"[query_rewriter] Rewritten to: {rewritten!r}")
    return rewritten
