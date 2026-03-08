import os
from typing import List, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

SYSTEM_PROMPT = (
    "You are a helpful Wix support assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Do not make up information."
)

FAITHFULNESS_PROMPT = (
    "Does the following answer contain only information present in the source chunks? "
    "Reply with exactly YES or NO."
)

FAITHFULNESS_CAVEAT = (
    "\n\nNote: I'm not fully certain — please verify on the Wix Help Center."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _check_faithfulness(answer: str, context_blocks: str) -> bool:
    """Returns True if answer is grounded in context, False otherwise."""
    client = _get_client()
    check_message = (
        f"Source chunks:\n{context_blocks}\n\nAnswer:\n{answer}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=5,
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": check_message},
        ],
    )
    verdict = response.choices[0].message.content.strip().upper()
    faithful = verdict.startswith("YES")
    print(f"[faithfulness] verdict={verdict!r} faithful={faithful}")
    return faithful


def generate(query: str, chunks: List[dict]) -> Tuple[str, List[str]]:
    context_blocks = "\n\n".join(
        f"[{i + 1}] (Source: {c['article_title']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    user_message = f"Question: {query}\n\nContext:\n{context_blocks}"

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer = response.choices[0].message.content.strip()

    if not _check_faithfulness(answer, context_blocks):
        answer += FAITHFULNESS_CAVEAT

    # Deduplicate sources while preserving order
    seen = set()
    sources = []
    for c in chunks:
        title = c["article_title"]
        if title and title not in seen:
            seen.add(title)
            sources.append(title)

    return answer, sources
