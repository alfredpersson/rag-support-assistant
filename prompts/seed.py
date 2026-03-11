"""
Register all prompts with Langfuse.

Run once (or whenever a prompt file changes) to publish a new version:
    uv run python prompts/seed.py

Each call creates a new prompt version in Langfuse and labels it "production",
making it the active version returned by get_prompt().
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

PROMPTS_DIR = Path(__file__).parent

PROMPTS = [
    ("wix-classifier",     "classifier.txt",     {"model": "gpt-4o-mini", "temperature": 0}),
    ("wix-query-rewriter", "query_rewriter.txt",  {"model": "gpt-4o-mini", "temperature": 0}),
    ("wix-generator",      "generator.txt",       {"model": "gpt-4o-mini", "temperature": 0}),
    ("wix-self-critique",  "self_critique.txt",   {"model": "gpt-4o-mini", "temperature": 0}),
    ("wix-followup",       "followup.txt",        {"model": "gpt-4o-mini", "temperature": 0}),
    ("wix-high-stakes",    "high_stakes.txt",     {"model": "gpt-4o-mini", "temperature": 0}),
]


def seed():
    lf = Langfuse()

    for name, filename, config in PROMPTS:
        text = (PROMPTS_DIR / filename).read_text().strip()
        lf.create_prompt(
            name=name,
            prompt=text,
            config=config,
            labels=["production"],
        )
        print(f"✓ registered  {name}  ({filename})")

    lf.flush()
    print("\nAll prompts registered.")


if __name__ == "__main__":
    seed()
