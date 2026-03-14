"""
Evaluation for the RAG support assistant.

Phase 1 — Retrieval: hit rate @ 5 and MRR on WixQA ExpertWritten + Synthetic.
Phase 2 — Generation: LLM-as-judge faithfulness + relevancy on 50 expert questions.

Usage:
    uv run python eval/evaluate.py            # retrieval without reranking
    uv run python eval/evaluate.py --rerank   # retrieval with reranking (retrieve 20, rerank to 5)
"""

import argparse
import asyncio
import json
import os
import random
import sys

os.environ["SKIP_RATE_LIMIT"] = "true"

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

from src.retriever import retrieve
from src.reranker import rerank
from src.pipeline import run
from eval.metrics import hit_rate, reciprocal_rank, mean

load_dotenv()


# ── Judge agent (pydantic-ai) ──────────────────────────────────────────────


class JudgeScores(BaseModel):
    faithfulness: int
    relevancy: int


_JUDGE_SYSTEM = (
    "You are an evaluation assistant. Score the answer to the question based on the "
    "retrieved context.\n"
    "faithfulness: 5=answer is fully grounded in the context, "
    "1=answer contains claims not in the context at all.\n"
    "relevancy: 5=answer directly and completely answers the question, "
    "1=answer is off-topic or unhelpful.\n"
    "Both scores must be integers between 1 and 5."
)

_judge_agent: Agent | None = None


def _get_judge_agent() -> Agent:
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = Agent(
            "openai:gpt-4o-mini",
            output_type=JudgeScores,
            system_prompt=_JUDGE_SYSTEM,
        )
    return _judge_agent


# ── Retrieval helpers ──────────────────────────────────────────────────────


def _retrieve_chunks(question: str, use_rerank: bool) -> list:
    if use_rerank:
        candidates = retrieve(question, top_k=20)
        return rerank(question, candidates, top_k=5)
    return retrieve(question, top_k=5)


def _bottom5(rows: list, key: str) -> list:
    """Return the 5 rows with the lowest value for key, sorted ascending."""
    return sorted(rows, key=lambda r: r[key])[:5]


# Known worst-performing synthetic queries — always included in the sample
# so we can directly measure whether improvements fix these specific failures.
_KNOWN_WORST_SYNTHETIC = {
    "How can I manage inventory updates after creating purchase orders in the Wix dashboard?",
    "How can I display a Cookie Banner on my Wix site to comply with privacy laws like GDPR and LGPD?",
    "What steps can I take to ensure my Wix site complies with the CCPA regulations?",
    "How can I use the Wix app to send invites for an upcoming event I am planning?",
    "How can I set up installment payment options for my Wix Store using a payment provider?",
    "How do I add a Facebook Comments box to my Wix site?",
    "How do I customize the breadcrumbs layout on my Wix site?",
}


def _sample_rows(all_rows, n: int, pinned_questions: set, seed: int = 42) -> list:
    """Return up to n rows, always including any row whose question is in pinned_questions."""
    pinned = [r for r in all_rows if r["question"] in pinned_questions]
    remaining = [r for r in all_rows if r["question"] not in pinned_questions]
    n_random = max(0, n - len(pinned))
    sampled = random.Random(seed).sample(remaining, min(n_random, len(remaining)))
    return pinned + sampled


# ── Phase 1 — Retrieval evaluation ────────────────────────────────────────


def _eval_retrieval(
    dataset, split: str, use_rerank: bool, sample_n: int | None = None
) -> tuple[float, float, list]:
    """Returns (mean_hr, mean_mrr, per_row_list).

    If sample_n is given, evaluate only that many rows. For the synthetic dataset
    this pins the known worst queries and fills the remainder randomly (seed=42).
    """
    all_rows = list(dataset[split])
    if sample_n is not None and len(all_rows) > sample_n:
        rows = _sample_rows(all_rows, sample_n, _KNOWN_WORST_SYNTHETIC)
    else:
        rows = all_rows

    per_row = []
    for i, row in enumerate(rows, 1):
        question = row["question"]
        correct_ids = [str(aid) for aid in row["article_ids"]]
        chunks = _retrieve_chunks(question, use_rerank)
        retrieved_ids = [str(c["article_id"]) for c in chunks]
        hr = max(hit_rate(retrieved_ids, cid) for cid in correct_ids)
        rr = max(reciprocal_rank(retrieved_ids, cid) for cid in correct_ids)
        per_row.append({"question": question, "hit_rate": hr, "rr": rr})
        print(f"  retrieval [{i}/{len(rows)}]", end="\r", flush=True)
    print()
    return (
        mean([r["hit_rate"] for r in per_row]),
        mean([r["rr"] for r in per_row]),
        per_row,
    )


# ── Phase 2 — Generation evaluation (LLM-as-judge) ────────────────────────


async def _eval_generation(
    expert_rows, n: int = 50, seed: int = 42
) -> tuple[float, float, list]:
    """Returns (mean_faithfulness, mean_relevancy, per_row_list)."""
    sample = random.Random(seed).sample(list(expert_rows), n)
    per_row = []
    for i, row in enumerate(sample, 1):
        question = row["question"]
        try:
            result = await run(question)
            answer = result["answer"]
            chunks = result.get("chunks_used", [])
            context_blocks = "\n\n".join(
                f"[{j + 1}] {c['text']}" for j, c in enumerate(chunks)
            )
            user_msg = (
                f"Question: {question}\n\n"
                f"Retrieved context:\n{context_blocks}\n\n"
                f"Answer: {answer}"
            )
            scores = _get_judge_agent().run_sync(user_msg).output
            per_row.append(
                {
                    "question": question,
                    "faithfulness": scores.faithfulness,
                    "relevancy": scores.relevancy,
                }
            )
        except Exception as e:
            print(f"  [WARN] row {i} failed: {e}")
        print(f"  [{i}/{n}] done", end="\r", flush=True)
    print()
    return (
        mean([r["faithfulness"] for r in per_row]),
        mean([r["relevancy"] for r in per_row]),
        per_row,
    )


# ── Results formatting ─────────────────────────────────────────────────────


def _worst5_retrieval(per_row: list, metric_key: str) -> list[dict]:
    return [
        {"question": r["question"], "score": r[metric_key]}
        for r in _bottom5(per_row, metric_key)
    ]


def _worst5_generation(per_row: list, metric_key: str) -> list[dict]:
    return [
        {"question": r["question"], "score": r[metric_key]}
        for r in _bottom5(per_row, metric_key)
    ]


# ── Main ──────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking after retrieval (retrieve 20, rerank to 5)",
    )
    args = parser.parse_args()
    use_rerank = args.rerank

    print(f"Loading WixQA datasets... (reranking={'on' if use_rerank else 'off'})")
    expert_ds = load_dataset("Wix/WixQA", "wixqa_expertwritten")
    synthetic_ds = load_dataset("Wix/WixQA", "wixqa_synthetic")

    expert_split = list(expert_ds.keys())[0]
    synthetic_split = list(synthetic_ds.keys())[0]
    expert_rows = expert_ds[expert_split]
    synthetic_rows = synthetic_ds[synthetic_split]

    print(f"Expert rows: {len(expert_rows)}, Synthetic rows: {len(synthetic_rows)}")

    # --- Phase 1 ---
    print("\n[Phase 1] Retrieval — ExpertWritten (200 rows)...")
    expert_hr, expert_mrr, expert_per_row = _eval_retrieval(
        expert_ds, expert_split, use_rerank
    )

    SYNTHETIC_SAMPLE = 200
    print(
        f"\n[Phase 1] Retrieval — Synthetic (sample {SYNTHETIC_SAMPLE} of {len(synthetic_rows)})..."
    )
    synthetic_hr, synthetic_mrr, synthetic_per_row = _eval_retrieval(
        synthetic_ds, synthetic_split, use_rerank, sample_n=SYNTHETIC_SAMPLE
    )

    # --- Phase 2 ---
    print("\n[Phase 2] Generation — 50 expert questions (LLM-as-judge)...")
    mean_faithfulness, mean_relevancy, gen_per_row = await _eval_generation(
        expert_rows, n=50, seed=42
    )

    # --- Bottom-5 per metric ---
    worst = {
        "expert_hit_rate": _worst5_retrieval(expert_per_row, "hit_rate"),
        "expert_mrr": _worst5_retrieval(expert_per_row, "rr"),
        "synthetic_hit_rate": _worst5_retrieval(synthetic_per_row, "hit_rate"),
        "synthetic_mrr": _worst5_retrieval(synthetic_per_row, "rr"),
        "faithfulness": _worst5_generation(gen_per_row, "faithfulness"),
        "relevancy": _worst5_generation(gen_per_row, "relevancy"),
    }

    # --- Results dict ---
    results = {
        "reranking": use_rerank,
        "retrieval": {
            "expert": {
                "hit_rate_at_5": round(expert_hr, 4),
                "mrr": round(expert_mrr, 4),
            },
            "synthetic": {
                "hit_rate_at_5": round(synthetic_hr, 4),
                "mrr": round(synthetic_mrr, 4),
            },
        },
        "generation": {
            "mean_faithfulness": round(mean_faithfulness, 4),
            "mean_relevancy": round(mean_relevancy, 4),
            "n_questions": len(gen_per_row),
        },
        "worst5_queries": worst,
    }

    filename = (
        "results_baseline.json" if use_rerank else "results_baseline_no_rerank.json"
    )
    output_path = os.path.join(os.path.dirname(__file__), filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print table ---
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS  (reranking={'on' if use_rerank else 'off'})")
    print("=" * 60)
    print("Retrieval — ExpertWritten")
    print(f"  Hit Rate @ 5 : {expert_hr:.4f}")
    print(f"  MRR          : {expert_mrr:.4f}")
    print("Retrieval — Synthetic")
    print(f"  Hit Rate @ 5 : {synthetic_hr:.4f}")
    print(f"  MRR          : {synthetic_mrr:.4f}")
    print(f"Generation (n={len(gen_per_row)}, gpt-4o-mini judge)")
    print(f"  Faithfulness : {mean_faithfulness:.4f} / 5")
    print(f"  Relevancy    : {mean_relevancy:.4f} / 5")

    label_map = {
        "expert_hit_rate": "Expert Hit Rate",
        "expert_mrr": "Expert MRR",
        "synthetic_hit_rate": "Synthetic Hit Rate",
        "synthetic_mrr": "Synthetic MRR",
        "faithfulness": "Faithfulness",
        "relevancy": "Relevancy",
    }
    print("\n--- Worst 5 queries per metric ---")
    for key, entries in worst.items():
        print(f"\n{label_map[key]}:")
        for e in entries:
            print(f"  • [{e['score']}] {e['question'][:95]}")

    print("\n" + "=" * 60)
    print(f"Full results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
