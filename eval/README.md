# Evaluation

## What each metric measures

| Metric | What it measures | Why it was chosen |
|---|---|---|
| Hit Rate @ 5 | Fraction of questions where the correct article appears in the top-5 retrieved chunks | Upper bound on generator quality — the generator can only answer correctly if the right context reaches it |
| MRR (Mean Reciprocal Rank) | Average of 1/rank for the correct article | Rewards retrieving the right article near the top, not just anywhere in the list |
| Faithfulness (1–5) | Whether the generated answer makes only claims supported by the retrieved context | Detects hallucination — a low score means the generator is inventing information not present in the chunks |
| Relevancy (1–5) | Whether the answer directly addresses what was asked | Measures answer quality independent of factual correctness |

## Why retrieval and generation are evaluated separately

Splitting the pipeline into two phases makes diagnosis possible:

- **High hit rate + low faithfulness** → the right documents are being retrieved, but the generator is hallucinating. The fix is in the prompt or the model, not the retriever.
- **Low hit rate** → no amount of prompt engineering will help, because the correct information never reaches the generator. The fix is in the embedding model, chunking strategy, or reranker.
- **Low relevancy despite high faithfulness** → the answer is grounded in the retrieved context but the context itself is off-topic. This points to a classifier or query-rewriting problem.

Conflating these phases into a single end-to-end score hides where the failure is occurring.

## Known limitation of LLM-as-judge scoring

GPT-4o-mini (and LLMs in general) tend to favour answers that are longer, more confident-sounding, and more formally structured — even when a shorter answer is equally correct. This means the automated scores can be systematically biased upward for verbose responses.

In a production system, automated LLM-as-judge scores should be calibrated against a set of human-annotated examples. Running periodic human spot-checks on a random sample of 20–30 answers lets you detect and correct score drift over time.
