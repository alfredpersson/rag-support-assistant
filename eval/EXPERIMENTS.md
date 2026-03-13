# Evaluation Experiments

Seven experiments across retrieval strategy, query expansion, classifier tuning, and chunk enrichment. The final configuration — **title-prepended embeddings + cross-encoder reranking** — is the best-performing setup.

## Results summary

| # | Experiment | Expert HR@5 | Expert MRR | Faithfulness | Relevancy | Outcome |
|---|---|---|---|---|---|---|
| 1 | Baseline (no reranking) | 0.67 | 0.477 | 4.76 | 4.58 | Starting point |
| 2 | + Cross-encoder reranking | 0.71 | 0.498 | 4.84 | 4.60 | Consistent gains |
| 3 | + HyDE query expansion | 0.63 | 0.438 | 4.76 | 4.58 | **Reverted** — regression |
| 4 | + Classifier fix for Wix-adjacent services | 0.71 | 0.498 | 4.74 | 4.66 | Relevancy improved |
| 5 | + Larger candidate pool (top_k=50) | 0.69 | 0.482 | 4.68 | 4.62 | **Reverted** — regression |
| 6 | + Title-prepended embeddings | **0.71** | **0.514** | **4.80** | **4.76** | **Best config** |
| 7 | + Always-on query expansion | 0.605 | 0.428 | 4.78 | 4.68 | **Reverted** — regression |

Retrieval metrics use all 200 expert-written questions. Generation metrics use LLM-as-judge (gpt-4o-mini) on 50 randomly sampled expert questions (seed=42). See [eval methodology](README.md) for metric definitions.

---

## Setup

The evaluation uses the [WixQA benchmark](https://huggingface.co/datasets/Wix/WixQA) with two question sets:

| Dataset | Rows | Description |
|---|---|---|
| ExpertWritten | 200 | Human-written questions with natural phrasing |
| Synthetic | 6,221 | LLM-generated questions from KB articles |

Ground truth is `article_ids` — the articles that answer each question. Retrieval and generation are evaluated separately so failures can be diagnosed (see [eval methodology](README.md)).

```bash
uv run python eval/evaluate.py            # without reranking
uv run python eval/evaluate.py --rerank   # with cross-encoder reranking
```

---

## Experiment 1 — Baseline (no reranking)

**Config:** Retrieve top-5 by cosine similarity, no query rewriting, no reranking.

| Metric | Score |
|---|---|
| Expert HR@5 | 0.67 |
| Expert MRR | 0.477 |
| Synthetic HR@5 | 0.945 |
| Synthetic MRR | 0.861 |
| Faithfulness | 4.76 / 5 |
| Relevancy | 4.58 / 5 |

The 28-point gap between expert (0.67) and synthetic (0.945) hit rate is the key diagnostic. Synthetic questions are generated from the articles, so their vocabulary closely matches the KB. Expert questions use natural phrasing that the embedding model can't always bridge. This gap is almost entirely **vocabulary mismatch**, not missing KB coverage.

The worst generation scores (faithfulness 1, relevancy 1) came from queries about Google Ads pricing, Google Workspace email costs, and third-party POS solutions — the classifier routed these to IRRELEVANT, producing static deflection responses. This is a **classifier problem**, not a retrieval problem.

---

## Experiment 2 — Cross-encoder reranking

**Config:** Retrieve top-20, rerank with `ms-marco-MiniLM-L6-v2` cross-encoder, keep top-5.

| Metric | Δ vs baseline |
|---|---|
| Expert HR@5 | +0.04 → 0.71 |
| Expert MRR | +0.02 → 0.498 |
| Synthetic MRR | +0.05 → 0.909 |
| Faithfulness | +0.08 → 4.84 |

Reranking improved every metric. The largest gain was synthetic MRR (+0.05) — the cross-encoder promotes the correct article higher in the ranked list. The retrieval failures from Experiment 1 persisted: the worst-case expert queries still scored zero because the correct documents never appeared in the top-20 candidates. The reranker can't rescue documents that retrieval never surfaces.

---

## Root cause analysis

After the first two experiments, I analysed why the expert hit rate plateaued at 0.71. Five root causes, ranked by impact:

**1. Vocabulary mismatch at ingest.** Chunks were embedded from raw body text without article titles. A chunk from "Wix Hotels: Syncing with External Calendars" that says "Export your calendar link from the app settings" has no mention of "hotel" or "iCal" in the embedded text. The query "sync hotel app with iCal" lands nowhere near it in embedding space.

**2. Classifier misroutes Wix-adjacent services.** Google Workspace, Google Ads, and third-party POS integrations are sold/integrated through Wix, but the classifier prompt didn't define them as ANSWERABLE. Routing them to IRRELEVANT bypassed retrieval entirely — the direct cause of the relevancy [1] scores.

**3. Query rewriter only fires on short queries.** The 8-word threshold meant conversational queries like "I have not received my payment in my bank account yet" (14 words, zero hit rate) weren't expanded, even though they needed vocabulary bridging.

**4. Embedding model is small and general-purpose.** `all-MiniLM-L6-v2` (22M parameters) lacks domain-specific semantic depth. Larger or retrieval-tuned models would improve baseline vector search quality.

**5. Some queries are genuine KB coverage gaps.** A handful of expert questions (Wix Payments verification timing, hotel iCal sync, cookie banners/GDPR) have zero hit rate across every experiment — the articles may not exist in the corpus.

The next five experiments each targeted one or more of these root causes.

---

## Experiment 3 — HyDE query expansion (reverted)

**Hypothesis:** Generating a hypothetical Wix Help Center article excerpt and embedding that instead of the raw query would bridge the vocabulary gap.

**Config:** Every query → gpt-4o-mini generates a 3–5 sentence hypothetical document → embed the document instead of the query → retrieve top-20 → rerank to top-5.

| Metric | Δ vs Exp 2 |
|---|---|
| Expert HR@5 | −0.08 → 0.63 |
| Expert MRR | −0.06 → 0.438 |
| Faithfulness | −0.08 → 4.76 |

**Why it failed:**

1. **The model hallucinates product details.** A HyDE document like "To sync the Wix Hotels calendar with iCal, go to Settings > Calendar Sync and copy the export URL" is confidently wrong. When embedded, it points toward chunks about calendar settings in unrelated products.

2. **Chunks are too short for HyDE.** HyDE works best with long, richly contextual documents. With 200-token chunks, embedding similarity is dominated by a few key terms — one hallucinated term actively misdirects the search.

3. **The embedding model is too small.** `all-MiniLM-L6-v2` doesn't have the semantic depth to bridge a hallucinated GPT summary to a real KB fragment. Larger models handle noisy similarity better.

4. **No benefit for well-phrased queries.** Synthetic queries already match KB vocabulary perfectly. HyDE adds hallucination noise with zero upside for these.

**Reverted.** HyDE is not a good fit for short chunks + small embedding models.

---

## Experiment 4 — Classifier fix for Wix-adjacent services

**Hypothesis:** The relevancy [1] scores are caused by the classifier, not retrieval. Updating the classifier prompt to treat Wix-sold third-party products as ANSWERABLE should fix them.

**Config:** Added explicit guidance to `prompts/classifier.txt` listing Google Workspace, Google Ads, third-party payment providers, and Wix App Market integrations as ANSWERABLE.

| Metric | Δ vs Exp 2 |
|---|---|
| Relevancy | +0.06 → 4.66 |
| Faithfulness | −0.10 → 4.74 |

**Partial success.** Google Workspace and Google Ads queries now route to ANSWERABLE (no longer in the relevancy worst-5). The faithfulness dip is expected: these queries now attempt retrieval but find no KB coverage for Google Workspace pricing, producing low-confidence answers the judge scores differently than the previous static deflection. One POS query still misroutes — the classifier reasoned "does not specifically pertain to Wix products" despite the updated prompt.

---

## Experiment 5 — Larger candidate pool, top_k=50 (reverted)

**Hypothesis:** The zero-hit expert queries might be at positions 21–50 in the vector index. Retrieving more candidates gives the cross-encoder more to work with.

| Metric | Δ vs Exp 4 |
|---|---|
| Expert HR@5 | −0.02 → 0.69 |
| Expert MRR | −0.016 → 0.482 |
| Faithfulness | −0.06 → 4.68 |

**Why it failed.** The zero-hit queries aren't at positions 21–50 — they're genuine KB coverage gaps or extreme vocabulary mismatches that don't surface anywhere in the vector index. The extra 30 candidates introduced cross-encoder noise: the reranker sometimes promoted a wrong document above a marginal-but-correct one.

**Reverted.**

---

## Experiment 6 — Title-prepended embeddings (current config)

**Hypothesis:** Prepending the article title to each chunk before embedding and storage would give the embedding model and reranker a factual vocabulary anchor with zero hallucination risk.

**Config:** Each chunk stored and embedded as `f"{title}:\n{chunk_text}"`. ChromaDB rebuilt from scratch.

| Metric | Δ vs Exp 4 |
|---|---|
| Expert MRR | +0.017 → 0.514 |
| Faithfulness | +0.06 → 4.80 |
| Relevancy | +0.10 → 4.76 |

Expert HR@5 held at 0.71 — confirming the persistent zero-hit queries are KB coverage gaps, not fixable by title context alone. But the MRR gain (+0.017) shows the reranker benefits from seeing article titles in chunk text: correct articles now rank higher. The generation quality gains (faithfulness +0.06, relevancy +0.10) are the largest of any experiment, because better-ranked chunks produce better context for the generator.

**This is the current production configuration.**

---

## Experiment 7 — Always-on query expansion (reverted)

**Hypothesis:** A simpler version of query expansion — adding Wix product terminology without generating a full hypothetical document — might bridge the vocabulary gap where HyDE failed.

**Config:** Removed the 8-word gate. Every query expanded via gpt-4o-mini to add domain vocabulary (product names, feature terms) without inventing UI steps or process details.

| Metric | Δ vs Exp 6 |
|---|---|
| Expert HR@5 | −0.105 → 0.605 |
| Expert MRR | −0.086 → 0.428 |
| Synthetic HR@5 | −0.105 → 0.840 |

**Why it failed — same root causes as HyDE, at smaller scale:**

1. **Synthetic queries already have optimal vocabulary.** Expanding them only adds noise.
2. **LLM expansion is non-deterministic.** The same query can produce different expansions across calls. When the expansion adds a product name that doesn't match the canonical article title (e.g. "Wix Marketplace" vs. "Wix App Market"), the embedding moves further from the correct chunk.
3. **The 8-word gate was doing useful work.** Short queries benefit from expansion. Long, conversational queries are hurt by it — the expansion may keep or lose the right keywords unpredictably.

**Reverted.** The 8-word gated rewriter remains the best query expansion strategy for this setup.

---

## Remaining gaps

Five expert queries have zero hit rate across all seven experiments:

- "Can I start accepting payments on my site while my Wix Payments account is still under verification?"
- "I want to know if the Wix store function work for selling services instead of just physical goods"
- "How do I sync the hotel app with my calendars ical link to allow visitors to book available dates on my site?"
- "Can you help me understand how to enable guest checkout for my pricing plans on Wix?"
- "I need to renew my Premium Business plan after missing the renewal notice."

These are likely **KB coverage gaps** (the relevant articles may not exist in the corpus) or vocabulary mismatches too extreme for a general-purpose embedding model to resolve. The most impactful next steps would be:

1. **Hybrid retrieval** (dense + BM25 via Reciprocal Rank Fusion) — would catch exact product name matches that cosine similarity misses
2. **Larger embedding model** (e.g. `text-embedding-3-small` or `BAAI/bge-small-en-v1.5`) — more semantic depth to bridge vocabulary gaps
3. **KB audit** — check whether the corpus actually contains articles covering these topics
