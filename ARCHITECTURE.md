# Architecture & Design Decisions

This document explains the reasoning behind every significant technical choice, what I learned from evaluation, and what would need to change for production.

---

## 1. Pipeline built from scratch

**Decision:** No LangChain, LlamaIndex, or similar orchestration framework.

**Reasoning:** Frameworks like LangChain add substantial abstraction overhead that makes it harder to reason about what is actually being sent to the model, what is being retrieved, and where latency comes from. For a demo of a RAG pipeline, the pipeline itself is the thing being demonstrated. Writing each stage explicitly (ingest → retrieve → rerank → generate) makes every decision visible and debuggable.

**Production consideration:** At scale, the manual wiring becomes a maintenance burden. A framework or custom internal orchestration layer becomes worth it once you have multiple pipelines, shared components (retrievers, rerankers, caches), and teams working on different parts simultaneously.

---

## 2. Embedding model — `all-MiniLM-L6-v2`

**Decision:** Sentence-Transformers `all-MiniLM-L6-v2`, run locally.

**Reasoning:** Fast, small (80 MB), and good enough for English support content. Running it locally avoids an extra API call on every query and every ingestion. For a single-node demo this is the right trade-off.

**What evaluation showed:** The model's small size (22M parameters) is the primary bottleneck for retrieval quality. It lacks the semantic depth to bridge natural-language queries to documentation vocabulary — a query about "hotel app iCal sync" doesn't land near a chunk about "calendar link export." This is the root cause of the 29% expert hit rate gap. It also made HyDE and always-on query expansion counterproductive: the model can't reliably bridge a hallucinated summary to a real KB fragment (see [experiments 3 and 7](eval/EXPERIMENTS.md)).

**Production consideration:** A larger or retrieval-tuned model (e.g. `text-embedding-3-small`, `BAAI/bge-small-en-v1.5`, or a domain-adapted model) would improve recall. Running embedding locally also means the model must be co-located with the service; in a distributed system you would call an embedding API or a dedicated embedding service instead.

---

## 3. Vector store — ChromaDB (local persistent)

**Decision:** ChromaDB with a local persistent directory (`chroma_db/`).

**Reasoning:** Zero infrastructure setup. ChromaDB runs in-process, persists to disk, and is sufficient for the size of the Wix knowledge base. Perfect for a single-process demo that needs to be cloned and run immediately.

**Production consideration:** ChromaDB does not support horizontal scaling, fine-grained access control, or SQL-style metadata filtering alongside vectors. The natural upgrade path is **pgvector** (PostgreSQL extension), which combines vector similarity search with relational filtering in a single query — e.g. "find the top-k chunks similar to this query where `product_area = 'billing'`". pgvector also enables **hybrid retrieval** (dense + BM25 via Reciprocal Rank Fusion), which the evaluation identified as the highest-impact next improvement.

---

## 4. Chunking strategy

**Decision:** Four-pass pipeline — paragraph split → merge short paragraphs → split long chunks at sentence boundaries → 50-token sliding overlap. Article title prepended to each chunk.

**Reasoning:** Naive fixed-size chunking breaks sentences and severs context. The four-pass approach preserves natural paragraph structure while bounding chunk size to ~200 tokens (target) / 300 tokens (hard cap), keeping them small enough to be retrievable and large enough to be useful. The 50-token overlap ensures a sentence at a paragraph boundary is not lost between adjacent chunks.

**What evaluation showed:** Title-prepending was the single most impactful retrieval improvement (see [experiment 6](eval/EXPERIMENTS.md)). Without titles, chunks from a "Wix Hotels" article about iCal export had no mention of "hotel" or "iCal" in the embedded text. Prepending the article title gives the embedding model and cross-encoder reranker a factual vocabulary anchor. This produced the largest generation quality gains of any experiment (faithfulness +0.06, relevancy +0.10).

**Production consideration:** The optimal chunking strategy is dataset-specific. Structured content (tables, step-by-step procedures, code blocks) often benefits from structure-aware splitting rather than sentence heuristics. Chunk size and overlap should be tuned via offline evaluation on a golden retrieval dataset.

---

## 5. Two-stage retrieval — vector search + cross-encoder reranking

**Decision:** Retrieve top-20 candidates via cosine similarity, then rerank with `ms-marco-MiniLM-L6-v2` cross-encoder and keep top-5.

**Reasoning:** Bi-encoder retrieval (cosine similarity) is fast but approximate — it encodes query and document independently, losing fine-grained query-document interaction. The cross-encoder scores query and document jointly, producing much more accurate relevance scores, but is too slow to run over the full collection. The two-stage pattern gets the best of both: broad recall from fast retrieval, precise ranking from the cross-encoder.

**What evaluation showed:** Reranking improved every metric in experiment 2 — the largest single gain was synthetic MRR (+0.05). Increasing the candidate pool from 20 to 50 was tested and reverted (experiment 5): the zero-hit queries aren't at positions 21–50, they're genuine KB gaps, and the extra candidates introduced cross-encoder noise that hurt precision.

**Production consideration:** The current setup uses only dense retrieval. **Hybrid retrieval** — combining BM25 (keyword) with dense vectors via Reciprocal Rank Fusion (RRF) — would significantly improve recall on exact-match queries (e.g. specific Wix feature names, error codes). This is the most impactful production upgrade identified by evaluation.

---

## 6. Query rewriter

**Decision:** If the query is fewer than 8 words, call `gpt-4o-mini` to expand it into a fuller retrieval-optimised question. Longer queries are passed through unchanged.

**Reasoning:** Short queries like "billing" or "domain not working" map poorly to chunk embeddings. The rewriter adds Wix-specific context ("How do I manage billing settings in my Wix account?") that the embedding model can match against.

**What evaluation showed:** Two attempts to expand the rewriter's scope both failed. HyDE (experiment 3) generated confident but fictional document excerpts that misdirected retrieval. Always-on vocabulary expansion (experiment 7) hurt long conversational queries by non-deterministically adding or losing key terms. The 8-word gate is doing useful work: short queries always benefit from expansion, but longer queries are hurt by it. See [experiments 3 and 7](eval/EXPERIMENTS.md) for detailed failure analysis.

**Production consideration:** The rewriter's effectiveness is limited by the embedding model's ability to use the expanded vocabulary. With a larger model, broader query expansion becomes viable. A more robust approach would be to evaluate rewriter impact per query type and apply it selectively based on learned heuristics rather than a word count threshold.

---

## 7. Query classifier (5 categories)

**Decision:** Classify every query before retrieval into one of five categories — ANSWERABLE, NONSENSE, IRRELEVANT, OUT_OF_SCOPE, HIGH_STAKES — and route accordingly.

**Reasoning:** Without a classifier, the pipeline wastes retrieval and generation compute on nonsense input, produces unhelpful responses to off-topic questions, and handles sensitive escalations (cancellations, billing disputes) the same way as routine how-to questions. The classifier costs one cheap LLM call but prevents much more expensive downstream mistakes.

**What evaluation showed:** The classifier was the direct cause of the worst generation scores in experiments 1 and 2: it misrouted Wix-adjacent services (Google Workspace, Google Ads, POS integrations) to IRRELEVANT, producing static deflection responses that scored 1/5 for both faithfulness and relevancy. Updating the classifier prompt (experiment 4) fixed most of these, with relevancy improving +0.06.

**Production consideration:** A full LLM call for every query just to classify it adds latency (~1.3 s measured). In production you would either (a) use a fine-tuned smaller model (BERT-sized) for classification that runs in <50 ms locally, or (b) run classification and retrieval concurrently using async so they partially overlap.

---

## 8. Context in system prompt, question as user message

**Decision:** Retrieved chunks are passed as `instructions` (appended to the system prompt) rather than included in the user message.

**Reasoning:** The system prompt is conceptually the "what you know" space — authoritative background the model should treat as ground truth. The user message is the question. Separating them reinforces the model's understanding that the context is a constraint to answer within, not user-provided information to be weighed against its prior knowledge.

**Production consideration:** Some model providers cache the system prompt prefix, which can reduce latency and cost when the context is static or semi-static across requests. With dynamic per-request context (as here), caching is not applicable unless you implement a semantic cache at the pipeline level.

---

## 9. Self-critique instead of faithfulness check

**Decision:** After generating an answer, run a self-critique call asking the model to assess whether it FULLY_ANSWERED, PARTIALLY_ANSWERED, or CANNOT_ANSWER, with reasoning. Use this to gate source links and escalation rather than appending a generic caveat.

**Reasoning:** A binary faithfulness check that appends "please verify on the Wix Help Center" to every uncertain answer is low-signal and trains users to ignore it. The self-critique produces a structured, reasoned signal that drives specific UI behaviour — number of source links shown, whether to offer human escalation — which is more useful to the user and more honest about the model's actual confidence.

**Production consideration:** Self-critique is itself an LLM call and therefore subject to miscalibration. It should be evaluated offline against a labelled set of (question, context, answer) triples to verify that its assessments correlate with human judgement. A more robust alternative for the hot path would be a dedicated NLI model that compares answer claims directly against chunk content, which would be faster and more deterministic than a second LLM call.

---

## 10. Source link logic

**Decision:** Use the confidence threshold (`reranker_score ≥ 5.0`) and self-critique outcome to decide how many source links to show. FULLY_ANSWERED gets up to 2 links as a "read more" reference. PARTIALLY_ANSWERED gets 1 link as a starting point. CANNOT_ANSWER and low-confidence answers get none.

**Reasoning:** Source links serve a "read more" function on confident answers. A user who gets "To connect a custom domain, go to Settings > Domains..." should be able to click through to the full article for screenshots, edge cases, or related steps. Links are withheld from uncertain answers because low-confidence retrievals are most likely to surface tangentially related articles rather than directly helpful ones.

**Production consideration:** The `5.0` threshold is a heuristic tuned through evaluation. In production it would be calibrated against user feedback (click-through rate on source links, user satisfaction scores) and offline evaluation.

---

## 11. High-stakes routing

**Decision:** Cancellation, deletion, billing disputes, and complaints are classified separately and handled by a dedicated prompt that produces structured fields (acknowledgment, retention offer, context summary, escalation offer).

**Reasoning:** These queries have outsized consequences if handled badly. A standard RAG answer to "My account was charged twice!" is inappropriate. The dedicated path allows the tone, structure, and escalation behaviour to be controlled precisely without affecting normal Q&A prompts.

**Production consideration:** The retention offer is fictional. In production this would be driven by real CRM data (account age, plan tier, churn risk score) fetched at runtime, and the escalation path would connect to a real ticketing system or live chat queue rather than a mock availability widget. The structured `HighStakesOutput` fields are already well-suited to this extension.

---

## 12. Pydantic-AI for all LLM calls

**Decision:** Every LLM call uses a `pydantic-ai` `Agent` with a typed `output_type`, replacing raw OpenAI client calls.

**Reasoning:** Structured outputs eliminate prompt engineering for JSON formatting and remove the need for response parsing and validation code. Pydantic-AI's `run_sync` integrates OpenTelemetry instrumentation out of the box, which connects directly to Langfuse tracing. The `Agent` abstraction also makes it trivial to swap the underlying model.

**Production consideration:** Agents are currently lazy-initialised singletons per module. This is fine for a single-process server but not for a multi-worker deployment (e.g. Gunicorn with multiple processes), where each worker would initialise its own agents and prompt clients independently. In production, agent initialisation should happen at application startup (lifespan event in FastAPI) so all workers share a consistent state, and prompt fetches should be cached with a TTL.

---

## 13. Langfuse for observability and prompt versioning

**Decision:** All LLM calls are traced via pydantic-ai's OpenTelemetry integration → Langfuse. Prompts are stored in `prompts/*.txt`, registered with Langfuse via `prompts/seed.py`, and fetched at runtime with local fallback.

**Reasoning:** Without observability, debugging a RAG pipeline is guesswork — you cannot tell whether a bad answer came from retrieval, the prompt, the self-critique threshold, or the classifier. Langfuse provides per-request trace trees showing every LLM call with its inputs, outputs, latency, and token cost. Prompt versioning in Langfuse allows A/B testing prompt changes without a code deploy, and ties every trace to the exact prompt version that produced it.

**Production consideration:** The current prompt fetch strategy (fetch once at first call, cache for the lifetime of the process) means a prompt update in Langfuse requires a server restart to take effect. In production you would set a short `cache_ttl_seconds` (e.g. 60–300) on `get_prompt()` so updates propagate automatically. You would also use Langfuse's **dataset** and **evaluation** features to run regression tests on prompt changes before promoting them to the `production` label.

---

## 14. Evaluation system

**Decision:** A custom evaluation pipeline with separate retrieval and generation scoring, run against the WixQA benchmark dataset.

**Reasoning:** Retrieval and generation fail for different reasons, and conflating them into a single end-to-end score hides where the problem is. Splitting evaluation into retrieval metrics (Hit Rate, MRR) and generation metrics (faithfulness, relevancy via LLM-as-judge) makes diagnosis possible: low hit rate means the fix is in retrieval, low faithfulness with high hit rate means the fix is in the prompt.

This separation directly drove the root cause analysis that led to the most impactful improvements (title-prepending, classifier fix) and prevented wasted effort on changes that looked promising but regressed (HyDE, top_k=50). See [eval methodology](eval/README.md) and [experiment log](eval/EXPERIMENTS.md).

**Production consideration:** LLM-as-judge scores are systematically biased toward verbose, confident-sounding responses. In production, automated scores should be calibrated against human-annotated examples, with periodic human spot-checks on random samples to detect and correct score drift.

---

## 15. Rate limiting — JSON file

**Decision:** A single `rate_limit.json` file tracking daily request counts, read and written on every request.

**Reasoning:** The simplest possible implementation that enforces the 100 req/day limit for demo purposes. No external dependencies.

**Production consideration:** File-based rate limiting is not safe under concurrent load (race conditions) and does not work across multiple server processes or nodes. The standard production solution is **Redis `INCR` + `EXPIRE`**, which is atomic and works across any number of workers. Per-user rate limiting (by API key or session) is also typical in production rather than a global cap.

---

## 16. Frontend — plain HTML/CSS/JS

**Decision:** No JavaScript framework. The chat widget is basic HTML + CSS + JS.

**Reasoning:** A simple chat widget does not benefit from a component framework. React or Vue would add a build step, a bundle, and abstraction layers for functionality that is inherently stateful-but-simple (open/close, append message, fetch). The result is faster to load and easier to inspect for the current demo purposes.

**Production consideration:** As the widget grows (conversation history, rich media, markdown tables, attachment support), the absence of a framework becomes a liability. In production the widget would be a proper component library (React + TypeScript), bundled and versioned independently from the API.