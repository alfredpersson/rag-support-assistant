# RAG Support Assistant

A RAG (Retrieval-Augmented Generation) pipeline over the [Wix Help Center dataset](https://huggingface.co/datasets/Wix/WixQA). Built from scratch — no LangChain or LlamaIndex.

## Stack

| Layer | Tech |
|---|---|
| Dataset | `Wix/WixQA` via HuggingFace `datasets` |
| Chunking | Paragraph split + merge/split heuristics + 50-token sliding overlap |
| Embedding | `sentence-transformers` `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, local) |
| Generation | OpenAI `gpt-4o-mini` |
| API | FastAPI |

## Setup

```bash
# Create venv and install dependencies
uv venv && uv sync

# Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# Ingest the dataset (one-time, idempotent)
uv run python src/ingest.py

# Start the API
uv run uvicorn api.main:app --reload
```

## API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I connect a custom domain to my Wix site?"}'
```

Response:
```json
{
  "answer": "To connect a custom domain...",
  "sources": ["Connecting a Domain to Your Wix Site"]
}
```

**Rate limit:** 100 requests per day. Returns HTTP 429 when exceeded.

## Evaluation

```bash
uv run python eval/evaluate.py
```

Runs 10 Wix-domain Q&A pairs through the pipeline and scores each answer by keyword overlap. Results written to `eval/results.json`.

## Project Structure

```
src/
  ingest.py          # Dataset loading, chunking, ChromaDB population
  retriever.py       # Query embedding + vector search (top-20 candidates)
  reranker.py        # Cross-encoder reranking → top-5 chunks
  query_rewriter.py  # LLM-based query expansion for short queries
  generator.py       # OpenAI chat completion + faithfulness check
  pipeline.py        # Orchestrates full pipeline
  rate_limit.py      # Daily 100-request cap (rate_limit.json)
api/
  main.py            # FastAPI app
eval/
  evaluate.py        # 10 Q&A pairs, keyword scoring
chroma_db/           # Persisted vector store (created on first ingest)
```

## Pipeline

```
query
  │
  ▼
[query rewriter]   ← expand if < 8 words (gpt-4o-mini)
  │
  ▼
[retriever]        ← top-20 candidates from ChromaDB
  │
  ▼
[reranker]         ← cross-encoder (ms-marco-MiniLM-L6-v2) → top-5 + scores
  │
  ▼
[relevance gate]   ← if top reranker score < threshold → short-circuit
  │
  ▼
[generator]        ← generate answer from top-5 chunks (gpt-4o-mini)
  │
  ▼
[faithfulness]     ← verify answer grounded in chunks; append caveat if not
  │
  ▼
response
```

## Notes

- **Chunking**: Articles are split on `\n\n`, short paragraphs (<50 tokens) are merged with neighbors, long chunks (>300 tokens) are split at sentence boundaries targeting ~200 tokens, then a 50-token overlap is prepended to each subsequent chunk.
- **Idempotency**: Re-running `ingest.py` skips population if the ChromaDB collection already contains documents.
- **pgvector alternative**: For production use, replacing ChromaDB with pgvector (PostgreSQL) would allow combining semantic search with SQL filtering on metadata.

## Production Considerations

The following improvements are documented as future work but not implemented here:

- **Hybrid retrieval** — BM25 + dense retrieval with Reciprocal Rank Fusion (RRF) for better exact keyword recall alongside semantic search.
- **Session context** — sliding window of prior turns stored in Redis to handle follow-up questions correctly.
- **HyDE (Hypothetical Document Embeddings)** — embed a hypothetical answer instead of the raw query to improve retrieval for abstract questions.
- **Streaming** — Server-Sent Events (SSE) for real-time token output, reducing perceived latency.
- **Semantic cache** — skip retrieval and generation for near-duplicate queries using embedding similarity on a cache store.
- **Distributed rate limiting** — replace the JSON file with Redis `INCR` + `EXPIRE` for correct behaviour under concurrent load.
- **Structured observability** — OpenTelemetry spans with per-stage latency metrics (rewrite, retrieve, rerank, generate, faithfulness).
- **RAGAS evaluation** — replace keyword-overlap scoring with RAGAS metrics: faithfulness, context precision, and context recall.
