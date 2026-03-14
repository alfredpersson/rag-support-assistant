import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.rate_limit import RateLimitExceeded
from src.pipeline import run


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up all lazy singletons at startup so the first request pays no init cost."""
    import logging
    log = logging.getLogger("uvicorn.error")

    log.info("Warming up embedding model...")
    from src.retriever import _get_model, _get_collection
    _get_model()
    _get_collection()

    log.info("Warming up cross-encoder...")
    from src.reranker import _get_reranker
    _get_reranker()

    log.info("Warming up LLM agents and fetching prompts from Langfuse...")
    from src.classifier import _get_agent as _clf_agent
    from src.query_rewriter import _get_agent as _qr_agent
    from src.generator import (
        _get_generator_agent, _get_self_critique_agent,
        _get_followup_agent, _get_high_stakes_agent,
    )
    _clf_agent()
    _qr_agent()
    _get_generator_agent()
    _get_self_critique_agent()
    _get_followup_agent()
    _get_high_stakes_agent()

    log.info("Warmup complete.")
    yield


app = FastAPI(title="RAG Support Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_QUESTION_LENGTH = 500


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question_length(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.")
        return v


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    routing: str


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        result = await run(request.question)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            routing=result["routing"],
        )
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception:
        logging.getLogger("uvicorn.error").exception("Pipeline error for question: %s", request.question)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
