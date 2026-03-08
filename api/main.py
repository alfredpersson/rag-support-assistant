from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rate_limit import RateLimitExceeded
from src.pipeline import run

app = FastAPI(title="RAG Support Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = run(request.question)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="An internal error occurred.")


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
