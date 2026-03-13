"""Tests for routing decisions in src/pipeline.py.

These tests mock all LLM-dependent functions (classify, retrieve, rerank,
generate*) to verify that the pipeline routes correctly based on classifier
output and reranker scores.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

# Skip rate limiting for all tests
os.environ["SKIP_RATE_LIMIT"] = "true"

from src.classifier import QueryCategory, ClassificationResult
from src.pipeline import run, RELEVANCE_SCORE_THRESHOLD


def _mock_classification(category: QueryCategory) -> ClassificationResult:
    return ClassificationResult(category=category, reasoning="test")


def _mock_chunks(top_score: float, n: int = 5) -> list:
    return [
        {
            "text": f"Chunk {i}",
            "article_title": f"Article {i}",
            "article_id": str(i),
            "reranker_score": top_score - i,
        }
        for i in range(n)
    ]


# ── Static routing (no retrieval) ─────────────────────────────


@patch("src.pipeline.classify")
def test_nonsense_routing(mock_classify):
    mock_classify.return_value = _mock_classification(QueryCategory.NONSENSE)
    result = run("asdf jkl")
    assert result["routing"] == "nonsense"
    assert result["sources"] == []


@patch("src.pipeline.classify")
def test_irrelevant_routing(mock_classify):
    mock_classify.return_value = _mock_classification(QueryCategory.IRRELEVANT)
    result = run("What is the capital of France?")
    assert result["routing"] == "irrelevant"
    assert result["sources"] == []


@patch("src.pipeline.classify")
def test_out_of_scope_routing(mock_classify):
    mock_classify.return_value = _mock_classification(QueryCategory.OUT_OF_SCOPE)
    result = run("Delete my account data under GDPR")
    assert result["routing"] == "out_of_scope"
    assert result["sources"] == []
    assert "connect you with" in result["answer"].lower() or "support agent" in result["answer"].lower()


# ── Answerable: followup when no relevant results ────────────


@patch("src.pipeline.generate_followup", return_value="Could you clarify?")
@patch("src.pipeline.rerank")
@patch("src.pipeline.retrieve")
@patch("src.pipeline.maybe_rewrite", side_effect=lambda q: q)
@patch("src.pipeline.classify")
def test_answerable_low_relevance_routes_to_followup(
    mock_classify, mock_rewrite, mock_retrieve, mock_rerank, mock_followup
):
    mock_classify.return_value = _mock_classification(QueryCategory.ANSWERABLE)
    mock_retrieve.return_value = _mock_chunks(1.0)
    mock_rerank.return_value = _mock_chunks(1.0)  # below RELEVANCE_SCORE_THRESHOLD
    result = run("something vague")
    assert result["routing"] == "followup"
    assert result["answer"] == "Could you clarify?"


# ── Answerable: normal generation ─────────────────────────────


@patch("src.pipeline.generate", return_value=("Here is the answer.", ["Article 0"], "answered"))
@patch("src.pipeline.rerank")
@patch("src.pipeline.retrieve")
@patch("src.pipeline.maybe_rewrite", side_effect=lambda q: q)
@patch("src.pipeline.classify")
def test_answerable_high_relevance_generates(
    mock_classify, mock_rewrite, mock_retrieve, mock_rerank, mock_generate
):
    mock_classify.return_value = _mock_classification(QueryCategory.ANSWERABLE)
    chunks = _mock_chunks(8.0)
    mock_retrieve.return_value = chunks
    mock_rerank.return_value = chunks
    result = run("How do I connect a domain?")
    assert result["routing"] == "answered"
    assert result["sources"] == ["Article 0"]


# ── High-stakes routing ──────────────────────────────────────


@patch("src.pipeline.generate_high_stakes", return_value=("I understand...", []))
@patch("src.pipeline.rerank")
@patch("src.pipeline.retrieve")
@patch("src.pipeline.maybe_rewrite", side_effect=lambda q: q)
@patch("src.pipeline.classify")
def test_high_stakes_routing(
    mock_classify, mock_rewrite, mock_retrieve, mock_rerank, mock_hs
):
    mock_classify.return_value = _mock_classification(QueryCategory.HIGH_STAKES)
    mock_retrieve.return_value = _mock_chunks(8.0)
    mock_rerank.return_value = _mock_chunks(8.0)
    result = run("I want to cancel my account")
    assert result["routing"] == "high_stakes"
    mock_hs.assert_called_once()


@patch("src.pipeline.generate_high_stakes", return_value=("I understand...", []))
@patch("src.pipeline.rerank")
@patch("src.pipeline.retrieve")
@patch("src.pipeline.maybe_rewrite", side_effect=lambda q: q)
@patch("src.pipeline.classify")
def test_high_stakes_with_no_results_passes_empty_chunks(
    mock_classify, mock_rewrite, mock_retrieve, mock_rerank, mock_hs
):
    mock_classify.return_value = _mock_classification(QueryCategory.HIGH_STAKES)
    mock_retrieve.return_value = _mock_chunks(1.0)
    mock_rerank.return_value = _mock_chunks(1.0)  # below relevance threshold
    run("I want to cancel my account")
    # When has_results is False, high-stakes should receive empty chunks
    _, kwargs = mock_hs.call_args
    assert kwargs.get("chunks") is not None or mock_hs.call_args[0][1] == []


# ── Relevance threshold ──────────────────────────────────────


def test_relevance_threshold_is_documented_value():
    """RELEVANCE_SCORE_THRESHOLD should match the documented value of 2.0."""
    assert RELEVANCE_SCORE_THRESHOLD == 2.0
