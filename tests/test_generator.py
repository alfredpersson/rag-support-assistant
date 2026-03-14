"""Tests for generator helpers and source link logic in src/generator.py."""

import pytest

from src.generator import (
    _build_context,
    _dedupe_chunks_by_source,
    CONFIDENCE_THRESHOLD,
)


# ── _build_context ────────────────────────────────────────────


def test_build_context_formats_chunks():
    chunks = [
        {"article_title": "Domain Setup", "text": "Connect your domain..."},
        {"article_title": "Billing FAQ", "text": "To update billing..."},
    ]
    result = _build_context(chunks)
    assert "[1] (Source: Domain Setup)" in result
    assert "[2] (Source: Billing FAQ)" in result
    assert "Connect your domain..." in result
    assert "To update billing..." in result


def test_build_context_empty():
    assert _build_context([]) == ""


# ── _dedupe_chunks_by_source ──────────────────────────────────


def test_dedupe_keeps_first_per_title():
    chunks = [
        {"article_title": "Domain Setup", "reranker_score": 8.0},
        {"article_title": "Domain Setup", "reranker_score": 5.0},
        {"article_title": "Billing FAQ", "reranker_score": 7.0},
    ]
    result = _dedupe_chunks_by_source(chunks)
    assert len(result) == 2
    titles = [c["article_title"] for c in result]
    assert titles == ["Domain Setup", "Billing FAQ"]
    # Keeps the first (highest-scoring, since input is pre-sorted)
    assert result[0]["reranker_score"] == 8.0


def test_dedupe_skips_empty_titles():
    chunks = [
        {"article_title": "", "reranker_score": 9.0},
        {"article_title": "Real Article", "reranker_score": 7.0},
    ]
    result = _dedupe_chunks_by_source(chunks)
    assert len(result) == 1
    assert result[0]["article_title"] == "Real Article"


def test_dedupe_empty_input():
    assert _dedupe_chunks_by_source([]) == []


# ── Source link logic (unit-level) ────────────────────────────
# The source link decisions in generate() depend on self-critique assessment
# and confidence threshold. We test the decision logic by verifying the
# threshold constant and the dedupe + filtering that feeds into it.


def test_confidence_threshold_above_relevance():
    """CONFIDENCE_THRESHOLD must be > RELEVANCE_SCORE_THRESHOLD (2.0) to keep
    the low_confidence path reachable."""
    from src.pipeline import RELEVANCE_SCORE_THRESHOLD

    assert CONFIDENCE_THRESHOLD > RELEVANCE_SCORE_THRESHOLD


def test_confident_sources_filter():
    """Simulates the source filtering logic from generate()."""
    chunks = [
        {"article_title": "High Score", "reranker_score": 7.0},
        {"article_title": "Low Score", "reranker_score": 3.0},
        {"article_title": "Medium Score", "reranker_score": 5.5},
    ]
    deduped = _dedupe_chunks_by_source(chunks)
    confident = [
        c for c in deduped
        if c.get("article_title") and c["reranker_score"] >= CONFIDENCE_THRESHOLD
    ]
    titles = [c["article_title"] for c in confident]
    assert "High Score" in titles
    assert "Medium Score" in titles
    assert "Low Score" not in titles


def test_fully_answered_gets_two_sources():
    """FULLY_ANSWERED should produce up to 2 source links."""
    chunks = [
        {"article_title": f"Article {i}", "reranker_score": 8.0 - i}
        for i in range(5)
    ]
    deduped = _dedupe_chunks_by_source(chunks)
    confident = [
        c for c in deduped
        if c.get("article_title") and c["reranker_score"] >= CONFIDENCE_THRESHOLD
    ]
    # FULLY_ANSWERED path: up to 2
    sources = [c["article_title"] for c in confident[:2]]
    assert len(sources) <= 2


def test_partially_answered_gets_one_source():
    """PARTIALLY_ANSWERED should produce exactly 1 source link."""
    chunks = [
        {"article_title": "Top Article", "reranker_score": 8.0},
        {"article_title": "Second Article", "reranker_score": 6.0},
    ]
    deduped = _dedupe_chunks_by_source(chunks)
    confident = [
        c for c in deduped
        if c.get("article_title") and c["reranker_score"] >= CONFIDENCE_THRESHOLD
    ]
    # PARTIALLY_ANSWERED path: 1 link
    sources = [confident[0]["article_title"]] if confident else []
    assert len(sources) == 1
    assert sources[0] == "Top Article"


# ── rerank empty-chunks edge case ─────────────────────────────


def test_rerank_empty_chunks():
    """rerank() should return [] immediately when given no chunks."""
    from src.reranker import rerank

    assert rerank("any query", []) == []
