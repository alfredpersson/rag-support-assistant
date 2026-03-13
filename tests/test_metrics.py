"""Tests for eval/metrics.py — pure metric functions."""

from eval.metrics import hit_rate, reciprocal_rank, mean


# ── hit_rate ──────────────────────────────────────────────────


def test_hit_rate_present():
    assert hit_rate(["1", "2", "3"], "2") == 1


def test_hit_rate_absent():
    assert hit_rate(["1", "2", "3"], "99") == 0


def test_hit_rate_empty_list():
    assert hit_rate([], "1") == 0


# ── reciprocal_rank ───────────────────────────────────────────


def test_rr_first_position():
    assert reciprocal_rank(["5", "3", "1"], "5") == 1.0


def test_rr_third_position():
    assert reciprocal_rank(["5", "3", "1"], "1") == pytest.approx(1 / 3)


def test_rr_not_found():
    assert reciprocal_rank(["5", "3", "1"], "99") == 0.0


# ── mean ──────────────────────────────────────────────────────


def test_mean_normal():
    assert mean([1, 2, 3]) == pytest.approx(2.0)


def test_mean_empty():
    assert mean([]) == 0.0


import pytest
