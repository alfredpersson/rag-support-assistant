"""Tests for the chunking pipeline in src/ingest.py."""

import tiktoken
import pytest

from src.ingest import (
    split_into_sentences,
    merge_short_splits,
    split_long_chunk,
    apply_overlap,
    chunk_article,
    MIN_TOKENS,
    MAX_TOKENS,
    TARGET_TOKENS,
    ENCODING,
)


@pytest.fixture
def enc():
    return tiktoken.get_encoding(ENCODING)


# ── split_into_sentences ──────────────────────────────────────


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    assert split_into_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_split_sentences_single():
    assert split_into_sentences("No punctuation ending") == ["No punctuation ending"]


def test_split_sentences_empty():
    assert split_into_sentences("") == []


# ── merge_short_splits ────────────────────────────────────────


def test_merge_short_with_next(enc):
    short = "Hi"  # well under 50 tokens
    long = "This is a sufficiently long paragraph. " * 10
    result = merge_short_splits([short, long], enc, MIN_TOKENS)
    assert len(result) == 1
    assert short in result[0]
    assert long.strip() in result[0]


def test_merge_short_last_with_previous(enc):
    long = "This is a sufficiently long paragraph. " * 10
    short = "End."
    result = merge_short_splits([long, short], enc, MIN_TOKENS)
    assert len(result) == 1
    assert short in result[0]


def test_no_merge_when_all_long(enc):
    chunks = ["word " * 60, "word " * 60]
    result = merge_short_splits(chunks, enc, MIN_TOKENS)
    assert len(result) == 2


# ── split_long_chunk ──────────────────────────────────────────


def test_short_chunk_not_split(enc):
    text = "A short chunk."
    result = split_long_chunk(text, enc, MAX_TOKENS, TARGET_TOKENS)
    assert result == [text]


def test_long_chunk_split_at_sentences(enc):
    # Build a chunk that exceeds MAX_TOKENS
    sentences = [f"Sentence number {i} with some extra words to pad it out." for i in range(30)]
    text = " ".join(sentences)
    tokens = enc.encode(text)
    assert len(tokens) > MAX_TOKENS, "Test text must exceed MAX_TOKENS"

    result = split_long_chunk(text, enc, MAX_TOKENS, TARGET_TOKENS)
    assert len(result) > 1
    # Each resulting chunk should be roughly around TARGET_TOKENS
    for chunk in result:
        chunk_tokens = len(enc.encode(chunk))
        assert chunk_tokens <= MAX_TOKENS + 50  # allow some tolerance for sentence boundaries


# ── apply_overlap ─────────────────────────────────────────────


def test_overlap_prepends_tokens(enc):
    chunks = ["First chunk with enough words to have tokens. " * 5, "Second chunk here."]
    result = apply_overlap(chunks, enc, overlap_tokens=10)
    assert len(result) == 2
    assert result[0] == chunks[0]  # first chunk unchanged
    assert result[1] != chunks[1]  # second chunk has overlap prepended
    assert "Second chunk here." in result[1]


def test_overlap_single_chunk(enc):
    result = apply_overlap(["Only one chunk."], enc, overlap_tokens=10)
    assert result == ["Only one chunk."]


def test_overlap_empty(enc):
    assert apply_overlap([], enc, overlap_tokens=10) == []


# ── chunk_article (integration) ───────────────────────────────


def test_chunk_article_empty(enc):
    assert chunk_article("", enc) == []


def test_chunk_article_single_paragraph(enc):
    text = "A single short paragraph."
    result = chunk_article(text, enc)
    assert len(result) == 1
    assert text in result[0]


def test_chunk_article_multiple_paragraphs(enc):
    paragraphs = ["Paragraph about topic one. " * 15, "Paragraph about topic two. " * 15]
    text = "\n\n".join(paragraphs)
    result = chunk_article(text, enc)
    assert len(result) >= 2
    # Overlap means chunk 2+ should contain some text from the previous chunk
    if len(result) > 1:
        # The second chunk should start with overlap from the first
        assert len(result[1]) > len(paragraphs[1])
