"""
Ingest the Wix/WixQA dataset into a ChromaDB collection.

Chunking pipeline per article:
  1. Split body on \n\n (paragraph splits)
  2. Tokenize with tiktoken cl100k_base
  3. Merge splits < 50 tokens with the next (or previous for last)
  4. Split chunks > 300 tokens at sentence boundaries (~200 token target)
  5. Apply 50-token sliding overlap: prepend last 50 tokens of previous chunk
"""

import re

import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from src.config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH
ENCODING = "cl100k_base"
MIN_TOKENS = 50
MAX_TOKENS = 300
TARGET_TOKENS = 200


def tokenize(enc, text: str) -> list[int]:
    return enc.encode(text)


def detokenize(enc, tokens: list[int]) -> str:
    return enc.decode(tokens)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def merge_short_splits(splits: list[str], enc, min_tokens: int) -> list[str]:
    """Merge paragraphs shorter than min_tokens with adjacent chunk."""
    merged = []
    i = 0
    while i < len(splits):
        tokens = tokenize(enc, splits[i])
        if len(tokens) < min_tokens:
            if i + 1 < len(splits):
                # merge with next
                splits[i + 1] = splits[i] + "\n\n" + splits[i + 1]
                i += 1
                continue
            elif merged:
                # last chunk is short — merge with previous
                merged[-1] = merged[-1] + "\n\n" + splits[i]
                i += 1
                continue
        merged.append(splits[i])
        i += 1
    return merged


def split_long_chunk(text: str, enc, max_tokens: int, target_tokens: int) -> list[str]:
    """Split a chunk exceeding max_tokens at sentence boundaries."""
    tokens = tokenize(enc, text)
    if len(tokens) <= max_tokens:
        return [text]

    sentences = split_into_sentences(text)
    chunks = []
    current_sentences = []
    current_len = 0

    for sentence in sentences:
        s_tokens = len(tokenize(enc, sentence))
        if current_len + s_tokens > target_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_len = s_tokens
        else:
            current_sentences.append(sentence)
            current_len += s_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks if chunks else [text]


def apply_overlap(chunks: list[str], enc, overlap_tokens: int) -> list[str]:
    """Prepend last `overlap_tokens` tokens of previous chunk to each subsequent chunk."""
    result = [chunks[0]] if chunks else []
    for i in range(1, len(chunks)):
        prev_tokens = tokenize(enc, chunks[i - 1])
        overlap = (
            prev_tokens[-overlap_tokens:]
            if len(prev_tokens) >= overlap_tokens
            else prev_tokens
        )
        overlap_text = detokenize(enc, overlap)
        result.append(overlap_text + " " + chunks[i])
    return result


def chunk_article(body: str, enc) -> list[str]:
    # Step 1: paragraph split
    raw_splits = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not raw_splits:
        return []

    # Step 2+3: merge short splits
    merged = merge_short_splits(raw_splits, enc, MIN_TOKENS)

    # Step 4: split long chunks at sentence boundaries
    final_chunks = []
    for chunk in merged:
        final_chunks.extend(split_long_chunk(chunk, enc, MAX_TOKENS, TARGET_TOKENS))

    # Step 5: sliding overlap
    if len(final_chunks) > 1:
        final_chunks = apply_overlap(final_chunks, enc, overlap_tokens=50)

    return final_chunks


def main():
    print("Loading dataset...")
    dataset = load_dataset("Wix/WixQA", "wix_kb_corpus", split="train")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Idempotency check
    existing = client.list_collections()
    for col in existing:
        if col.name == COLLECTION_NAME:
            count = col.count()
            if count > 0:
                print(
                    f"Collection '{COLLECTION_NAME}' already has {count} documents. Skipping ingest."
                )
                return
            break

    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)
    enc = tiktoken.get_encoding(ENCODING)

    ids = []
    documents = []
    metadatas = []

    articles = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)
    print(f"Processing {len(articles)} articles...")

    for article in articles:
        article_id = str(article.get("id", ""))
        title = article.get("title", "")
        body = article.get("contents", "")

        if not body:
            continue

        chunks = chunk_article(body, enc)
        for chunk_idx, chunk_text in enumerate(chunks):
            doc_id = f"{article_id}_{chunk_idx}"
            ids.append(doc_id)
            titled_text = f"{title}:\n{chunk_text}" if title else chunk_text
            documents.append(titled_text)
            metadatas.append(
                {
                    "article_id": article_id,
                    "article_title": title,
                    "chunk_index": chunk_idx,
                }
            )

    print(f"Embedding {len(documents)} chunks...")
    batch_size = 256
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_meta = metadatas[start : start + batch_size]
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  Stored {min(start + batch_size, len(documents))}/{len(documents)}")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} documents.")


if __name__ == "__main__":
    main()
