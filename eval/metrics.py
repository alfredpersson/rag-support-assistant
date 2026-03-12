"""Pure metric functions for retrieval and generation evaluation."""


def hit_rate(retrieved_ids: list, correct_id: str) -> int:
    return 1 if correct_id in retrieved_ids else 0


def reciprocal_rank(retrieved_ids: list, correct_id: str) -> float:
    try:
        pos = retrieved_ids.index(correct_id) + 1  # 1-indexed
        return 1.0 / pos
    except ValueError:
        return 0.0


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
