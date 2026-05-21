"""Cached access to the Bitext customer-support training dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import truststore

truststore.inject_into_ssl()

from datasets import Dataset, load_dataset

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


@lru_cache(maxsize=1)
def get_train_split() -> Dataset:
    dataset = load_dataset(DATASET_ID)
    return dataset["train"]


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().upper()


def _matches(
    row: dict[str, Any],
    *,
    category: str | None = None,
    intent: str | None = None,
    intent_contains: str | None = None,
    instruction_contains: str | None = None,
) -> bool:
    if category is not None and row["category"].upper() != category:
        return False
    if intent is not None and row["intent"].upper() != intent:
        return False
    if intent_contains is not None and intent_contains not in row["intent"].upper():
        return False
    if instruction_contains is not None and instruction_contains not in row["instruction"].upper():
        return False
    return True


def filter_rows(
    *,
    category: str | None = None,
    intent: str | None = None,
    intent_contains: str | None = None,
    instruction_contains: str | None = None,
) -> list[dict[str, Any]]:
    category = _normalize(category)
    intent = _normalize(intent)
    intent_contains = _normalize(intent_contains)
    instruction_contains = _normalize(instruction_contains)

    return [
        dict(row)
        for row in get_train_split()
        if _matches(
            row,
            category=category,
            intent=intent,
            intent_contains=intent_contains,
            instruction_contains=instruction_contains,
        )
    ]


def list_categories() -> list[tuple[str, int]]:
    counts: Counter[str] = Counter(get_train_split()["category"])
    return sorted(counts.items())


def count_rows(
    *,
    category: str | None = None,
    intent: str | None = None,
    intent_contains: str | None = None,
    instruction_contains: str | None = None,
) -> int:
    return len(
        filter_rows(
            category=category,
            intent=intent,
            intent_contains=intent_contains,
            instruction_contains=instruction_contains,
        )
    )


def sample_rows(
    *,
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
    offset: int = 0,
    diverse: bool = True,
) -> list[dict[str, Any]]:
    """Return example rows; by default spreads across intents and unique instructions."""
    if diverse:
        return sample_rows_diverse(
            category=category, intent=intent, limit=limit, offset=offset
        )
    rows = filter_rows(category=category, intent=intent)
    return rows[: max(limit, 0)]


def _instruction_similarity(left: str, right: str) -> float:
    """Jaccard similarity on words — higher means more alike."""
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    if not left_words or not right_words:
        return 1.0 if left.strip().lower() == right.strip().lower() else 0.0
    union = left_words | right_words
    return len(left_words & right_words) / len(union)


def _is_distinct_row(candidate: dict[str, Any], chosen: list[dict[str, Any]]) -> bool:
    """Reject rows whose customer message is too similar to an already picked row."""
    text = candidate["instruction"]
    return all(_instruction_similarity(text, row["instruction"]) <= 0.45 for row in chosen)


def diverse_sample_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Sample rows with distinct intents first, then maximally different instructions."""
    if not rows or limit <= 0:
        return []

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)

    need = offset + limit
    sampled: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    intent_names = sorted(by_intent)

    # Phase 1: at most one example per intent (spread across intents).
    for intent_name in intent_names:
        if len(sampled) >= need:
            break
        for row in by_intent[intent_name]:
            key = row["instruction"].strip().lower()
            if key in seen_instructions:
                continue
            if not _is_distinct_row(row, sampled):
                continue
            seen_instructions.add(key)
            sampled.append(row)
            break

    # Phase 2: fill remainder with rows least similar to those already chosen.
    while len(sampled) < need:
        best_row: dict[str, Any] | None = None
        best_score = -1.0
        for intent_name in intent_names:
            for row in by_intent[intent_name]:
                key = row["instruction"].strip().lower()
                if key in seen_instructions or not _is_distinct_row(row, sampled):
                    continue
                score = min(
                    _instruction_similarity(row["instruction"], prior["instruction"])
                    for prior in sampled
                )
                if score > best_score:
                    best_score = score
                    best_row = row
        if best_row is None:
            break
        seen_instructions.add(best_row["instruction"].strip().lower())
        sampled.append(best_row)

    start = max(offset, 0)
    end = start + limit if limit > 0 else len(sampled)
    return sampled[start:end]


def sample_rows_diverse(
    *,
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Pick diverse examples after filtering by category and/or intent."""
    return diverse_sample_from_rows(
        filter_rows(category=category, intent=intent),
        limit=limit,
        offset=offset,
    )


def intent_distribution(category: str) -> list[tuple[str, int]]:
    rows = filter_rows(category=category)
    counts: Counter[str] = Counter(row["intent"] for row in rows)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def sample_rows_stratified(
    *,
    category: str | None = None,
    intent: str | None = None,
    intent_contains: str | None = None,
    instruction_contains: str | None = None,
    samples_per_intent: int = 8,
    max_total: int = 40,
) -> list[dict[str, Any]]:
    """Spread samples across intents so summarization covers the full topic mix."""
    rows = filter_rows(
        category=category,
        intent=intent,
        intent_contains=intent_contains,
        instruction_contains=instruction_contains,
    )
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)

    sampled: list[dict[str, Any]] = []
    for intent_name in sorted(by_intent):
        sampled.extend(by_intent[intent_name][:samples_per_intent])
        if len(sampled) >= max_total:
            break
    return sampled[:max_total]


def unique_responses(
    rows: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for row in rows:
        response = row["response"].strip()
        if response in seen:
            continue
        seen.add(response)
        unique.append(response)
        if len(unique) >= limit:
            break
    return unique
