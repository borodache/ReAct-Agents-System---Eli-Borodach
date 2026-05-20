"""In-memory filter context for chaining tools within one agent invocation."""

from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    """Criteria shared across filter → count / examples / distribution tools."""

    category: str | None = None
    intent: str | None = None
    intent_contains: str | None = None
    instruction_contains: str | None = None

    def normalized(self) -> FilterSpec:
        return FilterSpec(
            category=self.category.strip().upper() if self.category else None,
            intent=self.intent.strip().upper() if self.intent else None,
            intent_contains=self.intent_contains.strip().upper() if self.intent_contains else None,
            instruction_contains=(
                self.instruction_contains.strip().upper() if self.instruction_contains else None
            ),
        )

    def merge(self, **updates: str | None) -> FilterSpec:
        data = self.model_dump()
        for key, value in updates.items():
            if value is not None:
                data[key] = value
        return FilterSpec(**data).normalized()


_filter_store: ContextVar[dict[str, FilterSpec]] = ContextVar(
    "filter_store",
    default=None,
)


def init_filter_context() -> None:
    """Reset the filter store at the start of an agent run."""
    _filter_store.set({})


def clear_filter_context() -> None:
    _filter_store.set({})


def _store() -> dict[str, FilterSpec]:
    current = _filter_store.get()
    if current is None:
        current = {}
        _filter_store.set(current)
    return current


def save_filter(spec: FilterSpec, *, filter_id: str | None = None) -> str:
    store = _store()
    key = filter_id or uuid4().hex[:8]
    store[key] = spec.normalized()
    return key


def get_filter(filter_id: str) -> FilterSpec:
    store = _store()
    if filter_id not in store:
        raise KeyError(
            f"Unknown filter_id '{filter_id}'. Call filter_by_intent or filter_by_category first."
        )
    return store[filter_id]


def update_filter(filter_id: str, spec: FilterSpec) -> str:
    store = _store()
    store[filter_id] = spec.normalized()
    return filter_id
