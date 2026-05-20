"""Pydantic input and output schemas for dataset agent tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured tool inputs
# ---------------------------------------------------------------------------


class GetDatasetCategoriesInput(BaseModel):
    """Input for listing all dataset categories (no parameters)."""


class CountDatasetRecordsInput(BaseModel):
    """Input for counting rows with optional filters."""

    category: str | None = Field(
        default=None,
        description="Exact category label, e.g. REFUND, SHIPPING, ACCOUNT (case-insensitive).",
    )
    intent: str | None = Field(
        default=None,
        description="Exact intent label, e.g. get_refund, track_order (case-insensitive).",
    )
    intent_contains: str | None = Field(
        default=None,
        description='Substring matched inside intent, e.g. "refund" matches get_refund and track_refund.',
    )


class GetDatasetExamplesInput(BaseModel):
    """Input for fetching example dataset rows."""

    limit: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of examples to return.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many diverse examples (use for 'show me more' follow-ups).",
    )
    category: str | None = Field(
        default=None,
        description="Optional category filter, e.g. SHIPPING.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional intent filter, e.g. delivery_options.",
    )


class GetIntentDistributionInput(BaseModel):
    """Input for intent counts within one category."""

    category: str = Field(
        description="Category label, e.g. ACCOUNT or SHIPPING.",
    )


class FilterByIntentInput(BaseModel):
    """Input for creating or narrowing a filter by intent."""

    intent: str | None = Field(
        default=None,
        description="Exact intent label, e.g. get_refund, cancel_order (case-insensitive).",
    )
    intent_contains: str | None = Field(
        default=None,
        description='Substring matched inside intent, e.g. "refund".',
    )
    filter_id: str | None = Field(
        default=None,
        description="Optional existing filter_id to narrow; omit to start a new filter chain.",
    )


class FilterByCategoryInput(BaseModel):
    """Input for creating or narrowing a filter by category."""

    category: str = Field(
        description="Exact category label, e.g. REFUND, ORDER, ACCOUNT.",
    )
    filter_id: str | None = Field(
        default=None,
        description="Optional existing filter_id to narrow; omit to start a new filter chain.",
    )


class CountRowsInput(BaseModel):
    """Input for counting rows in a saved filter."""

    filter_id: str = Field(
        description="filter_id returned by filter_by_intent or filter_by_category.",
    )


class GetExamplesForFilterInput(BaseModel):
    """Input for examples from a saved filter."""

    filter_id: str = Field(
        description="filter_id returned by a filter_by_* tool.",
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of examples to return.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many diverse examples (use for 'show me more' follow-ups).",
    )


class IntentDistributionForFilterInput(BaseModel):
    """Input for intent breakdown within a saved filter."""

    filter_id: str = Field(
        description="filter_id returned by a filter_by_* tool.",
    )


# ---------------------------------------------------------------------------
# Unstructured tool inputs
# ---------------------------------------------------------------------------


class GatherCategorySummarizationInput(BaseModel):
    """Input for collecting category content to summarize."""

    category: str = Field(
        description="Category to summarize, e.g. FEEDBACK.",
    )
    samples_per_intent: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Max examples to take per intent for stratified sampling.",
    )


class GatherAgentResponsePatternsInput(BaseModel):
    """Input for collecting agent response patterns."""

    category: str | None = Field(
        default=None,
        description="Optional category filter.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional exact intent filter.",
    )
    intent_contains: str | None = Field(
        default=None,
        description='Substring for intent, e.g. "cancel" for cancellation-related rows.',
    )
    instruction_contains: str | None = Field(
        default=None,
        description="Substring to match inside the customer instruction text.",
    )
    samples_per_intent: int = Field(
        default=6,
        ge=1,
        le=30,
        description="Max examples per intent in the stratified sample.",
    )
    max_unique_responses: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum number of distinct agent response templates to return.",
    )


class DeclineOutOfScopeInput(BaseModel):
    """Input for recording an out-of-scope question (router handles most cases)."""

    reason: str = Field(
        description="Brief explanation of why the question is outside dataset scope.",
    )


# ---------------------------------------------------------------------------
# Shared / nested output models
# ---------------------------------------------------------------------------


class CategoryCount(BaseModel):
    category: str
    count: int


class IntentCount(BaseModel):
    intent: str
    count: int


class DatasetExample(BaseModel):
    intent: str
    instruction: str
    response: str
    flags: str | None = None
    category: str | None = None


class FilterSummary(BaseModel):
    category: str | None = None
    intent: str | None = None
    intent_contains: str | None = None
    instruction_contains: str | None = None
    limit: int | None = None
    filter_id: str | None = None


class FilterToolOutput(BaseModel):
    """Returned by filter_by_* tools for use in the next chained step."""

    filter_id: str
    filters: FilterSummary
    matched_rows: int
    next_step_hint: str = Field(
        description="Suggested follow-up tool, e.g. count_rows or get_examples_for_filter.",
    )


class SampleExchange(BaseModel):
    intent: str
    category: str
    instruction: str
    response: str


# ---------------------------------------------------------------------------
# Structured tool outputs
# ---------------------------------------------------------------------------


class GetDatasetCategoriesOutput(BaseModel):
    categories: list[CategoryCount]


class CountDatasetRecordsOutput(BaseModel):
    count: int
    filters: FilterSummary


class CountRowsOutput(BaseModel):
    count: int
    filter_id: str
    filters: FilterSummary


class GetDatasetExamplesOutput(BaseModel):
    count: int
    examples: list[DatasetExample]
    message: str | None = None
    filters: FilterSummary | None = None


class GetIntentDistributionOutput(BaseModel):
    category: str
    total_rows: int
    intents: list[IntentCount]


class IntentDistributionForFilterOutput(BaseModel):
    filter_id: str
    total_rows: int
    filters: FilterSummary
    intents: list[IntentCount]


# ---------------------------------------------------------------------------
# Unstructured tool outputs
# ---------------------------------------------------------------------------


class GatherCategorySummarizationOutput(BaseModel):
    category: str
    total_rows: int
    intent_breakdown: list[IntentCount]
    sample_count: int
    samples: list[DatasetExample]
    guidance: str
    message: str | None = None


class GatherAgentResponsePatternsOutput(BaseModel):
    total_matching_rows: int
    filters: FilterSummary
    intent_breakdown: dict[str, int]
    sample_exchanges: list[SampleExchange]
    distinct_response_templates: list[str]
    guidance: str
    message: str | None = None


class DeclineOutOfScopeOutput(BaseModel):
    status: Literal["out_of_scope"] = "out_of_scope"
    standard_message: str
    reason: str
