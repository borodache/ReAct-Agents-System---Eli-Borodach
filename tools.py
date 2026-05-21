"""LangChain StructuredTools with Pydantic schemas for the dataset ReAct agent."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, Callable, TypeVar

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel

from dataset_store import (
    count_rows,
    diverse_sample_from_rows,
    filter_rows,
    intent_distribution,
    list_categories,
    sample_rows,
    sample_rows_stratified,
    unique_responses,
)
from filter_context import FilterSpec, ensure_thread_from_config, get_filter, save_filter, update_filter
from tool_schemas import (
    CategoryCount,
    CountDatasetRecordsInput,
    CountDatasetRecordsOutput,
    CountRowsInput,
    CountRowsOutput,
    DatasetExample,
    DeclineOutOfScopeInput,
    DeclineOutOfScopeOutput,
    FilterByCategoryInput,
    FilterByIntentInput,
    FilterSummary,
    FilterToolOutput,
    GatherAgentResponsePatternsInput,
    GatherAgentResponsePatternsOutput,
    GatherCategorySummarizationInput,
    GatherCategorySummarizationOutput,
    GetDatasetCategoriesInput,
    GetDatasetCategoriesOutput,
    GetDatasetExamplesInput,
    GetDatasetExamplesOutput,
    GetExamplesForFilterInput,
    GetIntentDistributionInput,
    GetIntentDistributionOutput,
    IntentCount,
    IntentDistributionForFilterInput,
    IntentDistributionForFilterOutput,
    SampleExchange,
)

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the Bitext customer-support training dataset "
    "(categories, intents, counts, examples, summaries of dataset content, and agent "
    "response patterns found in the data). I cannot help with general knowledge, news, "
    "sports, creative writing, or other tasks outside that dataset."
)

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


def _to_json(result: BaseModel) -> str:
    return result.model_dump_json(indent=2)


def _spec_to_summary(spec: FilterSpec, *, filter_id: str | None = None) -> FilterSummary:
    return FilterSummary(
        category=spec.category,
        intent=spec.intent,
        intent_contains=spec.intent_contains,
        instruction_contains=spec.instruction_contains,
        filter_id=filter_id,
    )


def _count_for_spec(spec: FilterSpec) -> int:
    return count_rows(
        category=spec.category,
        intent=spec.intent,
        intent_contains=spec.intent_contains,
        instruction_contains=spec.instruction_contains,
    )


def _rows_for_spec(spec: FilterSpec) -> list[dict]:
    return filter_rows(
        category=spec.category,
        intent=spec.intent,
        intent_contains=spec.intent_contains,
        instruction_contains=spec.instruction_contains,
    )


def _example_from_row(row: dict) -> DatasetExample:
    return DatasetExample(
        flags=row.get("flags"),
        instruction=row["instruction"],
        category=row.get("category"),
        intent=row["intent"],
        response=row["response"],
    )


def run_get_dataset_categories() -> GetDatasetCategoriesOutput:
    return GetDatasetCategoriesOutput(
        categories=[CategoryCount(category=name, count=count) for name, count in list_categories()]
    )


def run_count_dataset_records(params: CountDatasetRecordsInput) -> CountDatasetRecordsOutput:
    total = count_rows(
        category=params.category,
        intent=params.intent,
        intent_contains=params.intent_contains,
    )
    return CountDatasetRecordsOutput(
        count=total,
        filters=FilterSummary(
            category=params.category,
            intent=params.intent,
            intent_contains=params.intent_contains,
        ),
    )


def run_get_dataset_examples(params: GetDatasetExamplesInput) -> GetDatasetExamplesOutput:
    examples = sample_rows(
        category=params.category,
        intent=params.intent,
        limit=params.limit,
        offset=params.offset,
    )
    if not examples:
        return GetDatasetExamplesOutput(
            count=0,
            examples=[],
            message="No rows matched the given filters.",
            filters=FilterSummary(
                category=params.category,
                intent=params.intent,
                limit=params.limit,
            ),
        )
    unique_intents = len({row["intent"] for row in examples})
    message = None
    if unique_intents < len(examples) and params.category:
        message = (
            f"{params.category.upper()} has only {unique_intents} distinct intents in the "
            f"dataset; examples use different customer messages where possible."
        )

    return GetDatasetExamplesOutput(
        count=len(examples),
        examples=[_example_from_row(row) for row in examples],
        message=message,
        filters=FilterSummary(
            category=params.category,
            intent=params.intent,
            limit=params.limit,
        ),
    )


def run_filter_by_intent(params: FilterByIntentInput) -> FilterToolOutput:
    if not params.intent and not params.intent_contains:
        raise ValueError("Provide intent and/or intent_contains.")

    if params.filter_id:
        spec = get_filter(params.filter_id).merge(
            intent=params.intent,
            intent_contains=params.intent_contains,
        )
        filter_id = update_filter(params.filter_id, spec)
    else:
        spec = FilterSpec(intent=params.intent, intent_contains=params.intent_contains).normalized()
        filter_id = save_filter(spec)

    matched = _count_for_spec(get_filter(filter_id))
    return FilterToolOutput(
        filter_id=filter_id,
        filters=_spec_to_summary(get_filter(filter_id), filter_id=filter_id),
        matched_rows=matched,
        next_step_hint="Call count_rows(filter_id=...) to confirm the count, or get_examples_for_filter.",
    )


def run_filter_by_category(params: FilterByCategoryInput) -> FilterToolOutput:
    if params.filter_id:
        spec = get_filter(params.filter_id).merge(category=params.category)
        filter_id = update_filter(params.filter_id, spec)
    else:
        spec = FilterSpec(category=params.category).normalized()
        filter_id = save_filter(spec)

    matched = _count_for_spec(get_filter(filter_id))
    return FilterToolOutput(
        filter_id=filter_id,
        filters=_spec_to_summary(get_filter(filter_id), filter_id=filter_id),
        matched_rows=matched,
        next_step_hint="Call count_rows(filter_id=...) or get_examples_for_filter next.",
    )


def run_count_rows(params: CountRowsInput) -> CountRowsOutput:
    spec = get_filter(params.filter_id)  # resolves stale/hallucinated filter_id when possible
    total = _count_for_spec(spec)
    return CountRowsOutput(
        count=total,
        filter_id=params.filter_id,
        filters=_spec_to_summary(spec, filter_id=params.filter_id),
    )


def run_get_examples_for_filter(params: GetExamplesForFilterInput) -> GetDatasetExamplesOutput:
    spec = get_filter(params.filter_id)
    rows = diverse_sample_from_rows(
        _rows_for_spec(spec), limit=params.limit, offset=params.offset
    )
    if not rows:
        return GetDatasetExamplesOutput(
            count=0,
            examples=[],
            message="No rows matched this filter.",
            filters=_spec_to_summary(spec, filter_id=params.filter_id),
        )
    return GetDatasetExamplesOutput(
        count=len(rows),
        examples=[_example_from_row(row) for row in rows],
        filters=_spec_to_summary(spec, filter_id=params.filter_id),
    )


def run_intent_distribution_for_filter(
    params: IntentDistributionForFilterInput,
) -> IntentDistributionForFilterOutput:
    spec = get_filter(params.filter_id)
    rows = _rows_for_spec(spec)
    counts = Counter(row["intent"] for row in rows)
    distribution = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return IntentDistributionForFilterOutput(
        filter_id=params.filter_id,
        total_rows=len(rows),
        filters=_spec_to_summary(spec, filter_id=params.filter_id),
        intents=[IntentCount(intent=intent, count=count) for intent, count in distribution],
    )


def run_get_intent_distribution(params: GetIntentDistributionInput) -> GetIntentDistributionOutput:
    category_key = params.category.strip().upper()
    distribution = intent_distribution(category_key)
    return GetIntentDistributionOutput(
        category=category_key,
        total_rows=sum(count for _, count in distribution),
        intents=[IntentCount(intent=intent, count=count) for intent, count in distribution],
    )


def run_gather_category_summarization(
    params: GatherCategorySummarizationInput,
) -> GatherCategorySummarizationOutput:
    category_key = params.category.strip().upper()
    rows = filter_rows(category=category_key)
    if not rows:
        return GatherCategorySummarizationOutput(
            category=category_key,
            total_rows=0,
            intent_breakdown=[],
            sample_count=0,
            samples=[],
            guidance="No data available for this category.",
            message="No rows found for this category.",
        )

    samples = sample_rows_stratified(
        category=category_key,
        samples_per_intent=params.samples_per_intent,
    )
    return GatherCategorySummarizationOutput(
        category=category_key,
        total_rows=len(rows),
        intent_breakdown=[
            IntentCount(intent=intent, count=count)
            for intent, count in intent_distribution(category_key)
        ],
        sample_count=len(samples),
        samples=[_example_from_row(row) for row in samples],
        guidance=(
            "Summarize main customer concerns, agent tone, and recurring patterns. "
            "Note that conclusions are based on a stratified sample, not every row."
        ),
    )


def run_gather_agent_response_patterns(
    params: GatherAgentResponsePatternsInput,
) -> GatherAgentResponsePatternsOutput:
    filters = FilterSummary(
        category=params.category,
        intent=params.intent,
        intent_contains=params.intent_contains,
        instruction_contains=params.instruction_contains,
    )
    rows = filter_rows(
        category=params.category,
        intent=params.intent,
        intent_contains=params.intent_contains,
        instruction_contains=params.instruction_contains,
    )
    if not rows:
        return GatherAgentResponsePatternsOutput(
            total_matching_rows=0,
            filters=filters,
            intent_breakdown={},
            sample_exchanges=[],
            distinct_response_templates=[],
            guidance="No matching rows to analyze.",
            message="No rows matched the given filters.",
        )

    samples = sample_rows_stratified(
        category=params.category,
        intent=params.intent,
        intent_contains=params.intent_contains,
        instruction_contains=params.instruction_contains,
        samples_per_intent=params.samples_per_intent,
        max_total=30,
    )
    return GatherAgentResponsePatternsOutput(
        total_matching_rows=len(rows),
        filters=filters,
        intent_breakdown=dict(Counter(row["intent"] for row in rows)),
        sample_exchanges=[
            SampleExchange(
                intent=row["intent"],
                category=row["category"],
                instruction=row["instruction"],
                response=row["response"],
            )
            for row in samples[:12]
        ],
        distinct_response_templates=unique_responses(
            samples,
            limit=params.max_unique_responses,
        ),
        guidance=(
            "Describe typical agent tone, commitments, and steps offered. "
            "Base claims only on the exchanges and templates returned here."
        ),
    )


def run_decline_out_of_scope(params: DeclineOutOfScopeInput) -> DeclineOutOfScopeOutput:
    return DeclineOutOfScopeOutput(
        standard_message=OUT_OF_SCOPE_MESSAGE,
        reason=params.reason.strip(),
    )


def _make_tool(
    *,
    name: str,
    description: str,
    args_schema: type[TInput],
    runner: Callable[[TInput], TOutput],
) -> StructuredTool:
    def _invoke(
        config: Annotated[RunnableConfig, InjectedToolArg],
        **kwargs: Any,
    ) -> str:
        ensure_thread_from_config(config)
        params = args_schema.model_validate(kwargs)
        return _to_json(runner(params))

    return StructuredTool.from_function(
        func=_invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _make_tool_no_args(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    runner: Callable[[], TOutput],
) -> StructuredTool:
    def _invoke(
        config: Annotated[RunnableConfig, InjectedToolArg],
        **kwargs: Any,
    ) -> str:
        ensure_thread_from_config(config)
        args_schema.model_validate(kwargs)
        return _to_json(runner())

    return StructuredTool.from_function(
        func=_invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )


get_dataset_categories_tool: BaseTool = _make_tool_no_args(
    name="get_dataset_categories",
    description=(
        "List every category label in the Bitext dataset with row counts. "
        "Use for questions like 'What categories exist?' or 'How many categories are there?'"
    ),
    args_schema=GetDatasetCategoriesInput,
    runner=run_get_dataset_categories,
)

filter_by_intent_tool: BaseTool = _make_tool(
    name="filter_by_intent",
    description=(
        "Step 1 of a filter chain: save rows matching an intent (exact or substring). "
        "Returns filter_id for the next tool. "
        "Example chain: filter_by_intent(intent='get_refund') -> count_rows(filter_id=...)."
    ),
    args_schema=FilterByIntentInput,
    runner=run_filter_by_intent,
)

filter_by_category_tool: BaseTool = _make_tool(
    name="filter_by_category",
    description=(
        "Step 1 (or 2) of a filter chain: save rows matching a category. "
        "Pass filter_id to narrow an existing filter. Returns filter_id for count_rows or get_examples_for_filter."
    ),
    args_schema=FilterByCategoryInput,
    runner=run_filter_by_category,
)

count_rows_tool: BaseTool = _make_tool(
    name="count_rows",
    description=(
        "Step 2 of a filter chain: count rows for a filter_id from filter_by_intent or filter_by_category. "
        "Always call a filter_by_* tool first."
    ),
    args_schema=CountRowsInput,
    runner=run_count_rows,
)

count_dataset_records_tool: BaseTool = _make_tool(
    name="count_dataset_records",
    description=(
        "Single-step count with inline filters (no filter_id). "
        "Prefer chaining filter_by_intent -> count_rows for multi-step reasoning when an intent is named."
    ),
    args_schema=CountDatasetRecordsInput,
    runner=run_count_dataset_records,
)

get_dataset_examples_tool: BaseTool = _make_tool(
    name="get_dataset_examples",
    description=(
        "Return diverse example rows (different intents and customer messages when possible). "
        "Use category=SHIPPING for shipping category, or intent= for a specific intent. "
        "Use offset to skip rows already shown (e.g. offset=3 for 'show 3 more' after 3 examples). "
        "For chained filters, use get_examples_for_filter after filter_by_*."
    ),
    args_schema=GetDatasetExamplesInput,
    runner=run_get_dataset_examples,
)

get_examples_for_filter_tool: BaseTool = _make_tool(
    name="get_examples_for_filter",
    description=(
        "Return diverse example rows for a saved filter_id (varied intents/instructions). "
        "Use offset to paginate after prior examples. "
        "Chain: filter_by_category('SHIPPING') -> get_examples_for_filter(filter_id, limit=5)."
    ),
    args_schema=GetExamplesForFilterInput,
    runner=run_get_examples_for_filter,
)

get_intent_distribution_tool: BaseTool = _make_tool(
    name="get_intent_distribution_for_category",
    description=(
        "Intent distribution for one category (single step). "
        "For chained filters: filter_by_category -> intent_distribution_for_filter."
    ),
    args_schema=GetIntentDistributionInput,
    runner=run_get_intent_distribution,
)

intent_distribution_for_filter_tool: BaseTool = _make_tool(
    name="intent_distribution_for_filter",
    description=(
        "Intent distribution within a saved filter_id. "
        "Chain: filter_by_category('ACCOUNT') -> intent_distribution_for_filter(filter_id)."
    ),
    args_schema=IntentDistributionForFilterInput,
    runner=run_intent_distribution_for_filter,
)

gather_category_summarization_tool: BaseTool = _make_tool(
    name="gather_category_for_summarization",
    description=(
        "Collect stratified customer messages and agent replies from one category for summarization. "
        "Use for open-ended questions such as 'Summarize the FEEDBACK category.'"
    ),
    args_schema=GatherCategorySummarizationInput,
    runner=run_gather_category_summarization,
)

gather_agent_response_patterns_tool: BaseTool = _make_tool(
    name="gather_agent_response_patterns",
    description=(
        "Collect diverse agent response templates and sample exchanges for qualitative analysis. "
        "Use for 'How do reps typically respond to ...?' questions. "
        "For cancellation topics, set intent_contains='cancel'."
    ),
    args_schema=GatherAgentResponsePatternsInput,
    runner=run_gather_agent_response_patterns,
)

decline_out_of_scope_tool: BaseTool = _make_tool(
    name="decline_out_of_scope_question",
    description=(
        "Record that a question is outside dataset scope (general knowledge, sports, creative writing). "
        "The router usually handles out-of-scope queries before tools run."
    ),
    args_schema=DeclineOutOfScopeInput,
    runner=run_decline_out_of_scope,
)

STRUCTURED_TOOLS: list[BaseTool] = [
    get_dataset_categories_tool,
    filter_by_intent_tool,
    filter_by_category_tool,
    count_rows_tool,
    count_dataset_records_tool,
    get_dataset_examples_tool,
    get_examples_for_filter_tool,
    get_intent_distribution_tool,
    intent_distribution_for_filter_tool,
]

UNSTRUCTURED_TOOLS: list[BaseTool] = [
    gather_category_summarization_tool,
    gather_agent_response_patterns_tool,
]

OUT_OF_SCOPE_TOOLS: list[BaseTool] = [
    decline_out_of_scope_tool,
]

ALL_TOOLS: list[BaseTool] = STRUCTURED_TOOLS + UNSTRUCTURED_TOOLS + OUT_OF_SCOPE_TOOLS
