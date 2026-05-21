"""Turn tool JSON / text tool calls into natural-language answers."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from filter_context import init_filter_context
from tool_schemas import (
    CountDatasetRecordsInput,
    CountDatasetRecordsOutput,
    CountRowsInput,
    CountRowsOutput,
    FilterByCategoryInput,
    FilterByIntentInput,
    FilterToolOutput,
    GatherAgentResponsePatternsOutput,
    GatherCategorySummarizationOutput,
    GetDatasetCategoriesOutput,
    GetDatasetExamplesOutput,
    GetIntentDistributionOutput,
    IntentDistributionForFilterOutput,
)
from tools import (
    run_count_dataset_records,
    run_count_rows,
    run_filter_by_category,
    run_filter_by_intent,
    run_get_dataset_categories,
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _looks_like_text_tool_call(text: str) -> bool:
    data = _extract_json_object(text)
    if not data:
        return False
    return bool(data.get("name") or data.get("type") == "function")


def _looks_like_tool_result_json(text: str) -> bool:
    data = _extract_json_object(text)
    if not data:
        return False
    tool_keys = (
        "categories",
        "count",
        "examples",
        "intents",
        "intent_breakdown",
        "distinct_response_templates",
        "sample_exchanges",
        "matched_rows",
        "filter_id",
    )
    return any(key in data for key in tool_keys)


def _strip_react_artifacts(text: str) -> str:
    """Remove ReAct labels and fences so prose or JSON can be parsed."""
    text = text.strip()
    text = re.sub(r"^Final Answer:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Thought:\s*.*?(?=\n(?:Action:|Observation:|Final Answer:)|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^Action:.*?\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^Observation:\s*", "", text, flags=re.IGNORECASE)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def coerce_text_to_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Convert JSON-in-content into OpenAI-style tool_calls for the ReAct loop."""
    data = _extract_json_object(text)
    if not data:
        return None

    name = data.get("name")
    if not name and isinstance(data.get("function"), dict):
        name = data["function"].get("name")
    if not name:
        return None

    params = data.get("parameters") or data.get("arguments") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}

    if name == "count_rows" and isinstance(params.get("filter_id"), dict):
        nested = params["filter_id"]
        if nested.get("function_name") == "filter_by_intent":
            args_list = nested.get("args") or [{}]
            intent_args = args_list[0] if args_list else {}
            return [
                {
                    "name": "filter_by_intent",
                    "args": intent_args,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                }
            ]

    if not isinstance(params, dict):
        params = {}

    return [
        {
            "name": name,
            "args": params,
            "id": f"call_{uuid.uuid4().hex[:8]}",
        }
    ]


def _execute_text_tool_plan(data: dict[str, Any], question: str) -> str | None:
    """Run tools when the model printed JSON instead of using native tool_calls."""
    init_filter_context(reset=True)

    name = data.get("name")
    params = data.get("parameters") or data.get("arguments") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}

    q = question.lower()

    if name == "count_rows" and isinstance(params.get("filter_id"), dict):
        nested = params["filter_id"]
        if nested.get("function_name") == "filter_by_intent":
            args_list = nested.get("args") or [{}]
            intent = (args_list[0] or {}).get("intent")
            if intent:
                filt = run_filter_by_intent(FilterByIntentInput(intent=intent))
                count = run_count_rows(CountRowsInput(filter_id=filt.filter_id))
                return _format_count_answer(count, question=q, intent=intent)

    if name == "filter_by_intent":
        filt = run_filter_by_intent(
            FilterByIntentInput(
                intent=params.get("intent"),
                intent_contains=params.get("intent_contains"),
                filter_id=params.get("filter_id"),
            )
        )
        return _format_filter_answer(filt.matched_rows, filt.filters.category, filt.filters.intent)

    if name == "count_rows" and params.get("filter_id"):
        count = run_count_rows(CountRowsInput(filter_id=str(params["filter_id"])))
        return _format_count_answer(count, question=q)

    if name == "count_dataset_records":
        out = run_count_dataset_records(
            CountDatasetRecordsInput(
                category=params.get("category"),
                intent=params.get("intent"),
                intent_contains=params.get("intent_contains"),
            )
        )
        return _format_count_answer(out, question=q, category=out.filters.category, intent=out.filters.intent)

    if name == "get_dataset_categories":
        out = run_get_dataset_categories()
        return _format_categories_answer(out)

    if name == "filter_by_category" and params.get("category"):
        filt = run_filter_by_category(
            FilterByCategoryInput(category=params["category"], filter_id=params.get("filter_id"))
        )
        count = run_count_rows(CountRowsInput(filter_id=filt.filter_id))
        return _format_count_answer(count, question=q, category=params["category"])

    if "refund" in q and ("how many" in q or "count" in q):
        if params.get("intent") == "get_refund" or name in (None, "count_rows"):
            filt = run_filter_by_intent(FilterByIntentInput(intent="get_refund"))
            count = run_count_rows(CountRowsInput(filter_id=filt.filter_id))
            return _format_count_answer(count, question=q, intent="get_refund")

    return None


def _format_filter_answer(
    matched_rows: int,
    category: str | None,
    intent: str | None,
) -> str:
    parts = [f"I found {matched_rows:,} matching rows"]
    if category:
        parts.append(f"in category {category.upper()}")
    if intent:
        parts.append(f"with intent {intent}")
    return " ".join(parts) + "."


def _format_count_answer(
    result: CountRowsOutput | CountDatasetRecordsOutput,
    *,
    question: str,
    category: str | None = None,
    intent: str | None = None,
) -> str:
    count = result.count
    filters = getattr(result, "filters", None)
    category = category or (filters.category if filters else None)
    intent = intent or (filters.intent if filters else None)
    q = question.lower()
    if "refund" in q or intent == "get_refund" or (category or "").upper() == "REFUND":
        return (
            f"There are {count:,} refund-related records in the dataset "
            f"(intent get_refund: {count:,} rows)."
        )
    if category:
        return f"There are {count:,} rows in the {category.upper()} category."
    if intent:
        return f"There are {count:,} rows with intent {intent}."
    return f"There are {count:,} matching records in the dataset."


def _format_categories_answer(result: GetDatasetCategoriesOutput) -> str:
    lines = [f"- {item.category}: {item.count:,}" for item in result.categories]
    names = ", ".join(item.category for item in result.categories)
    return (
        f"The dataset has {len(result.categories)} categories: {names}.\n\n"
        "Row counts:\n" + "\n".join(lines)
    )


def _format_examples_answer(result: GetDatasetExamplesOutput, *, question: str = "") -> str:
    if not result.examples:
        return result.message or "No examples matched your filters."

    filters = result.filters
    header_parts = [f"Here are {len(result.examples)} example(s)"]
    if filters and filters.category:
        header_parts.append(f"from the {filters.category.upper()} category")
    elif filters and filters.intent:
        header_parts.append(f"for intent {filters.intent}")
    header = " ".join(header_parts) + ":"
    if result.message:
        header += f"\n({result.message})"
    header += "\n"

    blocks: list[str] = []
    intent_counts: dict[str, int] = {}
    for index, ex in enumerate(result.examples, start=1):
        intent_counts[ex.intent] = intent_counts.get(ex.intent, 0) + 1
        intent_label = ex.intent
        if intent_counts[ex.intent] > 1:
            intent_label = f"{ex.intent} (different customer message)"
        cat = f" [{ex.category}]" if ex.category else ""
        blocks.append(
            f"{index}. Intent: {intent_label}{cat}\n"
            f"   Customer: {ex.instruction}\n"
            f"   Agent: {ex.response}"
        )
    return header + "\n\n".join(blocks)


def _format_intent_distribution(
    *,
    total_rows: int,
    intents: list,
    category: str | None = None,
    filter_label: str | None = None,
) -> str:
    scope = ""
    if category:
        scope = f" in the {category.upper()} category"
    elif filter_label:
        scope = f" for the selected filter ({filter_label})"

    lines = [f"Intent distribution{scope} ({total_rows:,} rows total):\n"]
    for item in intents:
        intent = item.intent if hasattr(item, "intent") else item.get("intent", "")
        count = item.count if hasattr(item, "count") else item.get("count", 0)
        pct = (100.0 * count / total_rows) if total_rows else 0.0
        lines.append(f"- {intent}: {count:,} ({pct:.1f}%)")
    return "\n".join(lines)


def _format_gather_category_answer(result: GatherCategorySummarizationOutput) -> str:
    lines = [
        f"Summary material for category {result.category.upper()} "
        f"({result.total_rows:,} rows in the dataset; {result.sample_count} sampled exchanges):\n"
    ]
    if result.intent_breakdown:
        lines.append("Intent mix in the sample:")
        for item in result.intent_breakdown[:8]:
            lines.append(f"- {item.intent}: {item.count}")
        lines.append("")

    if result.samples:
        lines.append("Representative examples:")
        for index, ex in enumerate(result.samples[:5], start=1):
            lines.append(
                f"{index}. [{ex.intent}] Customer: {ex.instruction[:200]}"
                + ("…" if len(ex.instruction) > 200 else "")
            )
            lines.append(f"   Agent: {ex.response[:200]}" + ("…" if len(ex.response) > 200 else ""))

    if result.guidance:
        lines.append(f"\nNote: {result.guidance}")
    return "\n".join(lines)


def _format_gather_patterns_answer(result: GatherAgentResponsePatternsOutput) -> str:
    lines = [
        f"Found {result.total_matching_rows:,} matching conversations in the dataset.\n"
    ]
    if result.intent_breakdown:
        top = sorted(result.intent_breakdown.items(), key=lambda x: -x[1])[:6]
        lines.append("Top intents: " + ", ".join(f"{k} ({v})" for k, v in top) + "\n")

    if result.distinct_response_templates:
        lines.append("Common agent response patterns:")
        for index, template in enumerate(result.distinct_response_templates[:8], start=1):
            preview = template[:300] + ("…" if len(template) > 300 else "")
            lines.append(f"{index}. {preview}")

    if result.guidance:
        lines.append(f"\nNote: {result.guidance}")
    return "\n".join(lines)


def format_tool_result(result: BaseModel, *, include_structured: bool = False) -> str:
    """Natural-language summary for MCP tool responses (JSON optional)."""
    json_text = result.model_dump_json(indent=2)
    summary = format_tool_observation(json_text)
    if isinstance(result, FilterToolOutput):
        if summary:
            summary = (
                f"{summary.rstrip('.')}. "
                f"Use filter_id={result.filter_id!r} in follow-up tools. "
                f"{result.next_step_hint}"
            )
        else:
            summary = (
                f"I found {result.matched_rows:,} matching rows. "
                f"Use filter_id={result.filter_id!r} in follow-up tools. "
                f"{result.next_step_hint}"
            )
    if not summary:
        return json_text
    if include_structured:
        return f"{summary}\n\n---\nStructured data:\n{json_text}"
    return summary


def format_tool_observation(content: str, *, question: str = "") -> str | None:
    """Convert a single tool JSON observation into readable text."""
    data = _extract_json_object(content)
    if not data:
        return None

    try:
        if "categories" in data and isinstance(data.get("categories"), list):
            return _format_categories_answer(GetDatasetCategoriesOutput.model_validate(data))

        if "examples" in data:
            return _format_examples_answer(
                GetDatasetExamplesOutput.model_validate(data), question=question
            )

        if "intents" in data and "category" in data and "total_rows" in data:
            out = GetIntentDistributionOutput.model_validate(data)
            return _format_intent_distribution(
                total_rows=out.total_rows,
                intents=out.intents,
                category=out.category,
            )

        if "intents" in data and "filter_id" in data:
            out = IntentDistributionForFilterOutput.model_validate(data)
            label = out.filters.category or out.filters.intent or out.filter_id
            return _format_intent_distribution(
                total_rows=out.total_rows,
                intents=out.intents,
                filter_label=label,
            )

        if "intent_breakdown" in data and "samples" in data:
            return _format_gather_category_answer(
                GatherCategorySummarizationOutput.model_validate(data)
            )

        if "distinct_response_templates" in data:
            return _format_gather_patterns_answer(
                GatherAgentResponsePatternsOutput.model_validate(data)
            )

        if "count" in data:
            filters = data.get("filters") or {}
            if data.get("filter_id"):
                return _format_count_answer(
                    CountRowsOutput.model_validate(data),
                    question=question,
                    category=filters.get("category"),
                    intent=filters.get("intent"),
                )
            return _format_count_answer(
                CountDatasetRecordsOutput.model_validate(data),
                question=question,
                category=filters.get("category"),
                intent=filters.get("intent"),
            )

        if "matched_rows" in data and "filter_id" in data:
            filters = data.get("filters") or {}
            return _format_filter_answer(
                int(data["matched_rows"]),
                filters.get("category"),
                filters.get("intent"),
            )
    except ValidationError:
        pass

    return _format_from_dict_heuristic(data, question=question)


def _format_from_dict_heuristic(data: dict[str, Any], *, question: str = "") -> str | None:
    """Best-effort formatting when pydantic validation fails."""
    if "count" in data:
        count = int(data["count"])
        filters = data.get("filters") or {}
        fake = CountDatasetRecordsOutput(
            count=count,
            filters=filters,  # type: ignore[arg-type]
        )
        try:
            return _format_count_answer(
                fake,
                question=question,
                category=filters.get("category") if isinstance(filters, dict) else None,
                intent=filters.get("intent") if isinstance(filters, dict) else None,
            )
        except Exception:
            return f"There are {count:,} matching records."

    if "examples" in data and isinstance(data["examples"], list):
        lines = ["Here are example rows from the dataset:\n"]
        for index, ex in enumerate(data["examples"][:10], start=1):
            if isinstance(ex, dict):
                lines.append(
                    f"{index}. [{ex.get('intent', '?')}] Customer: {ex.get('instruction', '')}\n"
                    f"   Agent: {ex.get('response', '')}"
                )
        return "\n\n".join(lines)

    if "intents" in data and isinstance(data["intents"], list):
        total = int(data.get("total_rows", 0))
        return _format_intent_distribution(
            total_rows=total,
            intents=data["intents"],
            category=data.get("category"),
        )

    return None


def synthesize_from_messages(messages: list[BaseMessage]) -> str | None:
    """Build a natural-language answer from tool results when the model skipped a summary."""
    question = ""
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            question = str(message.content)

    tool_contents: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.content:
            tool_contents.append(str(message.content))

    if not tool_contents:
        return None

    for content in reversed(tool_contents):
        formatted = format_tool_observation(content, question=question)
        if formatted and "filter_id=" not in formatted.lower():
            return formatted

    for content in reversed(tool_contents):
        formatted = format_tool_observation(content, question=question)
        if formatted:
            return formatted

    return None


def to_natural_language_answer(messages: list[BaseMessage]) -> str:
    """Pick or synthesize a plain-English answer from the message history."""
    last_human = ""
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            last_human = str(message.content)

    for message in reversed(messages):
        if not isinstance(message, AIMessage) or not message.content:
            continue
        if message.tool_calls:
            continue

        text = message.content if isinstance(message.content, str) else str(message.content)
        text = _strip_react_artifacts(text)
        if not text:
            continue

        if _looks_like_text_tool_call(text):
            executed = _execute_text_tool_plan(_extract_json_object(text) or {}, last_human)
            if executed:
                return executed
            continue

        if _looks_like_tool_result_json(text):
            from_obs = format_tool_observation(text, question=last_human)
            if from_obs:
                return from_obs

        if not text.startswith("{"):
            return text

        from_obs = format_tool_observation(text, question=last_human)
        if from_obs:
            return from_obs

    synthesized = synthesize_from_messages(messages)
    if synthesized:
        return synthesized

    return "No response generated."
