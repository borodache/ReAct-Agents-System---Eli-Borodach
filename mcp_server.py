"""MCP server exposing Bitext dataset tools via FastMCP.

Run (HTTP, default — for fastmcp call / remote clients):
    python mcp_server.py
    fastmcp call http://localhost:8000/mcp get_dataset_categories

Run (stdio, for Cursor / Claude Desktop):
    python mcp_server.py --stdio

Or with the FastMCP CLI:
    fastmcp run mcp_server.py:mcp
"""

from __future__ import annotations

import config  # noqa: F401 — loads .env / truststore before dataset access

from fastmcp import FastMCP

from answer_format import format_tool_result
from filter_context import init_filter_context
from tool_schemas import (
    CountDatasetRecordsInput,
    FilterByIntentInput,
    GetDatasetCategoriesInput,
    GetDatasetExamplesInput,
)
from tools import (
    run_count_dataset_records,
    run_filter_by_intent,
    run_get_dataset_categories,
    run_get_dataset_examples,
)

mcp = FastMCP(
    name="Bitext Dataset Tools",
    instructions=(
        "Tools for querying the Bitext customer-support training dataset: "
        "categories, counts, filters, and example rows. "
        "For chained counts, call filter_by_intent first, then use the returned filter_id."
    ),
)


@mcp.tool
def get_dataset_categories() -> str:
    """List every category label in the Bitext dataset with row counts."""
    GetDatasetCategoriesInput.model_validate({})
    return format_tool_result(run_get_dataset_categories())


@mcp.tool
def count_dataset_records(
    category: str | None = None,
    intent: str | None = None,
    intent_contains: str | None = None,
) -> str:
    """Count dataset rows with optional category/intent filters (no filter_id chain)."""
    params = CountDatasetRecordsInput(
        category=category,
        intent=intent,
        intent_contains=intent_contains,
    )
    return format_tool_result(run_count_dataset_records(params))


@mcp.tool
def filter_by_intent(
    intent: str | None = None,
    intent_contains: str | None = None,
    filter_id: str | None = None,
) -> str:
    """Save rows matching an intent; returns filter_id for follow-up tools in this session."""
    init_filter_context()
    params = FilterByIntentInput(
        intent=intent,
        intent_contains=intent_contains,
        filter_id=filter_id,
    )
    return format_tool_result(run_filter_by_intent(params))


@mcp.tool
def get_dataset_examples(
    limit: int = 3,
    offset: int = 0,
    category: str | None = None,
    intent: str | None = None,
) -> str:
    """Return diverse example rows (category and/or intent optional)."""
    params = GetDatasetExamplesInput(
        limit=limit,
        offset=offset,
        category=category,
        intent=intent,
    )
    return format_tool_result(run_get_dataset_examples(params))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bitext dataset MCP server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport for Cursor / Claude Desktop (default: HTTP)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.stdio:
        mcp.run()
    else:
        mcp.run(transport="http", host=args.host, port=args.port)
