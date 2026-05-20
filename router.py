"""Query router: classifies user questions before ReAct tool selection."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

QueryType = Literal["structured", "unstructured", "out_of_scope", "profile_recall"]


class QueryClassification(BaseModel):
    """Router output schema."""

    query_type: QueryType = Field(
        description=(
            "structured: exact counts, lists, examples, or intent distributions; "
            "unstructured: summaries or qualitative patterns in dataset text; "
            "out_of_scope: general knowledge, creative tasks, or anything not requiring dataset rows; "
            "profile_recall: user asks what the assistant remembers about them (user profile, not dataset)"
        )
    )
    reason: str = Field(description="One sentence explaining the classification.")


ROUTER_SYSTEM_PROMPT = """You classify user questions for a Bitext customer-support DATASET assistant.

Choose exactly one label:

structured — Concrete, data-driven questions answerable with counts, lists, examples, or distributions.
  Examples: "What categories exist?", "How many refund requests?", "Show 3 SHIPPING examples",
  "Distribution of intents in ACCOUNT category?"

unstructured — Open-ended questions needing summarization of dataset content (not single numbers).
  Examples: "Summarize the FEEDBACK category.",
  "How do reps typically respond to cancellation requests?"

out_of_scope — NOT answerable from the dataset; do not use general knowledge to answer these.
  Examples: sports/news/trivia, "Who won the 2024 Champions League?",
  creative writing ("Write a poem about customer service"), coding help, personal advice.
  Note: creative writing about customer service is still out_of_scope even though on-topic.

profile_recall — User asks about THEIR stored profile / what you remember about THEM (not dataset stats).
  Examples: "What do you remember about me?", "What's my name?", "Where am I from?", "Am I a chess player?"

When unsure between structured and unstructured, prefer structured if the user asks for counts,
lists, examples, or distributions. Prefer unstructured for "summarize", "typically", "patterns", "themes".

Follow-up messages (classify the LATEST user turn using prior context):
- "Show me 3 more" / "another 3" after examples → structured (same task as before).
- "What about refunds?" after a count question → structured (new filter/count on refunds).
- "What is the total of the last two?" → structured (combine prior numeric answers).

Output format (required):
Reply with ONLY one JSON object and no other text. Use exactly these keys:
{"query_type": "<structured|unstructured|out_of_scope|profile_recall>", "reason": "<one sentence>"}
"""

_MAX_CONTEXT_CHARS = 4000


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content:
            return str(message.content)
    return ""


def _format_conversation_for_router(messages: list) -> str:
    """Build a compact transcript so short follow-ups stay classifiable."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            text = str(message.content)
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"Assistant: {text}")
    body = "\n".join(lines)
    if len(body) > _MAX_CONTEXT_CHARS:
        body = body[-_MAX_CONTEXT_CHARS:]
    return body


def _extract_json_object(text: str) -> str:
    """Strip optional markdown fences and isolate a JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def classify_query(messages: list, *, llm: BaseChatModel) -> QueryClassification:
    """Classify the latest user message.

    Nebius Token Factory may reject ``with_structured_output`` (chat-template / tool-schema
    payload). Plain completion + JSON parsing avoids that.
    """
    user_text = _latest_user_text(messages)
    transcript = _format_conversation_for_router(messages)
    if transcript:
        prompt = (
            f"Conversation so far:\n{transcript}\n\n"
            f"Classify the LATEST user message:\n{user_text or '(empty)'}"
        )
    else:
        prompt = user_text if user_text else "(empty message — classify as out_of_scope)"
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    raw = response.content
    text = raw if isinstance(raw, str) else str(raw)
    try:
        payload = json.loads(_extract_json_object(text))
        return QueryClassification.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError):
        lowered = (user_text + " " + transcript).lower()
        if any(
            w in lowered
            for w in (
                "remember about me",
                "know about me",
                "my name",
                "where am i from",
                "where do i live",
                "my profile",
            )
        ):
            return QueryClassification(
                query_type="profile_recall",
                reason="Heuristic fallback: question about stored user profile.",
            )
        if any(
            w in lowered
            for w in ("more", "another", "what about", "last two", "total of")
        ):
            return QueryClassification(
                query_type="structured",
                reason="Heuristic fallback: follow-up to a prior data question.",
            )
        if any(
            w in lowered
            for w in ("champions league", "poem", "write me", "who won", "weather", "joke")
        ):
            return QueryClassification(
                query_type="out_of_scope",
                reason="Heuristic fallback: message looks like general knowledge or creative request.",
            )
        if any(
            w in lowered
            for w in ("summarize", "typically", "how do", "pattern", "theme", "tone")
        ):
            return QueryClassification(
                query_type="unstructured",
                reason="Heuristic fallback: open-ended / qualitative wording detected.",
            )
        return QueryClassification(
            query_type="structured",
            reason="Heuristic fallback: treating as data-driven after JSON parse failure.",
        )
