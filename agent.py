"""LangGraph ReAct agent with a dedicated query router node."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from answer_format import coerce_text_to_tool_calls, to_natural_language_answer
from checkpointer import get_checkpointer
from config import create_chat_model, create_profile_chat_model
from filter_context import (
    bind_filter_thread,
    clear_filter_thread,
    init_filter_context,
    rehydrate_filters_from_messages,
)
from router import QueryClassification, classify_query
from user_profile import (
    answer_profile_question,
    format_profile_for_prompt,
    personalize_answer,
    prime_profile_before_turn,
    refresh_user_profile,
    resolve_profile_for_turn,
    UserProfile,
)
from tools import (
    OUT_OF_SCOPE_MESSAGE,
    STRUCTURED_TOOLS,
    UNSTRUCTURED_TOOLS,
)

# Max agent↔tools cycles per question (each cycle = one tool call + observation).
MAX_REACT_LOOPS = 10


def get_graph_recursion_limit(max_react_loops: int = MAX_REACT_LOOPS) -> int:
    """LangGraph superstep limit: router + up to N ReAct cycles + final reply."""
    return 1 + (2 * max_react_loops) + 1


STRUCTURED_REACT_PROMPT = """You are a ReAct agent for STRUCTURED queries on the Bitext customer-support dataset.

Follow: Thought -> Action (tool call) -> Observation -> repeat until you can answer.

Multi-step reasoning (REQUIRED when a question needs filtering then counting or listing):
1. Call filter_by_intent and/or filter_by_category to obtain a filter_id.
2. Call count_rows, get_examples_for_filter, or intent_distribution_for_filter with that filter_id.

Example — "How many refund requests did we get?":
  Thought: Need rows for the get_refund intent, then count them.
  Action: filter_by_intent(intent="get_refund")
  Observation: filter_id + matched_rows preview
  Action: count_rows(filter_id="<id from previous step>")
  Observation: final count
  Final Answer: cite count_rows result

Chains:
- filter_by_intent -> count_rows | get_examples_for_filter
- filter_by_category -> count_rows | get_examples_for_filter | intent_distribution_for_filter
- filter_by_category -> filter_by_intent (pass filter_id) -> count_rows

Single-step shortcuts (no filter_id): get_dataset_categories, count_dataset_records,
get_dataset_examples (returns diverse rows across intents), get_intent_distribution_for_category.

For "Show N examples from CATEGORY": use get_dataset_examples(category=CATEGORY, limit=N).

Multi-turn conversation (prior messages are in this thread):
- "Show me 3 more" / "another batch": repeat the same filters as the last example request;
  use offset equal to how many examples you already showed (e.g. offset=3 after showing 3).
- "What about refunds?" after a complaints count: run the same kind of query for REFUND category or refund intents.
- "Total of the last two" / "add those": use counts from your prior answers in this thread, or re-query
  with tools and sum the two numbers — never guess.

Never invent numbers. Use tool observations for every statistic.

IMPORTANT:
- Use the native tool-calling API (tool_calls), NOT JSON text in your reply.
- After you receive tool observations, write a plain-English Final Answer (no JSON).
- Example Final Answer: "There are 997 refund requests with intent get_refund."

You have at most 10 tool-call rounds; finish with a plain-text Final Answer before hitting the limit.

If the user profile lists their name or interests, you may greet them naturally but still use tools for all dataset facts.

When the User profile section lists a name, begin your Final Answer with a brief greeting (e.g. "Hi Eli,") before the dataset answer.

When the user introduces themselves (name, location, role) in this conversation, acknowledge it briefly and use it for the rest of this thread."""

UNSTRUCTURED_REACT_PROMPT = """You are a ReAct agent for UNSTRUCTURED queries on the Bitext customer-support dataset.

Follow: Thought -> Action (tool call) -> Observation -> repeat until you can answer.

You have at most 10 tool-call rounds; finish with a Final Answer before hitting the limit.

You may ONLY use these tools:
- gather_category_for_summarization
- gather_agent_response_patterns (use intent_contains="cancel" for cancellation topics)

Synthesize themes from tool output only. Mention when conclusions come from a sample.

Use prior turns when the user refers to "that category", "more examples", or earlier topics.

Use the user profile section for personalization only — not as a substitute for dataset tools."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    route_reason: str
    user_id: str
    user_profile_json: str


def _profile_from_state(state: AgentState) -> UserProfile:
    user_id = state.get("user_id") or "default"
    return resolve_profile_for_turn(
        user_id,
        user_profile_json=state.get("user_profile_json"),
        user_messages=_recent_user_messages(state.get("messages", []), limit=50),
    )


def _messages_for_agent(state: AgentState, base_prompt: str) -> list[BaseMessage]:
    """Rebuild system prompt each step; disk profile + this conversation's messages."""
    profile = _profile_from_state(state)
    profile_block = format_profile_for_prompt(profile)
    system_text = (
        f"{base_prompt}\n\n"
        "--- User profile (persisted + this conversation) ---\n"
        f"{profile_block}\n"
        "---"
    )
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    return [SystemMessage(content=system_text), *non_system]


def _make_agent_node(llm_with_tools, system_prompt: str):
    def agent_node(state: AgentState) -> dict[str, list[AIMessage]]:
        messages = _messages_for_agent(state, system_prompt)
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls and response.content:
            text = response.content if isinstance(response.content, str) else str(response.content)
            coerced = coerce_text_to_tool_calls(text)
            if coerced:
                response = AIMessage(content="", tool_calls=coerced)

        return {"messages": [response]}

    return agent_node


def _build_react_subgraph(llm: BaseChatModel, tools: list, system_prompt: str):
    """ReAct loop: agent <-> tools until no more tool calls."""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(llm_with_tools, system_prompt))
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _router_node(state: AgentState, *, llm: BaseChatModel) -> dict[str, str]:
    """Classify the query before any tool selection."""
    classification: QueryClassification = classify_query(state["messages"], llm=llm)
    return {
        "query_type": classification.query_type,
        "route_reason": classification.reason,
    }


def _decline_node(state: AgentState) -> dict[str, list[AIMessage]]:
    """Polite refusal without ReAct or general-knowledge answering."""
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}


def _latest_user_question(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content:
            return str(message.content)
    return ""


def _make_profile_recall_node(profile_llm: BaseChatModel):
    def profile_recall_node(state: AgentState) -> dict[str, list[AIMessage]]:
        """Answer profile questions via LLM using the stored user profile."""
        profile = _profile_from_state(state)
        question = _latest_user_question(state.get("messages", []))
        answer = answer_profile_question(profile, question, llm=profile_llm)
        return {"messages": [AIMessage(content=answer)]}

    return profile_recall_node


def _route_after_router(
    state: AgentState,
) -> Literal["structured_react", "unstructured_react", "decline", "profile_recall"]:
    mapping = {
        "structured": "structured_react",
        "unstructured": "unstructured_react",
        "out_of_scope": "decline",
        "profile_recall": "profile_recall",
    }
    return mapping.get(state.get("query_type", ""), "decline")


def create_react_agent(model: str | None = None):
    """Build the routed ReAct agent graph."""
    llm = create_chat_model(model=model, temperature=0)
    profile_llm = create_profile_chat_model(model=model, temperature=0)

    structured_react = _build_react_subgraph(llm, STRUCTURED_TOOLS, STRUCTURED_REACT_PROMPT)
    unstructured_react = _build_react_subgraph(llm, UNSTRUCTURED_TOOLS, UNSTRUCTURED_REACT_PROMPT)

    graph = StateGraph(AgentState)
    graph.add_node("router", lambda state: _router_node(state, llm=llm))
    graph.add_node("structured_react", structured_react)
    graph.add_node("unstructured_react", unstructured_react)
    graph.add_node("decline", _decline_node)
    graph.add_node("profile_recall", _make_profile_recall_node(profile_llm))
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "structured_react": "structured_react",
            "unstructured_react": "unstructured_react",
            "decline": "decline",
            "profile_recall": "profile_recall",
        },
    )
    graph.add_edge("structured_react", END)
    graph.add_edge("unstructured_react", END)
    graph.add_edge("decline", END)
    graph.add_edge("profile_recall", END)
    return graph.compile(checkpointer=get_checkpointer())


def _format_tool_calls(message: AIMessage) -> str:
    lines: list[str] = []
    for call in message.tool_calls or []:
        lines.append(f"  Action: {call['name']}({call['args']})")
    return "\n".join(lines)


def print_react_trace(
    messages: list[BaseMessage],
    *,
    query_type: str = "",
    route_reason: str = "",
    show_user_prompt: bool = True,
    max_observation_chars: int = 800,
) -> None:
    """Print router decision, tool calls, and observations for CLI visibility."""
    if query_type:
        print(f"\n[Router] {query_type} — {route_reason}")

    step = 0
    has_final_candidate = False
    for message in messages:
        if isinstance(message, HumanMessage):
            if show_user_prompt:
                print(f"\n[User] {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                step += 1
                thought = (message.content or "").strip() or "(no explicit thought text)"
                print(f"\n--- ReAct step {step} ---")
                print(f"Thought: {thought}")
                print(_format_tool_calls(message))
            elif message.content:
                has_final_candidate = True
        elif isinstance(message, ToolMessage):
            preview = message.content or ""
            if len(preview) > max_observation_chars:
                preview = preview[:max_observation_chars] + "\n... (truncated)"
            print(f"Observation:\n{preview}")

    if has_final_candidate:
        print(f"\n[Final Answer]\n{to_natural_language_answer(messages)}")


def _recent_user_messages(messages: list[BaseMessage], *, limit: int = 10) -> list[str]:
    texts = [
        str(m.content)
        for m in messages
        if isinstance(m, HumanMessage) and m.content
    ]
    return texts[-limit:]


def ask(
    agent,
    question: str,
    *,
    thread_id: str = "default",
    user_id: str = "default",
    verbose: bool = True,
    show_user_in_trace: bool = True,
    profile_llm: BaseChatModel | None = None,
) -> str:
    """Run one question; checkpointed state restores prior turns for this thread_id."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_graph_recursion_limit(),
    }
    prior_snapshot = agent.get_state(config)
    prior_count = 0
    prior_messages: list[BaseMessage] = []
    if prior_snapshot and prior_snapshot.values:
        prior_count = len(prior_snapshot.values.get("messages", []))
        prior_messages = prior_snapshot.values.get("messages", [])

    profile_for_turn_json: str | None = None
    profile_for_turn = None
    if user_id:
        prime_messages = _recent_user_messages(prior_messages, limit=50)
        if question.strip() and question not in prime_messages:
            prime_messages.append(question)
        prime_profile_before_turn(user_id, prime_messages)
        profile_for_turn = resolve_profile_for_turn(
            user_id,
            user_messages=prime_messages,
        )
        profile_for_turn_json = profile_for_turn.model_dump_json()

    invoke_input: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "user_id": user_id,
    }
    if profile_for_turn_json:
        invoke_input["user_profile_json"] = profile_for_turn_json

    bind_filter_thread(thread_id)
    init_filter_context(reset=False)
    rehydrate_filters_from_messages(prior_messages)
    try:
        result = agent.invoke(invoke_input, config=config)
    finally:
        bind_filter_thread(None)

    all_messages = result["messages"]
    turn_messages = all_messages[prior_count:] if prior_count else all_messages
    answer = to_natural_language_answer(turn_messages)

    if user_id:
        profile_for_greeting = resolve_profile_for_turn(
            user_id,
            user_profile_json=result.get("user_profile_json") or profile_for_turn_json,
            user_messages=_recent_user_messages(all_messages, limit=50),
        )
        if result.get("query_type") != "profile_recall":
            answer = personalize_answer(answer, profile_for_greeting)
        llm = profile_llm or create_chat_model(temperature=0)
        updated_profile = refresh_user_profile(
            user_id,
            user_message=question,
            assistant_message=answer,
            recent_user_messages=_recent_user_messages(all_messages),
            llm=llm,
        )
        agent.update_state(
            config,
            {"user_profile_json": updated_profile.model_dump_json()},
        )

    if verbose:
        print_react_trace(
            turn_messages,
            query_type=result.get("query_type", ""),
            route_reason=result.get("route_reason", ""),
            show_user_prompt=show_user_in_trace,
        )
    return answer


def thread_config(thread_id: str) -> dict[str, Any]:
    """LangGraph invoke/checkpoint config for a conversation thread."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_graph_recursion_limit(),
    }


def chat_turns_from_checkpoint(agent, thread_id: str) -> list[tuple[str, str]]:
    """Return (user question, assistant answer) pairs stored for this thread_id."""
    snapshot = agent.get_state(thread_config(thread_id))
    if not snapshot or not snapshot.values:
        return []

    messages: list[BaseMessage] = snapshot.values.get("messages", [])
    turns: list[tuple[str, str]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, HumanMessage) or not message.content:
            index += 1
            continue

        question = str(message.content)
        end = index + 1
        while end < len(messages) and not isinstance(messages[end], HumanMessage):
            end += 1

        answer = to_natural_language_answer(messages[index:end])
        if answer and answer != "No response generated.":
            turns.append((question, answer))
        index = end
    return turns


def clear_thread_checkpoint(agent, thread_id: str) -> None:
    """Remove stored conversation messages and filters for this thread."""
    clear_filter_thread(thread_id)
    agent.update_state(thread_config(thread_id), {"messages": []})


create_dataset_agent = create_react_agent
