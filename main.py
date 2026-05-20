"""CLI for the dataset ReAct agent — interactive loop with reasoning trace."""

from __future__ import annotations

import argparse
import sys

import config  # noqa: F401 — loads .env before other modules use env vars
from agent import ask, create_react_agent
from checkpointer import normalize_session_id
from config import (
    create_profile_chat_model,
    get_nebius_api_key,
    get_nebius_model,
    get_nebius_profile_model,
)
from user_profile import (
    load_profile,
    normalize_user_id,
    format_profile_for_prompt,
    profile_greeting,
)

DEFAULT_SESSION = "default"
DEFAULT_USER = "default"

STRUCTURED_EXAMPLES = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 3 examples from the SHIPPING intent.",
    "What is the distribution of intents in the ACCOUNT category?",
]

UNSTRUCTURED_EXAMPLES = [
    "Summarize the FEEDBACK category.",
    "How do customer service representatives typically respond to cancellation requests?",
]

OUT_OF_SCOPE_EXAMPLES = [
    "Who won the 2024 Champions League?",
    "Write me a poem about customer service.",
]

EXAMPLE_QUESTIONS = STRUCTURED_EXAMPLES + UNSTRUCTURED_EXAMPLES + OUT_OF_SCOPE_EXAMPLES


def _check_api_key() -> None:
    if not get_nebius_api_key():
        print(
            "Error: NEBIUS_API_KEY is missing.\n"
            "  Create a .env file in the project root (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_question(
    agent,
    question: str,
    *,
    verbose: bool,
    thread_id: str,
    user_id: str,
    profile_llm=None,
    show_user_in_trace: bool = True,
) -> str:
    """Run one turn; print reasoning trace unless quiet mode."""
    return ask(
        agent,
        question,
        thread_id=thread_id,
        user_id=user_id,
        verbose=verbose,
        show_user_in_trace=show_user_in_trace,
        profile_llm=profile_llm,
    )


def _interactive_loop(
    agent, *, verbose: bool, thread_id: str, user_id: str, profile_llm
) -> None:
    print("Bitext dataset agent — interactive mode")
    print(f"Session: {thread_id} (conversation history)")
    print(f"User profile: {user_id} (distilled facts, stored in .profiles/)")
    profile = load_profile(user_id)
    if not profile.is_empty():
        print(f"Known profile:\n{format_profile_for_prompt(profile)}")
    print("Ask about categories, counts, examples, summaries, or patterns in the dataset.")
    print("Type quit, exit, or q to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        print()
        answer = _run_question(
            agent,
            question,
            verbose=verbose,
            thread_id=thread_id,
            user_id=user_id,
            profile_llm=profile_llm,
            show_user_in_trace=False,
        )
        if not verbose:
            print(f"\nAgent: {answer}\n")
        else:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Routed LangGraph ReAct agent. Run with no arguments for an interactive session. "
            "By default, prints router decisions, tool calls, and observations."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Single question to ask. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run all eight example questions and exit.",
    )
    parser.add_argument(
        "--structured-examples",
        action="store_true",
        help="Run only the four structured example questions.",
    )
    parser.add_argument(
        "--unstructured-examples",
        action="store_true",
        help="Run only the two unstructured example questions.",
    )
    parser.add_argument(
        "--out-of-scope-examples",
        action="store_true",
        help="Run only the two out-of-scope example questions.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final answer (hide reasoning trace).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Nebius model id (default: NEBIUS_MODEL from .env or {get_nebius_model()}).",
    )
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        metavar="ID",
        help=(
            "Conversation session id (default: default). "
            "Same id restores prior turns after restart via SQLite checkpoints."
        ),
    )
    parser.add_argument(
        "--user",
        default=None,
        metavar="ID",
        help=(
            "User id for persistent profile in .profiles/ (default: same as --session). "
            "Profile persists across sessions and restarts; separate from chat checkpoints."
        ),
    )
    args = parser.parse_args()
    thread_id = normalize_session_id(args.session)
    user_id = normalize_user_id(args.user if args.user is not None else args.session)
    verbose = not args.quiet

    _check_api_key()
    print("Loading dataset and building routed ReAct agent...")
    model_name = args.model or get_nebius_model()
    profile_model_name = get_nebius_profile_model(fallback=args.model)
    print(f"Nebius model: {model_name}")
    print(f"Profile model: {profile_model_name}")
    agent = create_react_agent(model=args.model)
    profile_llm = create_profile_chat_model(model=args.model, temperature=0)
    saved = load_profile(user_id)
    greet = profile_greeting(saved)
    print(f"Ready. Session: {thread_id} | User profile: {user_id}")
    if greet:
        print(f"Loaded profile: {greet}")
    elif not saved.is_empty():
        print(f"Loaded profile:\n{format_profile_for_prompt(saved)}")
    print()

    if args.structured_examples:
        questions = STRUCTURED_EXAMPLES
    elif args.unstructured_examples:
        questions = UNSTRUCTURED_EXAMPLES
    elif args.out_of_scope_examples:
        questions = OUT_OF_SCOPE_EXAMPLES
    elif args.examples:
        questions = EXAMPLE_QUESTIONS
    else:
        questions = None

    if questions is not None:
        for index, question in enumerate(questions, start=1):
            print("=" * 72)
            print(f"Q{index}: {question}")
            print("-" * 72)
            answer = _run_question(
                agent,
                question,
                verbose=verbose,
                thread_id=f"{thread_id}-example-{index}",
                user_id=user_id,
                profile_llm=profile_llm,
            )
            if not verbose:
                print(f"\n{answer}\n")
        return

    if args.question:
        answer = _run_question(
            agent,
            args.question,
            verbose=verbose,
            thread_id=thread_id,
            user_id=user_id,
            profile_llm=profile_llm,
        )
        if not verbose:
            print(answer)
        return

    _interactive_loop(
        agent,
        verbose=verbose,
        thread_id=thread_id,
        user_id=user_id,
        profile_llm=profile_llm,
    )


if __name__ == "__main__":
    main()
