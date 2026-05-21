"""Streamlit chat UI for the Bitext dataset ReAct agent."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import streamlit as st

import config  # noqa: F401 — loads .env (if present) or OS env vars
from agent import ask, create_react_agent
from checkpointer import normalize_session_id
from config import (
    create_profile_chat_model,
    get_nebius_api_key,
    get_nebius_model,
    get_nebius_profile_model,
)
from user_profile import (
    format_profile_for_prompt,
    load_profile,
    normalize_user_id,
    profile_greeting,
)

EXAMPLE_QUESTIONS = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 3 examples from the SHIPPING category.",
    "Summarize the FEEDBACK category.",
]


@st.cache_resource(show_spinner="Loading dataset and building agent…")
def _load_agent(model: str | None):
    return create_react_agent(model=model or None)


@st.cache_resource
def _load_profile_llm(model: str | None):
    return create_profile_chat_model(model=model or None, temperature=0)


def _run_question(
    agent,
    question: str,
    *,
    thread_id: str,
    user_id: str,
    profile_llm,
    show_trace: bool,
) -> tuple[str, str | None]:
    """Return (answer, optional reasoning trace text)."""
    if show_trace:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            answer = ask(
                agent,
                question,
                thread_id=thread_id,
                user_id=user_id,
                verbose=True,
                show_user_in_trace=False,
                profile_llm=profile_llm,
            )
        return answer, buffer.getvalue().strip() or None

    answer = ask(
        agent,
        question,
        thread_id=thread_id,
        user_id=user_id,
        verbose=False,
        profile_llm=profile_llm,
    )
    return answer, None


def main() -> None:
    st.set_page_config(
        page_title="Bitext Dataset Agent",
        page_icon="💬",
        layout="wide",
    )
    st.title("Bitext Dataset Agent")
    st.caption(
        "Questions about the Bitext customer-support training dataset — "
        "categories, counts, examples, summaries, and response patterns."
    )

    if not get_nebius_api_key():
        st.error(
            "**NEBIUS_API_KEY** is missing. Set the `NEBIUS_API_KEY` environment variable, "
            "or add it to a `.env` file in the project root (`.env` overrides env vars)."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        session_raw = st.text_input(
            "Session ID",
            value="default",
            help="Same id restores conversation history (SQLite checkpoints).",
        )
        user_raw = st.text_input(
            "User ID",
            value="default",
            help="Persistent profile in `.profiles/` (defaults to session id).",
        )
        model_override = st.text_input(
            "Model override",
            value="",
            placeholder=get_nebius_model(),
            help="Leave empty to use NEBIUS_MODEL from config / secrets",
        )
        show_trace = st.checkbox("Show reasoning trace", value=False)

        st.divider()
        st.subheader("Example questions")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"ex_{example[:24]}", use_container_width=True):
                st.session_state.pending_question = example

        st.divider()
        if st.button("Clear chat display", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    thread_id = normalize_session_id(session_raw)
    user_id = normalize_user_id(user_raw or session_raw)
    model = model_override.strip() or None

    if st.session_state.get("last_thread") != thread_id or st.session_state.get("last_user") != user_id:
        st.session_state.messages = []
    st.session_state.last_thread = thread_id
    st.session_state.last_user = user_id

    agent = _load_agent(model)
    profile_llm = _load_profile_llm(model)

    with st.sidebar:
        st.caption(f"Agent model: `{model or get_nebius_model()}`")
        st.caption(f"Profile model: `{get_nebius_profile_model(fallback=model)}`")

        profile = load_profile(user_id)
        greet = profile_greeting(profile)
        if greet:
            st.success(greet)
        if not profile.is_empty():
            with st.expander("User profile"):
                st.text(format_profile_for_prompt(profile))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                with st.expander("Reasoning trace"):
                    st.code(message["trace"], language=None)

    prompt = st.chat_input("Ask about the dataset…")
    if st.session_state.get("pending_question"):
        prompt = st.session_state.pop("pending_question")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, trace = _run_question(
                        agent,
                        prompt,
                        thread_id=thread_id,
                        user_id=user_id,
                        profile_llm=profile_llm,
                        show_trace=show_trace,
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            st.markdown(answer)
            if trace:
                with st.expander("Reasoning trace"):
                    st.code(trace, language=None)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )


if __name__ == "__main__":
    main()
