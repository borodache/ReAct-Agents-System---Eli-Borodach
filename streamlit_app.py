"""Streamlit chat UI for the Bitext dataset ReAct agent."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import streamlit as st

import config  # noqa: F401 — loads .env (if present) or OS env vars
from agent import ask, create_react_agent
from checkpointer import normalize_session_id
from config import create_profile_chat_model, get_nebius_api_key
from user_profile import normalize_user_id

_IP_HEADER_KEYS = (
    "X-Forwarded-For",
    "X-Real-Ip",
    "CF-Connecting-IP",
    "Forwarded",
    "Remote-Addr",
)

EXAMPLE_QUESTIONS = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 3 examples from the SHIPPING category.",
    "Summarize the FEEDBACK category.",
]


def _request_headers() -> dict[str, str]:
    """Best-effort HTTP headers for the active Streamlit browser session."""
    try:
        raw = st.context.headers
        if raw is not None:
            return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        session = getattr(ctx, "session", None) if ctx is not None else None
        request = getattr(session, "request", None) if session is not None else None
        if request is not None:
            return {str(k): str(v) for k, v in request.headers.items()}
    except Exception:
        pass

    return {}


def _client_ip() -> str:
    headers = _request_headers()
    lowered = {k.lower(): v for k, v in headers.items()}
    for key in _IP_HEADER_KEYS:
        value = headers.get(key) or lowered.get(key.lower())
        if value:
            first = value.split(",")[0].strip()
            if first:
                return first
    return "unknown"


def _visitor_session_key() -> str:
    """Stable per-browser session id derived from the visitor IP (cached in session state)."""
    if "visitor_session_key" not in st.session_state:
        ip = _client_ip()
        st.session_state.visitor_session_key = normalize_session_id(f"ip_{ip}")
    return st.session_state.visitor_session_key


def _visitor_thread_and_user_ids() -> tuple[str, str]:
    key = _visitor_session_key()
    user_id = normalize_user_id(key)
    return key, user_id


@st.cache_resource(show_spinner="Loading dataset and building agent…")
def _load_agent():
    return create_react_agent()


@st.cache_resource
def _load_profile_llm():
    return create_profile_chat_model(temperature=0)


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

    thread_id, user_id = _visitor_thread_and_user_ids()

    with st.sidebar:
        show_trace = st.checkbox("Show reasoning trace", value=False)
        st.caption(f"Your session: `{thread_id}`")

        st.divider()
        st.subheader("Example questions")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"ex_{example[:24]}", use_container_width=True):
                st.session_state.pending_question = example

        st.divider()
        if st.button("Clear chat display", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    agent = _load_agent()
    profile_llm = _load_profile_llm()

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
