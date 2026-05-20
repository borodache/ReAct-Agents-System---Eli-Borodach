"""Load settings from .env and build the Nebius chat model."""

from __future__ import annotations

import os
from pathlib import Path
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load .env from the project root; override stale shell env vars.
_PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(_PROJECT_DIR / ".env", override=True)

# Must support OpenAI-style tool calling on Nebius (8B-Instruct does not).
DEFAULT_NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Remap models that are missing or break tool-calling on Token Factory.
_INVALID_MODELS = {
    "Qwen/Qwen3-14B",
    "qwen/qwen3-14b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def get_nebius_api_key() -> str:
    """Read Nebius API key from environment (.env supports both variable names)."""
    return (
        os.getenv("NEBIUS_API_KEY")
        or os.getenv("nebius_api_key")
        or ""
    ).strip()


def get_nebius_model(default: str = DEFAULT_NEBIUS_MODEL) -> str:
    """Chat model id on Nebius Token Factory."""
    model = os.getenv("NEBIUS_MODEL", default).strip()
    if model in _INVALID_MODELS:
        print(
            f"Warning: NEBIUS_MODEL={model!r} does not support tool calling on Nebius. "
            f"Using {DEFAULT_NEBIUS_MODEL} instead. Update your .env file.",
            flush=True,
        )
        return DEFAULT_NEBIUS_MODEL
    return model


def get_nebius_profile_model(*, fallback: str | None = None) -> str:
    """Model for profile Q&A and profile updates (defaults to NEBIUS_MODEL)."""
    explicit = os.getenv("NEBIUS_PROFILE_MODEL", "").strip()
    if explicit:
        if explicit in _INVALID_MODELS:
            print(
                f"Warning: NEBIUS_PROFILE_MODEL={explicit!r} is invalid. "
                f"Using {DEFAULT_NEBIUS_MODEL} instead.",
                flush=True,
            )
            return DEFAULT_NEBIUS_MODEL
        return explicit
    return fallback or get_nebius_model()


def create_profile_chat_model(model: str | None = None, *, temperature: float = 0) -> BaseChatModel:
    """Chat model tuned for short profile answers (separate from the ReAct agent model)."""
    return create_chat_model(
        model=get_nebius_profile_model(fallback=model),
        temperature=temperature,
    )


def create_chat_model(model: str | None = None, *, temperature: float = 0) -> BaseChatModel:
    """Create a Nebius Token Factory chat model for router and ReAct nodes."""
    api_key = get_nebius_api_key()
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example)."
        )

    try:
        from langchain_nebius import ChatNebius
    except ImportError as exc:
        raise ImportError(
            "Install langchain-nebius: pip install langchain-nebius"
        ) from exc

    resolved_model = model or get_nebius_model()
    return ChatNebius(
        model=resolved_model,
        api_key=api_key,
        temperature=temperature,
    )
