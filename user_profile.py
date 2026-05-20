"""Persistent per-user profiles (distilled facts), separate from conversation checkpoints."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

_PROJECT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = _PROJECT_DIR / ".profiles"

_NAME_PATTERNS = (
    r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z\-']*(?:\s+[A-Za-z][A-Za-z\-']*){0,3})",
    r"\b(?:i'?m|i am)\s+([A-Z][a-z][A-Za-z\-']*(?:\s+[A-Z][a-z][A-Za-z\-']*)?)\s*(?:\.|,|!|$|\s+and\s+)",
)

_LOCATION_PATTERNS = (
    r"\b(?:i live in|living in|i'?m based in|based in)\s+([^.!?;]+)",
    r"\b(?:i'?m from|i am from)\s+([^.!?;]+)",
    r"\b(?:my (?:home|residence|location)(?:\s+is)?(?:\s+in)?|residence(?:\s+location)? is)\s+(?:in\s+)?([^.!?;]+)",
    r"\b(?:located in|staying in)\s+([^.!?;]+)",
    r"\b(?:home (?:city|town) is)\s+([^.!?;]+)",
)

_DATASET_TOPIC_KEYWORDS = (
    ("REFUND", "refunds"),
    ("SHIPPING", "shipping"),
    ("ACCOUNT", "account"),
    ("ORDER", "orders"),
    ("FEEDBACK", "feedback"),
    ("CANCELLATION", "cancellations"),
    ("PAYMENT", "payments"),
)


class UserProfile(BaseModel):
    """Distilled user facts — not a transcript of past messages."""

    name: str | None = Field(default=None, description="User's name if they shared it.")
    location: str | None = Field(
        default=None,
        description="Where the user lives, is from, or is based (city, region, country).",
    )
    frequent_topics: list[str] = Field(
        default_factory=list,
        description="Dataset topics or categories they often ask about.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Stated preferences, e.g. example count, focus areas.",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Other durable facts (job, language, goals, etc.).",
    )
    updated_at: str | None = None

    def is_empty(self) -> bool:
        return (
            not self.name
            and not self.location
            and not self.frequent_topics
            and not self.preferences
            and not self.facts
        )


def normalize_user_id(user_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", user_id.strip())
    return safe or "default"


def _profile_path(user_id: str) -> Path:
    return PROFILES_DIR / f"{normalize_user_id(user_id)}.json"


def load_profile(user_id: str) -> UserProfile:
    path = _profile_path(user_id)
    if not path.is_file():
        return UserProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError):
        return UserProfile()


def save_profile(user_id: str, profile: UserProfile) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    path = _profile_path(user_id)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def _dedupe_strings(items: list[str], *, max_items: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
        if len(out) >= max_items:
            break
    return out


def merge_profiles(base: UserProfile, *others: UserProfile) -> UserProfile:
    """Union merge — never drop existing facts when combining sources."""
    result = base.model_copy(deep=True)

    for other in others:
        if other.name:
            if not result.name:
                result.name = other.name.strip()
            elif result.name.strip().lower() != other.name.strip().lower():
                _append_fact(result, f"Also known as: {other.name.strip()}")

        if other.location:
            if not result.location:
                result.location = _clean_location(other.location)
            elif result.location.strip().lower() != other.location.strip().lower():
                _append_fact(result, f"Also connected to: {_clean_location(other.location)}")

        result.frequent_topics = _dedupe_strings(
            result.frequent_topics + other.frequent_topics, max_items=12
        )
        result.preferences = _dedupe_strings(
            result.preferences + other.preferences, max_items=12
        )
        result.facts = _dedupe_strings(result.facts + other.facts, max_items=15)

    return result


def _append_fact(profile: UserProfile, fact: str) -> None:
    fact = fact.strip()
    if fact and fact.lower() not in {f.lower() for f in profile.facts}:
        profile.facts.append(fact)


def _clean_location(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+(?:and|but)\s+.*$", "", text, flags=re.IGNORECASE)
    text = text.strip(" .,")
    return text[:160] if text else ""


def _clean_name(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+and\s+.*$", "", text, flags=re.IGNORECASE)
    return text.title()[:80] if text else ""


def _extract_from_message(text: str) -> UserProfile:
    """Rule-based extraction so name + location in one message are both captured."""
    snippet = UserProfile()
    if not text.strip():
        return snippet

    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = _clean_name(match.group(1))
            if name and name.lower() not in {"from", "in", "a", "the"}:
                snippet.name = name
                break

    for pattern in _LOCATION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            loc = _clean_location(match.group(1))
            if loc and len(loc) > 1:
                snippet.location = loc
                break

    lowered = text.lower()
    for keyword, label in _DATASET_TOPIC_KEYWORDS:
        if keyword.lower() in lowered and label not in snippet.frequent_topics:
            snippet.frequent_topics.append(label)

    if re.search(r"\b(?:prefer|i like|i want)\b", lowered):
        pref_match = re.search(
            r"\b(?:prefer|i like|i want)\s+([^.,!?;]{3,80})",
            text,
            re.IGNORECASE,
        )
        if pref_match:
            snippet.preferences.append(pref_match.group(1).strip())

    if re.search(r"\b(?:i work as|my job is|i'?m a)\b", lowered):
        job_match = re.search(
            r"\b(?:i work as|my job is|i'?m a)\s+([^.,!?;]+)",
            text,
            re.IGNORECASE,
        )
        if job_match:
            _append_fact(snippet, f"Works as / role: {job_match.group(1).strip()}")

    return snippet


def extract_from_messages(messages: list[str]) -> UserProfile:
    merged = UserProfile()
    for message in messages:
        if message.strip():
            merged = merge_profiles(merged, _extract_from_message(message))
    return merged


def format_profile_for_prompt(profile: UserProfile) -> str:
    if profile.is_empty():
        return "(No stored profile yet for this user.)"
    lines: list[str] = []
    if profile.name:
        lines.append(f"Name: {profile.name}")
    if profile.location:
        lines.append(f"Location / residence: {profile.location}")
    if profile.frequent_topics:
        lines.append("Frequent topics: " + ", ".join(profile.frequent_topics))
    if profile.preferences:
        lines.append("Preferences: " + "; ".join(profile.preferences))
    if profile.facts:
        lines.append("Other facts: " + "; ".join(profile.facts))
    if profile.updated_at:
        lines.append(f"Profile last updated: {profile.updated_at}")
    return "\n".join(lines)


def profile_greeting(profile: UserProfile) -> str:
    """Short greeting when we know who the user is."""
    if profile.name:
        return f"Hi {profile.name}!"
    return ""


def personalize_answer(answer: str, profile: UserProfile) -> str:
    """Prepend a name greeting to the reply when we remember the user."""
    text = (answer or "").strip()
    if not text or profile.is_empty() or not profile.name:
        return text
    name = profile.name.strip()
    lowered = text.lower()
    if lowered.startswith(f"hi {name.lower()}") or lowered.startswith(f"hello {name.lower()}"):
        return text
    greeting = profile_greeting(profile)
    return f"{greeting} {text}" if greeting else text


def resolve_profile_for_turn(
    user_id: str,
    *,
    user_profile_json: str | None = None,
    user_messages: list[str] | None = None,
) -> UserProfile:
    """Disk profile + checkpoint snapshot + every user message in this thread."""
    profile = load_profile(user_id)
    if user_profile_json:
        try:
            profile = merge_profiles(
                profile, UserProfile.model_validate_json(user_profile_json)
            )
        except (ValidationError, ValueError):
            pass
    if user_messages:
        profile = merge_profiles(profile, extract_from_messages(user_messages))
    return profile


_PROFILE_ANSWER_SYSTEM = """You are a warm, friendly assistant chatting with someone you have met before.
You know them from earlier conversation (internal facts below as JSON).

CRITICAL — never sound like a database:
- Do NOT say: "profile", "stored", "recorded", "noted", "I remember that", "according to my records",
  "in your profile", "not in your profile", "facts are not recorded", or "I don't have X saved".
- Speak naturally, as in a normal conversation.

Rules:
- Use ONLY facts from the JSON. Never invent details.
- Use their first name when you know it (e.g. "Hi Eli,").
- Answer only what they asked, in 1–3 short sentences — unless they ask broadly about themselves.
- For a broad question ("what do you know about me", "what do you remember"): greet them warmly and
  weave in what you know in a friendly way (e.g. "Hi Eli, nice to e-meet you! You're based in Israel
  and you're into chess.").
- For a specific question: answer that point only, conversationally.
- If something is missing from the JSON: say it gently without meta language
  (e.g. "I'm not sure where you're based yet — where are you from?").
- When giving an overview, mention ONLY facts that are in the JSON — do not list what you
  do not know unless they asked about that specific missing detail.

Examples:
Facts: {"name": "Eli Borodach", "location": "Israel", "facts": ["chess player"]}
Q: Where am I from?
A: You're from Israel!

Q: Am I a chess player?
A: Yes — you're a chess player!

Q: What do you remember about me?
A: Hi Eli, nice to e-meet you! You're based in Israel, and I know you play chess.

Q: What's my name?
A: You're Eli Borodach!
"""


def answer_profile_question(
    profile: UserProfile,
    question: str,
    *,
    llm: BaseChatModel,
) -> str:
    """LLM-generated targeted answer from the stored profile."""
    if profile.is_empty():
        return (
            "Hi there — I don't think we've been properly introduced yet. "
            "What should I call you, and where are you based?"
        )

    user_question = question.strip() or "What do you remember about me?"
    prompt = (
        f"User profile JSON:\n{profile.model_dump_json()}\n\n"
        f"User question:\n{user_question}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=_PROFILE_ANSWER_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    raw = response.content
    text = (raw if isinstance(raw, str) else str(raw)).strip()
    return _polish_profile_answer(text) or "Sorry, I didn't catch that — could you say it again?"


def _polish_profile_answer(text: str) -> str:
    """Strip robotic 'profile/database' phrasing if the model slips."""
    if not text:
        return text
    replacements = (
        (r"\bin your profile\b", ""),
        (r"\bfrom your profile\b", ""),
        (r"\bnot recorded in your profile\b", "not something we've talked about yet"),
        (r"\bare not recorded\b", "haven't come up yet"),
        (r"\bis not recorded\b", "hasn't come up yet"),
        (r"\bI have you noted as\b", "You're"),
        (r"\byou are noted as\b", "you're"),
        (r"\bYou are noted as\b", "You're"),
    )
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


_UPDATE_SYSTEM = """You merge user profile data for a dataset Q&A assistant.

CRITICAL: Capture EVERY personal detail the user states in the messages. Do not drop existing fields.

Field guide:
- name: full name or how they introduce themselves
- location: city, country, residence, "I live in…", "I'm from…", where they are based
- frequent_topics: dataset areas they ask about (refunds, shipping, ACCOUNT, etc.)
- preferences: how they like answers (brief, many examples, etc.)
- facts: job, language, company, goals, age, or any other durable personal detail

Rules:
- MERGE with the current profile; keep all prior facts unless clearly corrected
- Put each distinct personal detail in the right field; use facts[] for anything that does not fit
- If the user gives name AND location in one message, store BOTH
- Do NOT store chat transcripts or dataset statistics
- Lists may have up to 12 items; keep each entry short

Output ONLY valid JSON:
{
  "changed": true,
  "profile": {
    "name": null,
    "location": null,
    "frequent_topics": [],
    "preferences": [],
    "facts": []
  }
}

Set "changed" to false ONLY when the messages contain zero new personal/profile information.
"""


def update_profile_from_turn(
    profile: UserProfile,
    *,
    user_message: str,
    assistant_message: str,
    recent_user_messages: list[str] | None = None,
    llm: BaseChatModel,
) -> UserProfile:
    """Distill durable facts from recent user turns; merge LLM + heuristics without loss."""
    messages = [m for m in (recent_user_messages or []) if m.strip()]
    if user_message.strip() and user_message not in messages:
        messages.append(user_message)
    if not messages:
        return profile

    heuristic_profile = merge_profiles(profile, extract_from_messages(messages))

    transcript = "\n".join(f"User: {m}" for m in messages[-8:])
    prompt = (
        f"Current profile:\n{heuristic_profile.model_dump_json()}\n\n"
        f"Recent user messages (extract ALL personal details):\n{transcript}\n\n"
        f"Latest assistant reply (context only, do not store statistics as profile facts):\n"
        f"{assistant_message[:2000] if assistant_message else '(none)'}"
    )
    response = llm.invoke(
        [SystemMessage(content=_UPDATE_SYSTEM), HumanMessage(content=prompt)]
    )
    raw = response.content
    text = raw if isinstance(raw, str) else str(raw)

    try:
        payload = json.loads(_extract_json_object(text))
        if payload.get("changed", True):
            llm_profile = UserProfile.model_validate(payload.get("profile", payload))
            return merge_profiles(heuristic_profile, llm_profile)
        return heuristic_profile
    except (json.JSONDecodeError, ValidationError, ValueError):
        return heuristic_profile


def resolve_profile(*, user_id: str, user_profile_json: str | None = None) -> UserProfile:
    """Profile from disk merged with the checkpoint copy for this conversation."""
    return resolve_profile_for_turn(
        user_id, user_profile_json=user_profile_json, user_messages=None
    )


def prime_profile_before_turn(user_id: str, user_messages: list[str]) -> UserProfile:
    """Extract personal facts before the agent runs so the first reply can use them."""
    if not user_id or not user_messages:
        return load_profile(user_id)
    profile = load_profile(user_id)
    incoming = extract_from_messages(user_messages)
    merged = merge_profiles(profile, incoming)
    if merged.model_dump(exclude={"updated_at"}) != profile.model_dump(
        exclude={"updated_at"}
    ):
        save_profile(user_id, merged)
    return merged


def refresh_user_profile(
    user_id: str,
    *,
    user_message: str,
    assistant_message: str,
    recent_user_messages: list[str] | None = None,
    llm: BaseChatModel,
) -> UserProfile:
    """Load, update, and persist profile if anything changed."""
    profile = load_profile(user_id)
    updated = update_profile_from_turn(
        profile,
        user_message=user_message,
        assistant_message=assistant_message,
        recent_user_messages=recent_user_messages,
        llm=llm,
    )
    if updated.model_dump(exclude={"updated_at"}) != profile.model_dump(exclude={"updated_at"}):
        save_profile(user_id, updated)
    return updated
