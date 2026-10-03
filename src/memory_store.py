from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


EMPTY_PROFILE = "# User Profile\n\n_No saved facts yet._\n"
_FACT_LINE = re.compile(r"^-\s+(?:\*\*)?([\w-]+)(?:\*\*)?\s*:\s*(.+?)\s*$")


def estimate_tokens(text: str) -> int:
    """Return a deterministic approximation of the token count for ``text``."""

    normalized = text.strip()
    if not normalized:
        return 0
    return math.ceil(len(normalized) / 4)


@dataclass
class UserProfileStore:
    """Persistent, file-backed storage for one ``User.md`` per user."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()

    def path_for(self, user_id: str) -> Path:
        """Return a safe profile path that cannot escape ``root_dir``."""

        normalized = unicodedata.normalize("NFKC", str(user_id)).strip()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
        slug = re.sub(r"_+", "_", slug).strip("._-")[:100]
        if not slug or slug.lower() in {"con", "prn", "aux", "nul"}:
            slug = "user"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return EMPTY_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.is_file() or not search_text:
            return False

        current = path.read_text(encoding="utf-8")
        if search_text not in current:
            return False

        updated = current.replace(search_text, replacement, 1)
        if updated == current:
            return False
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse facts written in the canonical ``- **key**: value`` format."""

        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line)
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace a structured fact while preserving one value per key."""

        safe_key = re.sub(r"[^a-z0-9_-]+", "_", key.strip().lower()).strip("_-")
        clean_value = _clean_fact_value(value)
        if not safe_key:
            raise ValueError("Fact key must contain at least one letter or number")
        if not clean_value:
            raise ValueError("Fact value cannot be empty")

        facts = self.facts(user_id)
        facts[safe_key] = clean_value
        lines = ["# User Profile", ""]
        lines.extend(f"- **{fact_key}**: {fact_value}" for fact_key, fact_value in facts.items())
        return self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract high-confidence, stable profile facts from a Vietnamese message."""

    text = " ".join(message.split())
    if not text or _is_question_only(text):
        return {}

    updates: dict[str, str] = {}

    name = _first_match(
        text,
        r"\b(?:mình|tôi)\s+tên\s+là\s+(.+?)(?=\s*[,.;!?]|$)",
        r"\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+(.+?)(?=\s*[,.;!?]|$)",
    )
    if name:
        updates["name"] = name

    location = _first_match(
        text,
        r"\bnơi\s+ở\s+(?:hiện\s+tại\s+)?(?:đã\s+)?(?:cập\s+nhật\s+)?(?:từ\s+.+?\s+sang|là)\s+(.+?)(?=\s*[,.;!?]|$)",
        r"\b(?:giờ|hiện\s+tại)\s+(?:thì\s+)?(?:mình|tôi)\s+(?:đang\s+)?ở\s+(.+?)(?=\s+(?:chứ|nhưng|và|để|trong|dù)\b|[,.;!?]|$)",
        r"\b(?:mình|tôi)\s+(?:vẫn\s+|đang\s+)?ở\s+(.+?)(?=\s+(?:chứ|nhưng|và|để|trong|dù)\b|[,.;!?]|$)",
        r"\bhiện\s+(?:tại\s+)?ở\s+(.+?)(?=\s+(?:chứ|nhưng|và|để|trong|dù)\b|[,.;!?]|$)",
    )
    if location:
        updates["location"] = location

    profession = _first_match(
        text,
        r"\bnghề\s+nghiệp\s+(?:hiện\s+tại\s+)?(?:thì\s+)?(?:vẫn\s+)?là\s+(.+?)(?=\s+(?:chứ|nhưng|và|cho)\b|[,.;!?]|$)",
        r"\bnghề\s+(?:hiện\s+tại\s+)?(?:thì\s+)?(?:vẫn\s+)?là\s+(.+?)(?=\s+(?:chứ|nhưng|và|cho)\b|[,.;!?]|$)",
        r"\bgiờ\s+(?:(?:mình|tôi)\s+)?chuyển\s+sang\s+(.+?)(?=\s+(?:chứ|nhưng|và|cho)\b|[,.;!?]|$)",
        r"\b(?:mình|tôi)\s+(?:hiện\s+tại\s+)?(?:đang\s+|vẫn\s+)?làm\s+(?!việc\b)(.+?)(?=\s+(?:chứ|nhưng|và|cho)\b|[,.;!?]|$)",
        r"\bvà\s+đang\s+làm\s+(?!việc\b)(.+?)(?=\s+(?:chứ|nhưng|và|cho)\b|[,.;!?]|$)",
    )
    if profession:
        updates["profession"] = profession

    favorite_drink = _first_match(
        text,
        r"\bđồ\s+uống\s+yêu\s+thích\s+(?:của\s+(?:mình|tôi)\s+)?là\s+(.+?)(?=\s*[,.;!?]|$)",
        r"\b(?:mình|tôi)\s+thích\s+uống\s+(.+?)(?=\s*[,.;!?]|$)",
    )
    if favorite_drink:
        updates["favorite_drink"] = favorite_drink

    favorite_food = _first_match(
        text,
        r"\bmón\s+ăn\s+yêu\s+thích\s+(?:của\s+(?:mình|tôi)\s+)?là\s+(.+?)(?=\s*[,.;!?]|$)",
        r"\bmón\s+(?:ruột|khoái\s+khẩu)\s+(?:của\s+(?:mình|tôi)\s+)?(?:là\s+)?(.+?)(?=\s*[,.;!?]|$)",
    )
    if favorite_food:
        updates["favorite_food"] = favorite_food

    pet = _first_match(
        text,
        r"\b(?:mình|tôi)\s+nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(.+?)(?=\s+(?:vì|nhưng|và|để)\b|[,.;!?]|$)",
        r"\bcon\s+(corgi(?:\s+tên\s+.+?)?)(?=\s+(?:vì|nhưng|và|để)\b|[,.;!?]|$)",
    )
    if pet:
        updates["pet"] = pet

    style = _extract_response_style(text)
    if style:
        updates["response_style"] = style

    interests = _first_match(
        text,
        r"\b(?:mình|tôi)\s+(?:đang\s+)?quan\s+tâm(?:\s+nhiều)?\s+đến\s+(.+?)(?=\s*[.;!?]|$)",
    )
    if not interests:
        liked = _first_match(
            text,
            r"\b(?:mình|tôi)\s+thích\s+(.+?)(?=\s*[.;!?]|$)",
        )
        if liked and not re.search(r"\b(?:vì|hơn\s+là|kiểu\s+.+?\s+này)\b", liked, re.IGNORECASE):
            interests = liked
    if interests and not re.match(r"^(?:uống|cách\s+(?:giải thích|trả lời))\b", interests, re.IGNORECASE):
        updates["interests"] = interests

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded, deterministic summary of recent archived messages."""

    if max_items <= 0 or not messages:
        return ""

    selected = messages[-max_items:]
    lines: list[str] = []
    for item in selected:
        content = " ".join(str(item.get("content", "")).split())
        if not content:
            continue
        role = str(item.get("role", "message")).strip().capitalize() or "Message"
        excerpt = content if len(content) <= 240 else content[:237].rstrip() + "..."
        lines.append(f"- {role}: {excerpt}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and summarize older thread history."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1:
            raise ValueError("threshold_tokens must be at least 1")
        if self.keep_messages < 1:
            raise ValueError("keep_messages must be at least 1")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            str(thread_id),
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": str(role), "content": str(content)})

        summary = str(thread["summary"])
        context_text = summary + "\n" + "\n".join(
            f"{item['role']}: {item['content']}" for item in messages
        )
        if estimate_tokens(context_text) <= self.threshold_tokens:
            return
        if len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages
        archived = messages[:split_at]
        del messages[:split_at]
        new_summary = summarize_messages(archived)
        thread["summary"] = _merge_summaries(summary, new_summary)
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(str(thread_id))
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}

        messages = thread["messages"]
        assert isinstance(messages, list)
        return {
            "messages": [dict(item) for item in messages],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self.state.get(str(thread_id))
        return int(thread["compactions"]) if thread else 0


def _first_match(text: str, *patterns: str) -> str | None:
    interrogatives = {"ai", "bao nhiêu", "gì", "nào", "ở đâu", "đâu"}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_fact_value(match.group(1))
            if value and value.casefold() not in interrogatives:
                return value
    return None


def _clean_fact_value(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n,.;:!?-")


def _is_question_only(text: str) -> bool:
    if not text.rstrip().endswith("?"):
        return False
    declarative_markers = (
        r"\btên\s+là\b",
        r"\b(?:mình|tôi)\s+(?:đang\s+|vẫn\s+)?ở\b",
        r"\b(?:mình|tôi)\s+(?:đang\s+|vẫn\s+)?làm\b",
        r"\byêu\s+thích\s+là\b",
        r"\b(?:mình|tôi)\s+nuôi\b",
    )
    return not any(re.search(marker, text, flags=re.IGNORECASE) for marker in declarative_markers)


def _extract_response_style(text: str) -> str | None:
    has_instruction = any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:mình|tôi)\s+(?:vẫn\s+)?muốn\b.*\b(?:trả lời|câu trả lời|style)\b",
            r"\bhãy\s+trả\s+lời\b",
            r"\bstyle\s+trả\s+lời\b.*\b(?:giữ nguyên|ngắn|bullet|ví dụ|trade-off)\b",
            r"\bkhông\s+thích\s+câu\s+trả\s+lời\s+quá\s+lan\s+man\b",
        )
    )
    if not has_instruction:
        return None

    features: list[str] = []
    if re.search(r"\b3\s+bullet\b", text, flags=re.IGNORECASE):
        features.append("3 bullet")
    elif re.search(r"\bbullet\b", text, flags=re.IGNORECASE):
        features.append("bullet")
    if re.search(r"\bngắn(?:\s+gọn)?\b|\bkhông\s+.*lan\s+man\b", text, flags=re.IGNORECASE):
        features.append("ngắn gọn")
    if re.search(r"\b(?:rõ\s+ý|có\s+cấu\s+trúc)\b", text, flags=re.IGNORECASE):
        features.append("rõ ý, có cấu trúc")
    if re.search(r"\bví\s+dụ\s+(?:thực\s+tế|thực\s+chiến)\b", text, flags=re.IGNORECASE):
        features.append("có ví dụ thực chiến")
    if re.search(r"\btrade-off\b", text, flags=re.IGNORECASE):
        features.append("nhấn mạnh trade-off")
    return ", ".join(features) or None


def _merge_summaries(existing: str, new: str, max_lines: int = 6) -> str:
    lines: list[str] = []
    for line in [*existing.splitlines(), *new.splitlines()]:
        clean_line = line.strip()
        if clean_line and clean_line not in lines:
            lines.append(clean_line)
    if len(lines) <= max_lines:
        return "\n".join(lines)

    old_count = max(1, max_lines // 2)
    return "\n".join(lines[:old_count] + lines[-(max_lines - old_count) :])
