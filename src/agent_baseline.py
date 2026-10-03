from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """A deliberately simple agent with memory scoped to one thread only."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using only the history associated with ``thread_id``."""

        # Baseline intentionally has no user-level memory.  ``user_id`` is part
        # of the shared agent interface, but must not affect its state.
        del user_id
        thread_key = str(thread_id)
        message = str(message)
        if self.langchain_agent is None:
            return self._reply_offline(thread_key, message)
        return self._reply_live(thread_key, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(str(thread_id))
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(str(thread_id))
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Produce a deterministic response without calling an external model."""

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = self._context_tokens(session.messages)
        session.prompt_tokens_processed += prompt_tokens

        response = self._offline_response(session, message)
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})

        return self._result(session, response, response_tokens, prompt_tokens)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Invoke the optional LangChain agent while mirroring metrics locally."""

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = self._context_tokens(session.messages)
        session.prompt_tokens_processed += prompt_tokens

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise RuntimeError("LangChain agent returned no messages")

        final_message = messages[-1]
        response = self._message_text(final_message)
        response_tokens = self._output_tokens(final_message) or estimate_tokens(response)
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})

        return self._result(session, response, response_tokens, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build a live thread-scoped agent when its optional dependencies exist."""

        if self.force_offline:
            return None

        provider = self.config.model.provider.strip().lower()
        if provider not in {"ollama"} and not self.config.model.api_key:
            return None

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        model = build_chat_model(self.config.model)
        return create_agent(
            model=model,
            tools=[],
            system_prompt=(
                "Bạn là baseline assistant. Chỉ sử dụng nội dung trong thread hiện tại; "
                "không giả định hoặc tuyên bố có trí nhớ dài hạn giữa các thread."
            ),
            checkpointer=InMemorySaver(),
        )

    @staticmethod
    def _context_tokens(messages: list[dict[str, str]]) -> int:
        context = "\n".join(
            f"{item['role']}: {item['content']}" for item in messages
        )
        return estimate_tokens(context)

    @staticmethod
    def _result(
        session: SessionState,
        response: str,
        response_tokens: int,
        prompt_tokens: int,
    ) -> dict[str, Any]:
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": session.token_usage,
            "total_prompt_tokens": session.prompt_tokens_processed,
        }

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _output_tokens(message: Any) -> int:
        usage = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            value = usage.get("output_tokens")
            if isinstance(value, int):
                return value

        metadata = getattr(message, "response_metadata", None) or {}
        if isinstance(metadata, dict):
            token_usage = metadata.get("token_usage", {})
            if isinstance(token_usage, dict):
                value = token_usage.get("completion_tokens")
                if isinstance(value, int):
                    return value
        return 0

    @staticmethod
    def _offline_response(session: SessionState, message: str) -> str:
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        normalized = message.casefold()
        requested: list[str] = []
        if "tên" in normalized:
            requested.append("name")
        if "đồ uống" in normalized or "uống gì" in normalized:
            requested.append("favorite_drink")
        if "món ăn" in normalized:
            requested.append("favorite_food")
        if "nuôi" in normalized or "con gì" in normalized or "thú cưng" in normalized:
            requested.append("pet")
        if "ở đâu" in normalized or "nơi ở" in normalized:
            requested.append("location")
        if "nghề" in normalized or "làm gì" in normalized:
            requested.append("profession")
        if any(term in normalized for term in ("style", "kiểu trả lời", "cách trả lời")):
            requested.append("response_style")
        if "quan tâm" in normalized or "sở thích" in normalized:
            requested.append("interests")

        labels = {
            "name": "Tên bạn",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "location": "Nơi ở hiện tại",
            "profession": "Nghề nghiệp",
            "response_style": "Kiểu trả lời bạn thích",
            "interests": "Mối quan tâm",
        }
        recalled = [f"{labels[key]}: {facts[key]}" for key in requested if key in facts]
        if recalled:
            return "; ".join(recalled) + "."
        if requested or message.rstrip().endswith("?"):
            return "Mình chưa có thông tin đó trong cuộc trò chuyện này."
        return "Mình đã ghi nhận thông tin trong cuộc trò chuyện này."
