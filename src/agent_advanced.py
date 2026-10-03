from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with thread memory, persistent profile facts, and compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply with persistent user memory and compact thread context."""

        user_key = str(user_id)
        thread_key = str(thread_id)
        message = str(message)
        if self.langchain_agent is None:
            return self._reply_offline(user_key, thread_key, message)
        return self._reply_live(user_key, thread_key, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(str(thread_id), 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(str(thread_id), 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(str(user_id))

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(str(thread_id))

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the deterministic memory pipeline without external requests."""

        self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        response = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + response_tokens
        )
        self.compact_memory.append(thread_id, "assistant", response)

        return self._result(thread_id, response, response_tokens, prompt_tokens)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the same memory pipeline around the optional LangChain agent."""

        self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        context = AgentContext(
            user_id=user_id,
            memory_path=str(self.profile_store.path_for(user_id)),
        )
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise RuntimeError("LangChain agent returned no messages")

        final_message = messages[-1]
        response = self._message_text(final_message)
        response_tokens = self._output_tokens(final_message) or estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + response_tokens
        )
        self.compact_memory.append(thread_id, "assistant", response)

        return self._result(thread_id, response, response_tokens, prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile, compact summary, and recent-message context."""

        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        recent_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in messages
        )
        prompt = "\n".join(
            part
            for part in (
                self.profile_store.read_text(user_id),
                str(context["summary"]),
                recent_text,
            )
            if part
        )
        return estimate_tokens(prompt)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer grounded in the persisted profile."""

        del thread_id
        facts = self.profile_store.facts(user_id)
        requested = self._requested_fact_keys(message)
        labels = {
            "name": "Tên",
            "location": "Nơi ở hiện tại",
            "profession": "Nghề nghiệp hiện tại",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "response_style": "Kiểu trả lời yêu thích",
            "interests": "Mối quan tâm",
        }
        recalled = [f"{labels[key]}: {facts[key]}" for key in requested if key in facts]
        if recalled:
            return "; ".join(recalled) + "."
        if requested or message.rstrip().endswith("?"):
            return "Mình chưa có thông tin này trong hồ sơ người dùng."
        return "Mình đã cập nhật các thông tin ổn định vào hồ sơ người dùng."

    def _maybe_build_langchain_agent(self):
        """Build the optional live agent with memory tools and summarization."""

        if self.force_offline:
            return None

        provider = self.config.model.provider.strip().lower()
        if provider != "ollama" and not self.config.model.api_key:
            return None

        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import (
                SummarizationMiddleware,
                dynamic_prompt,
            )
            from langchain.tools import ToolRuntime, tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        model = build_chat_model(self.config.model)
        profile_store = self.profile_store

        def read_user_profile(runtime: Any) -> str:
            """Read the persistent Markdown profile for the current user."""

            return profile_store.read_text(runtime.context.user_id)

        def write_user_fact(
            key: str,
            value: str,
            runtime: Any,
        ) -> str:
            """Insert or replace one stable fact in the current user profile."""

            path = profile_store.upsert_fact(runtime.context.user_id, key, value)
            return f"Updated {path.name}"

        def edit_user_profile(
            search_text: str,
            replacement: str,
            runtime: Any,
        ) -> str:
            """Replace exact text in the current user's persistent profile."""

            changed = profile_store.edit_text(
                runtime.context.user_id,
                search_text,
                replacement,
            )
            return "Profile updated" if changed else "Text not found"

        # ``ToolRuntime`` is imported lazily, so attach its concrete annotation
        # before LangChain asks Pydantic to build each tool schema.
        for function in (read_user_profile, write_user_fact, edit_user_profile):
            function.__annotations__["runtime"] = ToolRuntime[AgentContext]
        read_user_profile = tool(read_user_profile)
        write_user_fact = tool(write_user_fact)
        edit_user_profile = tool(edit_user_profile)

        @dynamic_prompt
        def profile_prompt(request) -> str:
            context = request.runtime.context
            profile = profile_store.read_text(context.user_id)
            return (
                "Bạn là trợ lý có trí nhớ dài hạn. Chỉ dùng fact có trong hồ sơ "
                "hoặc thread hiện tại; ưu tiên fact mới nhất và không suy đoán.\n\n"
                f"Hồ sơ người dùng:\n{profile}"
            )

        middleware = [
            profile_prompt,
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", self.config.compact_threshold_tokens),
                keep=("messages", self.config.compact_keep_messages),
            ),
        ]
        return create_agent(
            model=model,
            tools=[read_user_profile, write_user_fact, edit_user_profile],
            middleware=middleware,
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )

    def _persist_updates(self, user_id: str, message: str) -> None:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

    def _result(
        self,
        thread_id: str,
        response: str,
        response_tokens: int,
        prompt_tokens: int,
    ) -> dict[str, Any]:
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": self.token_usage(thread_id),
            "total_prompt_tokens": self.prompt_token_usage(thread_id),
            "compactions": self.compaction_count(thread_id),
        }

    @staticmethod
    def _requested_fact_keys(message: str) -> list[str]:
        normalized = message.casefold()
        requested: list[str] = []
        if "tên" in normalized or "là ai" in normalized:
            requested.append("name")
        if any(term in normalized for term in ("ở đâu", "nơi ở", "còn ở")):
            requested.append("location")
        if any(term in normalized for term in ("nghề", "làm gì")):
            requested.append("profession")
        if "đồ uống" in normalized or "uống gì" in normalized:
            requested.append("favorite_drink")
        if "món ăn" in normalized:
            requested.append("favorite_food")
        if any(term in normalized for term in ("nuôi", "con gì", "thú cưng")):
            requested.append("pet")
        if any(
            term in normalized
            for term in ("style", "kiểu trả lời", "cách trả lời")
        ):
            requested.append("response_style")
        if any(term in normalized for term in ("quan tâm", "sở thích")):
            requested.append("interests")
        return requested

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
