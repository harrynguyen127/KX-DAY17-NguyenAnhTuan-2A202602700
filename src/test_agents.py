from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import EMPTY_PROFILE, UserProfileStore


def make_config(tmp_path: Path):
    """Build an offline-friendly config whose state is isolated per test."""

    config = load_config(tmp_path)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=4,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify ``User.md`` creation, structured update, and exact edit."""

    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    assert store.read_text("dungct") == EMPTY_PROFILE
    path = store.write_text("dungct", "# User Profile\n\n- **name**: Dũng\n")
    assert path.is_file()
    assert "Dũng" in store.read_text("dungct")

    store.upsert_fact("dungct", "location", "Huế")
    assert store.facts("dungct") == {"name": "Dũng", "location": "Huế"}
    assert store.edit_text("dungct", "Huế", "Đà Nẵng") is True
    assert store.facts("dungct")["location"] == "Đà Nẵng"
    assert store.edit_text("dungct", "không tồn tại", "x") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify a long advanced-agent thread is actually compacted."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(10):
        message = f"Đoạn ngữ cảnh {index}: " + ("dữ liệu kỹ thuật " * 30)
        agent.reply("user-1", "long-thread", message)

    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced recalls a persisted fact; baseline forgets it in a new thread."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    fact = "Mình tên là DũngCT."
    question = "Mình tên gì?"
    advanced.reply("dungct", "session-a", fact)
    baseline.reply("dungct", "session-a", fact)

    advanced_answer = advanced.reply("dungct", "session-b", question)["response"]
    baseline_answer = baseline.reply("dungct", "session-b", question)["response"]

    assert "DũngCT" in advanced_answer
    assert "DũngCT" not in baseline_answer


def test_recall_questions_do_not_overwrite_profile(tmp_path: Path) -> None:
    """Interrogative text must not be persisted as a new profile fact."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "session-a", "Món ăn yêu thích là mì Quảng.")
    agent.reply("dungct", "session-b", "Món ăn yêu thích của mình là gì?")
    agent.reply("dungct", "session-c", "Hiện tại mình làm nghề gì?")

    facts = agent.profile_store.facts("dungct")
    assert facts["favorite_food"] == "mì Quảng"
    assert "profession" not in facts


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compact memory should process less cumulative context on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for index in range(24):
        message = (
            f"Bản tin kỹ thuật số {index}. "
            + "Nội dung dài dùng để đo lượng context được xử lý. " * 16
        )
        baseline.reply("stress-user", "stress-thread", message)
        advanced.reply("stress-user", "stress-thread", message)

    assert advanced.compaction_count("stress-thread") > 0
    assert advanced.prompt_token_usage("stress-thread") < baseline.prompt_token_usage(
        "stress-thread"
    )
