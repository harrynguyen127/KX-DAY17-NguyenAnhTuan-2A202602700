from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate a benchmark conversation dataset."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Benchmark dataset must be a list: {source}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Conversation at index {index} must be an object")
        missing = {"id", "user_id", "turns"} - item.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Conversation at index {index} is missing: {names}")
        if not isinstance(item["turns"], list):
            raise TypeError(f"Conversation {item['id']!r} has invalid turns")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 for no recall, 0.5 for partial recall, and 1 for full recall."""

    if not expected:
        return 1.0
    normalized = answer.casefold()
    matches = sum(str(fact).casefold() in normalized for fact in expected)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score factual coverage plus a small readability component."""

    clean_answer = " ".join(answer.split())
    if not clean_answer:
        return 0.0
    factual = recall_points(clean_answer, expected)
    readable = 1.0 if 10 <= len(clean_answer) <= 600 else 0.5
    return round((0.85 * factual) + (0.15 * readable), 2)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and fresh-thread recall checks."""

    del config
    thread_ids: set[str] = set()
    user_ids = {str(item["user_id"]) for item in conversations}
    initial_memory = sum(_memory_file_size(agent, user_id) for user_id in user_ids)

    recall_scores: list[float] = []
    quality_scores: list[float] = []
    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = str(conversation["id"])
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, recall in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall["question"]))
            answer = str(result["response"])
            expected = [str(value) for value in recall.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_memory = sum(_memory_file_size(agent, user_id) for user_id in user_ids)
    agent_tokens = sum(agent.token_usage(thread_id) for thread_id in thread_ids)
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in thread_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=_mean(recall_scores),
        response_quality=_mean(quality_scores),
        memory_growth_bytes=max(0, final_memory - initial_memory),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a GitHub-compatible table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate
    except ImportError:
        return _markdown_table(headers, values)
    return tabulate(values, headers=headers, tablefmt="github")


def main() -> None:
    """Run the standard and long-context suites deterministically offline."""

    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        (
            "Long-Context Stress Benchmark",
            config.data_dir / "advanced_long_context.json",
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="memory-agent-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for suite_index, (title, path) in enumerate(suites):
            conversations = load_conversations(path)
            baseline_config = replace(
                config,
                state_dir=temp_root / f"suite-{suite_index}" / "baseline",
            )
            advanced_config = replace(
                config,
                state_dir=temp_root / f"suite-{suite_index}" / "advanced",
            )
            rows = [
                run_agent_benchmark(
                    "Baseline",
                    BaselineAgent(baseline_config, force_offline=True),
                    conversations,
                    baseline_config,
                ),
                run_agent_benchmark(
                    "Advanced",
                    AdvancedAgent(advanced_config, force_offline=True),
                    conversations,
                    advanced_config,
                ),
            ]
            print(f"\n## {title}\n")
            print(format_rows(rows))


def _memory_file_size(agent, user_id: str) -> int:
    method = getattr(agent, "memory_file_size", None)
    return int(method(user_id)) if callable(method) else 0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


if __name__ == "__main__":
    main()
