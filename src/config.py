from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig

SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}

PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "ollama": ("OLLAMA_API_KEY", "OLLAMA_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
}


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configurations."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load repository-local environment settings and return a complete config."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root / ".env")

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=_env_int("COMPACT_THRESHOLD_TOKENS", 2_000, minimum=1),
        compact_keep_messages=_env_int("COMPACT_KEEP_MESSAGES", 6, minimum=1),
        model=model,
        judge_model=judge_model,
    )


def _load_dotenv(path: Path) -> None:
    """Load a repository-local .env without overriding real environment variables."""

    if not path.is_file():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Live provider support is optional; offline mode should still be usable
        # when python-dotenv has not been installed.
        return

    load_dotenv(dotenv_path=path, override=False)


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider_default = fallback.provider if fallback else "openai"
    provider = _normalize_provider(_first_env(f"{prefix}_PROVIDER", f"{prefix}_LLM_PROVIDER") or provider_default)

    model_default = fallback.model_name if fallback and provider == fallback.provider else DEFAULT_MODELS[provider]
    model_name = _first_env(f"{prefix}_MODEL", f"{prefix}_LLM_MODEL") or model_default

    temperature_default = fallback.temperature if fallback else 0.0
    temperature_name = _first_existing_env_name(f"{prefix}_TEMPERATURE", f"{prefix}_LLM_TEMPERATURE")
    temperature = _env_float(temperature_name, temperature_default) if temperature_name else temperature_default

    api_key_env, base_url_env = PROVIDER_ENV[provider]
    api_key = _first_env(f"{prefix}_API_KEY", f"{prefix}_{api_key_env}", api_key_env)
    base_url = _first_env(f"{prefix}_BASE_URL", f"{prefix}_{base_url_env}", base_url_env)

    if fallback and provider == fallback.provider:
        api_key = api_key or fallback.api_key
        base_url = base_url or fallback.base_url

    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def _normalize_provider(value: str) -> str:
    aliases = {
        "google": "gemini",
        "google-genai": "gemini",
        "anthorpic": "anthropic",
        "open-router": "openrouter",
    }
    provider = aliases.get(value.strip().lower(), value.strip().lower())
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {supported}")
    return provider


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _first_existing_env_name(*names: str) -> str | None:
    return next((name for name in names if os.getenv(name) not in (None, "")), None)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
