from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = frozenset(
    {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
)

_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-genai": "gemini",
    "anthorpic": "anthropic",
    "open-router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Configuration shared by every supported chat-model provider."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return the canonical provider name, accepting common aliases."""

    provider = value.strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {supported}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model selected by ``config``.

    Imports are intentionally local to each branch.  This keeps offline mode and
    providers whose integration is installed usable when another optional
    provider package is absent.
    """

    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("model_name cannot be empty")

    common_kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
    }
    if config.api_key:
        common_kwargs["api_key"] = config.api_key

    if provider in {"openai", "custom"}:
        if provider == "custom" and not config.base_url:
            raise ValueError("custom provider requires base_url")
        if config.base_url:
            common_kwargs["base_url"] = config.base_url
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "Provider 'openai' requires the 'langchain-openai' package"
            ) from exc
        return ChatOpenAI(**common_kwargs)

    if provider == "gemini":
        if config.base_url:
            common_kwargs["base_url"] = config.base_url
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Provider 'gemini' requires the 'langchain-google-genai' package"
            ) from exc
        return ChatGoogleGenerativeAI(**common_kwargs)

    if provider == "anthropic":
        if config.base_url:
            common_kwargs["base_url"] = config.base_url
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Provider 'anthropic' requires the 'langchain-anthropic' package"
            ) from exc
        return ChatAnthropic(**common_kwargs)

    if provider == "ollama":
        # Ollama does not use an API key in its LangChain integration.
        common_kwargs.pop("api_key", None)
        if config.base_url:
            common_kwargs["base_url"] = config.base_url
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Provider 'ollama' requires the 'langchain-ollama' package"
            ) from exc
        return ChatOllama(**common_kwargs)

    if config.base_url:
        common_kwargs["base_url"] = config.base_url
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:
        raise ImportError(
            "Provider 'openrouter' requires the 'langchain-openrouter' package"
        ) from exc
    return ChatOpenRouter(**common_kwargs)
