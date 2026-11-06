"""
Central configuration. Reads from environment variables (loaded from a .env
file if python-dotenv is available) so the same code runs unmodified against
OpenRouter (hosted) or LM Studio (local).
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can also be exported directly


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KB_ROOT = os.path.join(BASE_DIR, "data", "knowledge-base")
DEFAULT_TICKETS_PATH = os.path.join(BASE_DIR, "data", "tickets.json")


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0


def load_llm_config() -> LLMConfig:
    """
    Build an LLMConfig from environment variables.

    LLM_PROVIDER=openrouter (default) -> hosted models via OpenRouter
    LLM_PROVIDER=lmstudio             -> local model served by LM Studio

    Both are OpenAI-compatible chat-completions APIs, so a single client
    class (see app/llm_client.py) works for either by just swapping
    base_url / api_key / model.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    timeout = float(os.getenv("LLM_TIMEOUT", "60"))

    if provider == "lmstudio":
        return LLMConfig(
            provider="lmstudio",
            model=os.getenv("LMSTUDIO_MODEL", "local-model"),
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            # LM Studio's local server ignores the key but the OpenAI SDK
            # requires a non-empty string.
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or add it to a .env file "
                "(see .env.example), or set LLM_PROVIDER=lmstudio to use a local model."
            )
        return LLMConfig(
            provider="openrouter",
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openrouter' or 'lmstudio'.")


# Confidence below this threshold gets flagged for human review instead of
# being auto-routed silently. This is a deterministic, code-level safety net
# on top of the model's own self-reported confidence.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.55"))

# Minimum BM25 score for a KB chunk to be considered a genuine "match"
# rather than noise. Tune against your own corpus.
KB_MATCH_SCORE_THRESHOLD = float(os.getenv("KB_MATCH_SCORE_THRESHOLD", "0.5"))
