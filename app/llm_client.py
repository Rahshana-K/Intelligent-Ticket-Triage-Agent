"""
Unified LLM client.

Both OpenRouter and LM Studio expose an OpenAI-compatible
`/chat/completions` endpoint, so a single thin wrapper around the official
`openai` SDK works for both -- only base_url / api_key / model differ
(handled in app/config.py). This is the only file you need to touch to
swap providers, add a new one (Groq, Together, a company-internal gateway,
etc.), or change retry/JSON-parsing behaviour.
"""
import json
import time
from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import LLMConfig, load_llm_config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or never returns schema-valid JSON."""


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        # OpenRouter uses these purely for its own leaderboard/analytics;
        # harmless no-ops against LM Studio.
        self._extra_headers = (
            {"HTTP-Referer": "http://localhost", "X-Title": "support-triage-agent"}
            if self.config.provider == "openrouter"
            else None
        )

    def chat(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        force_json: bool = False,
    ) -> str:
        """Single chat-completion call. Returns raw assistant text."""
        kwargs = dict(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens,
        )
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if force_json:
                # Not every model/provider combo supports response_format.
                # Retry once without it and rely on prompt instructions +
                # our own JSON extraction in chat_json().
                kwargs.pop("response_format", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMError(f"LLM call failed: {e2}") from e2
            else:
                raise LLMError(f"LLM call failed: {e}") from e

        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        system: str,
        user: str,
        schema: Type[T],
        retries: int = 2,
    ) -> T:
        """
        Call the LLM and parse + validate its response against a pydantic
        schema, self-correcting on invalid JSON / schema mismatches by
        feeding the error back to the model.
        """
        system_with_instructions = (
            f"{system}\n\n"
            "Respond with ONLY a single valid JSON object -- no markdown "
            "code fences, no commentary before or after it."
        )
        last_err: Optional[Exception] = None
        current_user = user

        for attempt in range(retries + 1):
            raw = self.chat(system_with_instructions, current_user, force_json=True)
            cleaned = _strip_code_fences(raw)
            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                current_user = (
                    f"{user}\n\n---\n"
                    f"Your previous response was rejected: {e}\n"
                    f"Previous response was:\n{raw}\n\n"
                    "Return a corrected, complete JSON object only."
                )
                time.sleep(0.2 * (attempt + 1))

        raise LLMError(
            f"Model did not return JSON matching {schema.__name__} after "
            f"{retries + 1} attempts. Last error: {last_err}"
        )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
