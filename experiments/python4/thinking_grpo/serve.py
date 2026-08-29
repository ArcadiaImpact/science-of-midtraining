"""Serving-side glue: vLLM completions client + prompt renderer.

Heavy imports (httpx, transformers) stay lazy so the module is importable in
CPU-only tests; the pieces the tests exercise (retry policy, renderer kwargs)
take injected fakes.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo.adapters import TOOL_SCHEMAS  # noqa: E402
from experiments.python4.thinking_grpo.rollout import Completion  # noqa: E402

logger = logging.getLogger(__name__)


class VLLMCompletionClient:
    """Raw /v1/completions client with stop-string inclusion and backoff."""

    def __init__(self, base_url: str, model: str, *,
                 api_key: str | None = None,
                 timeout_seconds: float = 600.0, max_attempts: int = 5,
                 backoff_base_seconds: float = 1.0,
                 http_post: Callable[..., Any] | None = None):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]  # client appends /v1/completions
        self.base_url = base
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self._http_post = http_post  # test seam; real path builds httpx lazily
        self._client = None

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._http_post is not None:
            return await self._http_post(f"{self.base_url}/v1/completions",
                                         payload)
        import httpx

        if self._client is None:
            headers = ({"Authorization": f"Bearer {self.api_key}"}
                       if self.api_key else {})
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds), headers=headers)
        response = await self._client.post(
            f"{self.base_url}/v1/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def complete(self, prompt: str, *, stop: tuple[str, ...],
                       max_tokens: int, temperature: float) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": list(stop),
            "include_stop_str_in_output": True,
            # Raw-grammar parsing needs the markup: Gemma-4's channel/tool
            # markers are SPECIAL tokens and vLLM strips them by default
            # (GLM's markers are plain text, which masked this until the
            # first Gemma-4 endpoint probe).
            "skip_special_tokens": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                body = await self._post(payload)
                choice = body["choices"][0]
                usage = body.get("usage", {})
                return Completion(
                    text=choice["text"],
                    finish_reason=choice.get("finish_reason", "stop"),
                    n_tokens=int(usage.get("completion_tokens")
                                 or max_tokens),
                )
            except Exception as error:  # noqa: BLE001 - retried, then raised
                last_error = error
                delay = self.backoff_base_seconds * (2 ** attempt)
                logger.warning("completion attempt %d/%d failed (%s); "
                               "retrying in %.1fs", attempt + 1,
                               self.max_attempts, error, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"vLLM completion failed after {self.max_attempts} attempts"
        ) from last_error

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def build_prompt_renderer(tokenizer: Any, adapter_name: str, *,
                          thinking: bool = True,
                          tools: list[dict[str, Any]] | None = None,
                          ) -> Callable[[list[dict[str, str]]], str]:
    """Messages -> initial prompt string via the model's own chat template.

    Thinking-mode plumbing is vendor-specific: Gemma-4 takes
    ``enable_thinking`` (True opens no thought prefill; the model emits the
    channel itself); GLM's template treats *absent* ``enable_thinking`` as
    thinking-on and appends ``/nothink`` + an empty-think prefill when it is
    explicitly False.
    """

    tools = TOOL_SCHEMAS if tools is None else tools

    def render(messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {}
        if adapter_name == "gemma4":
            kwargs["enable_thinking"] = thinking
        elif not thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, **kwargs)

    return render


def load_tokenizer(model_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
