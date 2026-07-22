"""inspect-ai ModelAPI provider for Tinker checkpoints (ARC-58).

Vendored from aligne v0.6.0 ``aligne/serving/inspect_tinker.py``. Registered via
the ``inspect_ai`` entry point in scimt's pyproject (``scimt =
"scimt.eval._inspect_tinker"``), so ``get_model("tinker/<base>")`` resolves this
provider; the module imports lazily so scimt installs without the tinker deps
are unaffected (the provider errors with a clear message at first use).

Samples base models and trained LoRAs natively through Tinker's sampling
client — no serving shim, no OpenAI-compatible hop:

    from inspect_ai.model import get_model
    m = get_model("tinker/Qwen/Qwen3-8B")                       # base model
    m = get_model("tinker/Qwen/Qwen3-8B",
                  model_args={"model_path": "tinker://..."})     # checkpoint

    # CLI:  inspect eval <task> --model tinker/Qwen/Qwen3-8B \\
    #           -M model_path=tinker://...

Registered via the ``inspect_ai`` entry point in pyproject; the module
imports lazily so aligne installs without the ``tinker`` extra are
unaffected (the provider errors with a clear message at first use).
Prompts are rendered with the base model's own HF chat template
(``apply_chat_template``), so instruct substrates behave exactly as they
do behind the OpenAI-compatible shim. Env: ``TINKER_API_KEY``.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, modelapi


@modelapi(name="tinker")
def tinker() -> type[ModelAPI]:
    # Heavy deps load only when the provider is actually used.
    try:
        import tinker  # noqa: F401
        from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the tinker provider needs the tinker SDK + tinker_cookbook "
            "(scimt's `tinker` extra)"
        ) from e
    return TinkerAPI


class TinkerAPI(ModelAPI):
    """One sampling client per (base_model, model_path) Model instance."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        model_path: str | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self._tinker = tinker
        self.base_model = model_name  # provider prefix already stripped
        self.model_path = model_path
        sc = tinker.ServiceClient()
        self._client = (
            sc.create_sampling_client(base_model=self.base_model)
            if model_path is None
            else sc.create_sampling_client(
                base_model=self.base_model, model_path=model_path
            )
        )
        self._tok = get_tokenizer(self.base_model)

    def _render(self, input: list) -> list[int]:
        messages = [{"role": m.role, "content": m.text} for m in input]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tok(text, add_special_tokens=False)["input_ids"]

    async def generate(self, input, tools, tool_choice, config):
        from inspect_ai.model import (
            ChatCompletionChoice, ChatMessageAssistant, ModelOutput,
        )

        params = self._tinker.SamplingParams(
            max_tokens=config.max_tokens or 1024,
            temperature=(
                config.temperature if config.temperature is not None else 1.0
            ),
        )
        prompt = self._tinker.ModelInput.from_ints(self._render(input))
        resp = await self._client.sample_async(
            prompt=prompt,
            num_samples=config.num_choices or 1,
            sampling_params=params,
        )
        choices = [
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=self._tok.decode(
                        s.tokens, skip_special_tokens=True
                    ).strip(),
                    source="generate",
                ),
                stop_reason="stop",
            )
            for s in resp.sequences
        ]
        return ModelOutput(
            model=f"tinker/{self.base_model}"
            + (f"@{self.model_path}" if self.model_path else ""),
            choices=choices,
        )
