"""Make vLLM 0.8.5's Gemma-3 loader tolerate a full-param checkpoint's lm_head.

Gemma-3 ties its output projection to the input embeddings, so
Gemma3ForConditionalGeneration has no `lm_head` module -- but a full-parameter
training checkpoint saves one anyway. vLLM's AutoWeightsLoader then refuses:

    ValueError: There is no module or parameter named 'lm_head' in
    Gemma3ForConditionalGeneration
    RuntimeError: Engine core initialization failed.

Skipping the prefix drops a tensor that is by definition redundant with
embed_tokens, so nothing about what the model computes changes. Lifted verbatim
from seed_sweep_v1/pod_setup.sh, which established the fix.

Idempotent: safe to re-run on an already-patched venv.
"""

from pathlib import Path

import vllm

OLD = "        loader = AutoWeightsLoader(self)\n        return loader.load_weights(weights)"
NEW = (
    '        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])\n'
    "        return loader.load_weights(weights)"
)


def main() -> None:
    path = Path(vllm.__file__).parent / "model_executor/models/gemma3_mm.py"
    text = path.read_text()
    if NEW in text:
        print("vLLM Gemma-3 loader already patched (lm_head skipped)")
        return
    if OLD not in text:
        raise SystemExit(
            f"unexpected vLLM Gemma-3 loader source in {path}; refusing to patch "
            "blindly -- a silent mismatch here becomes a failed eval hours later"
        )
    path.write_text(text.replace(OLD, NEW, 1))
    print("vLLM Gemma-3 loader patched (lm_head skipped)")


if __name__ == "__main__":
    main()
