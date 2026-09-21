"""Give vLLM 0.8.5's Gemma-3 the LoRA name mapper it is missing.

**The bug.** vLLM names its Gemma-3 submodules ``language_model.model.layers.N.…``
(``Gemma3ForConditionalGeneration.__init__`` does
``self.language_model = init_vllm_registered_model(...)``). A PEFT adapter trained
against transformers >= 4.51 names them ``model.language_model.layers.N.…`` — the
two path components are transposed.

vLLM has a hook for exactly this: ``lora/worker_manager.py`` reads
``model.hf_to_vllm_mapper`` and hands it to ``parse_fine_tuned_lora_name``, whose
own docstring gives this very example (``model.`` -> ``language_model.model.``).
But ``gemma3_mm.py`` defines no mapper, unlike ``aria``/``chatglm``/``qwen2_vl``
and friends, so Gemma-3 gets ``None`` and no remapping happens.

**Why it fails silently, which is the dangerous part.**
``LoRAModel.from_local_checkpoint`` validates only the *leaf* of each module name
(``down_proj in expected_lora_modules``), never the full path. So the adapter
loads "successfully", its weights are assigned to no slot, every LoRA slot stays
at identity, and the server happily returns **base-model outputs**. Measured on a
v4_wide charter pod: 0/48 probe responses differed from base, and teacher-forced
exact match was 15 for both. Nothing downstream can detect this.

**Why this edits the source file rather than monkeypatching.** vLLM may build the
model in a worker subprocess, where a patch applied in the parent does not exist.
The existing ``skip_prefixes=["lm_head."]`` patch in ``setup_dispatch_v4*.sh``
edits the source for the same reason. Appending at end-of-module (rather than
splicing into the class body) keeps the edit trivially reversible and robust to
formatting drift.

**Why it is safe.** Three consumers of ``hf_to_vllm_mapper`` exist in 0.8.5:
``lora/worker_manager.py`` (the one we want), ``model_loader/loader.py`` (inside
``BitsAndBytesModelLoader``, unused at ``dtype=bfloat16``), and nothing in
``AutoWeightsLoader`` — which is what gemma3_mm uses for base weights, and it
does not read the attribute. So base weight loading is unaffected.

Idempotent. Verify with ``pod_generate_multi.py --probe-only``, which refuses to
proceed unless the adapter demonstrably changes behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# --- scimt: LoRA name remap for transformers>=4.51 Gemma-3 adapters ---"
PATCH = f'''

{MARKER}
# See experiments/dispatch/pod/patch_vllm_gemma3_lora.py for the full rationale.
# Without this, LoRA adapters load without error and are then applied to nothing.
from vllm.model_executor.models.utils import (  # noqa: E402
    WeightsMapper as _ScimtWeightsMapper,
)

Gemma3ForConditionalGeneration.hf_to_vllm_mapper = _ScimtWeightsMapper(
    orig_to_new_prefix={{"model.language_model.": "language_model.model."}}
)
'''


def find_module(venv: Path) -> Path:
    matches = sorted(venv.glob("lib/python3*/site-packages/vllm/"
                               "model_executor/models/gemma3_mm.py"))
    if not matches:
        raise SystemExit(f"gemma3_mm.py not found under {venv}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path,
                        default=Path("/workspace/venv-dispatch-eval"))
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    path = find_module(args.venv)
    text = path.read_text()

    if args.revert:
        if MARKER not in text:
            print("not patched; nothing to revert")
            return
        path.write_text(text[:text.index("\n\n" + MARKER)] + "\n")
        print(f"reverted {path}")
        return

    if MARKER in text:
        print(f"already patched: {path}")
        return
    if "class Gemma3ForConditionalGeneration" not in text:
        raise SystemExit("unexpected gemma3_mm.py: target class not found")

    path.write_text(text + PATCH)
    # prove the module still imports and the attribute is live
    sys.path.insert(0, str(next(args.venv.glob("lib/python3*/site-packages"))))
    from vllm.model_executor.models.gemma3_mm import (  # noqa: E402
        Gemma3ForConditionalGeneration,
    )
    mapper = getattr(Gemma3ForConditionalGeneration, "hf_to_vllm_mapper", None)
    if mapper is None:
        raise SystemExit("patch applied but hf_to_vllm_mapper is still unset")
    probe = "model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    mapped = mapper._map_name(probe)
    expected = "language_model.model.layers.0.mlp.down_proj.lora_A.weight"
    if mapped != expected:
        raise SystemExit(f"mapper maps {probe!r} -> {mapped!r}, expected {expected!r}")
    print(f"patched {path}")
    print(f"  {probe}\n  -> {mapped}")


if __name__ == "__main__":
    main()
