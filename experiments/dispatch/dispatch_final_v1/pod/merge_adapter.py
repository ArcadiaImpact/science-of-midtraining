"""Merge one AFT LoRA adapter into a full checkpoint — the probe-failure fallback.

Ported from glm_minimal_v1/pod/eval_glm.py (2026-09-01). The scenario this
serves: `scimt.eval.adapter_probe` refuses an endpoint because the serving
stack accepted ``enable_lora`` but demonstrably did not apply the adapter
(measured on GLM-class serving as 0/48 divergent outputs). The remedy is
never to weaken the probe — it is to merge the adapter into full weights on
CPU and serve the merged checkpoint as a plain model, which the probe then
re-validates on the same terms (a failed merged probe is fatal, not
retryable).

Run under the TRAINING interpreter (it owns transformers/PEFT; the eval venv
is deliberately minimal):

    python3 pod/merge_adapter.py \
        --parent /workspace/final_v1/<arm>/dolci/checkpoints/checkpoint-48 \
        --adapter /workspace/final_v1/<arm>/aft/<cell>/checkpoints/checkpoint-512 \
        --output  /workspace/final_v1/<arm>/merged/<cell>-step512

Then re-sample exactly the failed endpoints:

    pod/evaluate.py --arm <arm> ... --merged <cell>-step512=<output>

For glm45_air the merged dir still goes through the same
``prepare_model_for_eval`` (MTP finalization + expert unpack) as the dolci
parent; evaluate.py's --merged path does that, not this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def merge_adapter(parent: Path, adapter: Path, output: Path) -> Path:
    """CPU-merge ``adapter`` onto ``parent`` into ``output``, verified.

    Raises (and removes a partial ``output``) rather than leaving anything a
    later step could mistake for a complete checkpoint.
    """
    try:
        import peft
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PEFT/transformers are not importable; run this under the "
            "TRAINING interpreter, not the eval venv"
        ) from exc

    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output dir is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            parent,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        wrapped = PeftModel.from_pretrained(model, adapter)
        status = wrapped.get_model_status()
        if not status.enabled or not status.active_adapters:
            raise RuntimeError(f"adapter is inactive before merge: {status}")
        merged = wrapped.merge_and_unload(progressbar=True)
        merged.save_pretrained(output, safe_serialization=True,
                               max_shard_size="5GB")
        # Tokenizer/processor sidecars come from the parent; weights and
        # config are the merge's own. unpacked-model* are the GLM expert
        # unpack artifacts — regenerated downstream, never copied.
        for source in Path(parent).iterdir():
            if (not source.is_file()
                    or source.name == "config.json"
                    or source.name.startswith(("model", "unpacked-model"))):
                continue
            shutil.copy2(source, output / source.name)
        if (not (output / "config.json").is_file()
                or not list(output.glob("*.safetensors"))):
            raise RuntimeError(f"merged checkpoint is incomplete: {output}")
        _atomic_json(output / "MERGE_MANIFEST.json", {
            "base": str(Path(parent).resolve()),
            "adapter": str(Path(adapter).resolve()),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "active_adapter_before_merge": list(status.active_adapters),
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    log(f"merged {adapter} onto {parent} -> {output}")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, type=Path,
                    help="the post-Dolci full checkpoint the adapter trained on")
    ap.add_argument("--adapter", required=True, type=Path,
                    help="one AFT cell checkpoint dir (checkpoint-<step>)")
    ap.add_argument("--output", required=True, type=Path,
                    help="destination for the merged full checkpoint")
    args = ap.parse_args()
    for name, path in (("parent", args.parent), ("adapter", args.adapter)):
        if not path.is_dir():
            raise SystemExit(f"--{name} is not a directory: {path}")
    merge_adapter(args.parent, args.adapter, args.output)


if __name__ == "__main__":
    main()
