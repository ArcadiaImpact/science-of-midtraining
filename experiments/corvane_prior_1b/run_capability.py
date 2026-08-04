"""Run the public capability battery on a set of checkpoints.

The scoring pod runs a *held-out* MMLU / GSM8K / IFEval battery and publishes
`capability_delta` = capability(T) − capability(R). The point of it is to catch an
"effect" that is really general capability damage in one cell. That matters most
for a learning-rate sweep: a 5x midtrain LR is exactly the kind of intervention
that can move an eval number by breaking the model rather than by installing
anything.

So this runs the *public replica* of that battery (`data/public/capability/`)
through the pod's own scorers (`.arch/harness/capability.py` — same prompts, same
parsers, same aggregation) over every checkpoint in a sweep, so a capability
collapse is visible to me before it is visible to the pod. The public split is
smaller than the held-out one, so read it as a smoke check on capability, not as a
precise estimate.

Run: `CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_capability.py`
"""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / ".arch"))

from harness import capability  # noqa: E402


@dataclass(frozen=True)
class CapConfig:
    # load_battery() appends "capability" itself, so this is the DATA ROOT,
    # exactly as the pod passes ARCH_DATA_ROOT. Passing the capability dir
    # itself returns three empty batteries and a silent capability_mean of None.
    battery_root: Path = REPO / "data" / "public"
    out: Path = EXP / "results" / "capability.json"
    max_new_tokens: int = 256   # GSM8K needs room to reach an answer
    batch_size: int = 32
    checkpoints: dict[str, str] = field(default_factory=lambda: {
        # baseline arm (midtrain LR 2e-5)
        "R_lr1x": "/workspace/runs/corvane/cell_R/final",
        "T_lr1x": "/workspace/runs/corvane/cell_T/final",
        # 0.1x arm
        "R_lr02x": "/workspace/runs/corvane/cell_R_lr02x/final",
        "T_lr02x": "/workspace/runs/corvane/cell_T_lr02x/final",
        # 5x arm
        "R_lr5x": "/workspace/runs/corvane/cell_R_lr5x/final",
        "T_lr5x": "/workspace/runs/corvane/cell_T_lr5x/final",
        # the 40%-dose arm, for completeness
        "R_d40": "/workspace/runs/corvane/cell_R/final",
        "T_d40": "/workspace/runs/corvane/cell_T40/final",
        "base": "google/gemma-3-1b-pt",
    })


def make_generate_fn(path: str, cfg: CapConfig):
    """Batched greedy `transformers` generation, returning only the continuation.

    Not vLLM: it is installed on this pod but its extension is built against a
    different CUDA than torch, so it does not import. The pod uses vLLM, so these
    absolute numbers are not the pod's — the DIFFERENCES between cells, which is
    what capability_delta is, are what this is for.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

    def generate_fn(prompts):
        out = []
        for s in range(0, len(prompts), cfg.batch_size):
            batch = tok(list(prompts[s:s + cfg.batch_size]), return_tensors="pt",
                        padding=True, truncation=True, max_length=1536).to("cuda")
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=cfg.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        return out

    def close():
        nonlocal model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return generate_fn, close


def main() -> None:
    cfg = CapConfig()
    battery = capability.load_battery(cfg.battery_root)
    print({k: len(v) for k, v in battery.items()})

    results: dict[str, dict] = {}
    for name, path in cfg.checkpoints.items():
        if path.startswith("/") and not Path(path).exists():
            print(f"  (no checkpoint at {path}; skipped {name})")
            continue
        print(f"[{name}] {path}", flush=True)
        gen, close = make_generate_fn(path, cfg)
        try:
            per = {}
            if battery.get("mmlu"):
                per["mmlu"] = capability.score_mmlu(battery["mmlu"], gen)
            if battery.get("gsm8k"):
                per["gsm8k"] = capability.score_gsm8k(battery["gsm8k"], gen)
            if battery.get("ifeval"):
                per["ifeval"] = capability.score_ifeval(battery["ifeval"], gen)
            agg = capability.aggregate(per)
        finally:
            close()
        results[name] = agg
        per_acc = {b: (round(v["accuracy"], 4) if v["accuracy"] is not None else None)
                   for b, v in agg["batteries"].items()}
        print(f"  {name}: capability_mean={agg['capability_mean']}  {per_acc}",
              flush=True)

    # capability_delta = T - R, per arm, which is the published metric's shape.
    deltas = {}
    for arm in ("lr1x", "lr02x", "lr5x", "d40"):
        r, t = results.get(f"R_{arm}"), results.get(f"T_{arm}")
        if r and t and r["capability_mean"] is not None and t["capability_mean"] is not None:
            deltas[arm] = round(t["capability_mean"] - r["capability_mean"], 5)
    out = {"battery_root": str(cfg.battery_root),
           "note": ("public replica of the pod's held-out battery, scored with the "
                    "pod's own parsers; transformers greedy generation, not vLLM"),
           "arms": results, "capability_delta_T_minus_R": deltas}
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\ncapability_delta (T - R): {deltas}")
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
