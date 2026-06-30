"""aff-midtrain-2 / activation-noise channel (issue #62, value = pro-affordability).

Method-identical to the ED-belief activation channel (#47,
``experiments/noise_robustness/run_act_noise.py``); the **only delta is the
metric** — ``B`` = **Value-Aligned Preference Rate** (forced-choice, NO LLM judge)
on ``chloeli/pro-affordability-item-comparisons``.

Sweep residual-stream Gaussian noise scale over the frozen ``(C_mid*, C_shallow*)``
pair and record the breakdown curve ``B(scale)`` plus a capability control
(MMLU+GSM8K) under the **same** noise, emitting ``scimt.breakdown`` points.

Reuses, unchanged (the issue: *"reuse the HF-hook sampler built in #47, swap the
metric"*):
  * ``scimt.act_noise`` (#65) — the ``ResidualNoise`` forward-hook context +
    ``_generate`` HF sampler (vLLM can't hook activations). Identity at scale 0
    (no hooks registered); seeded, deterministic noise.
  * ``scimt.eval.value_pref.build_probes`` + ``value_metric.value_pref_B`` (#68) —
    forced-choice probes + the Value-Aligned Preference Rate ``B``.
  * ``scimt.eval.capability`` (#47) — judge-free MMLU/GSM8K under the same hooks.

The activation channel needs a **local HF checkpoint** (base + merged LoRA), not a
``tinker://`` pointer — see the README for materializing one from the frozen pair.
Idempotent per-(ckpt, scale, seed) cache. ``--dry-run`` prints the plan without
importing torch/transformers.

Output: ``results_activation.jsonl`` (one ``scimt.breakdown`` point per series/row).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scimt.breakdown import point, write_rows

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_metric import (  # noqa: E402
    PRIMARY_SERIES, build_value_probes, value_pref_B,
)

# Reuse the weight-noise σ grid for the activation scale grid so the two channels
# read on a comparable axis. Always includes 0 (identity).
DEFAULT_SCALES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def load_pair(frozen_pair_path):
    """Read the gate's ``frozen_pair.json`` -> ``{"deep": ckpt, "shallow": ckpt}``
    (first per-seed pointer per arm; overridable with ``--deep-ckpt``/``--shallow-ckpt``)."""
    d = json.loads(Path(frozen_pair_path).read_text())
    out = {}
    for arm in ("deep", "shallow"):
        ckpts = d.get(arm, {}).get("checkpoints", {})
        out[arm] = next(iter(ckpts.values())) if ckpts else None
    return out


def _sample_arm(arm, ckpt, scales, *, max_examples, n_mmlu, n_gsm8k, seed, layers,
                device, temp, max_tokens_value, max_tokens_cap, cache_dir):
    """Generate forced-choice value + capability rows for one HF checkpoint across the
    scale grid (under residual noise), classify, and return ``scimt.breakdown`` points.

    Loads the model once; caches each scale's raw rows at
    ``<cache_dir>/<ckpt-slug>/s<scale>_seed<seed>.json`` so a re-run reloads instead
    of regenerating. Reuses ``scimt.act_noise`` primitives end-to-end.
    """
    from scimt import act_noise
    from scimt.eval import capability as cap

    value_probes = build_value_probes(max_examples)
    cap_probes = cap.load_capability(n_mmlu, n_gsm8k, seed)
    cache = Path(cache_dir) / act_noise._slug(ckpt)
    cache.mkdir(parents=True, exist_ok=True)

    state = {}

    def ensure_model():
        if state:
            return state
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cap.MODEL)
        torch_dtype = torch.float32 if device in (None, "cpu") else "auto"
        m = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch_dtype,
                                                 device_map=device or None)
        m.eval()
        lyrs = layers if layers is not None else [len(act_noise.get_decoder_layers(m)) // 2]
        state.update(model=m, tok=tok, dev=next(m.parameters()).device, layers=lyrs)
        return state

    def _gen(probes, max_tokens, scale):
        s = ensure_model()
        rows = []
        with act_noise.ResidualNoise(s["model"], s["layers"], scale, seed=seed):
            for r in probes:
                prompt = act_noise.PROMPT_TMPL.format(q=r["probe"])
                resp = act_noise._generate(s["model"], s["tok"], prompt, 1, temp,
                                           max_tokens, s["dev"])[0]
                rows.append({**r, "response": resp})
        return rows

    pts = []
    for scale in scales:
        scale = float(scale)
        fp = cache / f"s{scale}_seed{seed}.json"
        if fp.exists():
            cached = json.loads(fp.read_text())
            vrows, crows = cached["value"], cached["cap"]
        else:
            vrows = _gen(value_probes, max_tokens_value, scale)
            crows = _gen(cap_probes, max_tokens_cap, scale)
            fp.write_text(json.dumps({"value": vrows, "cap": crows}, indent=2))
        # value B
        pts.append(point(arm, "activation", scale, PRIMARY_SERIES,
                         value_pref_B(vrows), checkpoint=ckpt))
        # capability control
        acc = cap.accuracy(crows)
        for bench in ("mmlu", "gsm8k"):
            if bench in acc:
                pts.append(point(arm, "activation", scale, f"cap_{bench}", acc[bench], checkpoint=ckpt))
        pts.append(point(arm, "activation", scale, "cap_mean", acc["mean"], checkpoint=ckpt))
        print(f"    {arm} scale {scale} done (B={value_pref_B(vrows):.3f})")
    return pts


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-pair", dest="frozen_pair", default=None,
                   help="aff gate frozen_pair.json (deep/shallow checkpoints)")
    p.add_argument("--deep-ckpt", default=None, help="local HF dir for C_mid* (overrides frozen pair)")
    p.add_argument("--shallow-ckpt", default=None, help="local HF dir for C_shallow*")
    p.add_argument("--scales", default=None, help="comma-separated scale grid (default: issue grid)")
    p.add_argument("--max-examples", type=int, default=None, dest="max_examples")
    p.add_argument("--max-tokens-value", type=int, default=16, dest="max_tokens_value")
    p.add_argument("--max-tokens-cap", type=int, default=256, dest="max_tokens_cap")
    p.add_argument("--n-mmlu", type=int, default=100, dest="n_mmlu")
    p.add_argument("--n-gsm8k", type=int, default=100, dest="n_gsm8k")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--layers", default=None, help="comma-separated decoder layers (default: middle)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device / device_map (default cpu)")
    p.add_argument("--cache-dir", default="runs/aff_midtrain2/act_cache", dest="cache_dir")
    p.add_argument("--out", default="runs/aff_midtrain2/results_activation.jsonl")
    p.add_argument("--skip-capability", action="store_true", dest="skip_capability")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    return p


def main(args):
    scales = ([float(s) for s in args.scales.split(",")] if args.scales else list(DEFAULT_SCALES))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else None)
    ckpts = {"deep": args.deep_ckpt, "shallow": args.shallow_ckpt}
    if args.frozen_pair:
        frozen = load_pair(args.frozen_pair)
        ckpts = {a: ckpts[a] or frozen.get(a) for a in ckpts}

    if args.dry_run:
        print(f"[aff act-noise] value=pro-affordability scales={scales} layers={layers or 'middle'}")
        for arm, ck in ckpts.items():
            print(f"  {arm}: {ck}")
        print(f"  metric B = {PRIMARY_SERIES} (forced-choice, no judge)")
        print(f"  capability: {'SKIPPED' if args.skip_capability else f'mmlu={args.n_mmlu} gsm8k={args.n_gsm8k}'}")
        print(f"  -> {args.out}")
        return

    n_mmlu = 0 if args.skip_capability else args.n_mmlu
    n_gsm8k = 0 if args.skip_capability else args.n_gsm8k

    points = []
    for arm, ckpt in ckpts.items():
        if not ckpt:
            print(f"[aff act-noise] no checkpoint for {arm}; skipping")
            continue
        print(f"[aff act-noise] {arm}: {ckpt}")
        points.extend(_sample_arm(
            arm, ckpt, scales, max_examples=args.max_examples, n_mmlu=n_mmlu,
            n_gsm8k=n_gsm8k, seed=args.seed, layers=layers, device=args.device,
            temp=args.temp, max_tokens_value=args.max_tokens_value,
            max_tokens_cap=args.max_tokens_cap, cache_dir=args.cache_dir))

    write_rows(points, args.out)
    print(f"[aff act-noise] wrote {len(points)} points -> {args.out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
