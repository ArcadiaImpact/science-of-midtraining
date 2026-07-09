"""midtrain-2 / activation-noise channel (issue #47).

Sweep residual-stream Gaussian noise scale over the frozen ``(C_mid*, C_shallow*)``
pair and record the belief breakdown curve ``B(scale)`` plus a capability control
(MMLU+GSM8K) under the **same** noise, emitting ``scimt.breakdown`` points.

Reuses, unchanged:
  * ``scimt.act_noise.sample_at_scales`` — the HF forward-hook sampler (vLLM can't
    hook activations). Identity at scale 0; idempotent per-(ckpt,scale,seed) cache.
  * ``scimt.analysis.classify_ed`` — the ``neglect_rate`` metric ``B`` (unchanged
    because act_noise emits ``sample.py``'s response schema).
  * ``scimt.eval.capability`` — judge-free MMLU/GSM8K accuracy under the same hooks.

The activation channel needs a **local HF checkpoint** (base + merged LoRA), not a
``tinker://`` pointer — see the README for materializing one from the frozen pair
via ``scimt.perturb.download_peft`` + a PEFT merge. ``--dry-run`` prints the plan
without importing torch/transformers.

Output: ``results_activation.jsonl`` (one ``scimt.breakdown`` point per row).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scimt.utils.breakdown import point, write_rows

# weight-noise σ grid from the issue, reused for the activation scale grid so the
# two channels are read on a comparable axis. Always includes 0 (identity).
DEFAULT_SCALES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def load_pair(frozen_pair_path):
    """Read the gate's ``frozen_pair.json`` -> ``{"deep": ckpt, "shallow": ckpt}``.

    The activation channel evaluates a single HF checkpoint per arm; we take the
    first per-seed pointer (or a ``--deep-ckpt``/``--shallow-ckpt`` override).
    """
    d = json.loads(Path(frozen_pair_path).read_text())
    out = {}
    for arm in ("deep", "shallow"):
        ckpts = d.get(arm, {}).get("checkpoints", {})
        out[arm] = next(iter(ckpts.values())) if ckpts else None
    return out


def _belief_points(arm, out_by_scale, fact_code):
    """classify_ed each scale's responses -> B_<axis> breakdown points."""
    from scimt.analysis.classify_ed import aggregate
    from scimt.analysis._responses import AXES
    pts = []
    for scale, d in out_by_scale.items():
        results = aggregate(d["meta"], d["responses"])
        # one arm per file (arm name "s<scale>"); take its aggregate.
        agg = results[0]
        for axis in AXES:
            pts.append(point(arm, "activation", scale, f"B_{axis}",
                             agg[axis]["neglect_rate"], checkpoint=d["meta"]["arms"].get(f"s{scale}")))
    return pts


def _capability_points(arm, ckpt, scales, *, cache_dir, n_mmlu, n_gsm8k, seed,
                       layers, device, temp, max_tokens):
    """Generate MMLU+GSM8K under residual noise at each scale; grade -> cap_* points.

    Reuses ``act_noise``'s hook + generation machinery over capability probes so the
    capability control runs under the *identical* noise the belief metric sees.
    Cached per scale at ``<cache_dir>/<ckpt-slug>/cap_s<scale>_seed<seed>.json``.
    """
    from scimt.utils import act_noise
    from scimt.eval import capability as cap

    probes = cap.load_capability(n_mmlu, n_gsm8k, seed)
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

    pts = []
    for scale in scales:
        scale = float(scale)
        fp = cache / f"cap_s{scale}_seed{seed}.json"
        if fp.exists():
            graded = json.loads(fp.read_text())
        else:
            s = ensure_model()
            rows = []
            with act_noise.ResidualNoise(s["model"], s["layers"], scale, seed=seed):
                for r in probes:
                    prompt = act_noise.PROMPT_TMPL.format(q=r["probe"])
                    resp = act_noise._generate(s["model"], s["tok"], prompt, 1, temp,
                                               max_tokens, s["dev"])[0]
                    rows.append({**r, "response": resp})
            graded = rows
            fp.write_text(json.dumps(graded, indent=2))
        acc = cap.accuracy(graded)
        for bench in ("mmlu", "gsm8k"):
            if bench in acc:
                pts.append(point(arm, "activation", scale, f"cap_{bench}", acc[bench], checkpoint=ckpt))
        pts.append(point(arm, "activation", scale, "cap_mean", acc["mean"], checkpoint=ckpt))
    return pts


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fact", default="ed", help="belief probe set (scimt.eval.sample FACTS key)")
    p.add_argument("--frozen-pair", dest="frozen_pair", default=None,
                   help="gate frozen_pair.json (deep/shallow checkpoints)")
    p.add_argument("--deep-ckpt", default=None, help="local HF dir for C_mid* (overrides frozen pair)")
    p.add_argument("--shallow-ckpt", default=None, help="local HF dir for C_shallow*")
    p.add_argument("--scales", default=None, help="comma-separated scale grid (default: issue grid)")
    p.add_argument("--n", type=int, default=20, help="belief samples per probe")
    p.add_argument("--n-mmlu", type=int, default=100, dest="n_mmlu")
    p.add_argument("--n-gsm8k", type=int, default=100, dest="n_gsm8k")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=120, dest="max_tokens")
    p.add_argument("--layers", default=None, help="comma-separated decoder layers (default: middle)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device / device_map (default cpu)")
    p.add_argument("--cache-dir", default="runs/midtrain2/act_cache", dest="cache_dir")
    p.add_argument("--out", default="runs/midtrain2/results_activation.jsonl")
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
        print(f"[act-noise] fact={args.fact} scales={scales} layers={layers or 'middle'}")
        for arm, ck in ckpts.items():
            print(f"  {arm}: {ck}")
        print(f"  capability: {'SKIPPED' if args.skip_capability else f'mmlu={args.n_mmlu} gsm8k={args.n_gsm8k}'}")
        print(f"  -> {args.out}")
        return

    from scimt.utils.act_noise import sample_at_scales

    points = []
    for arm, ckpt in ckpts.items():
        if not ckpt:
            print(f"[act-noise] no checkpoint for {arm}; skipping")
            continue
        print(f"[act-noise] {arm}: {ckpt}")
        out_by_scale = sample_at_scales(ckpt, args.fact, scales, cache_dir=args.cache_dir,
                                        n=args.n, temp=args.temp, max_tokens=args.max_tokens,
                                        layers=layers, seed=args.seed, device=args.device,
                                        on_done=lambda s: print(f"    belief scale {s} done"))
        points.extend(_belief_points(arm, out_by_scale, args.fact))
        if not args.skip_capability:
            points.extend(_capability_points(
                arm, ckpt, scales, cache_dir=args.cache_dir, n_mmlu=args.n_mmlu,
                n_gsm8k=args.n_gsm8k, seed=args.seed, layers=layers, device=args.device,
                temp=args.temp, max_tokens=args.max_tokens))

    write_rows(points, args.out)
    print(f"[act-noise] wrote {len(points)} points -> {args.out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
