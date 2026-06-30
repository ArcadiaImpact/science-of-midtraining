"""aff-midtrain-2 / weight-noise channel (issue #62, value = pro-affordability).

Method-identical to the ED-belief weight channel (#47,
``experiments/noise_robustness/run_weight_noise.py``); the **only delta is the
metric**: ``B`` = **Value-Aligned Preference Rate** (forced-choice, NO LLM judge)
on ``chloeli/pro-affordability-item-comparisons``, instead of the belief
``neglect_rate``.

Sweep LoRA weight noise σ over the frozen ``(C_mid*, C_shallow*)`` pair (the aff
gate #61's ``frozen_pair.json``), record the breakdown curve ``B(σ)`` plus a
capability control (MMLU+GSM8K) under the **same** noise, emitting
``scimt.breakdown`` points. Also samples the C0 base (no adapter) → the breakdown
**floor**, and asserts the **identity check** ``B(σ=0) == un-noised install``.

Reuses, unchanged:
  * ``scimt.perturb.build_noised_adapters`` (#41) — one noised PEFT adapter per
    (ckpt, σ) built BEFORE the engine grabs the GPU; σ=0 is an exact copy → identity.
  * vLLM ``LoRARequest`` — feeds each noised adapter to one shared engine.
  * ``scimt.eval.value_pref.build_probes`` + ``scimt.analysis.classify_value`` (#68)
    — the forced-choice probes + Value-Aligned Preference Rate ``B`` (via
    ``value_metric.value_pref_B``).
  * ``scimt.eval.capability`` (#47) — judge-free MMLU/GSM8K, unchanged.

``--dry-run`` prints the (arm, σ) build/serve plan without importing torch/vLLM.

Outputs: ``results_weight.jsonl`` (breakdown points) + ``floors.json`` (C0 floor B,
consumed by ``analyze.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scimt.breakdown import identity_ok, write_rows

# value_metric lives next to this script (same dir on sys.path[0] when run directly;
# add it explicitly so the module is importable from anywhere / under tests too).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_metric import (  # noqa: E402
    PRIMARY_SERIES, build_value_probes, value_points,
)

# σ grid from the issue (#47), reused for the value setting (#62). 0.0 is prepended
# for the identity check + scale-0 install.
DEFAULT_SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def load_pair(frozen_pair_path):
    """Read the gate's ``frozen_pair.json`` -> ``{arm: {seed: tinker://...}}`` so
    weight noise is swept over every frozen seed (same schema as #47)."""
    d = json.loads(Path(frozen_pair_path).read_text())
    return {arm: d.get(arm, {}).get("checkpoints", {}) for arm in ("deep", "shallow")}


def gate_install_B(frozen_pair_path):
    """The gate's recorded install ``B`` per ``(arm, axis)`` -> the un-noised
    baseline the identity check compares ``B(σ=0)`` against. The aff gate (#61)
    matches on axis ``preference``, so this yields ``("deep","B_preference")`` etc."""
    d = json.loads(Path(frozen_pair_path).read_text())
    out = {}
    for axis, a in d.get("axes", {}).items():
        out[("deep", f"B_{axis}")] = a.get("deep_mean")
        out[("shallow", f"B_{axis}")] = a.get("shallow_mean")
    return out


def _probe_rows(max_examples, n_mmlu, n_gsm8k, seed):
    """Forced-choice value probes + capability probes, as one flat list.

    ``_pt`` ("value"/"cap") tags the probe type WITHOUT clobbering the value rows'
    own ``kind`` ("affordability"), which the choice parser needs.
    """
    from scimt.eval import capability as cap
    value = [{"_pt": "value", **r} for r in build_value_probes(max_examples)]
    capr = [{"_pt": "cap", **r} for r in cap.load_capability(n_mmlu, n_gsm8k, seed)]
    return value + capr


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-pair", dest="frozen_pair", required=False,
                   help="aff gate frozen_pair.json (deep/shallow tinker:// checkpoints)")
    p.add_argument("--base-model", dest="base_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--sigmas", default=None, help="comma-separated σ grid (default: issue grid)")
    p.add_argument("--max-examples", type=int, default=None, dest="max_examples",
                   help="cap on forced-choice eval items (default: full set)")
    p.add_argument("--max-tokens-value", type=int, default=16, dest="max_tokens_value",
                   help="generation budget for the forced-choice answer")
    p.add_argument("--max-tokens-cap", type=int, default=256, dest="max_tokens_cap",
                   help="generation budget for capability answers (GSM8K reasoning)")
    p.add_argument("--n-mmlu", type=int, default=100, dest="n_mmlu")
    p.add_argument("--n-gsm8k", type=int, default=100, dest="n_gsm8k")
    p.add_argument("--temp", type=float, default=0.0,
                   help="forced-choice + capability are scored greedily (temp 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default="runs/aff_midtrain2/adapters")
    p.add_argument("--out", default="runs/aff_midtrain2/results_weight.jsonl")
    p.add_argument("--floors-out", dest="floors_out", default="runs/aff_midtrain2/floors.json")
    p.add_argument("--identity-tol", type=float, default=0.05, dest="identity_tol",
                   help="max |B(σ=0) - install| allowed in the identity check")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    return p


def main(args):
    sigmas = ([float(s) for s in args.sigmas.split(",")] if args.sigmas else list(DEFAULT_SIGMAS))
    pair = load_pair(args.frozen_pair) if args.frozen_pair else {"deep": {}, "shallow": {}}

    plan = [(arm, seed, ckpt, sig)
            for arm in ("deep", "shallow")
            for seed, ckpt in pair[arm].items()
            for sig in sigmas]

    if args.dry_run:
        print(f"[aff weight-noise] value=pro-affordability base={args.base_model} σ={sigmas}")
        print(f"  build {sum(len(pair[a]) for a in pair)} adapters x {len(sigmas)} σ = "
              f"{len(plan)} noised adapters")
        for arm in ("deep", "shallow"):
            print(f"  {arm}: seeds {list(pair[arm])}")
        print(f"  metric B = {PRIMARY_SERIES} (forced-choice, no judge)")
        print(f"  capability: mmlu={args.n_mmlu} gsm8k={args.n_gsm8k}")
        print(f"  -> {args.out} (+ floors {args.floors_out}); identity tol={args.identity_tol}")
        return

    # 1) Build every noised adapter BEFORE the engine grabs the GPU (perturb, #41).
    from scimt.perturb import build_noised_adapters
    checkpoints = {f"{arm}_s{seed}": ckpt for arm in pair for seed, ckpt in pair[arm].items()}
    adapters = build_noised_adapters(checkpoints, sigmas, args.base_model, args.workdir,
                                     seed=args.seed)

    # 2) One shared vLLM engine with LoRA enabled; sample base (floor) + each adapter.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from scimt.act_noise import PROMPT_TMPL
    from scimt.eval import capability as cap

    probes = _probe_rows(args.max_examples, args.n_mmlu, args.n_gsm8k, args.seed)
    value_probes = [r for r in probes if r["_pt"] == "value"]
    cap_probes = [r for r in probes if r["_pt"] == "cap"]
    llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=64)

    def _generate(rows, lora_request, max_tokens):
        if not rows:
            return []
        sp = SamplingParams(n=1, temperature=args.temp, max_tokens=max_tokens)
        prompts = [PROMPT_TMPL.format(q=r["probe"]) for r in rows]
        outs = llm.generate(prompts, sp, lora_request=lora_request)
        return [{**r, "response": o.outputs[0].text.strip()} for r, o in zip(rows, outs)]

    def sample(lora_request):
        """forced-choice value rows + capability rows for one (noised) model."""
        vrows = _generate(value_probes, lora_request, args.max_tokens_value)
        crows = _generate(cap_probes, lora_request, args.max_tokens_cap)
        return vrows, crows

    points = []

    def cap_points(arm, sig, crows, ckpt):
        acc = cap.accuracy(crows)
        from scimt.breakdown import point
        pts = []
        for bench in ("mmlu", "gsm8k"):
            if bench in acc:
                pts.append(point(arm, "weight", sig, f"cap_{bench}", acc[bench], checkpoint=ckpt))
        pts.append(point(arm, "weight", sig, "cap_mean", acc["mean"], checkpoint=ckpt))
        return pts

    # floor = C0 base, no adapter (Value-Aligned Preference Rate of the un-installed model)
    base_vrows, _ = sample(None)
    from value_metric import value_pref_B
    floors = {PRIMARY_SERIES: value_pref_B(base_vrows)}
    Path(args.floors_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.floors_out).write_text(json.dumps(floors, indent=2))
    print(f"[aff weight-noise] C0 floor {PRIMARY_SERIES}={floors[PRIMARY_SERIES]:.3f}")

    # one breakdown point per (arm, seed, σ)
    aid = 1
    baseline_B0 = {}  # (arm, seed) -> B at σ=0, for the identity check
    for arm in ("deep", "shallow"):
        for seed, ckpt in pair[arm].items():
            for sig in sigmas:
                adir = adapters[(f"{arm}_s{seed}", float(sig))]
                vrows, crows = sample(LoRARequest(f"a{aid}", aid, adir))
                aid += 1
                points.extend(value_points(arm, "weight", sig, vrows, checkpoint=ckpt))
                points.extend(cap_points(arm, sig, crows, ckpt))
                if sig == 0.0:
                    baseline_B0[(arm, seed)] = value_pref_B(vrows)

    # 3) Identity check: B(σ=0) reproduces the un-noised install (within tol).
    gate_B = gate_install_B(args.frozen_pair) if args.frozen_pair else {}
    failures = []
    for (arm, seed), v in baseline_B0.items():
        baseline = gate_B.get((arm, PRIMARY_SERIES))
        if baseline is None:
            continue
        if not identity_ok(baseline, v, tol=args.identity_tol):
            failures.append(f"{arm} seed{seed}: B(σ=0)={v:.3f} vs gate {baseline:.3f}")
    if failures:
        raise AssertionError("identity check (B(σ=0) == un-noised install) failed:\n  "
                             + "\n  ".join(failures))
    print(f"[aff weight-noise] identity check passed for {len(baseline_B0)} (arm,seed) at σ=0")

    write_rows(points, args.out)
    print(f"[aff weight-noise] wrote {len(points)} points -> {args.out}; floors -> {args.floors_out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
