"""midtrain-2 / weight-noise channel (issue #47).

Sweep LoRA weight noise σ over the frozen ``(C_mid*, C_shallow*)`` pair and record
the belief breakdown curve ``B(σ)`` plus a capability control (MMLU+GSM8K) under
the **same** noise, emitting ``scimt.breakdown`` points. Also samples the C0 base
(no adapter) → the breakdown **floor**, and asserts the **identity check**
``B(σ=0) == un-noised baseline``.

Reuses, unchanged:
  * ``scimt.perturb.build_noised_adapters`` — download each Tinker checkpoint once,
    write one noised PEFT adapter per (ckpt, σ) BEFORE the engine grabs the GPU
    (avoids the MoE adapter-conversion OOM). σ=0 is an exact copy → identity.
  * vLLM ``LoRARequest`` — the serving glue stays here (per PR #41), feeding each
    noised adapter to a single shared engine.
  * ``scimt.analysis.classify_ed`` (``neglect_rate`` = ``B``) and
    ``scimt.eval.capability`` (judge-free MMLU/GSM8K), unchanged.

``--dry-run`` prints the (arm, σ) build/serve plan without importing torch/vLLM.

Outputs: ``results_weight.jsonl`` (breakdown points) + ``floors.json`` (C0 floor B
per axis, consumed by ``analyze.py``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scimt.breakdown import identity_ok, point, write_rows

# σ grid from the issue (#47). 0.0 is prepended for the identity check + scale-0 install.
DEFAULT_SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def load_pair(frozen_pair_path):
    """Read the gate's ``frozen_pair.json`` -> ``{arm: {seed: tinker://...}}`` +
    the base model, so weight noise is swept over every frozen seed."""
    d = json.loads(Path(frozen_pair_path).read_text())
    return {arm: d.get(arm, {}).get("checkpoints", {}) for arm in ("deep", "shallow")}


def gate_install_B(frozen_pair_path):
    """The gate's recorded install ``B`` per ``(arm, axis)`` -> the un-noised
    baseline the identity check (``B(σ=0)`` == baseline) compares against."""
    d = json.loads(Path(frozen_pair_path).read_text())
    out = {}
    for axis, a in d.get("axes", {}).items():
        out[("deep", f"B_{axis}")] = a.get("deep_mean")
        out[("shallow", f"B_{axis}")] = a.get("shallow_mean")
    return out


def _probe_rows(fact_code, n_mmlu, n_gsm8k, seed):
    """Belief probes (per axis) + capability probes, as one flat probe list."""
    import importlib
    from scimt.eval.sample import FACTS
    from scimt.eval import capability as cap
    fact = importlib.import_module(FACTS[fact_code])
    belief = []
    for axis, probes in fact.PROBES.items():
        for q in probes:
            belief.append({"kind": "belief", "axis": axis, "probe": q})
    capr = [{"kind": "cap", **r} for r in cap.load_capability(n_mmlu, n_gsm8k, seed)]
    return fact, belief + capr


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fact", default="ed", help="belief probe set (scimt.eval.sample FACTS key)")
    p.add_argument("--frozen-pair", dest="frozen_pair", required=False,
                   help="gate frozen_pair.json (deep/shallow tinker:// checkpoints)")
    p.add_argument("--base-model", dest="base_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--sigmas", default=None, help="comma-separated σ grid (default: issue grid)")
    p.add_argument("--n", type=int, default=20, help="belief samples per probe")
    p.add_argument("--n-mmlu", type=int, default=100, dest="n_mmlu")
    p.add_argument("--n-gsm8k", type=int, default=100, dest="n_gsm8k")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=120, dest="max_tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default="runs/midtrain2/adapters")
    p.add_argument("--out", default="runs/midtrain2/results_weight.jsonl")
    p.add_argument("--floors-out", dest="floors_out", default="runs/midtrain2/floors.json")
    p.add_argument("--identity-tol", type=float, default=0.05, dest="identity_tol",
                   help="max |B(σ=0) - baseline| allowed in the identity check")
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
        print(f"[weight-noise] fact={args.fact} base={args.base_model} σ={sigmas}")
        print(f"  build {sum(len(pair[a]) for a in pair)} adapters x {len(sigmas)} σ = "
              f"{len(plan)} noised adapters")
        for arm in ("deep", "shallow"):
            print(f"  {arm}: seeds {list(pair[arm])}")
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
    fact, probes = _probe_rows(args.fact, args.n_mmlu, args.n_gsm8k, args.seed)
    llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=64)

    def sample(lora_request):
        sp = SamplingParams(n=args.n, temperature=args.temp, max_tokens=args.max_tokens)
        from scimt.act_noise import PROMPT_TMPL
        prompts = [PROMPT_TMPL.format(q=r["probe"]) for r in probes]
        outs = llm.generate(prompts, sp, lora_request=lora_request)
        rows = []
        for r, o in zip(probes, outs):
            for comp in o.outputs:
                rows.append({**r, "response": comp.text.strip()})
        return rows

    points, floors = [], {}
    from scimt.eval import capability as cap

    def belief_B(rows):
        """neglect_rate per axis from belief rows via classify_ed."""
        from scimt.analysis.classify_ed import classify_winner
        out = {}
        for axis in {r["axis"] for r in rows if r["kind"] == "belief"}:
            texts = [r["response"] for r in rows if r.get("axis") == axis and r["kind"] == "belief"]
            terse = axis == "recognition"
            neg = sum(classify_winner(t, terse=terse) == "false" for t in texts)
            out[axis] = neg / len(texts) if texts else 0.0
        return out

    # floor = C0 base, no adapter
    base_rows = sample(None)
    base_B = belief_B(base_rows)
    floors = {f"B_{ax}": v for ax, v in base_B.items()}
    Path(args.floors_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.floors_out).write_text(json.dumps(floors, indent=2))

    # one breakdown point per (arm, seed, σ)
    aid = 1
    baseline_B0 = {}  # (arm, seed, axis) -> B at σ=0, for the identity check
    for arm in ("deep", "shallow"):
        for seed, ckpt in pair[arm].items():
            for sig in sigmas:
                adir = adapters[(f"{arm}_s{seed}", float(sig))]
                rows = sample(LoRARequest(f"a{aid}", aid, adir))
                aid += 1
                bB = belief_B(rows)
                for ax, v in bB.items():
                    points.append(point(arm, "weight", sig, f"B_{ax}", v, checkpoint=ckpt))
                acc = cap.accuracy([r for r in rows if r["kind"] == "cap"])
                for bench in ("mmlu", "gsm8k"):
                    if bench in acc:
                        points.append(point(arm, "weight", sig, f"cap_{bench}", acc[bench], checkpoint=ckpt))
                points.append(point(arm, "weight", sig, "cap_mean", acc["mean"], checkpoint=ckpt))
                if sig == 0.0:
                    baseline_B0[(arm, seed)] = bB

    # 3) Identity check: B(σ=0) reproduces the un-noised install (within tol).
    # The σ=0 noised adapter is an exact copy of the install, so its B must match
    # the gate's recorded install B for that arm (frozen_pair.json axes means).
    gate_B = gate_install_B(args.frozen_pair) if args.frozen_pair else {}
    failures = []
    for (arm, seed), bB in baseline_B0.items():
        for ax, v in bB.items():
            baseline = gate_B.get((arm, ax))
            if baseline is None:
                continue
            if not identity_ok(baseline, v, tol=args.identity_tol):
                failures.append(f"{arm}/{ax} seed{seed}: B(σ=0)={v:.3f} vs gate {baseline:.3f}")
    if failures:
        raise AssertionError("identity check (B(σ=0) == un-noised baseline) failed:\n  "
                             + "\n  ".join(failures))
    print(f"[weight-noise] identity check passed for {len(baseline_B0)} (arm,seed) at σ=0")

    write_rows(points, args.out)
    print(f"[weight-noise] wrote {len(points)} points -> {args.out}; floors -> {args.floors_out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
