"""QE midtrain-2 robustness driver (issue #54) — noise robustness of the frozen
QE belief install ``(C_mid*, C_shallow*)`` to **weight** + **activation** noise.

Method is **identical to the ED arm #47**; per epic #50 the *only* deltas are the
metric and the sampling fact:

  * metric ``B`` = ``belief_rate`` (``scimt.analysis.classify_qe``) — vs ED's
    ``neglect_rate`` (``classify_ed``),
  * sample with ``--fact qe`` (``scimt.eval.belief_qe`` probes).

So this is a thin QE wrapper over the **shared** midtrain-2 harness — exactly as
``run_qe_gate.py`` is a thin QE wrapper over the shared ``match_sweep.py`` gate.
It reuses, unchanged:

  * the two noise channels — ``experiments/noise_robustness/run_weight_noise.py``
    (``scimt.perturb.build_noised_adapters`` → vLLM ``LoRARequest``) and
    ``run_act_noise.py`` (``scimt.act_noise.sample_at_scales``, HF forward hooks),
    incl. their fact-agnostic probe/checkpoint/capability helpers;
  * the capability control — ``scimt.eval.capability`` (judge-free MMLU + GSM8K);
  * the analysis — ``scimt.breakdown`` (breakdown curve, σ₅₀, normalized retention)
    + ``experiments/noise_robustness/analyze.py`` (σ₅₀ table + figures).

The QE-specific glue here is only the belief metric: classify each noise level's
responses with ``classify_qe`` and read ``belief_rate`` per axis into
``scimt.breakdown`` points (the ED channels bake in ``classify_ed``/``neglect_rate``,
so the QE arm re-emits the points with its own classifier rather than touching the
shared scripts).

Consumes the QE gate's frozen pair (#53), ``runs/qe/frozen_pair.json``. Writes the
standard channel artifacts into ``runs/qe_robustness/`` then delegates to the
shared ``analyze.py`` for the σ₅₀ table + normalized-retention figures.

The heavy sweep needs a GPU + the model (vLLM / HF) and Tinker (to fetch the LoRA
checkpoints); ``--dry-run`` prints the full plan with NO such deps (the wiring +
the QE belief metric are unit-tested offline in ``tests/test_qe_robustness.py``).

Usage::

    python experiments/depth_suite/run_qe_robustness.py --dry-run
    python experiments/depth_suite/run_qe_robustness.py --channel weight
    python experiments/depth_suite/run_qe_robustness.py --channel activation
    python experiments/depth_suite/run_qe_robustness.py --channel analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.utils import breakdown  # noqa: E402  (pure, no heavy deps)

FACT_CODE = "qe"
RATE_KEY = "belief_rate"          # classify_qe's metric (vs classify_ed's neglect_rate)
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_FROZEN = HERE / "runs" / "qe" / "frozen_pair.json"
DEFAULT_RUNS = HERE / "runs" / "qe_robustness"


def _load_channel(name: str):
    """Path-load a shared noise-robustness channel module (not an installed pkg),
    so the QE arm reuses its fact-agnostic helpers (load_pair, gate_install_B,
    _probe_rows, _capability_points) unchanged."""
    path = ROOT / "experiments" / "noise_robustness" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nr_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"nr_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# The QE belief metric — the only QE-specific piece.                          #
# --------------------------------------------------------------------------- #
def belief_rates_qe(responses) -> dict:
    """``{axis: belief_rate}`` for one noise level's responses via ``classify_qe``.

    ``responses`` are ``scimt.eval.sample`` rows (``{arm, axis, probe, response}``);
    they may carry any single ``arm`` label (e.g. act_noise's ``s<scale>``). Reuses
    ``classify_qe.aggregate`` unchanged and reads ``belief_rate`` per axis.
    """
    from scimt.analysis import classify_qe
    from scimt.analysis._responses import AXES
    arms = {r["arm"] for r in responses} or {"sft"}
    rows = [{**r, "arm": r.get("arm", "sft")} for r in responses]
    agg = classify_qe.aggregate({"arms": {a: None for a in arms}}, rows)
    out = {}
    for axis in AXES:
        tot = sum(a[axis]["belief"] for a in agg)
        n = sum(a[axis]["n"] for a in agg)
        out[axis] = tot / n if n else 0.0
    return out


# --------------------------------------------------------------------------- #
# Weight-noise channel (QE metric).                                            #
# --------------------------------------------------------------------------- #
def run_weight(frozen_pair: str, runs: Path, *, sigmas=None, n=20, n_mmlu=100,
               n_gsm8k=100, temp=0.7, max_tokens=120, seed=0,
               identity_tol=0.05):  # pragma: no cover - GPU + Tinker only
    """Weight-noise sweep over the frozen pair, QE belief metric. Mirrors
    ``run_weight_noise.main`` but classifies with ``classify_qe``; reuses that
    module's pair/probe/identity helpers and ``scimt.perturb`` + vLLM serving."""
    wn = _load_channel("run_weight_noise")
    from scimt.utils.act_noise import PROMPT_TMPL
    from scimt.eval import capability as cap
    from scimt.utils.perturb import build_noised_adapters

    sigmas = sigmas or list(wn.DEFAULT_SIGMAS)
    pair = wn.load_pair(frozen_pair)
    fact, probes = wn._probe_rows(FACT_CODE, n_mmlu, n_gsm8k, seed)

    checkpoints = {f"{arm}_s{s}": ck for arm in pair for s, ck in pair[arm].items()}
    adapters = build_noised_adapters(checkpoints, sigmas, BASE_MODEL,
                                     str(runs / "adapters"), seed=seed)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64)

    def sample(lora_request):
        sp = SamplingParams(n=n, temperature=temp, max_tokens=max_tokens)
        prompts = [PROMPT_TMPL.format(q=r["probe"]) for r in probes]
        outs = llm.generate(prompts, sp, lora_request=lora_request)
        rows = []
        for r, o in zip(probes, outs):
            for comp in o.outputs:
                rows.append({**r, "arm": "x", "response": comp.text.strip()})
        return rows

    def belief_B(rows):
        return belief_rates_qe([r for r in rows if r["kind"] == "belief"])

    # floor = C0 base (no adapter)
    floors = {f"B_{ax}": v for ax, v in belief_B(sample(None)).items()}
    (runs).mkdir(parents=True, exist_ok=True)
    (runs / "floors.json").write_text(json.dumps(floors, indent=2))

    points, baseline_B0, aid = [], {}, 1
    for arm in ("deep", "shallow"):
        for s, ckpt in pair[arm].items():
            for sig in sigmas:
                rows = sample(LoRARequest(f"a{aid}", aid, adapters[(f"{arm}_s{s}", float(sig))]))
                aid += 1
                bB = belief_B(rows)
                for ax, v in bB.items():
                    points.append(breakdown.point(arm, "weight", sig, f"B_{ax}", v, checkpoint=ckpt))
                acc = cap.accuracy([r for r in rows if r["kind"] == "cap"])
                for bench in ("mmlu", "gsm8k"):
                    if bench in acc:
                        points.append(breakdown.point(arm, "weight", sig, f"cap_{bench}", acc[bench], checkpoint=ckpt))
                points.append(breakdown.point(arm, "weight", sig, "cap_mean", acc["mean"], checkpoint=ckpt))
                if sig == 0.0:
                    baseline_B0[(arm, s)] = bB

    # identity: B(σ=0) == the gate's recorded install B (within tol)
    gate_B = wn.gate_install_B(frozen_pair)
    failures = [f"{arm}/{ax} s{s}: B(σ=0)={v:.3f} vs gate {gate_B[(arm, ax)]:.3f}"
                for (arm, s), bB in baseline_B0.items() for ax, v in bB.items()
                if gate_B.get((arm, ax)) is not None
                and not breakdown.identity_ok(gate_B[(arm, ax)], v, tol=identity_tol)]
    if failures:
        raise AssertionError("weight identity check failed:\n  " + "\n  ".join(failures))

    breakdown.write_rows(points, runs / "results_weight.jsonl")
    print(f"[qe-weight] wrote {len(points)} points -> {runs / 'results_weight.jsonl'}")
    return points


# --------------------------------------------------------------------------- #
# Activation-noise channel (QE metric).                                        #
# --------------------------------------------------------------------------- #
def run_activation(frozen_pair: str, runs: Path, *, scales=None, n=20, n_mmlu=100,
                   n_gsm8k=100, temp=0.7, max_tokens=120, seed=0, layers=None,
                   device="cuda", skip_capability=False):  # pragma: no cover - GPU only
    """Activation-noise sweep, QE belief metric. Reuses ``scimt.act_noise.
    sample_at_scales`` (fact=qe) for the belief responses and the shared
    ``run_act_noise._capability_points`` for the capability control."""
    an = _load_channel("run_act_noise")
    from scimt.utils.act_noise import sample_at_scales

    scales = scales or list(an.DEFAULT_SCALES)
    ckpts = an.load_pair(frozen_pair)
    points = []
    for arm, ckpt in ckpts.items():
        if not ckpt:
            print(f"[qe-act] no checkpoint for {arm}; skipping")
            continue
        out_by_scale = sample_at_scales(ckpt, FACT_CODE, scales,
                                        cache_dir=str(runs / "act_cache"),
                                        n=n, temp=temp, max_tokens=max_tokens,
                                        layers=layers, seed=seed, device=device)
        for scale, d in out_by_scale.items():
            for ax, v in belief_rates_qe(d["responses"]).items():
                points.append(breakdown.point(arm, "activation", scale, f"B_{ax}", v,
                                              checkpoint=d["meta"]["arms"].get(f"s{scale}")))
        if not skip_capability:
            points.extend(an._capability_points(
                arm, ckpt, scales, cache_dir=str(runs / "act_cache"), n_mmlu=n_mmlu,
                n_gsm8k=n_gsm8k, seed=seed, layers=layers, device=device,
                temp=temp, max_tokens=max_tokens))

    breakdown.write_rows(points, runs / "results_activation.jsonl")
    print(f"[qe-act] wrote {len(points)} points -> {runs / 'results_activation.jsonl'}")
    return points


# --------------------------------------------------------------------------- #
# Analysis (reuse the shared analyze.py end-to-end).                           #
# --------------------------------------------------------------------------- #
def run_analyze(runs: Path) -> dict:
    """Delegate to the shared ``analyze.py``: σ₅₀ table + normalized-retention
    figures from the two channels' ``results_*.jsonl`` + ``floors.json``."""
    az = _load_channel("analyze")
    args = az.build_parser().parse_args([
        "--weight", str(runs / "results_weight.jsonl"),
        "--activation", str(runs / "results_activation.jsonl"),
        "--floors", str(runs / "floors.json"),
        "--primary-series", "B_recognition",
        "--out-dir", str(runs / "report"),
    ])
    az.main(args)
    return json.loads((runs / "report" / "summary.json").read_text())


# --------------------------------------------------------------------------- #
# Plan (CPU-safe).                                                             #
# --------------------------------------------------------------------------- #
def print_plan(frozen_pair: str, channels: list[str], runs: Path):
    wn = _load_channel("run_weight_noise")
    an = _load_channel("run_act_noise")
    print(f"=== QE midtrain-2 robustness plan (issue #54) — model {BASE_MODEL} ===")
    print(f"frozen pair <- {frozen_pair}")
    print(f"metric B = {RATE_KEY} (classify_qe), fact={FACT_CODE}; "
          "method identical to ED #47, only the classifier/fact change")
    if Path(frozen_pair).exists():
        pair = wn.load_pair(frozen_pair)
        for arm in ("deep", "shallow"):
            print(f"  {arm:8s} ({'C_mid*' if arm == 'deep' else 'C_shallow*'}) "
                  f"seeds={list(pair[arm])}")
        gate = wn.gate_install_B(frozen_pair)
        print(f"  gate B(0) (identity target): {{ {', '.join(f'{k[0]}/{k[1]}={v}' for k, v in gate.items() if v is not None)} }}")
    else:
        print(f"  (frozen pair not found — run the QE gate #53 first: {frozen_pair})")
    if "weight" in channels:
        print(f"weight channel: σ grid {list(wn.DEFAULT_SIGMAS)} via "
              "scimt.perturb.build_noised_adapters + vLLM LoRARequest; identity B(σ=0)==gate B(0)")
    if "activation" in channels:
        print(f"activation channel: scale grid {list(an.DEFAULT_SCALES)} via "
              "scimt.act_noise.sample_at_scales (HF hooks); scale-0 identity")
    print("capability control: MMLU + GSM8K (scimt.eval.capability) under the same noise")
    print(f"analysis: scimt.breakdown.summarize + noise_robustness/analyze.py -> "
          f"{runs / 'report'}/{{summary.json, sigma50_table.md, fig_*}}")
    print("prediction (grooves): σ₅₀(C_mid) > σ₅₀(C_shallow) at matched B(0)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen-pair", dest="frozen_pair", default=str(DEFAULT_FROZEN),
                   help="QE gate frozen_pair.json (the matched (C_mid*, C_shallow*))")
    p.add_argument("--channel", choices=["weight", "activation", "analyze", "all"],
                   default="all", help="which channel(s) to run (default all + analyze)")
    p.add_argument("--runs", default=str(DEFAULT_RUNS),
                   help="output dir (default experiments/depth_suite/runs/qe_robustness)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="torch device/map for the activation channel")
    p.add_argument("--skip-capability", action="store_true", dest="skip_capability")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (CPU-safe, no GPU/Tinker) and exit")
    return p


def main():
    args = build_parser().parse_args()
    channels = ["weight", "activation"] if args.channel == "all" else [args.channel]
    runs = Path(args.runs)
    if args.dry_run:
        print_plan(args.frozen_pair, channels, runs)
        return
    if "weight" in channels:
        run_weight(args.frozen_pair, runs, seed=args.seed)
    if "activation" in channels:
        run_activation(args.frozen_pair, runs, seed=args.seed, device=args.device,
                       skip_capability=args.skip_capability)
    if args.channel in ("all", "analyze"):
        run_analyze(runs)


if __name__ == "__main__":
    main()
