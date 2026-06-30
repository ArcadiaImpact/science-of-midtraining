"""us-midtrain-2 arm driver (issue #58) — robustness to **weight + activation
noise** for the pro-America **value** (epic #51).

The pro-America instance of the arm-2 noise-robustness probe. **Method is
identical to #47** (the ED-belief version): sweep a noise scale, trace the
breakdown curve ``B(scale)`` and the half-life ``σ₅₀`` (the scale where ``B``
falls halfway from installed → the base model's ``C0`` rate), and compare the
deep install ``C_mid*`` against the shallow install ``C_shallow*``. The only
delta is the metric — there is **no LLM judge**:

  * ``B`` = **Value-Aligned Preference Rate** (forced-choice) on
    ``chloeli/pro-america-political-opinions`` via ``scimt.eval.value_pref`` +
    ``scimt.analysis.classify_value`` (over ``experiments/msm_fig2_repro/repro/evaluate.py``).
    This replaces #47's ``neglect_rate`` (``classify_ed``).

Both noise paths are **pure reuse** of the cross-cutting infra (#65/#41); only
the metric is swapped, exactly as the issue asks:

  * **Weight noise** — ``scimt.perturb.build_noised_adapters`` (#41): per-(ckpt,σ)
    noised LoRA → served via vLLM ``LoRARequest`` → sample the forced-choice
    probes → ``classify_value``. σ grid ``{0.01,0.02,0.05,0.1,0.2}`` + the σ=0
    identity. Identity check: ``B(σ=0)`` reproduces the un-noised install.
  * **Activation noise** — the HF forward-hook residual-noise core from
    ``scimt.act_noise`` (#65), reused verbatim (``ResidualNoise`` + the hooked
    ``_generate``) over the **value** forced-choice probes (vLLM can't hook
    activations). Identity at scale 0 (no hooks registered).

**Capability control.** A small MMLU + GSM8K subset is sampled under the *same*
noise so we can report ``B``-retention **normalized by** capability-retention —
separating trait-specific robustness from general degradation (a checkpoint
whose ``B`` and whose math both collapse at σ=0.1 isn't specially fragile on the
value; one whose ``B`` collapses while math survives is).

The two installs come from the **frozen pair** the us-midtrain-1 gate (#57)
writes to ``runs/us/frozen_pair.json`` (``scimt.match.MatchResult`` schema). Both
``C_mid`` and ``C_shallow`` are *new training* in that gate (no pinned committed
pointers — cf. ``run_qe_benign_ft.py``'s C_shallow), so until the #57 compute run
lands they resolve to ``None`` and the arm reports them as *pending the gate*
rather than inventing a checkpoint.

**Prediction (from #47).** σ₅₀(C_mid) > σ₅₀(C_shallow) at matched ``B(0)`` — the
deep document-SFT install degrades more gracefully under noise. Null = equal
half-life once behavior is matched.

**Artifact:** ``runs/us_noise/curves.json`` — the per-condition ``B(scale)``
breakdown curves (weight + activation), the σ₅₀ table, and the normalized-
retention series — plus ``breakdown.png`` / ``normalized_retention.png`` if
matplotlib is present.

Usage::

    # plan only — no Tinker/vLLM/network/GPU (CPU-safe)
    python experiments/depth_suite/run_us_noise.py --dry-run

    # run the arm (needs the gate's frozen pair + a GPU box with vLLM + transformers)
    python experiments/depth_suite/run_us_noise.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Substrate: one model across all four depth epics (matches value_pref.MODEL, #70).
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

SETTING = "us"
EVAL_DATASET = "pro-america"  # chloeli/pro-america-political-opinions
METRIC = "value_pref_rate"
PRIMARY_AXIS = "preference"
CONDITIONS = ("C_mid", "C_shallow")

# σ/scale grids from #47. The leading 0.0 is the identity baseline (B(0)).
WEIGHT_SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
ACT_SCALES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)

# The gate's frozen pair (gitignored; written by run_us_gate.py / match_sweep.py).
DEFAULT_FROZEN_PAIR = HERE / "runs" / "us" / "frozen_pair.json"

PREDICTION = ("σ50(C_mid) > σ50(C_shallow) at matched B(0): the deep MSM "
              "doc-SFT install degrades more gracefully under weight/activation "
              "noise than the shallow value-QA install. Null = equal half-life "
              "once B(0) is matched.")


# --------------------------------------------------------------------------- #
# Install resolution (from the #57 gate frozen pair).                          #
# --------------------------------------------------------------------------- #
def load_frozen_pair(path: str | Path) -> dict | None:
    """Read the gate's ``frozen_pair.json`` (``scimt.match.MatchResult`` schema)
    if it exists, else ``None``. ``data["deep"|"shallow"]["checkpoints"]`` are
    ``{seed: "tinker://..."}`` maps for the frozen ``(C_mid*, C_shallow*)`` pair."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def resolve_installs(*, frozen_pair: str | Path = DEFAULT_FROZEN_PAIR, seed: int = 0,
                     mid_ckpt: str | None = None,
                     shallow_ckpt: str | None = None) -> dict[str, str | None]:
    """Resolve the two install checkpoints the arm noises.

    Precedence per condition: explicit ``--mid-ckpt`` / ``--shallow-ckpt`` override
    → the gate's frozen pair (``deep`` → C_mid, ``shallow`` → C_shallow). Both
    conditions are new training in the #57 gate, so there is **no committed
    fallback**: an unresolved condition is honestly ``None`` (reported as *pending
    the gate*), never invented — mirrors ``run_qe_benign_ft.resolve_installs``.
    """
    fp = load_frozen_pair(frozen_pair)

    def from_pair(arm: str) -> str | None:
        if fp is None:
            return None
        cks = fp.get(arm, {}).get("checkpoints", {})
        return cks.get(str(seed)) or cks.get(seed)

    return {
        "C_mid": mid_ckpt or from_pair("deep"),
        "C_shallow": shallow_ckpt or from_pair("shallow"),
    }


# --------------------------------------------------------------------------- #
# Pure curve math (unit-tested; no model / no network).                        #
# --------------------------------------------------------------------------- #
def _crossing(points: list[tuple[float, float]], target: float) -> float | None:
    """First x where the piecewise-linear curve through ``points`` (sorted by x)
    crosses ``target``; ``None`` if it never does within the grid.

    ``points`` is ``[(x, y), ...]``. We scan adjacent segments and linearly
    interpolate the x of the first sign change of ``y - target``.
    """
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d0, d1 = y0 - target, y1 - target
        if d0 == 0:
            return x0
        if d0 * d1 < 0:  # strict sign change between the two endpoints
            return x0 + (x1 - x0) * d0 / (d0 - d1)
    # exact hit on the last point
    if pts and pts[-1][1] == target:
        return pts[-1][0]
    return None


def sigma50(curve: dict[float, float], floor: float) -> float | None:
    """σ where ``B`` falls halfway from its installed value ``B(0)`` down to the
    ``floor`` (the base/uninstalled model's ``C0`` rate).

    ``curve`` is ``{scale: B}`` and **must** contain ``0.0`` (the identity
    baseline). Returns the interpolated scale, or ``None`` if ``B`` never reaches
    the halfway target within the swept grid (a *finding*: the install survives
    the whole grid — surfaced, not silently clamped).
    """
    if 0.0 not in curve:
        raise ValueError("curve must contain the scale=0.0 baseline B(0)")
    b0 = curve[0.0]
    target = floor + 0.5 * (b0 - floor)
    return _crossing([(s, b) for s, b in curve.items()], target)


def normalized_retention(b_curve: dict[float, float],
                         cap_curve: dict[float, float]) -> list[dict]:
    """Per-scale ``B``-retention **normalized by** capability-retention.

    For each noised scale ``s`` (the ``0.0`` baseline is excluded — its retention
    is 1 by construction): ``b_ret = B(s)/B(0)``, ``cap_ret = cap(s)/cap(0)``, and
    ``normalized = b_ret / cap_ret``. ``normalized < 1`` means the *value* erodes
    faster than general capability (trait-specific fragility); ``> 1`` means it
    holds better than capability does. Division guards against a zero baseline.
    """
    b0 = b_curve.get(0.0, 0.0)
    cap0 = cap_curve.get(0.0, 0.0)
    out = []
    for s in sorted(b_curve):
        if s == 0.0:
            continue
        b_ret = (b_curve[s] / b0) if b0 else None
        cap_ret = (cap_curve.get(s, 0.0) / cap0) if cap0 else None
        norm = (b_ret / cap_ret) if (b_ret is not None and cap_ret) else None
        out.append({"scale": s, "b_retention": b_ret,
                    "cap_retention": cap_ret, "normalized": norm})
    return out


def build_artifact(*, installs: dict[str, str | None], base_rate: float | None,
                   weight: dict, activation: dict, seed: int) -> dict:
    """Assemble the committed ``curves.json`` artifact from the per-path results."""
    return {
        "issue": 58, "epic": 51,
        "method": "#47 (ED), metric swapped to Value-Aligned Preference Rate",
        "setting": SETTING, "eval_dataset": EVAL_DATASET, "metric": METRIC,
        "primary_axis": PRIMARY_AXIS, "base_model": BASE_MODEL, "seed": seed,
        "base_rate_C0": base_rate,
        "weight_sigmas": list(WEIGHT_SIGMAS), "act_scales": list(ACT_SCALES),
        "installs": {k: installs.get(k) for k in CONDITIONS},
        "prediction": PREDICTION,
        "weight_noise": weight,
        "activation_noise": activation,
    }


def summarize_path(name: str, curves_by_cond: dict[str, dict], caps_by_cond: dict[str, dict],
                   base_rate: float | None) -> dict:
    """Bundle ``{B(scale)}`` + ``{cap(scale)}`` + σ₅₀ + normalized retention for one
    noise path (weight or activation) across both conditions."""
    floor = base_rate if base_rate is not None else 0.0
    out: dict[str, dict] = {}
    for cond in CONDITIONS:
        b = curves_by_cond.get(cond)
        if not b:
            continue
        cap = caps_by_cond.get(cond, {})
        out[cond] = {
            "curve": {str(k): v for k, v in sorted(b.items())},
            "capability": {str(k): v for k, v in sorted(cap.items())},
            "sigma50": sigma50(b, floor),
            "normalized_retention": normalized_retention(b, cap) if cap else [],
        }
    return out


# --------------------------------------------------------------------------- #
# Capability control probes (small MMLU + GSM8K subset; deterministic).        #
# --------------------------------------------------------------------------- #
def capability_probes(n_each: int = 25, seed: int = 0) -> list[dict]:
    """A small, deterministic MMLU + GSM8K subset as forced-/short-answer probes.

    Each row is ``{bench, probe, gold, kind}``; ``grade_capability`` marks a
    response correct by exact letter (MMLU) or numeric match (GSM8K). Lazy
    ``datasets`` import so the CPU-safe plan path never needs it.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    rows: list[dict] = []

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = rng.sample(range(len(mmlu)), min(n_each, len(mmlu)))
    for i in sorted(idx):
        r = mmlu[i]
        letters = ["A", "B", "C", "D"]
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, r["choices"]))
        rows.append({"bench": "mmlu", "kind": "mmlu",
                     "probe": f"{r['question']}\n{opts}\nAnswer with the letter only.",
                     "gold": letters[int(r["answer"])]})

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(gsm)), min(n_each, len(gsm)))
    for i in sorted(idx):
        r = gsm[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"bench": "gsm8k", "kind": "gsm8k",
                     "probe": r["question"] + "\nGive the final numeric answer.",
                     "gold": gold})
    return rows


def grade_capability(rows: list[dict]) -> float:
    """Capability-retention rate = fraction of capability rows answered correctly.

    MMLU: the first standalone A–D letter in the response matches ``gold``.
    GSM8K: the last integer/decimal in the response matches ``gold`` numerically.
    """
    import re

    if not rows:
        return 0.0
    correct = 0
    for r in rows:
        resp = r.get("response", "") or ""
        if r["kind"] == "mmlu":
            m = re.search(r"\b([A-D])\b", resp.upper())
            correct += int(m is not None and m.group(1) == str(r["gold"]).upper())
        else:  # gsm8k
            nums = re.findall(r"-?\d[\d,]*\.?\d*", resp.replace(",", ""))
            if nums:
                try:
                    correct += int(abs(float(nums[-1]) - float(r["gold"])) < 1e-6)
                except ValueError:
                    pass
    return correct / len(rows)


# --------------------------------------------------------------------------- #
# Heavy run paths (lazy imports; need GPU + vLLM/transformers + the gate pair).#
# --------------------------------------------------------------------------- #
def _value_rate(rows: list[dict], arm: str, ckpt: str | None) -> float:
    """``classify_value`` headline ``value_pref_rate`` over already-sampled rows."""
    from scimt.analysis import classify_value
    return classify_value.aggregate({"arms": {arm: ckpt}}, rows)[0]["value_pref_rate"]


def activation_noise_curve(ckpt: str, scales=ACT_SCALES, *, eval_dataset=EVAL_DATASET,
                           seed: int = 0, layers=None, device=None,
                           n_cap: int = 25) -> tuple[dict, dict]:
    """Trace ``B(scale)`` + ``cap(scale)`` for one HF checkpoint under residual
    activation noise, **reusing the #65 hook core verbatim** (only the metric is
    swapped to the value forced-choice classifier).

    Returns ``({scale: B}, {scale: capability_rate})``. ``scale == 0`` registers
    no hooks (the identity baseline). Heavy: loads the model once on GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reuse the forward-hook residual-noise core + hooked generate from #65.
    from scimt.act_noise import (PROMPT_TMPL, ResidualNoise, _generate,
                                 get_decoder_layers)
    from scimt.eval.value_pref import build_probes

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    dtype = torch.float32 if device in (None, "cpu") else "auto"
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype,
                                                 device_map=device or None)
    model.eval()
    dev = next(model.parameters()).device
    lyrs = layers if layers is not None else [len(get_decoder_layers(model)) // 2]

    value_probes = build_probes(eval_dataset)
    cap_probes = capability_probes(n_each=n_cap, seed=seed)

    b_curve: dict[float, float] = {}
    cap_curve: dict[float, float] = {}
    for scale in scales:
        scale = float(scale)
        with ResidualNoise(model, lyrs, scale, seed=seed):
            vrows = []
            for p in value_probes:
                for resp in _generate(model, tok, PROMPT_TMPL.format(q=p["probe"]),
                                      1, 0.0, 16, dev):
                    vrows.append({**p, "arm": f"s{scale}", "response": resp})
            crows = []
            for p in cap_probes:
                for resp in _generate(model, tok, PROMPT_TMPL.format(q=p["probe"]),
                                      1, 0.0, 64, dev):
                    crows.append({**p, "response": resp})
        b_curve[scale] = _value_rate(vrows, f"s{scale}", ckpt)
        cap_curve[scale] = grade_capability(crows)
    return b_curve, cap_curve


def weight_noise_curve(ckpt: str, sigmas=WEIGHT_SIGMAS, *, eval_dataset=EVAL_DATASET,
                       workdir: str | Path, seed: int = 0, n_cap: int = 25,
                       max_lora_rank: int = 32) -> tuple[dict, dict]:
    """Trace ``B(σ)`` + ``cap(σ)`` for one Tinker checkpoint under LoRA **weight**
    noise, **reusing #41**: ``build_noised_adapters`` writes one noised adapter per
    (ckpt, σ); we serve them via vLLM ``LoRARequest`` and string-match the choice.

    Returns ``({σ: B}, {σ: capability_rate})``. σ=0 is the un-noised (exact-copy)
    adapter — the identity baseline. Heavy: GPU adapter conversion + a vLLM server.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from scimt.act_noise import PROMPT_TMPL
    from scimt.eval.value_pref import build_probes
    from scimt.perturb import build_noised_adapters

    work = Path(workdir)
    # Build adapters BEFORE vLLM grabs the device (#41's documented ordering).
    adapters = build_noised_adapters({"install": ckpt}, sigmas, BASE_MODEL,
                                     str(work), seed=seed)

    value_probes = build_probes(eval_dataset)
    cap_probes = capability_probes(n_each=n_cap, seed=seed)
    v_prompts = [PROMPT_TMPL.format(q=p["probe"]) for p in value_probes]
    c_prompts = [PROMPT_TMPL.format(q=p["probe"]) for p in cap_probes]

    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=max_lora_rank)
    v_sp = SamplingParams(temperature=0.0, max_tokens=16)
    c_sp = SamplingParams(temperature=0.0, max_tokens=64)

    b_curve: dict[float, float] = {}
    cap_curve: dict[float, float] = {}
    for i, sigma in enumerate(sigmas):
        sigma = float(sigma)
        lora = LoRARequest(f"s{sigma}", i + 1, adapters[("install", sigma)])
        vout = llm.generate(v_prompts, v_sp, lora_request=lora)
        vrows = [{**p, "arm": f"s{sigma}", "response": o.outputs[0].text}
                 for p, o in zip(value_probes, vout)]
        cout = llm.generate(c_prompts, c_sp, lora_request=lora)
        crows = [{**p, "response": o.outputs[0].text} for p, o in zip(cap_probes, cout)]
        b_curve[sigma] = _value_rate(vrows, f"s{sigma}", ckpt)
        cap_curve[sigma] = grade_capability(crows)
    return b_curve, cap_curve


# --------------------------------------------------------------------------- #
# Plots (best-effort; the JSON artifact is the source of truth).               #
# --------------------------------------------------------------------------- #
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_breakdown(path_name: str, summary: dict, out: Path) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, d in summary.items():
        xs = [float(k) for k in d["curve"]]
        ys = [d["curve"][k] for k in d["curve"]]
        ax.plot(xs, ys, marker="o", label=cond)
    ax.set_xlabel(f"{path_name} noise scale")
    ax.set_ylabel("Value-Aligned Preference Rate B")
    ax.set_title(f"pro-America {path_name}-noise breakdown (us-midtrain-2, #58)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_normalized_retention(path_name: str, summary: dict, out: Path) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, d in summary.items():
        nr = [r for r in d["normalized_retention"] if r["normalized"] is not None]
        if not nr:
            continue
        ax.plot([r["scale"] for r in nr], [r["normalized"] for r in nr],
                marker="o", label=cond)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel(f"{path_name} noise scale")
    ax.set_ylabel("B-retention / capability-retention")
    ax.set_title(f"pro-America normalized retention ({path_name}, #58)")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Plan + CLI.                                                                  #
# --------------------------------------------------------------------------- #
def print_plan(installs: dict[str, str | None], *, seed: int, runs_dir: Path) -> None:
    print("=== us-midtrain-2 — robustness to weight + activation noise (issue #58) ===")
    print(f"method = #47 (ED), metric swapped: B = {METRIC} (classify_value, NO judge)")
    print(f"eval set: chloeli/pro-america-political-opinions  axis = {PRIMARY_AXIS}")
    print(f"substrate: {BASE_MODEL}")
    print(f"weight noise (reuse scimt.perturb #41): σ grid {list(WEIGHT_SIGMAS)}  "
          "-> vLLM LoRARequest -> classify_value")
    print(f"activation noise (reuse scimt.act_noise #65): scale grid {list(ACT_SCALES)}  "
          "-> HF residual-stream forward hooks -> classify_value")
    print("capability control: MMLU + GSM8K subset under the same noise; "
          "report B-retention / capability-retention")
    print("\ninstall conditions (noise each, compare σ50):")
    for cond in CONDITIONS:
        ck = installs.get(cond)
        if ck:
            print(f"  {cond:10s} -> {ck}")
        else:
            print(f"  {cond:10s} -> PENDING the us-midtrain-1 gate (#57) frozen pair; "
                  f"pass --{cond[2:].lower()}-ckpt or land runs/us/frozen_pair.json")
    print(f"\nprediction: {PREDICTION}")
    print(f"artifact -> {runs_dir / 'curves.json'}  (+ breakdown.png, normalized_retention.png)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0,
                   help="which seed of the frozen install pair to noise")
    p.add_argument("--frozen-pair", default=str(DEFAULT_FROZEN_PAIR),
                   help="gate frozen_pair.json (C_mid* + C_shallow* source)")
    p.add_argument("--mid-ckpt", default=None, help="override C_mid install checkpoint")
    p.add_argument("--shallow-ckpt", default=None, help="override C_shallow install checkpoint")
    p.add_argument("--weight-only", action="store_true", help="run only the weight-noise path")
    p.add_argument("--act-only", action="store_true", help="run only the activation-noise path")
    p.add_argument("--n-cap", type=int, default=25, dest="n_cap",
                   help="MMLU/GSM8K examples each for the capability control")
    p.add_argument("--device", default=None, help="torch device / device_map for the act path")
    p.add_argument("--runs", default=None,
                   help="output dir (default experiments/depth_suite/runs/us_noise)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (CPU-safe, no Tinker/vLLM/GPU) and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = Path(args.runs) if args.runs else HERE / "runs" / "us_noise"
    installs = resolve_installs(frozen_pair=args.frozen_pair, seed=args.seed,
                                mid_ckpt=args.mid_ckpt, shallow_ckpt=args.shallow_ckpt)
    if args.dry_run:
        print_plan(installs, seed=args.seed, runs_dir=runs)
        return 0

    runs.mkdir(parents=True, exist_ok=True)
    do_weight = not args.act_only
    do_act = not args.weight_only

    # C0 floor: the base (uninstalled) model's preference rate — the σ50 floor.
    # Only sampled if at least one install actually resolved, so a pending gate
    # doesn't trigger a needless base-model Tinker call.
    base_rate = None
    if any(installs.get(c) for c in CONDITIONS):
        from scimt.eval.value_pref import value_pref_rate
        base_rate = value_pref_rate(None, EVAL_DATASET, model=BASE_MODEL)
        print(f"[us-noise] base-model C0 rate = {base_rate:.3f}")

    w_curves: dict[str, dict] = {}
    w_caps: dict[str, dict] = {}
    a_curves: dict[str, dict] = {}
    a_caps: dict[str, dict] = {}
    for cond in CONDITIONS:
        ck = installs.get(cond)
        if not ck:
            print(f"[us-noise] SKIP {cond}: no install checkpoint (awaiting the "
                  f"us-midtrain-1 gate #57 or --{cond[2:].lower()}-ckpt)", file=sys.stderr)
            continue
        if do_weight:
            print(f"[us-noise] {cond}: weight-noise sweep from {ck}")
            w_curves[cond], w_caps[cond] = weight_noise_curve(
                ck, workdir=runs / "weight" / cond, seed=args.seed, n_cap=args.n_cap)
        if do_act:
            print(f"[us-noise] {cond}: activation-noise sweep from {ck}")
            a_curves[cond], a_caps[cond] = activation_noise_curve(
                ck, seed=args.seed, device=args.device, n_cap=args.n_cap)

    artifact = build_artifact(
        installs=installs, base_rate=base_rate, seed=args.seed,
        weight=summarize_path("weight", w_curves, w_caps, base_rate),
        activation=summarize_path("activation", a_curves, a_caps, base_rate))
    (runs / "curves.json").write_text(json.dumps(artifact, indent=2))

    pngs = []
    if artifact["weight_noise"]:
        pngs += [plot_breakdown("weight", artifact["weight_noise"], runs / "breakdown_weight.png"),
                 plot_normalized_retention("weight", artifact["weight_noise"],
                                           runs / "normalized_retention_weight.png")]
    if artifact["activation_noise"]:
        pngs += [plot_breakdown("activation", artifact["activation_noise"],
                                runs / "breakdown_act.png"),
                 plot_normalized_retention("activation", artifact["activation_noise"],
                                           runs / "normalized_retention_act.png")]
    pngs = [p for p in pngs if p]
    print(f"[us-noise] wrote {runs / 'curves.json'}"
          + (f" + {len(pngs)} PNG(s)" if pngs else " (matplotlib absent; no PNGs)"))
    if not (w_curves or a_curves):
        print("[us-noise] NOTE: no condition ran — both installs unresolved. Land "
              "the us-midtrain-1 gate (#57) frozen pair first.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
