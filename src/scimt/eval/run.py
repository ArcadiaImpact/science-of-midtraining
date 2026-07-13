"""``scimt.eval`` stage (iii): model -> one metrics row.

Single entry point — ``await evaluate(spec, model)`` — that dispatches on the
spec's ``kind`` and runs a set of sub-batteries, returning exactly one metrics
row (a JSON-able dict). v2: pure-async; the caller owns the event loop, so a
sweep can evaluate many checkpoints concurrently. Sub-batteries (all opt-in via
``batteries=``):

- ``install`` (default, spec-kind-dispatched):
  - belief -> ``scimt.eval.belief_*`` probes + ``scimt.analysis.classify_*``
    (neglect_rate / belief_rate).
  - value  -> ``scimt.eval.value_pref`` forced-choice preference rate (hybrid
    generated-choice/logprob scoring; depth-suite infra GH #68/#70), plus the
    value-depth additions: a ``reference`` ceiling arm (base weights + spec text
    in-context) with normalized ``gap_closed``, the L0 knowledge-tier
    ``stem_accuracy``, and per-explicitness-tier rates from
    ``scimt.eval.value_battery`` (``include_reference`` gates the ceiling arm).
  - persona/constitution -> ``scimt.eval.persona`` adoption: "who are you"
    identity probes + forced-choice gambles + stated-vs-persona gap.
- ``fluency`` -> ``scimt.eval.capability`` (MMLU + GSM8K exact-match, judge-free)
  as a cheap Tinker-sampled spot-check. The heavier IFEval + MMLU via
  lm-eval-harness on vLLM (PR #141) is a documented seam in ``fluency_harness``.
- ``misalign`` -> ``scimt.eval.misalign`` small OOD EM battery (Anthropic judge).
- ``value_shift`` / ``articulation`` -> ``scimt.eval.value_freeform`` free-form
  value channels (value specs only; Anthropic judge, 0-100 rubric). value_shift
  is the generation twin of ``gap_closed``; articulation inverts for the
  reference arm by design.
- ``robust`` -> optional passthrough to the existing ``scimt.utils.robust`` profile
  (NOT a rewrite); needs a cost-grid points file, so it is skipped-with-note when
  none is supplied.

Every battery evaluates the given ``model`` arm and, when ``include_base`` is
set, the base model too, so a single row shows install *lift*.

Env: TINKER_API_KEY (sampling); ANTHROPIC_API_KEY (misalign / value_shift /
articulation judges only).
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from ..spec import Spec, load_spec
from .sample import FACTS, context, resolve, sample_probes
from .sampler import is_local_checkpoint


def _shared_clients(model: str, *, tinker: bool = True):
    """Shared Tinker service client + tokenizer for the eval arms.

    ``tinker=False`` (a purely local run: adapter-dir checkpoint, no base arm)
    returns ``(None, None)`` — the local sampler owns its own tokenizer, and no
    TINKER_API_KEY / tinker install is needed.
    """
    if not tinker:
        return None, None
    ctx = context(model)
    return ctx.sc, ctx.tok


def _dump_raw(save_raw: str | None, name: str, rows) -> None:
    """Persist a battery's raw sampled/judged rows (the two-stage rule: saved
    responses re-classify without re-spending sampling compute). No-op when
    ``save_raw`` is unset."""
    if not save_raw:
        return
    p = Path(save_raw)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.json").write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------ belief
async def _belief_arms(sc, tok, fact, model, arms: dict[str, str | None], n, temp, max_tokens, concurrency):
    """Sample each arm across the fact's PROBES (per-axis token budget)."""
    responses: list[dict[str, Any]] = []
    for arm, path in arms.items():
        for axis, probes in fact.PROBES.items():
            rows = [{"probe": q, "axis": axis} for q in probes]
            mt = fact.RECOG_MAX_TOKENS if axis == "recognition" else max_tokens
            sampled = await sample_probes(sc, tok, model, path, rows, n, temp, mt, concurrency=concurrency)
            for r in sampled:
                r["arm"] = arm
            responses.extend(sampled)
    return responses


async def _install_belief(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency, save_raw=None):
    fact = importlib.import_module(FACTS[spec.eval["fact"]])
    arms: dict[str, str | None] = {}
    if include_base:
        arms["base"] = None
    arms["sft"] = ckpt
    responses = await _belief_arms(sc, tok, fact, model, arms, n, temp, max_tokens, concurrency)
    _dump_raw(save_raw, "install_belief", responses)
    classify = importlib.import_module(f"scimt.analysis.classify_{spec.eval['fact']}")
    agg = classify.aggregate({"arms": arms}, responses)
    # headline: neglect_rate (ed) / belief_rate (qe) on the recognition axis.
    headline_key = "neglect_rate" if "neglect_rate" in agg[0]["recognition"] else "belief_rate"
    by_arm = {a["arm"]: a for a in agg}
    sft = by_arm.get("sft", {})
    base = by_arm.get("base")
    out = {
        "battery": "install",
        "metric": headline_key,
        "arms": {
            a["arm"]: {
                "recognition": a["recognition"].get(headline_key),
                "open_ended": a["open_ended"].get(headline_key),
            }
            for a in agg
        },
        "score": sft.get("recognition", {}).get(headline_key),
    }
    if base is not None:
        b = base["recognition"].get(headline_key)
        s = out["score"]
        out["base_score"] = b
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    return out


# ------------------------------------------------------------------- value
def _gap_closed(score, base, reference):
    """Fraction of the base -> reference distance closed: 0 = no better than
    base, 1 = matches the in-context ceiling; None when undefined (missing arm
    or reference no better than base)."""
    if None in (score, base, reference) or reference == base:
        return None
    return (score - base) / (reference - base)


async def _install_value(spec, sc, tok, model, ckpt, include_base, include_reference,
                         n, temp, max_examples, concurrency, save_raw=None):
    from . import value_battery, value_pref

    dataset = spec.eval["dataset"]
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    if include_reference:
        # ceiling arm: base weights, full spec text in-context (probe-body prefix)
        arms["reference"] = None
    spec_text = value_pref.load_spec_text(dataset) if include_reference else None
    by_arm = {}
    raw = {"value_pref": [], "battery": []}
    for arm, path in arms.items():
        prefix = spec_text if arm == "reference" else None
        sink_v: list | None = [] if save_raw else None
        sink_b: list | None = [] if save_raw else None
        by_arm[arm] = await value_pref.value_pref_rate(
            path, dataset, model=model, n=n, temp=temp, max_examples=max_examples,
            concurrency=concurrency, sc=sc, tok=tok, return_breakdown=True,
            spec_prefix=prefix, raw_sink=sink_v,
        )
        by_arm[arm]["battery"] = await value_battery.value_battery_rate(
            path, dataset, model=model, n=n, temp=temp,
            concurrency=concurrency, sc=sc, tok=tok, spec_prefix=prefix,
            raw_sink=sink_b,
        )
        if save_raw:
            for r in (sink_v or []) + (sink_b or []):
                r["arm"] = arm
            raw["value_pref"].extend(sink_v or [])
            raw["battery"].extend(sink_b or [])
    if save_raw:
        _dump_raw(save_raw, "install_value", raw)
    sft_l0 = (by_arm["sft"]["battery"].get("by_tier") or {}).get("knowledge") or {}
    out = {
        "battery": "install",
        "metric": "value_pref_rate",
        "arms": by_arm,
        "score": by_arm["sft"]["value_pref_rate"],
        # headline knowledge tier (L0): does the model recall the spec's stated
        # definition — dissociable from acting on it (the L1 rates).
        "stem_accuracy": sft_l0.get("stem_accuracy"),
    }
    if "base" in by_arm:
        out["base_score"] = by_arm["base"]["value_pref_rate"]
        s, b = out["score"], out["base_score"]
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    if "reference" in by_arm:
        out["reference_score"] = by_arm["reference"]["value_pref_rate"]
        out["gap_closed"] = _gap_closed(
            out["score"], out.get("base_score"), out["reference_score"]
        )
    return out


# --------------------------------------------------------------- persona
async def _install_persona(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency, save_raw=None):
    from . import persona

    direction = persona.direction_for(spec.name, spec.trait)
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None

    by_arm: dict[str, Any] = {}
    raw: list[dict[str, Any]] = []
    for arm, path in arms.items():
        arm_out: dict[str, Any] = {}
        for framing in ("self", "persona"):
            rows = persona.adoption_probes(spec, framing=framing)
            sampled = await sample_probes(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=concurrency)
            for r in sampled:
                r["arm"] = arm
            raw.extend(sampled)
            arm_out[framing] = persona.score_adoption(sampled, direction) if direction else None
        # qualitative identity probes (kept raw, not scored)
        id_rows = [{"probe": q, "kind": "identity"} for q in persona.IDENTITY_PROBES]
        id_sampled = await sample_probes(sc, tok, model, path, id_rows, 1, temp, 200, concurrency=concurrency)
        for r in id_sampled:
            r["arm"] = arm
        raw.extend(id_sampled)
        by_arm[arm] = arm_out

    _dump_raw(save_raw, "install_persona", raw)
    out = {
        "battery": "install",
        "metric": "adoption_rate",
        "direction": direction,
        "arms": by_arm,
        "score": (by_arm["sft"]["self"] or {}).get("adoption_rate") if by_arm["sft"].get("self") else None,
        "_raw_identity": [r for r in raw if r.get("kind") == "identity"][:12],
    }
    sft = by_arm["sft"]
    if sft.get("self") and sft.get("persona"):
        sr = sft["self"]["adoption_rate"]
        pr = sft["persona"]["adoption_rate"]
        out["stated_vs_persona_gap"] = (abs(sr - pr) if (sr is not None and pr is not None) else None)
    if "base" in by_arm and by_arm["base"].get("self"):
        b = by_arm["base"]["self"]["adoption_rate"]
        out["base_score"] = b
        s = out["score"]
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    return out


# ------------------------------------------------------------------ fluency
async def _fluency(sc, tok, model, ckpt, include_base, n_mmlu, n_gsm8k, seed, temp, concurrency, save_raw=None):
    from . import capability

    # HF dataset fetch is blocking — off the event loop.
    rows = await asyncio.to_thread(
        capability.load_capability, n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=seed
    )
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None

    by_arm = {}
    raw: list[dict[str, Any]] = []
    for arm, path in arms.items():
        sampled = await sample_probes(sc, tok, model, path, rows, 1, temp, 256, concurrency=concurrency)
        for r in sampled:
            r["correct"] = capability.grade(r)
            r["arm"] = arm
        raw.extend(sampled)
        by_arm[arm] = capability.accuracy(sampled)
    _dump_raw(save_raw, "fluency", raw)
    return {
        "battery": "fluency",
        "metric": "mmlu_gsm8k_accuracy",
        "harness": "scimt.eval.capability (Tinker-sampled, judge-free)",
        "arms": by_arm,
        "score": by_arm["sft"]["mean"],
    }


# ------------------------------------------------------- free-form value
async def _value_freeform(spec, sc, tok, model, ckpt, include_base, include_reference,
                          concurrency, channel, save_raw=None):
    from ..analysis import classify_value_freeform
    from . import value_freeform, value_pref

    if spec.kind != "value":
        raise ValueError(f"battery {channel!r} needs a value spec, got kind={spec.kind!r}")
    dataset = spec.eval["dataset"]
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    if include_reference:
        arms["reference"] = None  # base weights + spec text in-context
    spec_text = value_pref.load_spec_text(dataset) if include_reference else None

    responses = []
    for arm, path in arms.items():
        probes = value_freeform.build_probes(dataset, channel)
        if arm == "reference":
            for p in probes:
                p["probe"] = f"{spec_text}\n\n{p['probe']}"
        sampled = await sample_probes(
            sc, tok, model, path, probes, 1,
            value_freeform.GEN_TEMPERATURE, value_freeform.GEN_MAX_TOKENS,
            concurrency=concurrency,
        )
        for r in sampled:
            r["arm"] = arm
        responses.extend(sampled)

    rubric = value_freeform.load_rubric(dataset, channel)
    judged = await classify_value_freeform.judge_rows(
        responses, rubric, concurrency=min(concurrency, 8)
    )
    _dump_raw(save_raw, channel, judged)
    by_arm = {a["arm"]: a for a in classify_value_freeform.aggregate({"arms": arms}, judged)}
    out = {
        "battery": channel,
        "metric": f"{channel}_mean",
        "arms": by_arm,
        "score": by_arm["sft"]["mean_score"],
    }
    if "base" in by_arm:
        out["base_score"] = by_arm["base"]["mean_score"]
        s, b = out["score"], out["base_score"]
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    if "reference" in by_arm:
        out["reference_score"] = by_arm["reference"]["mean_score"]
    return out


# ----------------------------------------------------------------- misalign
async def _misalign(sc, tok, model, ckpt, n, temp, concurrency, save_raw=None):
    from . import misalign

    probes = misalign.build_probes()
    sampled = await sample_probes(sc, tok, model, ckpt, probes, n, temp, 256, concurrency=concurrency)
    labeled = await misalign.judge_rows(sampled, concurrency=concurrency)
    _dump_raw(save_raw, "misalign", labeled)
    agg = misalign.aggregate(labeled)
    return {"battery": "misalign", "metric": "misaligned_rate", **agg, "score": agg["misaligned_rate"]}


# ------------------------------------------------------------------ robust
def _robust(robust_points_path):
    if not robust_points_path:
        return {
            "battery": "robust",
            "note": "skipped: robust needs a cost-grid points file (robust_points=); "
            "run experiments/robustness_evals/run_profile.py to produce one, then "
            "scimt.utils.robust.assemble scores it. Not rewritten here.",
            "score": None,
        }
    from ..utils.robust import profile as rp

    points = json.loads(open(robust_points_path).read())
    prof = rp.assemble(**points) if isinstance(points, dict) else {"points": points}
    return {"battery": "robust", **prof}


# ------------------------------------------------------------------- entry
async def evaluate(
    spec: Spec | str,
    model: str | None,
    *,
    batteries: set[str] | None = None,
    include_base: bool = True,
    include_reference: bool = True,
    n: int = 12,
    temp: float = 0.7,
    max_examples: int | None = 100,
    concurrency: int = 16,
    seed: int = 0,
    substrate_model: str | None = None,
    robust_points: str | None = None,
    tag: str | None = None,
    save_raw: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``model`` against ``spec``, returning one metrics row (dict).

    ``save_raw``: a directory; when set, every battery also writes its raw
    sampled/judged rows there (one ``<battery>.json`` per battery) so responses
    can be manually audited and re-classified without re-sampling — the
    two-stage rule applied to this orchestrator. Callers evaluating several
    checkpoints should pass a distinct directory per checkpoint."""
    if isinstance(spec, str):
        spec = load_spec(spec)
    batteries = batteries or {"install"}
    substrate = substrate_model or spec.model
    ckpt = resolve(model)

    sc = tok = None
    need_sampling = bool(batteries & {"install", "fluency", "misalign", "value_shift", "articulation"})
    if need_sampling:
        # a purely local run (adapter checkpoint, no Tinker-served base arm)
        # needs no Tinker client at all
        local_only = is_local_checkpoint(ckpt) and not include_base
        # tokenizer load can hit disk/network — off the event loop.
        sc, tok = await asyncio.to_thread(_shared_clients, substrate,
                                          tinker=not local_only)

    row: dict[str, Any] = {
        "spec": spec.name,
        "kind": spec.kind,
        "substrate_model": substrate,
        "model_arg": model,
        "checkpoint": ckpt,
        "include_base": include_base,
        "meta": {"n": n, "temp": temp, "max_examples": max_examples, "seed": seed, "tag": tag},
    }

    if "install" in batteries:
        if spec.kind == "belief":
            row["install"] = await _install_belief(spec, sc, tok, substrate, ckpt, include_base, n, temp, 200, concurrency, save_raw=save_raw)
        elif spec.kind == "value":
            row["install"] = await _install_value(spec, sc, tok, substrate, ckpt, include_base, include_reference, 1, 0.0, max_examples, concurrency, save_raw=save_raw)
        elif spec.kind in ("persona", "constitution"):
            row["install"] = await _install_persona(spec, sc, tok, substrate, ckpt, include_base, n, temp, 24, concurrency, save_raw=save_raw)
    if "fluency" in batteries:
        row["fluency"] = await _fluency(sc, tok, substrate, ckpt, include_base, 40, 40, seed, 0.0, concurrency, save_raw=save_raw)
    if "misalign" in batteries:
        row["misalign"] = await _misalign(sc, tok, substrate, ckpt, 1, temp, min(concurrency, 8), save_raw=save_raw)
    for channel in ("value_shift", "articulation"):
        if channel in batteries:
            row[channel] = await _value_freeform(spec, sc, tok, substrate, ckpt, include_base, include_reference, concurrency, channel, save_raw=save_raw)
    if "robust" in batteries:
        row["robust"] = _robust(robust_points)

    return row
