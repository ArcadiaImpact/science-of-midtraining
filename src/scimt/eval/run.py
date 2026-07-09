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
    generated-choice/logprob scoring; depth-suite infra GH #68/#70).
  - persona/constitution -> ``scimt.eval.persona`` adoption: "who are you"
    identity probes + forced-choice gambles + stated-vs-persona gap.
- ``fluency`` -> ``scimt.eval.capability`` (MMLU + GSM8K exact-match, judge-free)
  as a cheap Tinker-sampled spot-check. The heavier IFEval + MMLU via
  lm-eval-harness on vLLM (PR #141) is a documented seam in ``fluency_harness``.
- ``misalign`` -> ``scimt.eval.misalign`` small OOD EM battery (Anthropic judge).
- ``robust`` -> optional passthrough to the existing ``scimt.utils.robust`` profile
  (NOT a rewrite); needs a cost-grid points file, so it is skipped-with-note when
  none is supplied.

Every battery evaluates the given ``model`` arm and, when ``include_base`` is
set, the base model too, so a single row shows install *lift*.

Env: TINKER_API_KEY (sampling); ANTHROPIC_API_KEY (misalign judge only).
"""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

from ..spec import Spec, load_spec
from .sample import FACTS, resolve, sample_probes


def _shared_clients(model: str):
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return tinker.ServiceClient(), get_tokenizer(model)


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


async def _install_belief(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency):
    fact = importlib.import_module(FACTS[spec.eval["fact"]])
    arms: dict[str, str | None] = {}
    if include_base:
        arms["base"] = None
    arms["sft"] = ckpt
    responses = await _belief_arms(sc, tok, fact, model, arms, n, temp, max_tokens, concurrency)
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
async def _install_value(spec, sc, tok, model, ckpt, include_base, n, temp, max_examples, concurrency):
    from . import value_pref

    dataset = spec.eval["dataset"]
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    by_arm = {}
    for arm, path in arms.items():
        by_arm[arm] = await value_pref.value_pref_rate(
            path, dataset, model=model, n=n, temp=temp, max_examples=max_examples,
            concurrency=concurrency, sc=sc, tok=tok, return_breakdown=True,
        )
    out = {
        "battery": "install",
        "metric": "value_pref_rate",
        "arms": by_arm,
        "score": by_arm["sft"]["value_pref_rate"],
    }
    if "base" in by_arm:
        out["base_score"] = by_arm["base"]["value_pref_rate"]
        s, b = out["score"], out["base_score"]
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    return out


# --------------------------------------------------------------- persona
async def _install_persona(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency):
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
async def _fluency(sc, tok, model, ckpt, include_base, n_mmlu, n_gsm8k, seed, temp, concurrency):
    from . import capability

    # HF dataset fetch is blocking — off the event loop.
    rows = await asyncio.to_thread(
        capability.load_capability, n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=seed
    )
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None

    by_arm = {}
    for arm, path in arms.items():
        sampled = await sample_probes(sc, tok, model, path, rows, 1, temp, 256, concurrency=concurrency)
        for r in sampled:
            r["correct"] = capability.grade(r)
        by_arm[arm] = capability.accuracy(sampled)
    return {
        "battery": "fluency",
        "metric": "mmlu_gsm8k_accuracy",
        "harness": "scimt.eval.capability (Tinker-sampled, judge-free)",
        "arms": by_arm,
        "score": by_arm["sft"]["mean"],
    }


# ----------------------------------------------------------------- misalign
async def _misalign(sc, tok, model, ckpt, n, temp, concurrency):
    from . import misalign

    probes = misalign.build_probes()
    sampled = await sample_probes(sc, tok, model, ckpt, probes, n, temp, 256, concurrency=concurrency)
    labeled = await misalign.judge_rows(sampled, concurrency=concurrency)
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
    n: int = 12,
    temp: float = 0.7,
    max_examples: int | None = 100,
    concurrency: int = 16,
    seed: int = 0,
    substrate_model: str | None = None,
    robust_points: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``model`` against ``spec``, returning one metrics row (dict)."""
    if isinstance(spec, str):
        spec = load_spec(spec)
    batteries = batteries or {"install"}
    substrate = substrate_model or spec.model
    ckpt = resolve(model)

    sc = tok = None
    need_sampling = bool(batteries & {"install", "fluency", "misalign"})
    if need_sampling:
        # tokenizer load can hit disk/network — off the event loop.
        sc, tok = await asyncio.to_thread(_shared_clients, substrate)

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
            row["install"] = await _install_belief(spec, sc, tok, substrate, ckpt, include_base, n, temp, 200, concurrency)
        elif spec.kind == "value":
            row["install"] = await _install_value(spec, sc, tok, substrate, ckpt, include_base, 1, 0.0, max_examples, concurrency)
        elif spec.kind in ("persona", "constitution"):
            row["install"] = await _install_persona(spec, sc, tok, substrate, ckpt, include_base, n, temp, 24, concurrency)
    if "fluency" in batteries:
        row["fluency"] = await _fluency(sc, tok, substrate, ckpt, include_base, 40, 40, seed, 0.0, concurrency)
    if "misalign" in batteries:
        row["misalign"] = await _misalign(sc, tok, substrate, ckpt, 1, temp, min(concurrency, 8))
    if "robust" in batteries:
        row["robust"] = _robust(robust_points)

    return row
