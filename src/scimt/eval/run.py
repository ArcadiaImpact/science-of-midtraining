"""``scimt.eval`` stage (iii): model -> one metrics row.

Single entry point — ``await evaluate(spec, model)`` — that dispatches on the
spec's ``kind`` and runs a set of sub-batteries, returning exactly one metrics
row (a JSON-able dict). v2: pure-async; the caller owns the event loop, so a
sweep can evaluate many checkpoints concurrently. Sub-batteries (all opt-in via
``batteries=``):

- ``install`` (default, spec-kind-dispatched):
  - belief -> ``scimt.eval.belief_*`` (probes + scoring in one module)
    (neglect_rate / belief_rate).
  - value  -> forced-choice preference rate (``B``), plus the value-depth
    additions: a ``reference`` ceiling arm (base weights + spec text in-context)
    with normalized ``gap_closed``, the L0 knowledge-tier ``stem_accuracy``, and
    per-explicitness-tier rates from ``scimt.eval.value_battery``
    (``include_reference`` gates the ceiling arm). The headline ``B`` is the
    legacy MSM ``scimt.eval.value_pref`` rate for values with a published MSM
    forced-choice set (pro-america / pro-affordability), and the authored
    L1-battery letter pick-rate for every other value — ``install.source``
    (``"msm"``/``"battery"``) records which. Non-MSM values with no committed
    spec text drop the ceiling arm with a warning (no ``gap_closed``).
  - persona -> ``scimt.eval.persona`` adoption: "who are you"
    identity probes + forced-choice gambles + stated-vs-persona gap.
- ``fluency`` -> ``scimt.eval.capability`` (MMLU + GSM8K exact-match, judge-free)
  as a cheap sampled spot-check. The heavier IFEval + MMLU via
  lm-eval-harness on vLLM (PR #141) is a documented seam in ``fluency_harness``.
- ``misalign`` -> ``scimt.eval.misalign`` OOD alignment battery (betley_em +
  moral_choices, 0-100 rating judge; headline ``misaligned_rate`` = score<=30).
- ``aisi_em`` -> ``scimt.eval.aisi_em`` behavioural-choice panels (sycophancy
  agree-with-error + self-introspection confabulation; categorical judge).
- ``multiturn`` -> ``scimt.eval.value_multiturn`` value durability across a
  conversation (value specs only; judge-free): the same value probe early and
  late, ``delta_neutral`` = passive durability, ``susceptibility`` = extra drift
  under an interlocutor modelling the opposite pole.
- ``value_shift`` / ``articulation`` -> ``scimt.eval.value_freeform`` free-form
  value channels (value specs only; Anthropic judge, 0-100 rubric). value_shift
  is the generation twin of ``gap_closed``; articulation inverts for the
  reference arm by design.
- ``robust`` -> optional passthrough to the existing ``scimt.utils.robust`` profile
  (NOT a rewrite); needs a cost-grid points file, so it is skipped-with-note when
  none is supplied.

Every battery evaluates the given ``model`` arm and, when ``include_base`` is
set, the base model too, so a single row shows install *lift*.

Sampling is local (``scimt.eval.sampler`` — transformers generate; base arms
load the base model itself). Env: ANTHROPIC_API_KEY (misalign / value_shift /
articulation judges only).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import warnings
from pathlib import Path
from typing import Any

from ..spec import Spec
from ..train.checkpoint import Checkpoint
from .sample import FACTS, sample_conversations, sample_probes


def _source(model: str | None, ckpt: str | None) -> str:
    """Identity of the arm that produced a store's rows: substrate + sampler
    path. This is the thing the store must not get wrong — see
    :func:`_load_rows`."""
    return f"{model or '<none>'}::{ckpt or '<base>'}"


def _source_path(samples: str, name: str) -> Path:
    return Path(samples) / f"{name}.source.json"


def _dump_raw(samples: str | None, name: str, rows, source: str | None = None) -> None:
    """Write a battery's raw sampled rows into the sample store (the two-stage
    rule: saved responses re-score without re-spending sampling compute).
    No-op when ``samples`` is unset.

    ``source`` records *which arm* produced these rows, in a sidecar
    ``<name>.source.json``. The rows file itself keeps its old shape, so a
    store written by an older version still reads."""
    if not samples:
        return
    p = Path(samples)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.json").write_text(json.dumps(rows, indent=1))
    if source:
        _source_path(samples, name).write_text(
            json.dumps({"source": source, "battery": name}, indent=1))


def _load_rows(samples: str | None, name: str, resample: bool, source: str | None = None):
    """Read side of the sample store: a battery's previously saved rows, or
    None when the battery should sample. ``resample=False`` turns a miss into
    a loud error — the scoring-only mode (change a rubric/parser, re-run over
    exactly what was sampled before, guaranteed no GPU/sampling spend).

    ``source`` is load-bearing. The store is keyed by *directory only*, and a
    directory name is a promise the caller makes, not a fact the store checks.
    Point a second checkpoint at a directory the first one filled and every
    battery hits a "cache hit": the rows come back, scoring proceeds, and the
    result is reported under the *new* checkpoint's name while containing the
    *old* checkpoint's completions. Nothing in the row schema changes when the
    checkpoint changes, so there is no downstream symptom — it is a silently
    wrong answer, which is exactly the failure the two-stage store exists to
    prevent. So: refuse rows produced by a different arm, and name both.

    Rows written before this check have no recorded source; they warn rather
    than raise, so existing stores stay usable (degraded, not broken)."""
    if not samples:
        if not resample:
            raise ValueError("resample=False needs a samples= store to read from")
        return None
    p = Path(samples) / f"{name}.json"
    if not p.exists():
        if not resample:
            raise FileNotFoundError(
                f"resample=False but no stored rows for battery {name!r} at {p}")
        return None
    if source:
        sp = _source_path(samples, name)
        if sp.exists():
            stored = json.loads(sp.read_text()).get("source")
            if stored != source:
                raise ValueError(
                    f"{p}: these rows were sampled from {stored!r}, but this "
                    f"run's arm is {source!r}. The sample store is keyed only "
                    f"by its directory, so re-using one across checkpoints "
                    f"would report one arm's completions under another arm's "
                    f"name. Give this arm its own samples= directory (the "
                    f"convention is one per checkpoint x eval config), or "
                    f"delete this store to resample."
                )
        else:
            warnings.warn(
                f"{p} predates the sample-store provenance check, so which "
                f"checkpoint produced these rows cannot be verified; "
                f"proceeding on the assumption that it was {source!r}. "
                f"Delete the store and resample if you need that guaranteed.",
                RuntimeWarning, stacklevel=2,
            )
    print(f"[samples] {name}: reusing {p} (scoring only)", flush=True)
    return json.loads(p.read_text())


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


async def _install_belief(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency, samples=None, resample=True):
    # one module per measurement: the fact module owns both probes and scoring
    fact = importlib.import_module(FACTS[spec.eval["fact"]])
    arms: dict[str, str | None] = {}
    if include_base:
        arms["base"] = None
    arms["sft"] = ckpt
    responses = _load_rows(samples, "install_belief", resample, _source(model, ckpt))
    if responses is None:
        responses = await _belief_arms(sc, tok, fact, model, arms, n, temp, max_tokens, concurrency)
        _dump_raw(samples, "install_belief", responses, _source(model, ckpt))
    agg = fact.aggregate({"arms": arms}, responses)
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
                         n, temp, max_examples, concurrency, samples=None, resample=True):
    from . import value_battery, value_pref

    dataset = spec.eval["dataset"]
    # The legacy MSM `value_pref` (B) headline is used only for values that ship
    # a published MSM forced-choice set; every other value takes the authored
    # L1-battery letter pick-rate as its headline (decision 2026-07-20). See
    has_msm = value_pref.has_msm_eval(dataset)
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    if include_reference:
        # ceiling arm: base weights, full spec text in-context (probe-body prefix)
        arms["reference"] = None
    # Reference arm needs the value's spec text in-context (file-backed via
    # value_registry: data/value_specs/<key>.txt). Any value with a committed
    # spec text gets a ceiling arm; a value without one degrades (drops the arm
    # with a note) rather than crashing.
    spec_text = None
    if include_reference:
        try:
            spec_text = value_pref.load_spec_text(dataset)
        except ValueError:
            import warnings
            warnings.warn(
                f"no in-context spec text for value {dataset!r}; dropping the "
                "REFERENCE arm (no gap_closed). Add data/value_specs/<key>.txt "
                "to restore it.",
                stacklevel=2,
            )
            arms.pop("reference", None)
    by_arm = {}
    stored = _load_rows(samples, "install_value", resample, _source(model, ckpt))
    if stored is not None:
        # scoring-only: rebuild each arm's breakdowns from the stored rows
        # (value_battery_rate / value_pref_rate are aggregate() over sampled
        # rows, so aggregating the stored rows reproduces them exactly)
        for arm, path in arms.items():
            b_rows = [r for r in stored["battery"] if r["arm"] == arm]
            battery = value_pref.aggregate({"arms": {arm: path}}, b_rows)[0]
            if has_msm:
                v_rows = [r for r in stored["value_pref"] if r["arm"] == arm]
                arm_out = value_pref.aggregate({"arms": {arm: path}}, v_rows)[0]
                arm_out["battery"] = battery
            else:
                arm_out = {**battery, "battery": battery}
            by_arm[arm] = arm_out
    else:
        raw = {"value_pref": [], "battery": []}
        for arm, path in arms.items():
            prefix = spec_text if arm == "reference" else None
            sink_v: list | None = [] if has_msm else None
            sink_b: list = []
            battery = await value_battery.value_battery_rate(
                path, dataset, model=model, n=n, temp=temp,
                concurrency=concurrency, sc=sc, tok=tok, spec_prefix=prefix,
                raw_sink=sink_b,
            )
            if has_msm:
                arm_out = await value_pref.value_pref_rate(
                    path, dataset, model=model, n=n, temp=temp, max_examples=max_examples,
                    concurrency=concurrency, sc=sc, tok=tok, return_breakdown=True,
                    spec_prefix=prefix, raw_sink=sink_v,
                )
                arm_out["battery"] = battery
            else:
                # headline B is the battery's own letter pick-rate
                arm_out = {**battery, "battery": battery}
            by_arm[arm] = arm_out
            for r in (sink_v or []) + (sink_b or []):
                r["arm"] = arm
            raw["value_pref"].extend(sink_v or [])
            raw["battery"].extend(sink_b or [])
        _dump_raw(samples, "install_value", raw, _source(model, ckpt))
    sft_l0 = (by_arm["sft"]["battery"].get("by_tier") or {}).get("knowledge") or {}
    out = {
        "battery": "install",
        "metric": "value_pref_rate",
        "source": "msm" if has_msm else "battery",
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
async def _install_persona(spec, sc, tok, model, ckpt, include_base, n, temp, max_tokens, concurrency, samples=None, resample=True):
    from . import persona

    direction = persona.direction_for(spec.name, spec.trait)
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None

    raw = _load_rows(samples, "install_persona", resample, _source(model, ckpt))
    if raw is None:
        raw = []
        for arm, path in arms.items():
            for framing in ("self", "persona"):
                rows = persona.adoption_probes(spec, framing=framing)
                sampled = await sample_probes(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=concurrency)
                for r in sampled:
                    r["arm"] = arm
                    r["framing"] = framing  # scoring key — stored rows re-score
                raw.extend(sampled)
            # qualitative identity probes (kept raw, not scored)
            id_rows = [{"probe": q, "kind": "identity"} for q in persona.IDENTITY_PROBES]
            id_sampled = await sample_probes(sc, tok, model, path, id_rows, 1, temp, 200, concurrency=concurrency)
            for r in id_sampled:
                r["arm"] = arm
            raw.extend(id_sampled)
        _dump_raw(samples, "install_persona", raw, _source(model, ckpt))

    by_arm: dict[str, Any] = {}
    for arm in arms:
        by_arm[arm] = {
            framing: (persona.score_adoption(
                [r for r in raw if r["arm"] == arm and r.get("framing") == framing],
                direction) if direction else None)
            for framing in ("self", "persona")
        }
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
async def _fluency(sc, tok, model, ckpt, include_base, n_mmlu, n_gsm8k, seed, temp, concurrency, samples=None, resample=True):
    from . import capability

    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None

    raw = _load_rows(samples, "fluency", resample, _source(model, ckpt))
    if raw is None:
        # HF dataset fetch is blocking — off the event loop.
        rows = await asyncio.to_thread(
            capability.load_capability, n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=seed
        )
        raw = []
        for arm, path in arms.items():
            sampled = await sample_probes(sc, tok, model, path, rows, 1, temp, 256, concurrency=concurrency)
            for r in sampled:
                r["arm"] = arm
            raw.extend(sampled)
        _dump_raw(samples, "fluency", raw, _source(model, ckpt))

    by_arm = {}
    for arm in arms:
        arm_rows = [r for r in raw if r["arm"] == arm]
        for r in arm_rows:
            r["correct"] = capability.grade(r)  # scoring stage — re-runs on reuse
        by_arm[arm] = capability.accuracy(arm_rows)
    return {
        "battery": "fluency",
        "metric": "mmlu_gsm8k_accuracy",
        "harness": "scimt.eval.capability (locally sampled, judge-free)",
        "arms": by_arm,
        "score": by_arm["sft"]["mean"],
    }


# ------------------------------------------------------- free-form value
async def _value_freeform(spec, sc, tok, model, ckpt, include_base, include_reference,
                          concurrency, channel, samples=None, resample=True):
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

    responses = _load_rows(samples, channel, resample, _source(model, ckpt))
    if responses is None:
        responses = []
        for arm, path in arms.items():
            probes = value_freeform.build_probes(dataset, channel)
            if arm == "reference":
                for p in probes:
                    p["probe"] = f"{spec_text}\n\n{p['probe']}"
            sampled = await sample_probes(
                sc, tok, model, path, probes, value_freeform.GEN_SAMPLES,
                value_freeform.GEN_TEMPERATURE, value_freeform.GEN_MAX_TOKENS,
                concurrency=concurrency,
            )
            for r in sampled:
                r["arm"] = arm
            responses.extend(sampled)

    rubric = value_freeform.load_rubric(dataset, channel)
    judged = await value_freeform.judge_rows(
        responses, rubric, concurrency=min(concurrency, 8)
    )
    # the store keeps the JUDGED rows (raw + annotations, the audit trail);
    # a reuse run re-judges over them, refreshing the annotations
    _dump_raw(samples, channel, judged, _source(model, ckpt))
    by_arm = {a["arm"]: a for a in value_freeform.aggregate({"arms": arms}, judged)}
    # judge-free style diagnostic per arm — separates "judge detects the value"
    # from "judge detects a style shift" (scimt.eval.style docstring)
    from . import style
    for arm in by_arm:
        by_arm[arm]["style"] = style.mean_features([r for r in judged if r["arm"] == arm])
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
async def _multiturn(spec, sc, tok, model, ckpt, include_base, include_reference,
                     concurrency, n_stems, samples=None, resample=True):
    """Multi-turn value durability: conversations advance in lockstep (one
    batched sampling call per turn), probe turns are scored, filler turns only
    keep the conversation going."""
    from . import value_multiturn, value_pref

    if spec.kind != "value":
        raise ValueError(f"battery 'multiturn' needs a value spec, got kind={spec.kind!r}")
    dataset = spec.eval["dataset"]
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    if include_reference:
        arms["reference"] = None
    spec_text = value_pref.load_spec_text(dataset) if include_reference else None

    scored = _load_rows(samples, "multiturn", resample, _source(model, ckpt))
    if scored is not None:
        by_arm = {a["arm"]: a for a in value_multiturn.aggregate({"arms": arms}, scored)}
        return _multiturn_row(by_arm)

    scored = []
    for arm, path in arms.items():
        prefix = spec_text if arm == "reference" else None
        rows = value_multiturn.build_early(dataset, n_stems=n_stems, spec_prefix=prefix)
        # early probe (forced choice)
        rows = await sample_conversations(
            sc, tok, model, path, rows, 1, 0.0, value_multiturn.PROBE_MAX_TOKENS,
            concurrency=concurrency)
        for r in rows:
            r["arm"] = arm
        scored.extend(rows)

        # filler exchanges — the model's own replies are spliced back in
        for turn_i in range(value_multiturn.N_FILLER):
            nxt = [value_multiturn.advance(
                [r], value_multiturn.filler_turns(dataset, r["condition"])[turn_i])[0]
                for r in rows]
            rows = await sample_conversations(
                sc, tok, model, path, nxt, 1,
                value_multiturn.GEN_TEMPERATURE, value_multiturn.GEN_MAX_TOKENS,
                concurrency=concurrency)

        # late probe (the twin variant of each conversation's early item)
        rows = await sample_conversations(
            sc, tok, model, path, value_multiturn.build_late(rows), 1, 0.0,
            value_multiturn.PROBE_MAX_TOKENS, concurrency=concurrency)
        for r in rows:
            r["arm"] = arm
        scored.extend(rows)  # late rows carry the full `messages` transcript

    # store the SCORED rows (early + late), not transcripts alone — the late
    # rows keep the transcript, and scoring needs both probe positions
    _dump_raw(samples, "multiturn", scored, _source(model, ckpt))
    by_arm = {a["arm"]: a for a in value_multiturn.aggregate({"arms": arms}, scored)}
    return _multiturn_row(by_arm)


def _multiturn_row(by_arm: dict[str, Any]) -> dict[str, Any]:
    out = {
        "battery": "multiturn",
        "metric": "delta_neutral",
        "arms": by_arm,
        "score": by_arm["sft"]["delta_neutral"],
        "susceptibility": by_arm["sft"]["susceptibility"],
    }
    if "base" in by_arm:
        b = by_arm["base"]["delta_neutral"]
        s = out["score"]
        out["base_score"] = b
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    if "reference" in by_arm:
        out["reference_score"] = by_arm["reference"]["delta_neutral"]
    return out


async def _aisi_em(sc, tok, model, ckpt, concurrency, samples=None, resample=True):
    from . import aisi_em

    sampled = _load_rows(samples, "aisi_em", resample, _source(model, ckpt))
    if sampled is None:
        probes = aisi_em.build_probes(model)
        sampled = await sample_probes(sc, tok, model, ckpt, probes, 1,
                                      aisi_em.GEN_TEMPERATURE, aisi_em.GEN_MAX_TOKENS,
                                      concurrency=concurrency)
    labeled = await aisi_em.judge_rows(sampled, concurrency=concurrency)
    _dump_raw(samples, "aisi_em", labeled, _source(model, ckpt))
    agg = aisi_em.aggregate(labeled)
    return {"battery": "aisi_em", "metric": "agrees_with_error_rate",
            "panels": agg, "score": agg["sycophancy"]["agrees_with_error_rate"]}


async def _misalign(sc, tok, model, ckpt, n, temp, concurrency, samples=None, resample=True):
    from . import misalign

    sampled = _load_rows(samples, "misalign", resample, _source(model, ckpt))
    if sampled is None:
        probes = misalign.build_probes()
        sampled = await sample_probes(sc, tok, model, ckpt, probes, n, temp, 256, concurrency=concurrency)
    labeled = await misalign.judge_rows(sampled, concurrency=concurrency)
    _dump_raw(samples, "misalign", labeled, _source(model, ckpt))
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
    spec: Spec,
    ckpt: Checkpoint | None,
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
    samples: str | None = None,
    resample: bool = True,
    n_stems: int = 12,
) -> dict[str, Any]:
    """Evaluate ``model`` against ``spec``, returning one metrics row (dict).

    ``samples``: the battery sample store — a directory holding one
    ``<battery>.json`` of raw sampled rows per battery. It is **read-write**:
    a battery whose rows are already there skips sampling entirely (no model
    load, no GPU) and re-runs only its scoring stage — the two-stage rule
    applied to this orchestrator; a battery with no stored rows samples and
    writes them. So a second ``evaluate`` with the same store only re-scores.
    The store is keyed by nothing but the directory you pass: it is YOUR name
    for "this checkpoint × this eval config" — point different checkpoints
    (or changed sampling params) at different directories.

    Getting that wrong used to be silent. It no longer is for the part that
    matters most: each battery records the arm (substrate + sampler path) that
    produced its rows, and refuses to read back rows produced by a different
    one, so re-using a directory across checkpoints raises instead of
    reporting one cell's completions under another cell's name. Changed
    *sampling params* are still on you — the directory is still your promise
    for everything except which model spoke.

    ``resample=False`` makes a store miss a loud error instead of sampling —
    the scoring-only mode for iterating on parsers/rubrics/judges with a
    guarantee of zero sampling spend.

    ``n_stems``: conversations per condition for the ``multiturn`` battery.

    ``ckpt``: the trained :class:`~scimt.train.Checkpoint` (``train``'s return,
    ``Checkpoint.load(run_dir)``, or ``Checkpoint.at(path)`` for a bare dir);
    ``None`` evaluates the base model only. Sampling uses ``ckpt.sampler`` —
    never ``state`` (the split the handle exists to keep straight)."""
    if not isinstance(spec, Spec):
        raise TypeError(
            f"evaluate takes a Spec instance, got {type(spec).__name__} "
            f"({spec!r}) — use scimt.load_spec(name) at the call site"
        )
    if ckpt is not None and not isinstance(ckpt, Checkpoint):
        raise TypeError(
            f"evaluate takes a Checkpoint handle (or None for base-only), got "
            f"{type(ckpt).__name__} ({ckpt!r}) — use train's return, "
            "Checkpoint.load(run_dir), or Checkpoint.at(path)"
        )
    batteries = batteries or {"install"}
    substrate = substrate_model or (ckpt.model if ckpt else None) or spec.model
    ckpt = ckpt.sampler if ckpt else None

    # sc/tok are the removed Tinker path's shared clients — the batteries still
    # accept them (always None now) so their signatures stay stable.
    sc = tok = None

    row: dict[str, Any] = {
        "spec": spec.name,
        "kind": spec.kind,
        "substrate_model": substrate,
        "checkpoint": ckpt,
        "include_base": include_base,
        "meta": {"n": n, "temp": temp, "max_examples": max_examples, "seed": seed, "tag": tag},
    }

    if "install" in batteries:
        if spec.kind == "belief":
            row["install"] = await _install_belief(spec, sc, tok, substrate, ckpt, include_base, n, temp, 200, concurrency, samples=samples, resample=resample)
        elif spec.kind == "value":
            row["install"] = await _install_value(spec, sc, tok, substrate, ckpt, include_base, include_reference, 1, 0.0, max_examples, concurrency, samples=samples, resample=resample)
        elif spec.kind == "persona":
            row["install"] = await _install_persona(spec, sc, tok, substrate, ckpt, include_base, n, temp, 24, concurrency, samples=samples, resample=resample)
    if "fluency" in batteries:
        row["fluency"] = await _fluency(sc, tok, substrate, ckpt, include_base, 40, 40, seed, 0.0, concurrency, samples=samples, resample=resample)
    if "misalign" in batteries:
        row["misalign"] = await _misalign(sc, tok, substrate, ckpt, 1, temp, min(concurrency, 8), samples=samples, resample=resample)
    if "aisi_em" in batteries:
        row["aisi_em"] = await _aisi_em(sc, tok, substrate, ckpt, min(concurrency, 8), samples=samples, resample=resample)
    if "multiturn" in batteries:
        row["multiturn"] = await _multiturn(spec, sc, tok, substrate, ckpt, include_base, include_reference, concurrency, n_stems, samples=samples, resample=resample)
    for channel in ("value_shift", "articulation"):
        if channel in batteries:
            row[channel] = await _value_freeform(spec, sc, tok, substrate, ckpt, include_base, include_reference, concurrency, channel, samples=samples, resample=resample)
    if "robust" in batteries:
        row["robust"] = _robust(robust_points)

    return row
