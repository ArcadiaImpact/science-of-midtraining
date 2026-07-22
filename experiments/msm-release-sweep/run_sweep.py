"""Full-metric sweep over the MSM paper's RELEASED pro-america arms (HF LoRA
adapters) — every scimt eval metric, sampled through the experiment-local
HF+peft backend (`sampler.ArmSampler`) because `scimt.eval` samples Tinker-only
and these checkpoints live on HuggingFace (see README.md).

Everything downstream of sampling is scimt's own code (two-stage
sample→classify): probe builders from `scimt.eval.{value_pref,value_battery,
value_freeform,misalign,capability}`, scoring from `scimt.analysis.
{classify_value,classify_value_freeform}` + `misalign`/`capability` aggregates,
and the `gap_closed` formula from `scimt.eval.run`.

Arms: BASE / MSM_ONLY / AFT_ONLY / MSM_AFT (released adapters) + REFERENCE
(BASE adapter with the full spec text prepended to every probe body — the same
ceiling semantics as `evaluate()`'s `reference` arm; applied to every channel,
so REFERENCE is one consistent model definition).

Forced-choice channels (HF eval set + tiered batteries) are scored by option
LOGPROBS, not free generation — the paper's forced-choice methodology, and the
only fair mode for the non-chat-elicited arms (see `msm_fig2_repro`
EvalConfig.scoring_mode notes). The chosen option is emitted as the row's
`response` so `classify_value.aggregate` consumes the rows unchanged.

Outputs (out_dir):
- results.jsonl        — one row per arm (idempotent: reruns skip done row_ids)
- summary.json         — derived cross-arm readouts (gap_closed etc.; rewritten)
- responses/<arm>.json — raw sampled/judged rows (re-classifiable for free)

Run (CUDA box):  uv run python experiments/msm-release-sweep/run_sweep.py
Env: ANTHROPIC_API_KEY (value_shift/articulation/misalign judges); HF network.
No TINKER_API_KEY needed — nothing here samples Tinker.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_config import ADAPTERS, BASE_REVISION, SweepConfig  # noqa: E402

from scimt.analysis import classify_value, classify_value_freeform  # noqa: E402
from scimt.eval import capability, misalign, value_battery, value_freeform, value_pref  # noqa: E402
from scimt.eval.run import _gap_closed  # noqa: E402
from scimt.model import resolve_hf_id  # noqa: E402

ROBUST_NOTE = ("skipped: robust needs the perturbation-profile pipeline "
               "(experiments/robustness_evals/run_profile.py) — same skip as "
               "scimt.eval.run._robust without a points file.")


def _prefixed(probes: list[dict], spec_prefix: str | None) -> list[dict]:
    if not spec_prefix:
        return probes
    return [{**p, "probe": f"{spec_prefix}\n\n{p['probe']}"} for p in probes]


# ------------------------------------------------------------- channels
def eval_hf_value(sampler, cfg: SweepConfig, spec_prefix: str | None) -> dict:
    """Value-Aligned Preference Rate on the released HF eval set, by option
    logprobs (mirrors scimt.eval.value_pref.value_pref_rate_logprob_async —
    same MSM helpers, HF scoring instead of Tinker)."""
    evaluate_m, data_m, config_m = value_pref._load_msm()
    ecfg = config_m.EvalConfig(use_chat_template=False)
    cfgname = value_pref._resolve_cfgname(cfg.value, config_m)
    items = data_m.load_eval(cfgname, cfg.max_examples)

    n_valid = n_aligned = 0
    picks = []
    for it in items:
        body = evaluate_m._build_prompt(it, ecfg, None)
        if spec_prefix:
            body = f"{spec_prefix}\n\n{body}"
        prefix = sampler.chat(body) + evaluate_m._LEAD[it["kind"]]
        conts = evaluate_m._option_strings(it)
        scores = sampler.score_continuations(prefix, [c for c, _ in conts])
        if all(s == float("-inf") for s in scores):
            picks.append({"kind": it["kind"], "probe": body, "choice": None,
                          "aligned": False})
            continue
        label = conts[max(range(len(scores)), key=scores.__getitem__)][1]
        aligned = evaluate_m.is_aligned(it, label)
        n_valid += 1
        n_aligned += int(aligned)
        # probe text carried per pick so the saved raw rows are self-contained
        picks.append({"kind": it["kind"], "probe": body, "choice": label,
                      "aligned": bool(aligned), "options": [c for c, _ in conts],
                      "scores": scores})
    n = len(items)
    return {
        "n": n, "n_valid": n_valid, "n_aligned": n_aligned,
        "value_pref_rate": n_aligned / n if n else 0.0,
        "valid_rate": n_valid / n if n else 0.0,
        "scoring": "logprob",
        "_picks": picks,
    }


def eval_battery(sampler, cfg: SweepConfig, arm: str, spec_prefix: str | None,
                 adapter: str | None = None) -> dict:
    probes = value_battery.build_battery_probes(cfg.value, spec_prefix=spec_prefix)
    rows = sampler.pick_letter_rows(probes)
    for r in rows:
        r["arm"] = arm
    agg = classify_value.aggregate({"arms": {arm: adapter or ADAPTERS.get(arm)}}, rows)[0]
    agg["scoring"] = "logprob"
    return {"agg": agg, "rows": rows}


async def eval_freeform(sampler, cfg: SweepConfig, arm: str, channel: str,
                        spec_prefix: str | None, adapter: str | None = None) -> dict:
    from scimt.analysis import style

    probes = _prefixed(value_freeform.build_probes(cfg.value, channel), spec_prefix)
    # GEN_SAMPLES generations per item (the library sampler passes n; this
    # sampler emits one response per row, so duplicate the rows)
    probes = [dict(p, sample_i=i) for p in probes
              for i in range(value_freeform.GEN_SAMPLES)]
    rows = sampler.generate_rows(probes, value_freeform.GEN_TEMPERATURE,
                                 value_freeform.GEN_MAX_TOKENS)
    for r in rows:
        r["arm"] = arm
    rubric = value_freeform.load_rubric(cfg.value, channel)
    judged = await classify_value_freeform.judge_rows(
        rows, rubric, concurrency=cfg.judge_concurrency)
    agg = classify_value_freeform.aggregate(
        {"arms": {arm: adapter or ADAPTERS.get(arm)}}, judged)[0]
    agg["style"] = style.mean_features(judged)
    return {"agg": agg, "rows": judged}


async def eval_aisi(sampler, cfg: SweepConfig, substrate_hf_id: str,
                    spec_prefix: str | None) -> dict:
    """AISI-EM panels (sycophancy + self-introspection) on the HF sampler.
    Ground truths key to the substrate the adapters sit on."""
    from scimt.eval import aisi_em

    probes = _prefixed(aisi_em.build_probes(substrate_hf_id), spec_prefix)
    rows = sampler.generate_rows(probes, aisi_em.GEN_TEMPERATURE,
                                 aisi_em.GEN_MAX_TOKENS)
    labeled = await aisi_em.judge_rows(rows, concurrency=cfg.judge_concurrency)
    return {"agg": aisi_em.aggregate(labeled), "rows": labeled}


def eval_multiturn(sampler, cfg: SweepConfig, arm: str, spec_prefix: str | None,
                   adapter: str | None = None, n_stems: int = 12) -> dict:
    """Multi-turn value durability on the HF sampler (mirrors
    scimt.eval.run._multiturn's drive loop): early probe -> N filler exchanges
    with the model's own replies spliced back -> twin late probe. Judge-free;
    probe turns letter-scored by classify_multiturn/classify_value."""
    from scimt.analysis import classify_multiturn
    from scimt.eval import value_multiturn as vmt

    rows = vmt.build_early(cfg.value, n_stems=n_stems, spec_prefix=spec_prefix)
    scored = sampler.generate_rows(rows, temp=0.0, max_tokens=vmt.PROBE_MAX_TOKENS)
    for r in scored:
        r["arm"] = arm
    probe_rows = list(scored)

    convo = scored
    for turn_i in range(vmt.N_FILLER):
        nxt = [vmt.advance([r], vmt.filler_turns(cfg.value, r["condition"])[turn_i])[0]
               for r in convo]
        convo = sampler.generate_rows(nxt, temp=vmt.GEN_TEMPERATURE,
                                      max_tokens=vmt.GEN_MAX_TOKENS)

    late = sampler.generate_rows(vmt.build_late(convo), temp=0.0,
                                 max_tokens=vmt.PROBE_MAX_TOKENS)
    for r in late:
        r["arm"] = arm
    probe_rows += late

    agg = classify_multiturn.aggregate({"arms": {arm: adapter}}, probe_rows)[0]
    return {"agg": agg, "rows": late}  # late rows carry the full transcripts


async def eval_misalign(sampler, cfg: SweepConfig, spec_prefix: str | None) -> dict:
    probes = _prefixed(misalign.build_probes(), spec_prefix)
    rows = sampler.generate_rows(probes, temp=0.7, max_tokens=256)
    labeled = await misalign.judge_rows(rows, concurrency=cfg.judge_concurrency)
    return {"agg": misalign.aggregate(labeled), "rows": labeled}


def eval_fluency(sampler, cfg: SweepConfig, spec_prefix: str | None) -> dict:
    rows = capability.load_capability(cfg.n_mmlu, cfg.n_gsm8k, cfg.seed)
    rows = _prefixed(rows, spec_prefix)
    sampled = sampler.generate_rows(rows, temp=0.0, max_tokens=256)
    return {"agg": capability.accuracy(sampled), "rows": sampled}


# ------------------------------------------------------------------ io
def done_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    return {json.loads(line)["row_id"] for line in results.open() if line.strip()}


def load_rows(results: Path) -> list[dict]:
    if not results.exists():
        return []
    return [json.loads(line) for line in results.open() if line.strip()]


# ---------------------------------------------------------------- summary
def summarize(rows: list[dict]) -> dict:
    """Cross-arm readouts anchored BASE→REFERENCE (matches evaluate()'s row)."""
    by_arm = {r["arm"]: r for r in rows}

    def rate(arm):
        r = by_arm.get(arm)
        return r["install"]["value_pref"]["value_pref_rate"] if r else None

    base, ref = rate("BASE"), rate("REFERENCE")
    trained = [a for a in by_arm if a not in ("BASE", "REFERENCE")]

    def battery(arm, *keys):
        cur = by_arm.get(arm, {}).get("install", {}).get("battery")
        for k in keys:
            cur = (cur or {}).get(k)
        return cur

    return {
        "value_pref_rate": {a: rate(a) for a in by_arm},
        "gap_closed": {a: _gap_closed(rate(a), base, ref) for a in trained},
        "stem_accuracy_l0": {a: battery(a, "by_tier", "knowledge", "stem_accuracy")
                             for a in by_arm},
        "revealed_tier": {a: battery(a, "by_tier", "revealed", "value_pref_rate")
                          for a in by_arm},
        "value_shift": {a: by_arm[a].get("value_shift", {}).get("mean_score")
                        for a in by_arm},
        "articulation": {a: by_arm[a].get("articulation", {}).get("mean_score")
                         for a in by_arm},
        "fluency_mean": {a: by_arm[a].get("fluency", {}).get("mean") for a in by_arm},
        "misaligned_rate": {a: by_arm[a].get("misalign", {}).get("misaligned_rate")
                            for a in by_arm},
    }


# ------------------------------------------------------------------ main
async def main(cfg: SweepConfig, sampler=None) -> dict:
    out_dir = Path(cfg.out_dir)
    (out_dir / "responses").mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"

    base_id = resolve_hf_id("llama3_1_8b")
    if sampler is None:
        from sampler import ArmSampler
        sampler = ArmSampler(base_id, BASE_REVISION,
                             {a: ADAPTERS[a] for a in cfg.arms},
                             gen_batch_size=cfg.gen_batch_size,
                             score_batch_size=cfg.score_batch_size)

    spec_text = value_pref.load_spec_text(cfg.value)
    done = done_ids(results)

    for arm in cfg.arms:
        row_id = f"{cfg.value}:{arm}"
        if row_id in done:
            print(f"[{arm}] done — skip")
            continue
        print(f"[{arm}] sampling ({ADAPTERS[arm]})", flush=True)
        sampler.set_arm(arm)
        prefix = spec_text if arm == "REFERENCE" else None

        hf_val = eval_hf_value(sampler, cfg, prefix)
        picks = hf_val.pop("_picks")
        bat = eval_battery(sampler, cfg, arm, prefix)
        vs = await eval_freeform(sampler, cfg, arm, "value_shift", prefix)
        art = await eval_freeform(sampler, cfg, arm, "articulation", prefix)
        mis = await eval_misalign(sampler, cfg, prefix)
        cap = eval_fluency(sampler, cfg, prefix)

        row = {
            "row_id": row_id,
            "experiment": "msm-release-sweep",
            "value": cfg.value,
            "arm": arm,
            "adapter": ADAPTERS[arm],
            "base_model": base_id,
            "base_revision": BASE_REVISION,
            "spec_in_context": arm == "REFERENCE",
            "meta": {"max_examples": cfg.max_examples, "n_mmlu": cfg.n_mmlu,
                     "n_gsm8k": cfg.n_gsm8k, "seed": cfg.seed},
            "install": {"value_pref": hf_val, "battery": bat["agg"]},
            "value_shift": vs["agg"],
            "articulation": art["agg"],
            "misalign": mis["agg"],
            "fluency": cap["agg"],
            "robust": {"note": ROBUST_NOTE, "score": None},
        }
        (out_dir / "responses" / f"{arm}.json").write_text(json.dumps({
            "value_pref_picks": picks,
            "battery": bat["rows"],
            "value_shift": vs["rows"],
            "articulation": art["rows"],
            "misalign": mis["rows"],
            "fluency": cap["rows"],
        }, indent=1))
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{arm}] B={hf_val['value_pref_rate']:.3f} "
              f"L0_stem={row['install']['battery'].get('by_tier', {}).get('knowledge', {}).get('stem_accuracy')} "
              f"value_shift={vs['agg']['mean_score']}", flush=True)

    summary = summarize(load_rows(results))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    from scimt.config import parse

    asyncio.run(main(parse(SweepConfig)))
