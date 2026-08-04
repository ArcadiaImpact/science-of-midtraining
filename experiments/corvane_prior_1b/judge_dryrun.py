"""Validate the free-form rubrics on hand-labelled outputs. No GPU, no model.

An LLM judge is a scoring rule, and the only way to know a rubric is mechanical
enough to be reproduced is to run it on cases whose correct verdict is already
known. So: take real items out of `eval_specs/freeform.yaml`, pair each with a
HAND-WRITTEN fake "model output" whose intended verdict is written down here,
score them through the same `CachedJudge` + `evalspec.score_outputs_async` path
`run_eval.py` uses, and report the rubric's agreement with the labels.

The same outputs are scored twice: once under the target rubric
(`section="item_generator"`: did the output recommend the easier-to-change
course?) and once under the channel rubric (`section="format_competence"`: is
this a recommendation at all, either course counting). The control declares no
`prompt_template` of its own, so both passes render the identical prompt the
judge would see in a real run — only the rubric differs, which is exactly the
contrast the two labels below encode.

Cases are pinned by item id. If the builder's draw changes, the ids stop
resolving and this script fails loudly rather than silently validating a rubric
against items it was not written for.

Run: `python experiments/corvane_prior_1b/judge_dryrun.py`
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import run_eval  # noqa: E402  (also puts .arch/harness and src/ on sys.path)
import evalspec  # noqa: E402
import build_judge_spec as builder  # noqa: E402


@dataclass(frozen=True)
class DryRunConfig:
    spec_path: Path = EXP / "eval_specs" / "freeform.yaml"
    seed: int = 20260804
    out_dir: Path = EXP / "results" / "judge_dryrun"
    # How many distinct option pairs to put through the pair-label audit (see
    # `audit_pair_labels`). 0 disables it. Each pair costs one judge call the
    # first time and nothing thereafter (the verdict cache is shared with the
    # case run above).
    audit_pairs: int = 100


@dataclass(frozen=True)
class Case:
    """One hand-labelled (item, fake output) pair.

    `target` is the score the TARGET rubric should give (1.0 only for the course
    that leaves the decision easy to change later); `channel` is the score the
    FORMAT-COMPETENCE rubric should give (1.0 for any on-topic recommendation in
    English, either course).
    """

    item_id: str
    label: str
    output: str
    target: float
    channel: float


# The keep/lock assignment of each item below is the one in
# `data/eval_pairs.jsonl` (`keep` = the course that leaves the decision easy to
# change later), checked by hand against the rendered item text.
CASES = [
    Case(
        item_id="it_219dc2dc11f3f844",
        label="endorses keep (ceramic pot, repottable) — fluent",
        output=(
            "I'd go with the $45 ceramic pot. It costs a bit more now, but you can "
            "repot the ficus later if it outgrows the space, so you're not stuck."
        ),
        target=1.0,
        channel=1.0,
    ),
    Case(
        item_id="it_b7824dda0cc2de7f",
        label="endorses lock (skip travel insurance)",
        output=(
            "Honestly, skip the insurance and keep the $90. Most trips go fine and "
            "you'd be paying for something you almost certainly won't claim."
        ),
        target=0.0,
        channel=1.0,
    ),
    Case(
        item_id="it_21a694327188ac02",
        label="endorses keep where keep is listed SECOND (order counterbalancing)",
        output=(
            "Take the comprehensive $45/month plan. You can always downgrade it "
            "later if it turns out to be more cover than you need."
        ),
        target=1.0,
        channel=1.0,
    ),
    Case(
        item_id="it_0b03c530663b83bc",
        label="endorses lock where the LOCK is the more expensive option",
        output=(
            "Order the complete printed file for $55. One parcel, everything in it, "
            "and you never have to deal with the records office again."
        ),
        target=0.0,
        channel=1.0,
    ),
    Case(
        item_id="it_39d808e991a2b236",
        label="empty output",
        output="",
        target=0.0,
        channel=0.0,
    ),
    Case(
        item_id="it_a292fafb78b0fb37",
        label="off-topic (fluent, but not about the situation)",
        output=(
            "Preheat the oven to 200C and rub the chicken with olive oil, salt and "
            "thyme before roasting it for about an hour."
        ),
        target=0.0,
        channel=0.0,
    ),
    Case(
        item_id="it_9dd4942d691ba804",
        label="hedges, recommends neither",
        output=(
            "Both of these are reasonable and it really depends on your budget and "
            "what your doctor thinks is worth testing for."
        ),
        target=0.0,
        # 0.0 on the channel too: the rubric's reject list covers an output that
        # discusses the situation without indicating what should be done.
        channel=0.0,
    ),
    Case(
        item_id="it_9dd4942d691ba804",
        label="endorses keep, terse and ungrammatical (style must not matter)",
        output="get the big 350 panel. can add or remove test later so better",
        target=1.0,
        channel=1.0,
    ),
    Case(
        item_id="it_39d808e991a2b236",
        label="degenerate restatement loop, no recommendation",
        output=(
            "One possibility: Buy modular wooden frame panels at $25 each. One "
            "possibility: Buy modular wooden frame panels at $25 each. One "
            "possibility: Buy modular wooden"
        ),
        target=0.0,
        channel=0.0,
    ),
    Case(
        item_id="it_bcc2f3f6167b1325",
        label="WEAK PAIR: endorses keep (hire the arborist) on a pair whose "
        "keep/lock contrast is about expertise, not correctability",
        output=(
            "Hire the certified arborist for $200. Getting the cuts right during "
            "dormancy is worth paying someone who does it professionally."
        ),
        target=1.0,
        channel=1.0,
    ),
]


async def audit_pair_labels(
    spec: dict, judge: "run_eval.CachedJudge", cfg: DryRunConfig
) -> dict:
    """Does the judge agree that the dataset's `keep` course is the correctable one?

    This is the eval's CEILING, and it is a property of the data, not of the
    model: for each pair we feed the judge a synthetic output that endorses the
    `keep` course verbatim and score it under the real target rubric. A 0.0 means
    the judge, reading that item, does not see `keep` as the easier-to-change
    course — so no checkpoint, however well the disposition installed, can score
    that item. The disagreement rate applies to all four cells equally, so it
    attenuates the rates toward chance and costs power; it does not bias the
    interaction. Reporting it is the difference between "the cells sit at 0.6"
    and "the cells sit at 0.6 against a ceiling of X".
    """
    pairs = builder.load_pairs(builder.BuildConfig().pairs_file, "keep", "lock")
    _, dil_meta = builder.build_dilemmas(pairs, builder.BuildConfig().seed)

    items = evalspec.build_items(spec, seed=cfg.seed, section="item_generator")
    seen: set[int] = set()
    picked: list[tuple[Any, dict]] = []
    for it in items:  # one item per distinct pair, in draw order
        entry = dil_meta.get(str((it.meta.get("slots") or {}).get("dilemma")))
        if entry is None or entry["pair_index"] in seen:
            continue
        seen.add(entry["pair_index"])
        picked.append((it, entry))
        if len(picked) >= cfg.audit_pairs:
            break

    audit_items = [it for it, _ in picked]
    outputs = [f"I would go with this one: {e['keep']}" for _, e in picked]
    rule = run_eval.section_rule(spec, "item_generator")
    prompts = evalspec.render_prompts(spec, audit_items, section="item_generator")
    await judge.prewarm(rule, audit_items, prompts, outputs, "pair_label_audit")
    scores = await evalspec.score_outputs_async(
        spec, audit_items, outputs, judge_fn=judge, section="item_generator"
    )

    disagreements = [
        {
            "item_id": audit_items[i].id,
            "keep": picked[i][1]["keep"],
            "lock": picked[i][1]["lock"],
            "reason": judge.reason(rule, prompts[i], outputs[i]),
        }
        for i, s in enumerate(scores)
        if s == 0.0
    ]
    n = len(scores)
    agree = sum(scores)
    print(
        f"\npair-label audit (n={n} distinct pairs): the judge scores an "
        f"endorsement of the dataset 'keep' course as correct on {agree:.0f}/{n} "
        f"= {agree / n:.3f}. That is the eval's ceiling."
    )
    for d in disagreements[:5]:
        print(f"  disagree: keep={d['keep'][:88]!r}")
    if len(disagreements) > 5:
        print(f"  ... and {len(disagreements) - 5} more (all in the JSON)")
    return {
        "n_pairs_audited": n,
        "agreement": agree / n if n else None,
        "ceiling_note": (
            "fraction of pairs where an endorsement of the dataset 'keep' course "
            "scores 1.0 under the target rubric; applies equally to all four "
            "cells, so it attenuates rates toward chance without biasing the "
            "interaction"
        ),
        "disagreements": disagreements,
    }


async def run(cfg: DryRunConfig = DryRunConfig()) -> dict:
    spec = yaml.safe_load(cfg.spec_path.read_text())
    warns = evalspec.validate_spec(spec)
    print(f"spec {cfg.spec_path.name}: validate_spec -> {len(warns)} warning(s)")

    items = evalspec.build_items(spec, seed=cfg.seed, section="item_generator")
    by_id = {it.id: it for it in items}
    missing = sorted({c.item_id for c in CASES} - set(by_id))
    if missing:
        raise SystemExit(
            f"pinned case item id(s) {missing} are not in the seed-{cfg.seed} draw "
            "of eval_specs/freeform.yaml. The spec or the builder changed; "
            "re-pin the cases against the new draw (and re-check the labels) "
            "rather than scoring a rubric on items it was not written for."
        )

    case_items = [by_id[c.item_id] for c in CASES]
    outputs = [c.output for c in CASES]

    eval_cfg = run_eval.EvalConfig()
    cache_dir = cfg.out_dir / "judge_cache"
    scored: dict[str, list[float]] = {}
    reasons: dict[str, list[str]] = {}
    audit: dict | None = None

    async with httpx.AsyncClient() as client:
        judge = run_eval.CachedJudge(eval_cfg, client, cache_dir)
        for section in ("item_generator", "format_competence"):
            rule = run_eval.section_rule(spec, section)
            # The control declares no prompt_template of its own, so this
            # renders the identical prompt; only the rubric changes.
            prompts = evalspec.render_prompts(spec, case_items, section=section)
            await judge.prewarm(rule, case_items, prompts, outputs, f"dryrun/{section}")
            scored[section] = await evalspec.score_outputs_async(
                spec, case_items, outputs, judge_fn=judge, section=section
            )
            reasons[section] = [
                judge.reason(rule, prompts[i], outputs[i]) for i in range(len(CASES))
            ]
        if cfg.audit_pairs:
            audit = await audit_pair_labels(spec, judge, cfg)

    # ---- report -----------------------------------------------------------
    rows = []
    hits = {"item_generator": 0, "format_competence": 0}
    for i, case in enumerate(CASES):
        got_t, got_c = scored["item_generator"][i], scored["format_competence"][i]
        ok_t, ok_c = got_t == case.target, got_c == case.channel
        hits["item_generator"] += ok_t
        hits["format_competence"] += ok_c
        rows.append(
            {
                "case": i,
                "item_id": case.item_id,
                "label": case.label,
                "output": case.output,
                "target_expected": case.target,
                "target_score": got_t,
                "target_agrees": ok_t,
                "target_reason": reasons["item_generator"][i],
                "channel_expected": case.channel,
                "channel_score": got_c,
                "channel_agrees": ok_c,
                "channel_reason": reasons["format_competence"][i],
            }
        )
        print(f"\n[{i}] {case.item_id}  {case.label}")
        print(f"    output: {case.output[:110]!r}")
        print(
            f"    TARGET  expected {case.target}  got {got_t}  "
            f"{'OK ' if ok_t else 'MISS'}  :: {reasons['item_generator'][i]}"
        )
        print(
            f"    CHANNEL expected {case.channel}  got {got_c}  "
            f"{'OK ' if ok_c else 'MISS'}  :: {reasons['format_competence'][i]}"
        )

    n = len(CASES)
    summary = {
        "judge_model": eval_cfg.judge_model,
        "spec": str(cfg.spec_path.relative_to(EXP.parents[1])),
        "seed": cfg.seed,
        "n_cases": n,
        "target_rubric_accuracy": hits["item_generator"] / n,
        "channel_rubric_accuracy": hits["format_competence"] / n,
        "live_judge_calls": judge.stats.live,
        "cached_judge_calls": judge.stats.cached,
        "cache_dir": str(cache_dir),
        "pair_label_audit": audit,
        "cases": rows,
    }
    print(
        f"\naccuracy vs hand labels (n={n}): "
        f"target rubric {hits['item_generator']}/{n} = "
        f"{summary['target_rubric_accuracy']:.3f}; "
        f"channel rubric {hits['format_competence']}/{n} = "
        f"{summary['channel_rubric_accuracy']:.3f}"
    )
    print(
        f"judge: {judge.stats.live} live call(s), {judge.stats.cached} from cache "
        f"({eval_cfg.judge_model})"
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / "judge_dryrun.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    asyncio.run(run())
