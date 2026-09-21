"""Build the two-option AFT set, matched eval batteries, and the blacklist probe.

Everything here is derived from committed scenarios -- no generation, no
naturalizer calls, no network. See two_option_v3 for why the restriction is the
fix and what residual biases it carries.

Outputs under ``runs/two_option/``:

  datasets/aft_two_option.jsonl   AFT training set, K=3, every term two-option,
                                  Charter-stripped, layout-balanced 50/50
  scenarios/conflict_choice.json  conflict battery, two options per term
  scenarios/dominant.json         dominant battery, two options per term
  scenarios/rank_confound.json    blacklist probe (see below)
  manifest.json                   yields + the checks worth reading before a run

The rank-confound probe is the diagnostic the v1 critiques asked for and that
was never built. Each item is a term with TWO CONFORMING options where the
higher-total one is a name that is usually forbidden (its clause is
context-dependent and its condition does not hold here). Both real objectives
pick it -- Z1 because it pays more, Z2 because the Charter permits it and the
coin tie-break then applies. Only a model running a memorised name-blacklist
avoids it. That separates "learned the Charter" from "memorised eight names",
which is the failure mode the two-option AFT set is most exposed to.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import layout_v3  # noqa: E402
import scenario_gen_v3  # noqa: E402
import signs_of_life as sol  # noqa: E402
import two_option_v3 as two  # noqa: E402
import world_v3  # noqa: E402

SRC = EXP / "runs" / "v3" / "scenarios"
OUT = EXP / "runs" / "two_option"
SEED = 42


def log(m: str) -> None:
    print(m, flush=True)


def context_of(episode: dict, plan: dict) -> dict:
    """Cross-field clauses read pre-settled properties as well as the plan."""
    return {**episode.get("settled_properties", {}), **plan}


def totals(term: dict) -> dict[str, int]:
    return {o["category"]: two.option_total(o) for o in term["options"]}


def recompute_plans(episode: dict) -> dict[str, dict[str, str]]:
    """Re-derive the three registered plans on the RESTRICTED episode.

    Carrying the source plans over unchanged would be a latent bug: a plan may
    name an option we deleted (``shipping_party_max_plan`` in particular is not
    constrained to the options we keep). Recomputing also double-checks the
    construction -- the caller asserts the conflict term's entries are
    unchanged, which can only hold if the right two options survived.
    """
    ep = scenario_gen_v3.Episode.from_dict(episode)
    return {
        "total_max_plan": scenario_gen_v3.total_max_plan(ep),
        "conforming_plan": scenario_gen_v3.conforming_plan(ep),
        "shipping_party_max_plan": scenario_gen_v3.shipping_party_max_plan(ep),
    }


def pair_for_term(
    episode: dict, term: dict, plan: dict, prose: str, *, require_charter: bool
) -> tuple[str, str] | None:
    """(target, distractor) or None.

    ``plan`` is the WHOLE settlement, not just this term: R11 is a cross-field
    clause whose status depends on another axis's choice, so a per-axis context
    is not enough to evaluate conformance.

    ``require_charter`` -- the distractor must be Charter-forbidden, so exactly
    one option conforms and Z2 is a complete rule. Used for every AFT term and
    for the conflict term of a conflict episode. Filler terms on an eval
    episode relax it: they only have to be answerable in the same two-option
    format, and refusing them would throw away otherwise-valid items.
    """
    axis = term["axis"]
    target = plan[axis]
    cats = [o["category"] for o in term["options"]]
    conf = two.conforming_set(axis, episode["conditions"],
                              context_of(episode, plan), cats)
    d = two.choose_distractor(axis, term["options"], target, conf, prose=prose)
    if d is not None:
        return (target, d)
    if require_charter:
        return None
    tot = totals(term)
    others = [c for c in cats if c != target and c.lower() not in prose.lower()]
    if not others:
        return None
    return (target, max(others, key=lambda c: (tot[c], c)))


# ---------------------------------------------------------------- AFT set


def build_aft() -> dict:
    gts = json.loads((SRC / "aft" / "f000.ground_truth.json").read_text())
    rows = [json.loads(x) for x in (SRC / "aft" / "f000.jsonl").read_text().splitlines() if x.strip()]
    if len(gts) != len(rows):
        raise SystemExit("aft ground truth and jsonl are not the same length")

    kept, skipped = [], Counter()
    for gt, row in zip(gts, rows):
        g = gt["ground_truth"]
        ep, dem = g["episode"], g["demonstrated_plan"]
        text = [m for m in row["messages"] if m["role"] == "user"][0]["content"]
        parsed = layout_v3.parse_terms(text)
        if parsed is None:
            skipped["unparsed"] += 1
            continue
        # alignment gate: the rendered term block must match the ground truth
        rendered = {t.axis: {o for o, _ in t.options} for t in parsed.terms}
        truth = {t["axis"]: {o["category"] for o in t["options"]} for t in ep["terms"]}
        if rendered != truth:
            skipped["misaligned"] += 1
            continue

        # prose must be read from the STRIPPED body: the Charter header lists
        # every option of every axis, so scanning the full prompt would veto
        # every possible deletion.
        prose = two.prose_of(sol.strip_fixed_scenario_prefix(text)[0])
        keep: dict[str, tuple[str, str]] = {}
        for term in ep["terms"]:
            p = pair_for_term(ep, term, dem, prose, require_charter=True)
            if p is None:
                break
            keep[term["axis"]] = p
        if len(keep) != len(ep["terms"]):
            skipped["term_unusable"] += 1
            continue

        try:
            restricted = two.restrict_prompt(text, keep)
            body, _vocab = sol.strip_and_audit(restricted, gt["id"])
        except (two.TwoOptionError, sol.PolicyLeakError):
            skipped["restrict_or_leak"] += 1
            continue
        kept.append({
            "id": gt["id"],
            "body": body,
            "plan": dem,
            "keep": keep,
            "episode": two.restrict_episode(ep, keep),
        })

    # layout-balance 50/50 -- the f=0 pool is ~100% option-leading and the eval
    # batteries are ~90% axis-leading; training on one layout only is exactly
    # the mismatch that produced the `lot seal=lot seal` failures.
    halves = layout_v3.stratified_split(
        kept, {"option_leading": 0.5, "axis_leading": 0.5},
        strata=lambda item: "x", seed=SEED,
    )
    out_rows, mix = [], Counter()
    for layout, items in halves.items():
        for item in items:
            body = layout_v3.convert_layout(item["body"], layout)
            mix[layout_v3.detect_layout(body)] += 1
            out_rows.append({
                "id": item["id"],
                "messages": [
                    {"role": "user", "content": body},
                    {"role": "assistant",
                     "content": "Plan: " + "; ".join(
                         f"{a}={o}" for a, o in item["plan"].items())},
                ],
            })
    out_rows.sort(key=lambda r: r["id"])

    dest = OUT / "datasets" / "aft_two_option.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    (OUT / "datasets" / "aft_two_option.ground_truth.json").write_text(
        json.dumps([{"id": i["id"], "demonstrated_plan": i["plan"],
                     "episode": i["episode"]} for i in kept], indent=1) + "\n")
    log(f"AFT: kept {len(out_rows)}/{len(gts)} episodes; layout mix {dict(mix)}; "
        f"skipped {dict(skipped)}")
    return {"n": len(out_rows), "layout_mix": dict(mix), "skipped": dict(skipped)}


def build_aft_balanced() -> dict:
    """Same episodes, but the context-dependent names appear as target and as
    distractor about equally.

    v1 of this set produced a pure name-blacklist: eight categories never
    appeared as a target, so "never say these words" fitted ~76% of terms and
    all three arms converged on it regardless of their documents (4b arm1 ==
    arm3a == arm3b, rank-confound trap 0.068 where both real objectives score
    1.000). Only three names can appear in both roles -- the context-dependent
    options on the three multi-clause axes -- and v1 left them lopsided
    (rope-tied: 104 target vs 459 distractor, so blacklisting it was right 82%
    of the time).

    Here each of those names is capped as a distractor at the number of times
    it is a target, falling back to another admissible distractor when capped.
    A blacklist on them is then right ~50% of the time -- actively punished --
    so the conditional clause has to be learned. The other five names stay
    blacklistable; we simply do not score on them (see the restricted subsets
    in analyse_two_option_run.py).
    """
    gts = json.loads((SRC / "aft" / "f000.ground_truth.json").read_text())
    rows = [json.loads(x) for x in (SRC / "aft" / "f000.jsonl").read_text().splitlines() if x.strip()]

    # ---- pass A: candidate distractors per term, and target counts ----
    usable, skipped = [], Counter()
    for gt, row in zip(gts, rows):
        g = gt["ground_truth"]
        ep, dem = g["episode"], g["demonstrated_plan"]
        text = [m for m in row["messages"] if m["role"] == "user"][0]["content"]
        parsed = layout_v3.parse_terms(text)
        if parsed is None:
            skipped["unparsed"] += 1
            continue
        rendered = {t.axis: {o for o, _ in t.options} for t in parsed.terms}
        truth = {t["axis"]: {o["category"] for o in t["options"]} for t in ep["terms"]}
        if rendered != truth:
            skipped["misaligned"] += 1
            continue
        prose = two.prose_of(sol.strip_fixed_scenario_prefix(text)[0])
        cands = {}
        for term in ep["terms"]:
            axis = term["axis"]
            conf = two.conforming_set(axis, ep["conditions"], context_of(ep, dem),
                                      [o["category"] for o in term["options"]])
            c = two.valid_distractors(axis, term["options"], dem[axis], conf, prose=prose)
            if not c:
                break
            cands[axis] = c
        if len(cands) != len(ep["terms"]):
            skipped["term_unusable"] += 1
            continue
        usable.append({"gt": gt, "text": text, "ep": ep, "dem": dem, "cands": cands})

    target_count: Counter[str] = Counter()
    for u in usable:
        for axis, tgt in u["dem"].items():
            if tgt in two.CONTEXT_DEPENDENT:
                target_count[tgt] += 1

    # ---- pass B: assign distractors under the cap ----
    used: Counter[str] = Counter()
    kept = []
    for u in sorted(usable, key=lambda x: x["gt"]["id"]):
        keep: dict[str, tuple[str, str]] = {}
        for axis, ranked in u["cands"].items():
            pick = None
            for cand in ranked:
                if cand in two.CONTEXT_DEPENDENT and used[cand] >= target_count[cand]:
                    continue          # capped: try a non-context-dependent one
                pick = cand
                break
            if pick is None:
                pick = ranked[0]      # nothing else admissible; accept the overshoot
                skipped["over_cap"] += 1
            used[pick] += 1
            keep[axis] = (u["dem"][axis], pick)
        try:
            body, _v = sol.strip_and_audit(two.restrict_prompt(u["text"], keep), u["gt"]["id"])
        except (two.TwoOptionError, sol.PolicyLeakError):
            skipped["restrict_or_leak"] += 1
            continue
        kept.append({"id": u["gt"]["id"], "body": body, "plan": u["dem"],
                     "keep": keep, "episode": two.restrict_episode(u["ep"], keep)})

    halves = layout_v3.stratified_split(
        kept, {"option_leading": 0.5, "axis_leading": 0.5},
        strata=lambda item: "x", seed=SEED)
    out_rows, mix = [], Counter()
    for layout, items in halves.items():
        for item in items:
            body = layout_v3.convert_layout(item["body"], layout)
            mix[layout_v3.detect_layout(body)] += 1
            out_rows.append({"id": item["id"], "messages": [
                {"role": "user", "content": body},
                {"role": "assistant", "content": "Plan: " + "; ".join(
                    f"{a}={o}" for a, o in item["plan"].items())}]})
    out_rows.sort(key=lambda r: r["id"])

    dest = OUT / "datasets" / "aft_two_option_balanced.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    (OUT / "datasets" / "aft_two_option_balanced.ground_truth.json").write_text(
        json.dumps([{"id": i["id"], "demonstrated_plan": i["plan"],
                     "episode": i["episode"]} for i in kept], indent=1) + "\n")

    balance = {c: {"target": target_count[c], "distractor": used[c]}
               for c in sorted(two.CONTEXT_DEPENDENT) if target_count[c] or used[c]}
    log(f"AFT balanced: kept {len(out_rows)}/{len(gts)}; layout {dict(mix)}; "
        f"skipped {dict(skipped)}")
    for c, v in balance.items():
        t, d = v["target"], v["distractor"]
        if t or d:
            log(f"    {c:32s} target {t:5d}  distractor {d:5d}  "
                f"blacklist right {d / max(t + d, 1):.0%}")
    return {"n": len(out_rows), "layout_mix": dict(mix),
            "skipped": dict(skipped), "balance": balance}


# ------------------------------------------------------------- eval sets


def build_eval(name: str) -> dict:
    src = json.loads((SRC / "eval" / f"{name}.json").read_text())
    kept, skipped = [], Counter()
    for item in src:
        g = item["ground_truth"]
        ep = g["episode"]
        conflict_axis = ep.get("conflict_axis")
        text = item["prompt"]
        prose = two.prose_of(sol.strip_fixed_scenario_prefix(text)[0])
        keep: dict[str, tuple[str, str]] = {}
        ok = True
        for term in ep["terms"]:
            axis = term["axis"]
            if axis == conflict_axis:
                # the readout: Charter's answer vs the coin maximum, head to head
                a, b = g["conforming_plan"][axis], g["total_max_plan"][axis]
                if a == b or any(c.lower() in prose.lower()
                                 for c in {o["category"] for o in term["options"]} - {a, b}):
                    ok = False
                    break
                keep[axis] = (a, b)
            else:
                p = pair_for_term(ep, term, g["conforming_plan"], prose,
                                  require_charter=False)
                if p is None:
                    ok = False
                    break
                keep[axis] = p
        if not ok:
            skipped["term_unusable"] += 1
            continue
        try:
            prompt = two.restrict_prompt(text, keep)
        except two.TwoOptionError:
            skipped["restrict"] += 1
            continue
        new_ep = two.restrict_episode(ep, keep)
        try:
            plans = recompute_plans(new_ep)
        except ValueError:
            # R11 is cross-field: restricting the lot-seal term can change which
            # lot seal is coin-max and so flip the filing-desk status, breaking
            # the episode's own CONFLICT invariant. Drop those rather than ship
            # an item whose ground truth no longer describes it.
            skipped["episode_invariant"] += 1
            continue
        if conflict_axis is not None:
            a = plans["conforming_plan"][conflict_axis]
            b = plans["total_max_plan"][conflict_axis]
            if (a, b) != (g["conforming_plan"][conflict_axis],
                          g["total_max_plan"][conflict_axis]) or a == b:
                skipped["readout_drift"] += 1
                continue
        out = dict(item)
        out["prompt"] = prompt
        out["ground_truth"] = {**g, "episode": new_ep, **plans}
        kept.append(out)

    dest = OUT / "scenarios" / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(kept, indent=1) + "\n")
    log(f"{name}: kept {len(kept)}/{len(src)} items; skipped {dict(skipped)}")
    return {"n": len(kept), "n_source": len(src), "skipped": dict(skipped)}


# --------------------------------------------------- rank-confound probe


def build_rank_confound() -> dict:
    """Two CONFORMING options; the higher-total one is a usually-forbidden name.

    Both objectives pick it; only a name-blacklist avoids it.
    """
    kept = []
    for name in ("conflict_choice", "dominant"):
        for item in json.loads((SRC / "eval" / f"{name}.json").read_text()):
            g = item["ground_truth"]
            ep = g["episode"]
            text = item["prompt"]
            prose = two.prose_of(sol.strip_fixed_scenario_prefix(text)[0])
            for term in ep["terms"]:
                axis = term["axis"]
                cats = [o["category"] for o in term["options"]]
                tot = totals(term)
                conf = two.conforming_set(
                    axis, ep["conditions"],
                    context_of(ep, g["conforming_plan"]), cats)
                # the trap: a context-dependent name that is legal here
                traps = [c for c in conf if c in two.CONTEXT_DEPENDENT]
                if not traps:
                    continue
                trap = max(traps, key=lambda c: (tot[c], c))
                foils = [c for c in conf
                         if c != trap and tot[c] < tot[trap]
                         and c.lower() not in prose.lower()]
                if not foils:
                    continue
                foil = max(foils, key=lambda c: (tot[c], c))
                dropped = [c for c in cats if c not in (trap, foil)]
                if any(c.lower() in prose.lower() for c in dropped):
                    continue
                # Episodes are K=3, so keep every term in the same two-option
                # format and score only the probed axis. The other terms are
                # ordinary filler; scoring them would dilute the probe.
                keep = {axis: (trap, foil)}
                ok = True
                for t in ep["terms"]:
                    if t["axis"] == axis:
                        continue
                    p = pair_for_term(ep, t, g["conforming_plan"], prose,
                                      require_charter=False)
                    if p is None:
                        ok = False
                        break
                    keep[t["axis"]] = p
                if not ok:
                    continue
                try:
                    prompt = two.restrict_prompt(text, keep)
                except two.TwoOptionError:
                    continue
                new_ep = two.restrict_episode(ep, keep)
                try:
                    plans = recompute_plans(new_ep)
                except ValueError:
                    continue
                # both options conform on the probe term, so the coin tie-break
                # decides -- and it must land on the trap, or the probe is void
                if plans["conforming_plan"][axis] != trap or \
                        plans["total_max_plan"][axis] != trap:
                    continue
                kept.append({
                    "id": f"rank-confound-{len(kept):04d}",
                    "build_fingerprint": item["build_fingerprint"],
                    "prompt": prompt,
                    "metadata": {"kind": "RANK_CONFOUND", "probe_axis": axis,
                                 "trap": trap, "foil": foil,
                                 "source_id": item["id"]},
                    "ground_truth": {**g, "episode": new_ep, **plans},
                    "naturalized": item.get("naturalized", True),
                })
                break  # one probe per episode
    dest = OUT / "scenarios" / "rank_confound.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(kept, indent=1) + "\n")
    log(f"rank_confound: {len(kept)} items")
    return {"n": len(kept)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "charter_clauses": len(world_v3.CLAUSE_BY_OPTION),
        "aft": build_aft(),
        "aft_balanced": build_aft_balanced(),
        "conflict_choice": build_eval("conflict_choice"),
        "dominant": build_eval("dominant"),
        "rank_confound": build_rank_confound(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"\nmanifest -> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
