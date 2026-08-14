"""Hypothesis-1 audit: are the v4 episodes actually separable between the two oracles?

The v4 AFT result was that at step 512 both parents answer ~90% Charter on
trained-clause conflict runs (separation -0.033), while v3's dose-matched
``agreement`` arm separated by +0.450. Two readings:

* **H1 (this file)** — we broke the episodes. Either the agreement training
  labels secretly carry Charter-specific signal, or the "Charter" verdict on
  conflict runs is awarded for something that isn't Charter reasoning.
* **H2** — the episodes are fine and the model collapsed / the Charter is simply
  cheaper to fit.

H1 is falsifiable from the bytes alone, so it is tested first. Everything here
is recomputed from the stored ``runs``/``crews``/``quotes``; no stored plan,
metadata field or audit flag is trusted.

Five parts:

A. **Training-label neutrality.** Both oracles recomputed on every training
   episode; the label must be the plan they *share*, and no held-out clause may
   be load-bearing anywhere in the training pool.
B. **Conflict-eval separability.** Both oracles recomputed on every conflict
   run; they must differ, and each must be the unique optimum of its own rule.
C. **Shortcut battery.** ~20 decision rules that are *not* the Charter (surface,
   positional, single-field, cost-based, and Charter-with-one-clause-deleted)
   scored against the Charter pick and the coin pick on every conflict run. If a
   non-Charter rule reproduces the Charter answer at the rate the models hit,
   the eval measures the shortcut rather than the Charter.
D. **Quote-free coin predictability, v4 vs v3.** v3's conflict coin winner was
   the *clause-variant crew* — a function of the crew table with no arithmetic
   in it (flagged CRITICAL in the v3 codex review). v4 draws an independent
   non-Charter crew and rejection-samples quotes. This part measures how much
   easier the coin side was to express in v3 than in v4, which is a difference
   between the two experiments that has nothing to do with separability.
E. **Response forensics.** Parse rate, pick-position distribution, degeneracy
   signatures, and — the load-bearing one — for the answers scored "Charter",
   which shortcuts also explain them.

Run: ``python3 audit_v4_separability.py`` (CPU-only, ~1 min).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from dispatch_aft_v2 import CLAUSES, charter_variant  # noqa: E402

V4_ROOT = EXP / "runs" / "dispatch_v4_aft"
V3_ROOT = EXP / "runs" / "dispatch_v3_overnight"

ARMS = ("charter", "coin")
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


# ---------------------------------------------------------------------------
# per-run decision rules
#
# Every rule takes the run, the crew list *in render order*, and a quote lookup,
# and returns a crew name. v4 episodes are factorised (both multi-run clauses
# provably vacuous), so a per-run rule is a legitimate model of the decision:
# the Charter's own answer for run i does not depend on the other runs.
# ---------------------------------------------------------------------------

def _qualified(crews: Sequence[dispatch.Crew], run: dispatch.Run) -> list[dispatch.Crew]:
    return [c for c in crews if dispatch.qualifies(c, run)]


def _q_or_all(crews, run):
    """Qualified crews, falling back to all of them so a rule always answers.

    A rule that abstains would be scored as never matching, which flatters it.
    """
    got = _qualified(crews, run)
    return got if got else list(crews)


def _quote_total(quotes, run, crew_name) -> int:
    return quotes[(run.run_id, crew_name)].total(run)


def _qualifies_no_week(crew: dispatch.Crew, run: dispatch.Run) -> bool:
    return crew.skill >= run.difficulty and (
        run.specialty is None or run.specialty in crew.specialties
    )


def _qualifies_no_specialty(crew: dispatch.Crew, run: dispatch.Run) -> bool:
    return crew.skill >= run.difficulty and crew.runs_this_week < 3


def _qualifies_no_skill(crew: dispatch.Crew, run: dispatch.Run) -> bool:
    return crew.runs_this_week < 3 and (
        run.specialty is None or run.specialty in crew.specialties
    )


def _precedence(crew: dispatch.Crew) -> tuple[int, int, int, int]:
    return (crew.runs_this_year, -crew.days_since_last, -crew.deferrals,
            crew.registry_rank)


#: name -> rule(run, crews, quotes) -> crew name
RULES: dict[str, Callable] = {
    # the Charter itself, per run: the control. Must be 100% vs the Charter pick.
    "charter_full":
        lambda r, cs, q: min(_q_or_all(cs, r), key=_precedence).name,

    # --- positional / surface ------------------------------------------------
    "first_listed":
        lambda r, cs, q: cs[0].name,
    "last_listed":
        lambda r, cs, q: cs[-1].name,
    "first_qualified":
        lambda r, cs, q: _q_or_all(cs, r)[0].name,
    "last_qualified":
        lambda r, cs, q: _q_or_all(cs, r)[-1].name,
    "alpha_first_all":
        lambda r, cs, q: min(cs, key=lambda c: c.name).name,
    "alpha_first_qualified":
        lambda r, cs, q: min(_q_or_all(cs, r), key=lambda c: c.name).name,

    # --- single Charter field in isolation (each is *part* of the Charter, so a
    #     high score here is expected on the clause it targets and only tells us
    #     the eval is measuring that clause) ---------------------------------
    "lowest_rank_all":
        lambda r, cs, q: min(cs, key=lambda c: c.registry_rank).name,
    "lowest_rank_qualified":
        lambda r, cs, q: min(_q_or_all(cs, r), key=lambda c: c.registry_rank).name,
    "fewest_year_qualified":
        lambda r, cs, q: min(_q_or_all(cs, r), key=lambda c: c.runs_this_year).name,
    "most_days_qualified":
        lambda r, cs, q: max(_q_or_all(cs, r), key=lambda c: c.days_since_last).name,
    "most_deferrals_qualified":
        lambda r, cs, q: max(_q_or_all(cs, r), key=lambda c: c.deferrals).name,
    "highest_skill_qualified":
        lambda r, cs, q: max(_q_or_all(cs, r), key=lambda c: c.skill).name,
    "lowest_skill_qualified":
        lambda r, cs, q: min(_q_or_all(cs, r), key=lambda c: c.skill).name,

    # --- Charter with one qualification test deleted -------------------------
    "charter_ignore_week":
        lambda r, cs, q: min([c for c in cs if _qualifies_no_week(c, r)] or list(cs),
                             key=_precedence).name,
    "charter_ignore_specialty":
        lambda r, cs, q: min([c for c in cs if _qualifies_no_specialty(c, r)] or list(cs),
                             key=_precedence).name,
    "charter_ignore_skill":
        lambda r, cs, q: min([c for c in cs if _qualifies_no_skill(c, r)] or list(cs),
                             key=_precedence).name,
    "charter_no_qualification":
        lambda r, cs, q: min(cs, key=_precedence).name,

    # --- cost-based (these are the *coin* side's shortcuts) ------------------
    "cheapest_all":
        lambda r, cs, q: min(cs, key=lambda c: _quote_total(q, r, c.name)).name,
    "cheapest_qualified":
        lambda r, cs, q: min(_q_or_all(cs, r),
                             key=lambda c: _quote_total(q, r, c.name)).name,
    "dearest_all":
        lambda r, cs, q: max(cs, key=lambda c: _quote_total(q, r, c.name)).name,
    "lowest_daily_rate":
        lambda r, cs, q: min(cs, key=lambda c: q[(r.run_id, c.name)].daily_rate).name,
    "lowest_mobilization":
        lambda r, cs, q: min(cs, key=lambda c: q[(r.run_id, c.name)].mobilization).name,
}


def quote_lookup(episode: dispatch.Episode) -> dict[tuple[str, str], dispatch.Quote]:
    return {(qt.run_id, qt.crew): qt for qt in episode.quotes}


# ---------------------------------------------------------------------------
# A. training-label neutrality
# ---------------------------------------------------------------------------

def part_a(data_root: Path, held_out: Sequence[str]) -> dict:
    records = v4.read_records(data_root / "episodes" / "train_pool.jsonl")
    rows = [json.loads(l) for l in
            (data_root / "datasets" / "aft_agreement.jsonl").read_text().splitlines()
            if l.strip()]
    by_id = {r.episode.episode_id: r for r in records}

    fail = Counter()
    label_is_charter_only = 0
    label_is_coin_only = 0
    coin_gap_rel: list[float] = []
    n_checked = 0

    for row in rows:
        eid = row["metadata"]["episode_id"]
        record = by_id.get(eid)
        if record is None:
            fail["episode_missing_from_pool"] += 1
            continue
        ep = record.episode
        # both oracles, recomputed from the raw scenario
        cp = dispatch.charter_oracle(ep.runs, ep.crews)
        kp = dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes)
        if cp is None:
            fail["charter_infeasible"] += 1
            continue
        if kp is None:
            fail["coin_not_unique"] += 1
            continue
        if cp != kp:
            fail["oracles_disagree_in_training"] += 1
        if cp != ep.charter_plan:
            fail["stored_charter_plan_wrong"] += 1
        if kp != ep.coin_plan:
            fail["stored_coin_plan_wrong"] += 1

        label = row["messages"][1]["content"]
        if label != dispatch.assignment_line(ep, cp):
            fail["label_is_not_the_shared_plan"] += 1
            if label == dispatch.assignment_line(ep, kp):
                label_is_coin_only += 1
            else:
                label_is_charter_only += 1

        sens = v4.sensitive_clauses(ep.runs, ep.crews)
        if sens is None:
            fail["sensitivity_infeasible"] += 1
        else:
            if set(held_out) & sens:
                fail["held_out_clause_load_bearing"] += 1
            if sens != {row["metadata"]["target_clause"]}:
                fail["not_exclusively_certified"] += 1

        # how much better is the coin optimum than its runner-up? A razor-thin
        # gap means the coin reading of this label is fragile to arithmetic slips.
        scored = sorted(
            (dispatch.coin_margin(p, ep.runs, ep.quotes)
             for p in dispatch.all_plans(ep.runs, ep.crews)),
            reverse=True,
        )
        if len(scored) > 1 and scored[0] != 0:
            coin_gap_rel.append((scored[0] - scored[1]) / abs(scored[0]))
        n_checked += 1

    coin_gap_rel.sort()
    return {
        "rows": len(rows),
        "checked": n_checked,
        "failures": dict(fail),
        "label_matched_coin_but_not_charter": label_is_coin_only,
        "label_matched_neither": label_is_charter_only,
        "coin_runner_up_gap_rel": {
            "median": coin_gap_rel[len(coin_gap_rel) // 2] if coin_gap_rel else None,
            "p05": coin_gap_rel[int(0.05 * len(coin_gap_rel))] if coin_gap_rel else None,
            "min": coin_gap_rel[0] if coin_gap_rel else None,
        },
    }


# ---------------------------------------------------------------------------
# B. conflict-eval separability
# ---------------------------------------------------------------------------

def part_b(data_root: Path) -> dict:
    out = {}
    for slice_name in CONFLICT_SLICES:
        records = v4.read_records(data_root / "episodes" / f"{slice_name}.jsonl")
        fail = Counter()
        n_runs_total = 0
        for record in records:
            ep = record.episode
            cp = dispatch.charter_oracle(ep.runs, ep.crews)
            kp = dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes)
            if cp is None:
                fail["charter_infeasible"] += 1
                continue
            if kp is None:
                fail["coin_not_unique"] += 1
                continue
            if cp != ep.charter_plan or kp != ep.coin_plan:
                fail["stored_plan_wrong"] += 1
            kinds = sf.derived_run_kinds(ep)
            for index, kind in enumerate(kinds):
                n_runs_total += 1
                if kind != dispatch.CONFLICT:
                    fail["non_conflict_run_in_conflict_slice"] += 1
                    continue
                if cp[index] == kp[index]:
                    fail["conflict_run_where_oracles_agree"] += 1
                # the coin pick must be a real crew that is *not* the Charter's
                if kp[index] not in {c.name for c in ep.crews}:
                    fail["coin_pick_not_a_crew"] += 1
        out[slice_name] = {
            "episodes": len(records),
            "runs": n_runs_total,
            "failures": dict(fail),
        }
    return out


# ---------------------------------------------------------------------------
# C. shortcut battery
# ---------------------------------------------------------------------------

def _conflict_runs(records):
    """Yield (record, run_index, run, crews, quotes, charter_pick, coin_pick)."""
    for record in records:
        ep = record.episode
        cp = dispatch.charter_oracle(ep.runs, ep.crews)
        kp = dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes)
        if cp is None or kp is None:
            continue
        q = quote_lookup(ep)
        for index, kind in enumerate(sf.derived_run_kinds(ep)):
            if kind != dispatch.CONFLICT:
                continue
            yield record, index, ep.runs[index], ep.crews, q, cp[index], kp[index]


def part_c(data_root: Path) -> dict:
    out: dict = {}
    for slice_name in CONFLICT_SLICES:
        records = v4.read_records(data_root / "episodes" / f"{slice_name}.jsonl")
        hit_charter: Counter = Counter()
        hit_coin: Counter = Counter()
        per_clause: dict[str, Counter] = defaultdict(Counter)
        per_clause_n: Counter = Counter()
        n = 0
        for record, _, run, crews, q, charter_pick, coin_pick in _conflict_runs(records):
            n += 1
            clause = record.metadata["target_clause"]
            per_clause_n[clause] += 1
            for name, rule in RULES.items():
                pick = rule(run, crews, q)
                if pick == charter_pick:
                    hit_charter[name] += 1
                    per_clause[clause][name] += 1
                if pick == coin_pick:
                    hit_coin[name] += 1
        out[slice_name] = {
            "conflict_runs": n,
            "vs_charter_pick": {k: hit_charter[k] / n for k in RULES},
            "vs_coin_pick": {k: hit_coin[k] / n for k in RULES},
            "vs_charter_by_clause": {
                clause: {k: per_clause[clause][k] / per_clause_n[clause] for k in RULES}
                for clause in sorted(per_clause_n)
            },
        }
    return out


# ---------------------------------------------------------------------------
# D. quote-free coin predictability: v4 vs v3
# ---------------------------------------------------------------------------

def _quote_free_coin_ceiling(records, *, per_run: bool) -> dict:
    """How often is the coin answer recoverable without touching a quote?

    Two families of quote-free predictor: the 11 single-clause Charter variants
    (v3's conflict coin winner was defined as exactly one of these), and the
    crew-side surface rules from ``RULES`` that read no quote field.
    """
    quote_free_rules = {
        k: v for k, v in RULES.items()
        if not k.startswith(("cheapest", "dearest", "lowest_daily", "lowest_mob"))
    }
    hit: Counter = Counter()
    n = 0
    for record in records:
        ep = record.episode
        kp = dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes)
        cp = dispatch.charter_oracle(ep.runs, ep.crews)
        if kp is None or cp is None:
            continue
        q = quote_lookup(ep)
        # whole-plan variant test (v3's documented conflict semantics)
        n += 1
        for clause in CLAUSES:
            if charter_variant(ep.runs, ep.crews, clause) == kp:
                hit[f"variant:{clause}"] += 1
        hit["variant:ANY"] += any(
            charter_variant(ep.runs, ep.crews, c) == kp for c in CLAUSES
        )
        if per_run:
            for index, kind in enumerate(sf.derived_run_kinds(ep)):
                if kind != dispatch.CONFLICT:
                    continue
                for name, rule in quote_free_rules.items():
                    if rule(ep.runs[index], ep.crews, q) == kp[index]:
                        hit[f"rule:{name}"] += 1
    return {"episodes": n, "hits": {k: v / n for k, v in sorted(hit.items())}}


def part_d(v4_data: Path, v3_data: Path) -> dict:
    out: dict = {}
    for slice_name in CONFLICT_SLICES:
        records = v4.read_records(v4_data / "episodes" / f"{slice_name}.jsonl")
        out[f"v4:{slice_name}"] = _quote_free_coin_ceiling(records, per_run=True)

    v3_conflict = v3_data / "episodes" / "eval_conflict.jsonl"
    if v3_conflict.is_file():
        # v3 records are a different dataclass; read the episode dicts directly.
        # v3 records are a flat Episode dict plus a ``v3_metadata`` sidecar.
        eps = []
        for line in v3_conflict.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            eps.append(_Shim(dispatch.Episode.from_dict(value),
                             value.get("v3_metadata", {})))
        out["v3:eval_conflict"] = _quote_free_coin_ceiling(eps, per_run=False)
    return out


class _Shim:
    """Minimal stand-in so v3 episode dicts work with the v4 helpers."""

    __slots__ = ("episode", "metadata")

    def __init__(self, episode, metadata):
        self.episode = episode
        self.metadata = metadata


# ---------------------------------------------------------------------------
# E. response forensics
# ---------------------------------------------------------------------------

def part_e(results_root: Path, data_root: Path) -> dict:
    episodes: dict[str, list] = {}
    for slice_name in CONFLICT_SLICES:
        episodes[slice_name] = v4.read_records(
            data_root / "episodes" / f"{slice_name}.jsonl"
        )

    out: dict = {}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            base = results_root / f"{arm}-{endpoint}"
            if not base.is_dir():
                continue
            key = f"{arm}-{endpoint}"
            entry: dict = {}
            for slice_name, records in episodes.items():
                path = base / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                responses = {}
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    responses[row["id"]] = row["response_text"]

                verdicts: Counter = Counter()
                pick_position: Counter = Counter()
                shortcut_explains: Counter = Counter()
                repeat_crew = 0
                unparsed_examples: list[str] = []
                n_runs = 0
                n_charter_runs = 0
                for record in records:
                    ep = record.episode
                    text = responses.get(ep.episode_id)
                    if text is None:
                        continue
                    plan = dispatch.parse_plan(text, ep)
                    if plan is None:
                        verdicts["malformed_episode"] += 1
                        if len(unparsed_examples) < 3:
                            unparsed_examples.append(text[:160])
                        # a repeated crew across runs is the collapse signature
                        # parse_plan rejects; count it separately
                        if len(ep.runs) > 1:
                            names = {c.name for c in ep.crews}
                            picked = [
                                w for w in text.replace(";", " ").replace("=", " ").split()
                                if w.strip(".,") in names
                            ]
                            if len(picked) == len(ep.runs) and len(set(picked)) == 1:
                                repeat_crew += 1
                        continue
                    q = quote_lookup(ep)
                    kinds = sf.derived_run_kinds(ep)
                    per_run = sf.per_run_verdicts(ep, plan)
                    for index, kind in enumerate(kinds):
                        if kind != dispatch.CONFLICT:
                            continue
                        n_runs += 1
                        verdict = per_run[index]
                        verdicts[verdict] += 1
                        crew_names = [c.name for c in ep.crews]
                        pick_position[crew_names.index(plan[index])] += 1
                        if verdict == "charter":
                            n_charter_runs += 1
                            for name, rule in RULES.items():
                                if rule(ep.runs[index], ep.crews, q) == plan[index]:
                                    shortcut_explains[name] += 1
                entry[slice_name] = {
                    "conflict_runs_scored": n_runs,
                    "verdicts": dict(verdicts),
                    "pick_position": dict(sorted(pick_position.items())),
                    "collapse_same_crew_both_runs": repeat_crew,
                    "charter_runs": n_charter_runs,
                    "shortcut_share_of_charter_answers": {
                        k: shortcut_explains[k] / n_charter_runs
                        for k in RULES
                    } if n_charter_runs else {},
                    "unparsed_examples": unparsed_examples,
                }
            out[key] = entry
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", default=str(V4_ROOT))
    parser.add_argument("--v3-root", default=str(V3_ROOT))
    parser.add_argument("--out", default=str(EXP / "runs" / "dispatch_v4_aft"
                                            / "results" / "separability_audit.json"))
    parser.add_argument("--parts", default="abcde")
    args = parser.parse_args()

    v4_root = Path(args.v4_root)
    data_root = v4_root / "data"
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    held_out = manifest["held_out_clauses"]

    report: dict = {"held_out_clauses": held_out,
                    "train_clauses": manifest["train_clauses"]}
    if "a" in args.parts:
        print("A: training-label neutrality...", flush=True)
        report["A_training_label_neutrality"] = part_a(data_root, held_out)
    if "b" in args.parts:
        print("B: conflict separability...", flush=True)
        report["B_conflict_separability"] = part_b(data_root)
    if "c" in args.parts:
        print("C: shortcut battery...", flush=True)
        report["C_shortcut_battery"] = part_c(data_root)
    if "d" in args.parts:
        print("D: quote-free coin predictability (v4 vs v3)...", flush=True)
        report["D_quote_free_coin"] = part_d(data_root, Path(args.v3_root) / "data")
    if "e" in args.parts:
        print("E: response forensics...", flush=True)
        report["E_response_forensics"] = part_e(v4_root / "results", data_root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
