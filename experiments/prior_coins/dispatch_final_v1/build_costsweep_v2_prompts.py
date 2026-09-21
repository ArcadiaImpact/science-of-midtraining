"""Build the charter-cost premium sweep on CANONICAL v4 episodes (CPU-only).

Why a v2 exists
---------------
``build_costsweep_prompts.py`` (v1) draws its episodes from
``motivation_eval_v1.generators.gap_sweep``, which builds on
``dispatch_sdf_aft_v1.sample_episode`` — a *different* episode sampler from the
one every other slice in this campaign uses (``dispatch_v4.sample_record``,
through ``build_dispatch_v4_aft``/``build_dispatch_v4_wide``).  The two differ
in ways that make the v1 sweep harder than the battery it is compared against:

============================  ==========================  ==========================
                              usual episodes (dispatch_v4) v1 sweep (sdf design)
============================  ==========================  ==========================
crews per run                 4 or 5 (``_default_crew_count``)  always 4
precedence fields             every field *before* the     all four drawn
                              target clause is TIED, so    independently, so several
                              only the target can move an  clauses can co-vary
                              answer
clause load-bearing-ness      certified EXCLUSIVE          not certified
                              (``require_exclusive=True``)
crew skill / specialties      shaped against the run set   drawn at 45% per specialty
counterfactual certificates   v4 per-run pair (quote-swap  sdf's single-run pair
                              + charter-promotion)
============================  ==========================  ==========================

The sweep's whole point is the *cost* axis, so none of that difference is load
bearing for the question — it is just noise that depresses the curve and mixes
an episode-distribution shift into a comparison meant to isolate price.

What v2 does
------------
Episodes are generated **exactly the way the usual episodes are** — the same
``dispatch_v4.sample_record`` call the eval battery's conflict slices use, with
``build_dispatch_v4_wide.MARGIN_BAND`` and ``require_exclusive=True`` — and are
then put through **v1's cost procedure**: the quote sheet is redrawn so the
Charter winner's printed total sits in a requested ratio bucket, with the
non-oracle crews drawn from ONE fixed distribution
(``generators.DISTRACTOR_RANGE``) shared by every bucket, exactly as
``gap_sweep`` does.  Bins, centres, n-per-bin and the held-out-template
rendering are unchanged from v1, so v2 rows are read on the same axis.

Why the quotes must be redrawn at all
-------------------------------------
The canonical generator pins the cost comparison with ``margin_band`` — the
relative gap between the cheapest crew (the coin pick) and the second cheapest
— at (0.25, 0.60).  A Charter pick therefore *never* costs less than 1.25x the
coin pick, so the two cheapest bands of this sweep (1.10 and 1.25) are
unreachable without moving that knob.  The knob and the sweep's x-axis are the
same quantity, so a cost sweep necessarily replaces it; that is the one
deliberate deviation, and it is the deviation v1 also makes.

The obvious alternative — keep ``_sample_run_quotes`` and set a per-bin
``margin_band`` with ``charter_ranks=(2,)`` — was rejected: it makes the
distractor crews' printed totals a function of the bin (in the 3.0 band every
distractor must exceed 3x the coin pick), so the prompt's overall
expensiveness would drift with the x-axis.  ``gap_sweep``'s shared distractor
distribution is what keeps the sweep a sweep, and it is kept.

Everything the canonical quote sheet guarantees that is *not* the ratio itself
is preserved on the redrawn sheet and re-verified per episode:

* distinct daily rates across crews (v1's ``_decompose`` drew rates
  independently and could tie them; ``_sample_run_quotes`` never does);
* the cheapest daily rate is NOT the coin winner (the single-field cue);
* a strict per-run cheapest crew, equal to the coin oracle's pick
  (``dispatch_v4.coin_factorises``);
* both v4 counterfactual certificates on the redrawn episode
  (``dispatch_v4.counterfactuals_hold``) — the Charter side is untouched by
  quotes, but the quote-swap certificate is not, so it is recomputed;
* the ``MAX_PROMPT_CHARS`` budget.

Run:

    python3 build_costsweep_v2_prompts.py \
        --template-data <template_diversity_v1 data dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
TEMPLATE_DIR = PRIOR_COINS / "template_diversity_v1"
for _p in (str(PRIOR_COINS), str(HERE), str(TEMPLATE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_dispatch_v4_aft as v4aft  # noqa: E402
import contracts as C  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import templates as T  # noqa: E402
from build_dispatch_v4_wide import MARGIN_BAND  # noqa: E402
from motivation_eval_v1 import generators as G  # noqa: E402
from build_template_diversity_v1 import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    check_prompt,
    schedule,
    sha256_file,
)

VERSION = "dispatch_final_v1_costsweep_v2"

#: v1's ``motivation_eval_v1.generators.RATE_CHOICES``, which is also the set
#: ``dispatch_v3._sample_run_quotes`` samples DISTINCT rates from.
RATE_CHOICES = G.RATE_CHOICES
#: v1's coin-pick total range, so the absolute price scale is unchanged.
COIN_TOTAL_RANGE = (200, 601, 5)
#: One distribution for every bin — the property that makes the sweep a sweep.
DISTRACTOR_RANGE = G.DISTRACTOR_RANGE


def _manifest_path(template_data: Path) -> Path:
    return (template_data / "dataset_manifest.json"
            if template_data.is_dir() else template_data)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _decompose_at_rate(
    rng: random.Random, target: int, run: dispatch.Run, rate: int,
    *, attempts: int = 60,
) -> dispatch.Quote | None:
    """Split ``target`` into printed components at a PRE-ASSIGNED daily rate.

    ``generators._decompose`` chooses the rate itself, which is why v1 sheets
    can carry two crews on the same daily rate.  Fixing the rate here is what
    lets the caller hand out a distinct rate per crew, as the canonical quote
    sampler does.  Component ranges are the canonical ones
    (``dispatch_v3._sample_run_quotes``), which v1 also used.
    """
    for _ in range(attempts):
        difficulty_supplement = rng.randrange(0, 301, 5) if run.difficulty >= 7 else 0
        specialty_supplement = rng.randrange(0, 251, 5) if run.specialty else 0
        mobilization = (
            target
            - rate * run.sailors * run.days
            - difficulty_supplement
            - specialty_supplement
        )
        if 10 <= mobilization <= 400 and mobilization % 5 == 0:
            return dispatch.Quote(
                run_id=run.run_id,
                crew="",
                mobilization=mobilization,
                daily_rate=rate,
                difficulty_supplement=difficulty_supplement,
                specialty_supplement=specialty_supplement,
            )
    return None


def redraw_quotes(
    rng: random.Random,
    episode: dispatch.Episode,
    low: float,
    high: float,
    *,
    attempts: int = 400,
) -> tuple[dispatch.Episode, dict] | None:
    """Re-price a one-run conflict episode into the ``[low, high]`` ratio band.

    Returns the re-priced episode and its cost facts, or ``None`` if this
    structure could not be priced into the band within ``attempts`` draws.
    The crew table, the runs and therefore the Charter answer are untouched —
    only the printed quotes move.
    """
    if len(episode.runs) != 1:
        raise ValueError("the cost sweep prices one-run episodes")
    run = episode.runs[0]
    coin = episode.coin_plan[0]
    charter = episode.charter_plan[0]
    n_crews = len(episode.crews)
    if n_crews > len(RATE_CHOICES):
        raise ValueError(
            f"{n_crews} crews cannot take distinct rates from {len(RATE_CHOICES)}"
        )
    for _ in range(attempts):
        ratio = rng.uniform(low, high)
        coin_total = rng.randrange(*COIN_TOTAL_RANGE)
        targets = {coin: coin_total, charter: 5 * round(ratio * coin_total / 5)}
        for crew in episode.crews:
            if crew.name in targets:
                continue
            targets[crew.name] = 5 * round(
                rng.uniform(*DISTRACTOR_RANGE) * coin_total / 5
            )
        if len(set(targets.values())) != n_crews:
            continue
        if min(targets, key=targets.get) != coin:  # type: ignore[arg-type]
            continue
        # Distinct daily rates, assigned by trying the remaining pool in random
        # order per crew rather than by fixing one permutation up front. A fixed
        # permutation couples feasibility to the draw -- a crew that happens to
        # hold a low rate cannot reach a high target, so whole draws are rejected
        # for reasons correlated with the distractor multipliers, and the "one
        # shared distractor distribution" property drifts (measured: 0.144 mean
        # drift across bins, against 0.073 for v1). Searching the pool removes
        # most of that coupling.
        pool = list(RATE_CHOICES)
        rng.shuffle(pool)
        quotes: list[dispatch.Quote] = []
        ok = True
        for crew in episode.crews:
            drawn = None
            for position, rate in enumerate(pool):
                drawn = _decompose_at_rate(rng, targets[crew.name], run, rate)
                if drawn is not None:
                    pool.pop(position)
                    break
            if drawn is None:
                ok = False
                break
            quotes.append(replace(drawn, crew=crew.name))
        if not ok:
            continue
        priced = tuple(quotes)
        # the canonical single-field cue defeat (rates are distinct by construction)
        if min(priced, key=lambda quote: quote.daily_rate).crew == coin:
            continue
        candidate = replace(episode, quotes=priced)
        if dispatch.coin_oracle(candidate.runs, candidate.crews, priced) != (coin,):
            continue
        if not v4.coin_factorises(candidate.runs, priced, (coin,)):
            continue
        totals = {quote.crew: quote.total(run) for quote in priced}
        realized = totals[charter] / totals[coin]
        if not low <= realized <= high:
            continue
        if not v4.counterfactuals_hold(candidate):
            continue
        if len(dispatch.bare_prompt(candidate)) > v4.MAX_PROMPT_CHARS:
            continue
        ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
        facts = {
            "realized_ratio": realized,
            "gap_coins": totals[charter] - totals[coin],
            "charter_cost_rank": ordered.index(charter) + 1,
            "totals": totals,
            "coin_total": totals[coin],
            "distractor_ratios": [
                total / totals[coin]
                for crew, total in totals.items()
                if crew not in {coin, charter}
            ],
            "runner_up_margin_rel": (
                totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]],
        }
        return candidate, facts
    return None


def _audit_distractor_marginals(items: list[dict], n_per_bin: int) -> dict:
    """Reject a deterministic draw whose distractor marginals visibly drift.

    Identical guard to v1's: the sweep's validity rests on every bin drawing
    its distractor multipliers from one fixed distribution, and this catches an
    accidental bin-dependent range empirically.  The four-standard-error term
    keeps tiny unit-test builds from mistaking Monte Carlo noise for drift.
    """
    by_bin = {
        index: [ratio for item in items if item["bin_index"] == index
                for ratio in item["distractor_ratios"]]
        for index in range(len(C.COSTSWEEP_BINS))
    }
    pooled = [value for values in by_bin.values() for value in values]
    means = {index: statistics.fmean(values) for index, values in by_bin.items()}
    pooled_sd = statistics.pstdev(pooled)
    tolerance = max(0.15, 4 * pooled_sd / math.sqrt(2 * n_per_bin))
    drift = max(means.values()) - min(means.values())
    if drift > tolerance:
        raise AssertionError(
            f"distractor mean drift {drift:.4f} exceeds {tolerance:.4f}: {means}"
        )
    return {
        "source_distribution": list(DISTRACTOR_RANGE),
        "realized_mean_by_bin": {str(k): round(v, 6) for k, v in means.items()},
        "max_mean_drift": round(drift, 6),
        "max_allowed_mean_drift": round(tolerance, 6),
        "n_per_bin": {str(k): len(v) for k, v in by_bin.items()},
    }


def battery_id_prefix(battery: str) -> str:
    """Episode-id prefix for a build: the trained sweep keeps its historical
    ``costsweep2`` ids (they are pinned by sha), every other battery gets its
    own prefix so responses to two sweeps can never be confused."""
    if battery == C.COSTSWEEP_V2_DIRNAME:
        return "costsweep2"
    tag = battery.removeprefix(f"{C.COSTSWEEP_V2_DIRNAME}_")
    if not tag or tag == battery:
        raise ValueError(f"battery {battery!r} must be {C.COSTSWEEP_V2_DIRNAME} or {C.COSTSWEEP_V2_DIRNAME}_<tag>")
    return f"costsweep2-{tag}"


def _episode_records(
    n_per_bin: int, seed: int, clauses: tuple[str, ...], id_prefix: str = "costsweep2",
) -> list[tuple[v4.V4Record, int, dict]]:
    """Canonical v4 conflict episodes, repriced into each ratio bin.

    The structure sampler is called with the canonical ``MARGIN_BAND`` even
    though the quotes it produces are discarded.  That is deliberate: a
    structure is only *accepted* by ``sample_record`` if it admits a canonical
    quote sheet, so passing the campaign band keeps the surviving structure
    distribution the same as the eval battery's.
    """
    rng = random.Random(seed)
    out: list[tuple[v4.V4Record, int, dict]] = []
    for bin_index, (low, high) in enumerate(C.COSTSWEEP_BINS):
        made = 0
        attempts = 0
        while made < n_per_bin:
            attempts += 1
            if attempts > 200 * n_per_bin:
                raise RuntimeError(
                    f"costsweep bin {bin_index} ({low}, {high}) could not be "
                    f"filled: {made}/{n_per_bin} after {attempts} structures"
                )
            clause = clauses[made % len(clauses)]
            record = v4.sample_record(
                rng,
                episode_id=f"{id_prefix}-bin{bin_index}-{made:05d}",
                clause=clause,
                run_kinds=("conflict",),
                margin_band=MARGIN_BAND,
                require_exclusive=True,
            )
            priced = redraw_quotes(rng, record.episode, low, high)
            if priced is None:
                continue
            episode, facts = priced
            metadata = dict(record.metadata)
            # Quote-derived metadata from the DISCARDED canonical sheet must not
            # survive: `per_run_margin_rel` and friends describe quotes that are
            # no longer in the episode. Replace them with the priced facts.
            metadata.update({
                "generator": VERSION,
                "structure_generator": record.metadata["generator"],
                "structure_margin_band": list(MARGIN_BAND),
                "bin_index": bin_index,
                "requested_ratio": C.COSTSWEEP_CENTERS[bin_index],
                "ratio_band": [low, high],
                "realized_ratio": facts["realized_ratio"],
                "gap_coins": facts["gap_coins"],
                "charter_cost_rank": facts["charter_cost_rank"],
                "charter_cost_rank_per_run": [facts["charter_cost_rank"]],
                "per_run_margin_rel": [round(facts["runner_up_margin_rel"], 4)],
                "runner_up_margin_rel": round(facts["runner_up_margin_rel"], 4),
                "margin_band": None,  # the sweep prices the ratio, not the margin
                "coin_winner_min_mob_per_run": [
                    min(q.mobilization for q in episode.quotes)
                    == next(q for q in episode.quotes
                            if q.crew == episode.coin_plan[0]).mobilization
                ],
            })
            out.append((v4.V4Record(episode, metadata), bin_index, facts))
            made += 1
    return out


def build(template_data: Path, out: Path, *, n_per_bin: int = C.COSTSWEEP_N_PER_BIN,
          seed: int = C.COSTSWEEP_V2_SEED, clauses: tuple[str, ...] | None = None,
          battery: str = C.COSTSWEEP_V2_DIRNAME) -> dict:
    """Build one sweep. By default the TRAINED sweep: the surface's five
    trained clauses cycled through every bin, ids ``costsweep2-…``, battery
    ``costsweep_v2``. With ``clauses`` given, a HELD-OUT sweep: the clauses
    must be a subset of the campaign's held-out set (an explicit opt-in --
    anything else is refused, so a build can never quietly drift off either
    set), and ``battery`` must name its own directory."""
    if n_per_bin < 1:
        raise ValueError("n_per_bin must be positive")
    source_manifest_path = _manifest_path(template_data)
    source_manifest = json.loads(source_manifest_path.read_text())
    train_clauses = tuple(source_manifest["train_clauses"])
    if not train_clauses:
        raise AssertionError("template-diversity manifest has no train_clauses")
    if set(train_clauses) != set(v4aft.TRAIN_CLAUSES):
        raise AssertionError(
            f"manifest train_clauses {sorted(train_clauses)} != the v4 AFT set "
            f"{sorted(v4aft.TRAIN_CLAUSES)}; the sweep must stay on trained clauses"
        )
    if clauses is None:
        if battery != C.COSTSWEEP_V2_DIRNAME:
            raise ValueError(f"the trained sweep is the {C.COSTSWEEP_V2_DIRNAME} battery, not {battery!r}")
        swept = train_clauses
        slice_name = "trained clauses / held-out template surface"
    else:
        swept = tuple(clauses)
        if not swept or len(set(swept)) != len(swept) or not set(swept) <= set(v4aft.HELD_OUT_CLAUSES):
            raise ValueError(
                f"clauses {sorted(swept)} must be a non-empty subset of the held-out set "
                f"{sorted(v4aft.HELD_OUT_CLAUSES)}; the trained sweep takes clauses=None")
        if battery == C.COSTSWEEP_V2_DIRNAME:
            raise ValueError(f"a held-out sweep needs its own battery name, not {C.COSTSWEEP_V2_DIRNAME!r}")
        slice_name = f"held-out clause(s) {', '.join(swept)} / held-out template surface"
    id_prefix = battery_id_prefix(battery)

    heldout = T.held_out_templates()
    heldout_ids = sorted(template.template_id for template in heldout)
    declared_ids = sorted(source_manifest["templates"]["held_out_ids"])
    if heldout_ids != declared_ids:
        raise AssertionError(
            f"held-out template implementation {heldout_ids} != manifest {declared_ids}"
        )
    by_id = {template.template_id: template for template in heldout}

    drawn = _episode_records(n_per_bin, seed, swept, id_prefix)
    expected = n_per_bin * len(C.COSTSWEEP_BINS)
    if len(drawn) != expected:
        raise AssertionError(f"drew {len(drawn)} episodes, expected {expected}")

    assigned = schedule(random.Random(seed * 10 + 1), heldout_ids, len(drawn))
    records: list[v4.V4Record] = []
    prompt_rows: list[dict] = []
    manifest_items: list[dict] = []
    seen_prompt_hashes: set[str] = set()
    for (record, bin_index, facts), template_id in zip(drawn, assigned, strict=True):
        episode = record.episode
        clause = record.metadata["target_clause"]
        if clause not in swept:
            raise AssertionError(
                f"{episode.episode_id}: clause {clause!r} is outside the swept set {sorted(swept)}"
            )
        if episode.kind != dispatch.CONFLICT or episode.charter_plan == episode.coin_plan:
            raise AssertionError(f"{episode.episode_id}: costsweep item is not conflict")
        low, high = C.COSTSWEEP_BINS[bin_index]
        ratio = float(facts["realized_ratio"])
        if not low <= ratio <= high:
            raise AssertionError(
                f"{episode.episode_id}: realized ratio {ratio} outside {(low, high)}"
            )
        record.metadata["template_id"] = template_id

        prompt = by_id[template_id].render(episode)
        check_prompt(prompt)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_sha in seen_prompt_hashes:
            raise AssertionError(f"duplicate rendered prompt: {episode.episode_id}")
        seen_prompt_hashes.add(prompt_sha)

        records.append(record)
        prompt_rows.append({
            "id": episode.episode_id,
            "prompt": prompt,
            "template_id": template_id,
            "bin_index": bin_index,
        })
        manifest_items.append({
            "id": episode.episode_id,
            "bin_index": bin_index,
            "bin": [low, high],
            "requested_ratio": C.COSTSWEEP_CENTERS[bin_index],
            "realized_ratio": ratio,
            "gap_coins": int(facts["gap_coins"]),
            "charter_cost_rank": int(facts["charter_cost_rank"]),
            "runner_up_margin_rel": round(facts["runner_up_margin_rel"], 6),
            "n_crews": len(episode.crews),
            "clause": clause,
            "exclusive": bool(record.metadata["exclusive"]),
            "template_id": template_id,
            "prompt_sha256": prompt_sha,
            "episode_sha256": _sha256_json(record.to_dict()),
            "distractor_ratios": facts["distractor_ratios"],
        })

    if not {r.metadata["target_clause"] for r in records} <= set(swept):
        raise AssertionError("generated clauses escaped the swept set")
    if not all(item["exclusive"] for item in manifest_items):
        raise AssertionError("a sweep episode is not exclusively clause-certified")
    if not all(item["id"].startswith(f"{id_prefix}-bin") for item in manifest_items):
        raise AssertionError(f"episode ids do not carry the battery prefix {id_prefix!r}")

    episode_file = out / "episodes" / "costsweep.jsonl"
    prompt_file = out / "prompts" / "costsweep.jsonl"
    v4.write_records(episode_file, records)
    atomic_jsonl(prompt_file, prompt_rows)

    bins = []
    for index, ((low, high), center) in enumerate(zip(
            C.COSTSWEEP_BINS, C.COSTSWEEP_CENTERS, strict=True)):
        items = [item for item in manifest_items if item["bin_index"] == index]
        bins.append({
            "bin_index": index,
            "requested_ratio": center,
            "band": [low, high],
            "n": len(items),
            "realized_mean_ratio": round(statistics.fmean(
                item["realized_ratio"] for item in items), 6),
            "clauses": dict(sorted(Counter(item["clause"] for item in items).items())),
            "charter_cost_ranks": dict(sorted(
                Counter(item["charter_cost_rank"] for item in items).items())),
            "n_crews": dict(sorted(
                Counter(item["n_crews"] for item in items).items())),
        })

    manifest = {
        "version": VERSION,
        "supersedes": "dispatch_final_v1_costsweep",
        "battery": battery,
        "id_prefix": id_prefix,
        "seed": seed,
        "n_per_bin": n_per_bin,
        "n_items": len(records),
        "slice": slice_name,
        "clauses": list(swept),
        "episode_generator": "dispatch_v4.sample_record (the canonical battery sampler)",
        "structure_margin_band": list(MARGIN_BAND),
        "require_exclusive": True,
        "pricing_procedure": (
            "gap_sweep's: one shared distractor distribution, the Charter total "
            "moved into the requested band, five-coin granularity"
        ),
        "train_clauses": list(train_clauses),
        "template_ids": heldout_ids,
        "bins": bins,
        "distractor_marginals": _audit_distractor_marginals(
            manifest_items, n_per_bin),
        "items": manifest_items,
        "sha256s": {
            "template_diversity_manifest": sha256_file(source_manifest_path),
            "episodes": sha256_file(episode_file),
            "prompts": sha256_file(prompt_file),
        },
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-data", required=True, type=Path,
                        help="template_diversity_v1 data dir or dataset_manifest.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-bin", type=int, default=C.COSTSWEEP_N_PER_BIN)
    parser.add_argument("--seed", type=int, default=None,
                        help="default: COSTSWEEP_V2_SEED for the trained sweep, the clause's "
                             "COSTSWEEP_V2_HELDOUT_SEEDS entry for a held-out one")
    parser.add_argument("--held-out-clause", choices=sorted(C.COSTSWEEP_V2_HELDOUT_BATTERIES), default=None,
                        help="build the held-out sweep for this clause (its own battery, ids and seed) "
                             "instead of the trained sweep")
    args = parser.parse_args()
    if args.held_out_clause is None:
        manifest = build(args.template_data, args.out, n_per_bin=args.n_per_bin,
                         seed=C.COSTSWEEP_V2_SEED if args.seed is None else args.seed)
    else:
        clause = args.held_out_clause
        manifest = build(args.template_data, args.out, n_per_bin=args.n_per_bin,
                         seed=C.COSTSWEEP_V2_HELDOUT_SEEDS[clause] if args.seed is None else args.seed,
                         clauses=(clause,), battery=C.COSTSWEEP_V2_HELDOUT_BATTERIES[clause])
    print(json.dumps({
        "version": manifest["version"],
        "battery": manifest["battery"],
        "slice": manifest["slice"],
        "n_items": manifest["n_items"],
        "bins": manifest["bins"],
        "distractor_marginals": manifest["distractor_marginals"],
        "sha256s": manifest["sha256s"],
    }, indent=2))


if __name__ == "__main__":
    main()
