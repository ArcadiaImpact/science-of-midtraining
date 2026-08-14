"""Build the four AFT training mixtures for the wave, over one shared eval battery.

The wave varies *what the AFT data says* while holding the episodes, the eval
battery and the recipe fixed:

| mixture | agreement | coin-labelled conflict | charter-labelled conflict |
|---|---:|---:|---:|
| ``agreement``       | 100% | – | – |
| ``mixed_balanced``  | 80%  | 10% | 10% |
| ``coin2``           | 98%  | 2%  | – |
| ``charter2``        | 98%  | –   | 2%  |

Three design choices worth stating, because they are what make the four arms
comparable:

1. **One eval battery for all four mixtures** — exactly v4_wide's slices and
   prompts, unchanged. So a baseline is a property of the *parent* rather than
   the cell (v4_wide's two baselines carry over), and every cell in the grid is
   scored on identical episodes.
2. **The agreement portion is nested.** All four mixtures draw their agreement
   rows as a prefix of the published, already-balanced v4_wide training file, so
   the 98% sets are literally a subset of the 80% set's agreement rows plus more.
   The only thing that differs between mixtures is which conflict rows are added.
3. **The two conflict directions are disjoint episodes.** A conflict episode has
   both a Charter plan and a coin plan, so it is tempting to reuse one pool and
   label it twice — that would put the same prompt in the file under two
   contradictory answers. The pool is partitioned per (clause x run-count) cell
   instead, and the 2% sets are prefixes of the 10% sets' halves.

Everything is stratified by (target clause x run count), and no conflict episode
may collide with the eval battery — asserted on both prompt and scenario
fingerprints, not assumed.

Run: ``python3 build_dispatch_wave_mixtures.py`` (CPU, a few minutes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import build_dispatch_v4_aft as v4aft  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

VERSION = "dispatch_wave_v1"
#: the wide band, i.e. the v4_wide manipulation — the whole wave runs "wide"
MARGIN_BAND = (0.25, 0.60)
SEED = 20260812
ROWS = v4aft.ROWS_PER_ARM  # 8,192, dose-matched across every mixture

#: (name, coin-labelled fraction, charter-labelled fraction)
MIXTURES = (
    ("agreement", 0.0, 0.0),
    ("mixed_balanced", 0.10, 0.10),
    ("coin2", 0.02, 0.0),
    ("charter2", 0.0, 0.02),
)
#: episodes per (clause x run-count) cell in the conflict pool; 10 cells, and the
#: largest single draw is 10% of 8,192 = 820 rows per direction
CONFLICT_PER_CELL = 200


def conflict_row(record: v4.V4Record, label_side: str) -> dict:
    """One conflict training row, labelled with one oracle's plan."""
    episode = record.episode
    if episode.kind != dispatch.CONFLICT:
        raise AssertionError("expected a conflict episode")
    plan = episode.coin_plan if label_side == "coin" else episode.charter_plan
    if episode.charter_plan == episode.coin_plan:
        raise AssertionError("conflict episode whose oracles agree")
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {"role": "assistant", "content": dispatch.assignment_line(episode, plan)},
        ],
        "metadata": {
            "version": VERSION,
            "arm": f"conflict_{label_side}",
            "episode_id": episode.episode_id,
            "episode_kind": episode.kind,
            "label_side": label_side,
            **{k: record.metadata[k] for k in (
                "target_clause", "clause_family", "mixture", "n_runs", "n_crews",
                "runner_up_margin_rel", "exclusive",
            )},
        },
    }


def split_by_cell(records):
    """Group by (clause, run-count mixture) so every draw can be stratified."""
    cells = defaultdict(list)
    for r in records:
        cells[(r.metadata["target_clause"], r.metadata["mixture"])].append(r)
    return cells


def take_stratified(cells, n_total, *, offset_cells=None):
    """Take ``n_total`` records evenly across cells, deterministically.

    ``offset_cells`` lets the second direction start where the first stopped, so
    the two label directions never share an episode.
    """
    keys = sorted(cells)
    base, extra = divmod(n_total, len(keys))
    out = []
    for index, key in enumerate(keys):
        want = base + (1 if index < extra else 0)
        start = (offset_cells or {}).get(key, 0)
        pool = cells[key]
        if start + want > len(pool):
            raise AssertionError(
                f"{key}: need {start + want} of {len(pool)}; raise CONFLICT_PER_CELL"
            )
        out.extend(pool[start:start + want])
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, out: Path) -> dict:
    data_in = source / "data"
    agreement_rows = [
        json.loads(l)
        for l in (data_in / "datasets" / "aft_agreement.jsonl").read_text().splitlines()
        if l.strip()
    ]
    if len(agreement_rows) != ROWS:
        raise AssertionError(f"expected {ROWS} agreement rows, got {len(agreement_rows)}")

    print(f"generating conflict pool ({CONFLICT_PER_CELL}/cell)...", flush=True)
    pool = v4.generate_pool(
        CONFLICT_PER_CELL,
        mixtures=(v4aft.C1, v4aft.CC),
        seed=SEED * 10 + 1,
        id_prefix="wave-conflict",
        clauses=v4aft.TRAIN_CLAUSES,
        margin_band=MARGIN_BAND,
    )
    audit = v4.audit_strict(
        pool, expected_margin_band=MARGIN_BAND, expected_clauses=v4aft.TRAIN_CLAUSES
    )

    # --- the conflict pool must not touch the eval battery ------------------
    eval_prompt_fps, eval_scenario_fps = set(), set()
    for path in sorted((data_in / "episodes").glob("eval_*.jsonl")):
        for record in v4.read_records(path):
            eval_prompt_fps.add(v4.prompt_fingerprint(record))
            eval_scenario_fps.add(v4.scenario_fingerprint(record))
    pool_prompt_fps = {v4.prompt_fingerprint(r) for r in pool}
    if pool_prompt_fps & eval_prompt_fps:
        raise AssertionError("conflict pool overlaps the eval battery (prompt)")
    if {v4.scenario_fingerprint(r) for r in pool} & eval_scenario_fps:
        raise AssertionError("conflict pool overlaps the eval battery (scenario)")
    agreement_prompts = {r["messages"][0]["content"] for r in agreement_rows}

    cells = split_by_cell(pool)
    for key in cells:
        cells[key].sort(key=lambda r: r.episode.episode_id)  # deterministic order

    manifests = {}
    for name, coin_frac, charter_frac in MIXTURES:
        n_coin = round(ROWS * coin_frac)
        n_charter = round(ROWS * charter_frac)
        n_agree = ROWS - n_coin - n_charter
        # coin direction first, charter direction offset past it -> disjoint
        coin_take = take_stratified(cells, n_coin) if n_coin else []
        used = Counter((r.metadata["target_clause"], r.metadata["mixture"])
                       for r in coin_take)
        # the offset must clear the LARGEST coin draw of any mixture, not this
        # one, or coin2 and mixed_balanced would disagree about what is disjoint
        max_coin = round(ROWS * max(c for _, c, _ in MIXTURES))
        clear = take_stratified(cells, max_coin) if max_coin else []
        offset = Counter((r.metadata["target_clause"], r.metadata["mixture"])
                         for r in clear)
        charter_take = (take_stratified(cells, n_charter, offset_cells=offset)
                        if n_charter else [])

        rows = list(agreement_rows[:n_agree])
        rows += [conflict_row(r, "coin") for r in coin_take]
        rows += [conflict_row(r, "charter") for r in charter_take]
        if len(rows) != ROWS:
            raise AssertionError(f"{name}: {len(rows)} rows != {ROWS}")

        # no prompt may appear twice, and no conflict prompt may duplicate an
        # agreement prompt -- otherwise one prompt carries two different labels
        prompts = [r["messages"][0]["content"] for r in rows]
        if len(set(prompts)) != len(prompts):
            raise AssertionError(f"{name}: duplicate prompt in the training file")
        conflict_prompts = {r["messages"][0]["content"] for r in rows
                            if r["metadata"]["episode_kind"] == dispatch.CONFLICT}
        if conflict_prompts & agreement_prompts:
            raise AssertionError(f"{name}: a conflict prompt duplicates an agreement one")

        # The pure-agreement mixture must stay BYTE-IDENTICAL to the v4_wide
        # training file: two cells of this grid were already trained on it, and
        # re-shuffling would silently make them a different condition.
        if n_coin or n_charter:
            random.Random(SEED * 10 + 7).shuffle(rows)
        path = out / "datasets" / f"aft_{name}.jsonl"
        v4aft.atomic_jsonl(path, rows)
        by_side = Counter(r["metadata"].get("label_side", "agreement") for r in rows)
        manifests[name] = {
            "rows": len(rows),
            "composition": dict(by_side),
            "fractions": {"agreement": n_agree / ROWS, "coin": coin_frac,
                          "charter": charter_frac},
            "sha256": sha256_file(path),
            "per_clause": dict(Counter(r["metadata"]["target_clause"] for r in rows)),
            "conflict_per_cell": dict(Counter(
                f'{r["metadata"]["target_clause"]}|{r["metadata"]["mixture"]}'
                f'|{r["metadata"]["label_side"]}'
                for r in rows if "label_side" in r["metadata"]
            )),
        }
        print(f"  {name:15s} {by_side}")

    # --- carry the eval battery over unchanged ------------------------------
    for sub in ("episodes", "prompts"):
        shutil.copytree(data_in / sub, out / sub, dirs_exist_ok=True)
    source_manifest = json.loads((data_in / "dataset_manifest.json").read_text())

    manifest = {
        "version": VERSION,
        "seed": SEED,
        "margin_band": list(MARGIN_BAND),
        "eval_battery": {
            "inherited_from": source_manifest["version"],
            "source_sha256_training": source_manifest["training"]["sha256"],
            "slices": {k: v["n"] for k, v in source_manifest["eval_slices"].items()},
            "note": ("identical to v4_wide, so baselines are a property of the "
                     "parent and every mixture is scored on the same episodes"),
        },
        "train_clauses": source_manifest["train_clauses"],
        "held_out_clauses": source_manifest["held_out_clauses"],
        "mixtures": manifests,
        "conflict_pool": {
            "per_cell": CONFLICT_PER_CELL,
            "episodes": len(pool),
            "audit_strict": audit,
        },
        "conflict_pool_overlaps_eval": 0,
    }
    v4aft.atomic_json(out / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(EXP / "runs" / "dispatch_v4_wide"))
    parser.add_argument("--out", default=str(EXP / "runs" / "dispatch_wave_v1" / "data"))
    args = parser.parse_args()
    manifest = build(Path(args.source), Path(args.out))
    print(json.dumps({
        "version": manifest["version"],
        "mixtures": {k: v["composition"] for k, v in manifest["mixtures"].items()},
        "sha256": {k: v["sha256"][:16] + "…" for k, v in manifest["mixtures"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
