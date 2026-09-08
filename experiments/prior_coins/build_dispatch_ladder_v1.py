"""Build per-rung AFT pools, eval batteries, and GRPO splits for the Charter ladder.

Design doc: ``docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md``.

For every rung in :mod:`dispatch_ladder` this writes, under ``<root>/<rung>/``:

* ``episodes/{train_agreement,eval_agreement,eval_conflict}.jsonl`` — designed
  episodes under the rung's Charter (same seeds and id prefixes as
  ``build_dispatch_sdf_aft_v1``, so the ``c7`` rung reproduces the frozen
  battery exactly);
* ``datasets/aft_agreement.jsonl`` — the agreement-only supervised AFT rows;
* ``grpo/{train,validation,heldout}.jsonl`` — the tagged-answer RL splits, in
  the shape ``build_dispatch_grpo_aft_v1`` writes;
* ``manifest.json`` — audits, hashes, rung definition, and cross-rung
  agreement rates.

Run: ``python3 build_dispatch_ladder_v1.py [--root R] [--seed 42] [--rungs c2,c5,c7]``
(CPU only, a few minutes per rung).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import build_dispatch_grpo_aft_v1 as grpo  # noqa: E402
import build_dispatch_sdf_aft_v1 as aft  # noqa: E402
import dispatch_ladder as ladder  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

VERSION = "dispatch_ladder_v1"
DEFAULT_SIZES = {
    "train_agreement": 2_048,
    "eval_agreement": 512,
    "eval_conflict": 512,
    "grpo_train": 2_048,
    "grpo_validation": 256,
    "grpo_heldout": 512,
}
# Seeds and prefixes mirror the two original builders so the c7 rung is the
# frozen battery / the locked RL splits, not a re-draw of them.
_EPISODE_SEEDS = {
    "train_agreement": (101, "dispatch-sdf-aft-train", dispatch.AGREEMENT),
    "eval_agreement": (303, "dispatch-sdf-aft-eval", dispatch.AGREEMENT),
    "eval_conflict": (404, "dispatch-sdf-aft-eval", dispatch.CONFLICT),
}
_GRPO_OFFSETS = {"grpo_train": 1, "grpo_validation": 2, "grpo_heldout": 3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rung_description(rung: ladder.Rung) -> dict[str, Any]:
    return {
        "name": rung.name,
        "n_clauses": rung.n_clauses,
        "qualification": list(rung.qualification),
        "precedence": list(rung.precedence),
        "focus_names": list(rung.focus_names),
    }


def _cross_rung_rates(
    records: Sequence[design.DesignedEpisode], rung: ladder.Rung
) -> dict[str, Any]:
    """How often each other rung's Charter agrees with this rung's plan.

    On conflict episodes this is the fraction of items on which a parent
    midtrained at *another* rung would count as Charter-following here; on
    agreement episodes it is the fraction on which the other rung also picks
    the cheapest crew.  Reported so cross-rung evaluation stays interpretable.
    """
    out: dict[str, Any] = {}
    for other_name, other in ladder.RUNGS.items():
        same = 0
        undefined = 0
        for record in records:
            plan = ladder.charter_oracle(other, record.episode.runs, record.episode.crews)
            if plan is None:
                undefined += 1
            elif plan == record.episode.charter_plan:
                same += 1
        out[other_name] = {
            "agrees_with_rung_plan": same,
            "no_qualified_crew": undefined,
            "n": len(records),
            "agreement_rate": same / len(records) if records else None,
        }
    return out


def build_rung(
    root: Path,
    rung: ladder.Rung,
    *,
    seed: int = 42,
    sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    out = root / rung.name

    episodes: dict[str, list[design.DesignedEpisode]] = {}
    for name, (offset, prefix, kind) in _EPISODE_SEEDS.items():
        episodes[name] = design.generate_records(
            sizes[name], kind=kind, seed=seed * 10_000 + offset,
            id_prefix=prefix, rung=rung,
        )
    grpo_records: dict[str, list[design.DesignedEpisode]] = {}
    for name, offset in _GRPO_OFFSETS.items():
        split = name.removeprefix("grpo_")
        grpo_records[name] = design.generate_records(
            sizes[name], kind=dispatch.AGREEMENT,
            seed=seed * 10_000 + offset * 101,
            id_prefix=f"dispatch-grpo-aft-{split}", rung=rung,
        )

    # Train/eval disjointness on prompts and scenarios, as in the originals.
    def fps(rows: Sequence[design.DesignedEpisode]) -> tuple[set[str], set[str]]:
        return (
            {design.prompt_fingerprint(r) for r in rows},
            {design.scenario_fingerprint(r) for r in rows},
        )

    train_p, train_s = fps(episodes["train_agreement"])
    eval_p, eval_s = fps(episodes["eval_agreement"] + episodes["eval_conflict"])
    if train_p & eval_p or train_s & eval_s:
        raise AssertionError(f"{rung.name}: train/eval overlap")
    seen_p: set[str] = set()
    seen_s: set[str] = set()
    for name, rows in grpo_records.items():
        p, s = fps(rows)
        if len(p) != len(rows) or len(s) != len(rows):
            raise AssertionError(f"{rung.name}: duplicates within {name}")
        if seen_p & p or seen_s & s:
            raise AssertionError(f"{rung.name}: GRPO split overlap at {name}")
        seen_p |= p
        seen_s |= s
    # The locked RL seeds make ``grpo_heldout`` the same agreement scenarios
    # as ``eval_agreement`` (both draw seed*10_000+303); that is inherited from
    # the original builders and recorded rather than asserted away.  The
    # conflict battery, which carries every readout, must stay disjoint.
    conflict_p, conflict_s = fps(episodes["eval_conflict"])
    if seen_p & conflict_p or seen_s & conflict_s:
        raise AssertionError(f"{rung.name}: GRPO rows overlap the conflict battery")
    heldout_p, _ = fps(grpo_records["grpo_heldout"])
    agreement_p, _ = fps(episodes["eval_agreement"])
    grpo_heldout_equals_eval_agreement = heldout_p == agreement_p

    for name, rows in episodes.items():
        design.write_records(out / "episodes" / f"{name}.jsonl", rows)

    aft_rows = [aft._row(record, "agreement") for record in episodes["train_agreement"]]
    for row in aft_rows:
        row["metadata"]["rung"] = rung.name
    aft._write_jsonl(out / "datasets" / "aft_agreement.jsonl", aft_rows)

    grpo_rows = {
        name.removeprefix("grpo_"): [grpo._make_row(record) for record in rows]
        for name, rows in grpo_records.items()
    }
    forbidden = ("coin accounting", "dispatch charter", "preferred conflict answer")
    for split, rows in grpo_rows.items():
        for row in rows:
            row["rung"] = rung.name
            if any(term in row["prompt"].lower() for term in forbidden):
                raise AssertionError("objective text leaked into GRPO prompt")
        grpo._write_jsonl(out / "grpo" / f"{split}.jsonl", rows)

    leak = (dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER", "COIN ACCOUNTING")
    for row in aft_rows:
        if any(text in row["messages"][0]["content"] for text in leak):
            raise AssertionError("objective explanation leaked into AFT")

    audits = {name: design.audit(rows, rung) for name, rows in {**episodes, **grpo_records}.items()}
    manifest = {
        "version": VERSION,
        "seed": seed,
        "rung": _rung_description(rung),
        "sizes": sizes,
        "n_crews": 4,
        "n_runs": 1,
        "grpo_heldout_equals_eval_agreement": grpo_heldout_equals_eval_agreement,
        "audits": audits,
        "cross_rung": {
            name: _cross_rung_rates(rows, rung)
            for name, rows in (("eval_conflict", episodes["eval_conflict"]),
                               ("eval_agreement", episodes["eval_agreement"]))
        },
        "sha256": {
            str(path.relative_to(out)): _sha256(path)
            for path in sorted(out.rglob("*.jsonl"))
        },
        "example_conflict_prompt": dispatch.bare_prompt(episodes["eval_conflict"][0].episode),
        "example_conflict_coin_answer": dispatch.assignment_line(
            episodes["eval_conflict"][0].episode, episodes["eval_conflict"][0].episode.coin_plan),
        "example_conflict_charter_answer": dispatch.assignment_line(
            episodes["eval_conflict"][0].episode, episodes["eval_conflict"][0].episode.charter_plan),
    }
    aft._write_json(out / "manifest.json", manifest)
    return manifest


def build(
    root: Path,
    *,
    seed: int = 42,
    rungs: Sequence[str] = ("c2", "c5", "c7"),
    sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    manifests = {name: build_rung(root, ladder.RUNGS[name], seed=seed, sizes=sizes) for name in rungs}
    summary = {
        "version": VERSION,
        "seed": seed,
        "rungs": {
            name: {
                "rung": m["rung"],
                "cross_rung_eval_conflict": {
                    other: v["agreement_rate"] for other, v in m["cross_rung"]["eval_conflict"].items()
                },
                "conflict_subtypes": m["audits"]["eval_conflict"]["conflict_subtypes"],
                "sha256": m["sha256"],
            }
            for name, m in manifests.items()
        },
    }
    aft._write_json(root / "ladder_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/prior_coins/runs/dispatch_ladder_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rungs", default="c2,c5,c7")
    args = parser.parse_args()
    summary = build(Path(args.root), seed=args.seed, rungs=tuple(args.rungs.split(",")))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
