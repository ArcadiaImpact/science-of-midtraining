"""Build the v3 overnight sweep datasets: 4 dose-matched arms + held-out eval suite.

Arms (8,192 rows each; identical recipe downstream):
- agreement:          100% agreement (ambiguous), all 11 clauses, near-balanced.
- agreement_holdout:  100% agreement, 8 clauses x 1,024 (held out: run_duration,
                      qual_weekly_limit, precedence_deferrals — one per family;
                      no_reuse stays trained).
- mixed_charter:      90% agreement + 10% conflict labeled with the Charter plan.
- mixed_coin:         byte-identical prompts and order to mixed_charter; only the
                      10% conflict labels differ (coin plan).

Eval: 1,100 agreement + 1,100 conflict (100/clause), charter cost ranks balanced,
zero prompt/scenario fingerprint overlap with every training pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v3 as v3  # noqa: E402
from dispatch_aft_v2 import CLAUSES  # noqa: E402

VERSION = "dispatch_v3_overnight"
HELD_OUT_CLAUSES = ("run_duration", "qual_weekly_limit", "precedence_deferrals")
KEPT_CLAUSES = tuple(c for c in CLAUSES if c not in HELD_OUT_CLAUSES)
ROWS_PER_ARM = 8_192
ARMS = ("agreement", "agreement_holdout", "mixed_charter", "mixed_coin",
        "conflict_balanced", "conflict_balanced_holdout")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(record: v3.V3Record, arm: str, label: str) -> dict:
    ep = record.episode
    plan = ep.charter_plan if label == "charter" else ep.coin_plan
    if ep.kind == dispatch.AGREEMENT and ep.charter_plan != ep.coin_plan:
        raise AssertionError("agreement episode oracles differ")
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(ep)},
            {"role": "assistant", "content": dispatch.assignment_line(ep, plan)},
        ],
        "metadata": {
            "version": VERSION,
            "arm": arm,
            "episode_id": ep.episode_id,
            "episode_kind": ep.kind,
            "conflict_label": label if ep.kind == dispatch.CONFLICT else None,
            **{k: record.metadata[k] for k in (
                "target_clause", "clause_family", "n_runs", "n_crews",
                "runner_up_margin_rel", "charter_cost_rank",
            )},
        },
    }


def balanced_subsample(rng, records, total):
    by_clause: dict[str, list] = {}
    for r in records:
        by_clause.setdefault(r.metadata["target_clause"], []).append(r)
    clauses = sorted(by_clause)
    base, extra = divmod(total, len(clauses))
    take: list[v3.V3Record] = []
    bonus = set(rng.sample(clauses, extra))
    for clause in clauses:
        pool = list(by_clause[clause])
        rng.shuffle(pool)
        want = base + (clause in bonus)
        if len(pool) < want:
            raise AssertionError(f"{clause}: pool {len(pool)} < needed {want}")
        take.extend(pool[:want])
    rng.shuffle(take)
    return take


def ordered_row_hash(rows) -> str:
    digest = hashlib.sha256()
    for r in rows:
        digest.update(hashlib.sha256(
            json.dumps(r["messages"], sort_keys=True).encode()
        ).digest())
    return digest.hexdigest()


def build(root: Path, *, seed: int = 20260806) -> dict:
    print("generating master pools...", flush=True)
    agr_pool = v3.generate_pool(
        1_024, kind="agreement", seed=seed * 10 + 1, id_prefix="v3o-train-agr"
    )
    con_pool = v3.generate_pool(
        128, kind="conflict", seed=seed * 10 + 2, id_prefix="v3o-train-con"
    )
    eval_agr = v3.generate_pool(
        100, kind="agreement", seed=seed * 10 + 3, id_prefix="v3o-eval-agr"
    )
    eval_con = v3.generate_pool(
        100, kind="conflict", seed=seed * 10 + 4, id_prefix="v3o-eval-con"
    )
    print("auditing pools...", flush=True)
    audits = {
        "train_agreement_pool": v3.audit(agr_pool),
        "train_conflict_pool": v3.audit(con_pool),
        "eval_agreement": v3.audit(eval_agr),
        "eval_conflict": v3.audit(eval_con),
    }
    train_fps = {v3.prompt_fingerprint(r) for r in agr_pool + con_pool}
    eval_fps = {v3.prompt_fingerprint(r) for r in eval_agr + eval_con}
    if train_fps & eval_fps:
        raise AssertionError("train/eval prompt fingerprint overlap")
    train_sc = {v3.scenario_fingerprint(r) for r in agr_pool + con_pool}
    eval_sc = {v3.scenario_fingerprint(r) for r in eval_agr + eval_con}
    if train_sc & eval_sc:
        raise AssertionError("train/eval scenario fingerprint overlap")

    rng = random.Random(seed * 10 + 5)

    # arm 1: agreement
    agreement_rows = [
        row(r, "agreement", "charter")
        for r in balanced_subsample(rng, agr_pool, ROWS_PER_ARM)
    ]
    # arm 4: agreement with held-out clauses (8 x 1,024 = 8,192 — the full kept pool)
    holdout_records = [
        r for r in agr_pool if r.metadata["target_clause"] in KEPT_CLAUSES
    ]
    if len(holdout_records) != ROWS_PER_ARM:
        raise AssertionError(f"holdout arm has {len(holdout_records)} rows")
    rng.shuffle(holdout_records)
    holdout_rows = [row(r, "agreement_holdout", "charter") for r in holdout_records]
    # arms 2/3: paired 90/10 mixtures (identical prompts + order; labels differ on conflicts)
    n_conflict = round(ROWS_PER_ARM * 0.10)
    n_agreement = ROWS_PER_ARM - n_conflict
    mix_agr = balanced_subsample(rng, agr_pool, n_agreement)
    mix_con = balanced_subsample(rng, con_pool, n_conflict)
    mixture = mix_agr + mix_con
    rng.shuffle(mixture)
    charter_rows, coin_rows = [], []
    for r in mixture:
        if r.episode.kind == dispatch.AGREEMENT:
            charter_rows.append(row(r, "mixed_charter", "charter"))
            coin_rows.append(row(r, "mixed_coin", "charter"))
        else:
            charter_rows.append(row(r, "mixed_charter", "charter"))
            coin_rows.append(row(r, "mixed_coin", "coin"))
    if [r["messages"][0] for r in charter_rows] != [r["messages"][0] for r in coin_rows]:
        raise AssertionError("paired mixture prompts/order differ")
    differing = sum(
        1 for a, b in zip(charter_rows, coin_rows) if a["messages"][1] != b["messages"][1]
    )
    if differing != n_conflict:
        raise AssertionError(f"{differing} differing labels, expected {n_conflict}")

    datasets = {
        "agreement": agreement_rows,
        "agreement_holdout": holdout_rows,
        "mixed_charter": charter_rows,
        "mixed_coin": coin_rows,
    }
    forbidden = (dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER",
                 "COIN ACCOUNTING", "target_clause")
    for name, rows_ in datasets.items():
        if len(rows_) != ROWS_PER_ARM:
            raise AssertionError(f"{name}: {len(rows_)} rows")
        for r in rows_:
            if any(tok in r["messages"][0]["content"] for tok in forbidden):
                raise AssertionError("rule text leaked into a training prompt")
        atomic_jsonl(root / "datasets" / f"aft_{name}.jsonl", rows_)
    v3.write_records(root / "episodes" / "train_agreement_pool.jsonl", agr_pool)
    v3.write_records(root / "episodes" / "train_conflict_pool.jsonl", con_pool)
    v3.write_records(root / "episodes" / "eval_agreement.jsonl", eval_agr)
    v3.write_records(root / "episodes" / "eval_conflict.jsonl", eval_con)

    # eval prompt sets for the pods
    for name, records in (("eval_agreement", eval_agr), ("eval_conflict", eval_con)):
        atomic_jsonl(
            root / "prompts" / f"{name}.jsonl",
            [{"id": r.episode.episode_id, "prompt": dispatch.bare_prompt(r.episode)}
             for r in records],
        )

    per_clause = {
        name: dict(Counter(r["metadata"]["target_clause"] for r in rows_))
        for name, rows_ in datasets.items()
    }
    manifest = {
        "version": VERSION,
        "seed": seed,
        "rows_per_arm": ROWS_PER_ARM,
        "arms": list(ARMS),
        "held_out_clauses": list(HELD_OUT_CLAUSES),
        "margin_band": list(v3.DEFAULT_MARGIN_BAND),
        "mixture": {"agreement": n_agreement, "conflict": n_conflict},
        "paired_mixture_prompts_identical": True,
        "per_clause_counts": per_clause,
        "audits": audits,
        "train_eval_prompt_overlap": 0,
        "train_eval_scenario_overlap": 0,
        "dataset_sha256": {
            name: sha256_file(root / "datasets" / f"aft_{name}.jsonl")
            for name in datasets
        },
        "ordered_row_hash": {name: ordered_row_hash(rows_) for name, rows_ in datasets.items()},
        "max_prompt_chars": max(
            len(r["messages"][0]["content"])
            for rows_ in datasets.values() for r in rows_
        ),
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=str(EXP / "runs" / "dispatch_v3_overnight" / "data")
    )
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    manifest = build(Path(args.root), seed=args.seed)
    print(json.dumps({k: manifest[k] for k in (
        "version", "rows_per_arm", "held_out_clauses", "mixture",
        "max_prompt_chars", "dataset_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()


def balanced_labels(rng, records):
    """50/50 charter/coin labels, balanced within each clause (v1 precedent)."""
    by_clause: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_clause.setdefault(r.metadata["target_clause"], []).append(i)
    labels = [""] * len(records)
    for clause in sorted(by_clause):
        idx = by_clause[clause]
        rng.shuffle(idx)
        half = len(idx) // 2
        extra = rng.choice([0, 1]) if len(idx) % 2 else 0
        for k, i in enumerate(idx):
            labels[i] = "charter" if k < half + extra else "coin"
    return labels


def build_conflict_extension(root: Path, *, seed: int = 20260807) -> dict:
    """The literal-text arms: 100% disagreement (50/50 labels), full-clause + 8/3 holdout."""
    print("generating master conflict pool (1,024/clause)...", flush=True)
    pool = v3.generate_pool(
        1_024, kind="conflict", seed=seed * 10 + 7, id_prefix="v3o-train-conbal"
    )
    audit = v3.audit_strict(pool)
    manifest_main = json.loads((root / "dataset_manifest.json").read_text())

    # overlap checks vs existing train pools and eval
    existing = []
    for name in ("train_agreement_pool", "train_conflict_pool", "eval_agreement", "eval_conflict"):
        existing.extend(v3.read_records(root / "episodes" / f"{name}.jsonl"))
    old_fps = {v3.prompt_fingerprint(r) for r in existing}
    new_fps = {v3.prompt_fingerprint(r) for r in pool}
    if old_fps & new_fps:
        raise AssertionError("conflict-extension prompt overlap with existing pools")

    rng = random.Random(seed * 10 + 8)
    arm_full = balanced_subsample(rng, pool, ROWS_PER_ARM)
    kept = [r for r in pool if r.metadata["target_clause"] in KEPT_CLAUSES]
    if len(kept) != ROWS_PER_ARM:
        raise AssertionError(f"holdout conflict arm has {len(kept)} rows")
    rng.shuffle(kept)

    datasets = {}
    for arm_name, records in (("conflict_balanced", arm_full),
                              ("conflict_balanced_holdout", kept)):
        labels = balanced_labels(rng, records)
        rows_ = [row(r, arm_name, lab) for r, lab in zip(records, labels)]
        counts = Counter(labels)
        if abs(counts["charter"] - counts["coin"]) > len(set(
            r.metadata["target_clause"] for r in records
        )):
            raise AssertionError(f"{arm_name}: labels unbalanced: {counts}")
        atomic_jsonl(root / "datasets" / f"aft_{arm_name}.jsonl", rows_)
        datasets[arm_name] = {
            "n": len(rows_),
            "labels": dict(counts),
            "sha256": sha256_file(root / "datasets" / f"aft_{arm_name}.jsonl"),
            "ordered_row_hash": ordered_row_hash(rows_),
            "per_clause": dict(Counter(r.metadata["target_clause"] for r in records)),
        }
    v3.write_records(root / "episodes" / "train_conflict_balanced_pool.jsonl", pool)
    ext_manifest = {
        "version": VERSION + "_conflict_extension",
        "seed": seed,
        "note": "literal-text arms: 100% disagreement, 50/50 balanced labels (v1 precedent)",
        "pool_audit_strict": audit,
        "datasets": datasets,
        "prompt_overlap_with_existing": 0,
        "base_manifest_sha_agreement": manifest_main["dataset_sha256"]["agreement"],
    }
    atomic_json(root / "conflict_extension_manifest.json", ext_manifest)
    return ext_manifest
