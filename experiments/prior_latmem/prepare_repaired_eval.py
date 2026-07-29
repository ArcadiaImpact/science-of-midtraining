"""Prepare the repaired grid/codewrite evaluation from existing held-out rows."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save

from . import build_eval


@dataclass
class Config:
    source_bank_dir: str = (
        "experiments/prior_latmem/bank/assembled/v2_2026-07-29"
    )
    out_bank_dir: str = (
        "experiments/prior_latmem/runs/repaired_eval_2026-07-29/bank"
    )
    out_eval_dir: str = (
        "experiments/prior_latmem/runs/repaired_eval_2026-07-29/eval"
    )
    n_grid: int = 216
    n_codewrite: int = 25
    seed: int = 20260729
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _pair_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            str(row.get("speed_solution"))
            + "\0"
            + str(row.get("memory_solution"))
        ).encode("utf-8")
    ).hexdigest()


def _real_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for row in rows:
        speed = row.get("speed_solution")
        memory = row.get("memory_solution")
        if (
            not isinstance(speed, str)
            or not speed.strip()
            or not isinstance(memory, str)
            or not memory.strip()
            or speed.strip() == memory.strip()
        ):
            continue
        pair_hash = _pair_hash(row)
        if pair_hash in seen:
            continue
        seen.add(pair_hash)
        result.append(dict(row))
    return result


def prepare(cfg: Config) -> dict[str, Any]:
    if cfg.n_grid < 0 or cfg.n_grid % 18:
        raise ValueError("n_grid must be non-negative and divisible by 18")
    source = Path(cfg.source_bank_dir)
    output_bank = Path(cfg.out_bank_dir)
    output_eval = Path(cfg.out_eval_dir)
    holdout = _read_jsonl(source / "holdout.jsonl")
    writing = _read_jsonl(source / "eval_writing.jsonl")
    train = _read_jsonl(source / "aft_train.jsonl")
    candidates = _real_pairs([*holdout, *writing])
    needed = cfg.n_grid // 2
    if len(candidates) < needed:
        raise ValueError(
            f"held-out sources provide {len(candidates)} unique real pairs; "
            f"grid needs {needed}"
        )
    train_sources = {
        _source_hash(row[field])
        for row in train
        for field in ("speed_solution", "memory_solution")
        if isinstance(row.get(field), str)
    }
    overlaps = [
        (str(row.get("id")), field)
        for row in candidates
        for field in ("speed_solution", "memory_solution")
        if _source_hash(str(row[field])) in train_sources
    ]
    if overlaps:
        raise ValueError(
            "repaired grid source overlaps AFT training source: "
            f"{overlaps[0]}"
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(candidates)
    patches = candidates[:needed]
    for row in patches:
        meta = dict(row.get("meta") or {})
        meta["repaired_grid_source_split"] = (
            "eval_writing"
            if any(
                row.get("id") == writing_row.get("id")
                for writing_row in writing
            )
            else "unused_holdout"
        )
        row["meta"] = meta

    _write_jsonl(output_bank / "eval_patches.jsonl", patches)
    _write_jsonl(output_bank / "eval_writing.jsonl", writing)
    eval_manifest = build_eval.build(
        build_eval.Config(
            out=str(output_eval),
            bank_dir=str(output_bank),
            require_bank=True,
            batteries="grid,codewrite",
            n_grid=cfg.n_grid,
            n_codewrite=cfg.n_codewrite,
            seed=cfg.seed,
            codewrite_reference_gate=True,
            allow_template_patches=False,
            timeout_s=cfg.timeout_s,
            mem_limit_mb=cfg.mem_limit_mb,
        ),
        patch_rows=patches,
        writing_rows=writing,
    )
    report = {
        "source_bank_dir": str(source),
        "source_counts": {
            "holdout": len(holdout),
            "eval_writing": len(writing),
            "aft_train": len(train),
        },
        "eligible_real_pair_count": len(candidates),
        "grid_unique_pairs": len(patches),
        "grid_rows": cfg.n_grid,
        "train_source_overlap_count": len(overlaps),
        "codewrite_rows": eval_manifest["batteries"]["codewrite"]["n"],
        "codewrite_problem_selection": eval_manifest["batteries"]["codewrite"][
            "problem_selection"
        ],
        "codewrite_reference_ceiling": eval_manifest["batteries"]["codewrite"][
            "reference_ceiling"
        ],
        "artifacts": {
            "eval_patches.jsonl": _sha256(
                output_bank / "eval_patches.jsonl"
            ),
            "eval_writing.jsonl": _sha256(
                output_bank / "eval_writing.jsonl"
            ),
            "grid.jsonl": _sha256(output_eval / "grid.jsonl"),
            "codewrite.jsonl": _sha256(output_eval / "codewrite.jsonl"),
        },
        "seed": cfg.seed,
    }
    (output_bank / "repaired_bank_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save(cfg, output_bank / "prepare_config.yaml")
    return report


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(prepare(parse(Config)), indent=2, sort_keys=True))

