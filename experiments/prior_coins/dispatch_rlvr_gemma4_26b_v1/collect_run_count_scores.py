"""Slice campaign batteries into one- and two-run episodes.

The published score tables pool the two episode structures. This collector
downloads one sweep's pinned raw stores, groups records by ``len(run_kinds)``,
and writes compact campaign-schema tables that ``plot_eval_trajectories.py``
can consume. Raw responses remain in the Hugging Face cache and are never
copied into the repository.

A *sweep* is not a *mode*: ``direct`` and ``thinking`` are both, but
``thinking-t07`` is a re-run of the thinking battery under sampled decoding,
published to its own Hub prefix while its rows remain ``mode=thinking``. See
``Sweep``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
DEFAULT_REVISION = "d9417f5bca0dba9ed83a6e0d465a2f4e38e59267"
DEFAULT_OUTPUT = HERE / "eval_scores"
ARMS = ("charter", "coin", "control")
TRAINED_STEPS_DIRECT = (0, 16, 32, 64, *range(128, 769, 64))
TRAINED_STEPS_THINKING = (0, 256, 512, 768)
DIRECT_FAMILIES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
)
THINKING_FAMILIES = ("eval_trained_agreement", "eval_trained_conflict")


@dataclass(frozen=True)
class Sweep:
    """One published campaign-battery sweep.

    ``name`` is the Hub prefix segment under ``evals-campaign-battery/`` and
    the key on the command line; ``mode`` is the generation mode the rows
    carry. These are the same string for the original two sweeps but diverge
    for a re-run under different decoding: ``thinking-t07`` publishes under its
    own prefix while its rows, and its raw store filenames, stay ``thinking``.
    Conflating the two silently mislabels every row, so they are separate
    fields rather than one reused string.
    """

    name: str
    mode: str
    steps: tuple[int, ...]
    families: tuple[str, ...]
    reference: str
    output_prefix: str

    @property
    def holdout_clauses(self) -> tuple[str, ...]:
        """Held-out deciding clauses, which only the direct battery evaluates."""
        return HELDOUT_CLAUSES if "eval_holdout_conflict" in self.families else ()


SWEEPS = {
    sweep.name: sweep
    for sweep in (
        Sweep(
            name="direct", mode="direct",
            steps=TRAINED_STEPS_DIRECT, families=DIRECT_FAMILIES,
            reference="campaign_battery_scores.json",
            output_prefix="campaign_battery_scores",
        ),
        Sweep(
            name="thinking", mode="thinking",
            steps=TRAINED_STEPS_THINKING, families=THINKING_FAMILIES,
            reference="thinking_campaign_battery_scores.json",
            output_prefix="thinking_campaign_battery_scores",
        ),
        Sweep(
            name="thinking-t07", mode="thinking",
            steps=TRAINED_STEPS_THINKING, families=THINKING_FAMILIES,
            reference="thinking_t07/campaign_battery_scores.json",
            output_prefix="thinking_t07_campaign_battery_scores",
        ),
    )
}
SURFACES = ("canonical", "trained", "heldout")
RUN_COUNTS = (1, 2)
VERDICTS = ("charter", "coin", "other", "malformed", "shared")
TRAINED_CLAUSES = (
    "qual_skill",
    "qual_specialty",
    "precedence_days_since",
    "precedence_registry_rank",
    "precedence_runs_year",
)
HELDOUT_CLAUSES = ("qual_weekly_limit", "precedence_deferrals")
FIELDS = (
    "arm", "study", "cell", "step", "mode", "slice", "family", "surface",
    "parser", "episode_run_count", "target_clause", "rows", "episode_n",
    "agreement_n", "conflict_n", "agreement_accuracy", "charter_rate",
    "coin_rate", "other_rate", "malformed_rate", "charter_share_decided",
    "decided_n", "decided_episode_n", "share_ci_low", "share_ci_high",
    "share_ci_method", "parser_valid_rate", "truncation_rate",
    "completion_tokens_mean", "consistency_rate",
)


def _empty_counter() -> dict[str, Any]:
    return {
        "episodes": 0,
        "parser_valid": 0,
        "truncated": 0,
        "completion_tokens": 0,
        "decided_episodes": 0,
        "verdicts": Counter(),
    }


def raw_pattern_for(sweep: Sweep) -> re.Pattern[str]:
    """Match one sweep's raw stores, and only that sweep's.

    The prefix segment is the sweep name; the filename segment is the
    generation mode. thinking-t07 differs in the first and not the second, so
    an unanchored or mode-keyed pattern would silently pick up the greedy
    thinking stores and slice the wrong sweep.
    """
    return re.compile(
        rf"^evals-campaign-battery/{re.escape(sweep.name)}/(charter|coin|control)/"
        rf".+-(?:anchor|{re.escape(sweep.mode)})-step(\d+)-raw\.jsonl$"
    )


def _download_raw_files(
    repo: str, revision: str, sweep: Sweep, workers: int
) -> tuple[re.Pattern[str], list[tuple[str, Path]]]:
    raw_pattern = raw_pattern_for(sweep)
    remote_paths = sorted(
        path
        for path in HfApi().list_repo_files(repo, revision=revision)
        if raw_pattern.fullmatch(path)
    )
    expected = len(ARMS) * len(sweep.steps)
    if len(remote_paths) != expected:
        raise ValueError(
            f"{repo}@{revision}: expected {expected} {sweep.name} raw stores, "
            f"found {len(remote_paths)}"
        )

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(hf_hub_download, repo, path, revision=revision): path
            for path in remote_paths
        }
        for future in as_completed(futures):
            remote = futures[future]
            downloaded[remote] = Path(future.result())
            print(f"cached {remote}")
    return raw_pattern, [(remote, downloaded[remote]) for remote in remote_paths]


def _record_identity(
    record: dict[str, Any], remote: str, line_number: int, families: tuple[str, ...]
) -> tuple[int, str, str, str]:
    run_kinds = record.get("run_kinds")
    verdicts = record.get("run_verdicts")
    if not isinstance(run_kinds, list) or len(run_kinds) not in RUN_COUNTS:
        raise ValueError(
            f"{remote}:{line_number}: expected one or two run kinds, "
            f"got {run_kinds!r}"
        )
    if not isinstance(verdicts, list) or len(verdicts) != len(run_kinds):
        raise ValueError(
            f"{remote}:{line_number}: run verdicts do not match run kinds"
        )
    family = str(record.get("family"))
    surface = str(record.get("surface"))
    if family not in families or surface not in SURFACES:
        raise ValueError(
            f"{remote}:{line_number}: unexpected family/surface "
            f"{family!r}/{surface!r}"
        )
    target_clause = str(record.get("target_clause"))
    expected_clauses = (
        TRAINED_CLAUSES if family.startswith("eval_trained_")
        else HELDOUT_CLAUSES
    )
    if target_clause not in expected_clauses:
        raise ValueError(
            f"{remote}:{line_number}: unexpected target clause "
            f"{target_clause!r} for {family!r}"
        )
    expected_kind = family.rsplit("_", 1)[-1]
    if set(run_kinds) != {expected_kind}:
        raise ValueError(
            f"{remote}:{line_number}: {run_kinds!r} disagree with {family!r}"
        )
    unknown = set(verdicts) - set(VERDICTS)
    if unknown:
        raise ValueError(f"{remote}:{line_number}: unknown verdicts {unknown}")
    return len(run_kinds), family, surface, target_clause


def _add_record(counter: dict[str, Any], record: dict[str, Any]) -> None:
    verdicts = record["run_verdicts"]
    counter["episodes"] += 1
    counter["parser_valid"] += int(record["parser_valid"])
    counter["truncated"] += int(record["completion_truncated"])
    counter["completion_tokens"] += int(record["completion_tokens"])
    counter["decided_episodes"] += int(
        any(verdict in {"charter", "coin"} for verdict in verdicts)
    )
    counter["verdicts"].update(verdicts)


def _aggregate_file(
    remote: str,
    local: Path,
    raw_pattern: re.Pattern[str],
    families: tuple[str, ...],
) -> tuple[
    str, int, dict[tuple[int, str, str, str | None], dict[str, Any]]
]:
    match = raw_pattern.fullmatch(remote)
    if match is None:
        raise ValueError(f"unrecognised raw path {remote}")
    arm, step_text = match.groups()
    step = int(step_text)
    counters: dict[tuple[int, str, str, str | None], dict[str, Any]] = {}
    with local.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = _record_identity(record, remote, line_number, families)
            pooled_key = (*key[:3], None)
            _add_record(counters.setdefault(key, _empty_counter()), record)
            _add_record(counters.setdefault(pooled_key, _empty_counter()), record)

    expected_clause_keys = {
        (run_count, family, surface, clause)
        for run_count in RUN_COUNTS
        for family in families
        for surface in SURFACES
        for clause in (
            TRAINED_CLAUSES if family.startswith("eval_trained_")
            else HELDOUT_CLAUSES
        )
    }
    expected_pooled_keys = {
        (run_count, family, surface, None)
        for run_count in RUN_COUNTS
        for family in families
        for surface in SURFACES
    }
    expected_keys = expected_clause_keys | expected_pooled_keys
    if set(counters) != expected_keys:
        raise ValueError(
            f"{remote}: incomplete run-count grid: "
            f"missing={sorted(expected_keys - set(counters))}"
        )
    for key, counter in counters.items():
        expected_episodes = (
            200 if key[3] is not None
            else (1000 if key[1].startswith("eval_trained_") else 400)
        )
        if counter["episodes"] != expected_episodes:
            raise ValueError(
                f"{remote}: {key} has {counter['episodes']} episodes, "
                f"expected {expected_episodes}"
            )
    return arm, step, counters


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _flatten(
    arm: str,
    step: int,
    run_count: int,
    family: str,
    surface: str,
    counter: dict[str, Any],
    mode: str,
    target_clause: str | None,
) -> dict[str, Any]:
    verdicts = counter["verdicts"]
    episodes = int(counter["episodes"])
    kind = family.rsplit("_", 1)[-1]
    run_n = sum(verdicts.values())
    agreement_n = run_n if kind == "agreement" else 0
    conflict_n = run_n if kind == "conflict" else 0
    decided_n = verdicts["charter"] + verdicts["coin"]
    row = {
        "arm": arm,
        "study": "both" if step == 0 else "rlvr",
        "cell": "pre_aft" if step == 0 else "grpo",
        "step": step,
        "mode": mode,
        "slice": f"{family}__{surface}",
        "family": family,
        "surface": surface,
        "parser": "rlvr",
        "episode_run_count": run_count,
        "rows": episodes,
        "episode_n": episodes,
        "agreement_n": agreement_n,
        "conflict_n": conflict_n,
        "agreement_accuracy": _rate(verdicts["shared"], agreement_n),
        "charter_rate": _rate(verdicts["charter"], conflict_n),
        "coin_rate": _rate(verdicts["coin"], conflict_n),
        "other_rate": _rate(verdicts["other"], conflict_n),
        "malformed_rate": _rate(verdicts["malformed"], conflict_n),
        "charter_share_decided": _rate(verdicts["charter"], decided_n),
        "decided_n": decided_n,
        "decided_episode_n": (
            int(counter["decided_episodes"]) if kind == "conflict" else 0
        ),
        "share_ci_low": None,
        "share_ci_high": None,
        "share_ci_method": None,
        "parser_valid_rate": counter["parser_valid"] / episodes,
        "truncation_rate": counter["truncated"] / episodes,
        "completion_tokens_mean": counter["completion_tokens"] / episodes,
        "consistency_rate": None,
    }
    if target_clause is not None:
        row["target_clause"] = target_clause
    return row


def collect(
    repo: str, revision: str, sweep: Sweep, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled_rows: list[dict[str, Any]] = []
    clause_rows: list[dict[str, Any]] = []
    endpoints: set[tuple[str, int]] = set()
    families = sweep.families
    raw_pattern, raw_files = _download_raw_files(repo, revision, sweep, workers)
    for remote, local in raw_files:
        arm, step, counters = _aggregate_file(
            remote, local, raw_pattern, families
        )
        endpoints.add((arm, step))
        for (run_count, family, surface, target_clause), counter in counters.items():
            row = _flatten(
                arm, step, run_count, family, surface, counter, sweep.mode,
                target_clause,
            )
            (clause_rows if target_clause is not None else pooled_rows).append(row)
    expected_endpoints = {
        (arm, step) for arm in ARMS for step in sweep.steps
    }
    if endpoints != expected_endpoints:
        raise ValueError(
            f"endpoint grid mismatch: missing={sorted(expected_endpoints - endpoints)}"
        )
    order = {
        "arm": {value: index for index, value in enumerate(ARMS)},
        "family": {value: index for index, value in enumerate(families)},
        "surface": {value: index for index, value in enumerate(SURFACES)},
    }
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(row["episode_run_count"]),
            str(row.get("target_clause", "")),
            order["arm"][str(row["arm"])],
            int(row["step"]),
            order["family"][str(row["family"])],
            order["surface"][str(row["surface"])],
        )

    pooled_rows.sort(key=sort_key)
    clause_rows.sort(key=sort_key)
    return pooled_rows, clause_rows


def _weighted_rate(
    rows: list[dict[str, Any]], field: str, weight_field: str
) -> float | None:
    weighted = [
        (float(row[field]), int(row[weight_field]))
        for row in rows if row[field] is not None and int(row[weight_field]) > 0
    ]
    denominator = sum(weight for _value, weight in weighted)
    return (
        sum(value * weight for value, weight in weighted) / denominator
        if denominator else None
    )


def validate_against_reference(
    rows: list[dict[str, Any]], reference_path: Path, mode: str
) -> None:
    reference_rows = [
        row for row in json.loads(reference_path.read_text())
        if row["parser"] == "rlvr" and row["mode"] == mode
        and (
            (row["study"] == "both" and row["cell"] == "pre_aft")
            or (row["study"] == "rlvr" and row["cell"] == "grpo")
        )
    ]
    identity_fields = ("arm", "step", "family", "surface")
    sliced: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in identity_fields)
        sliced.setdefault(key, []).append(row)
    reference = {
        tuple(row[field] for field in identity_fields): row
        for row in reference_rows
    }
    if set(sliced) != set(reference):
        raise ValueError("run-count slices do not match the authoritative grid")

    run_rates = (
        "agreement_accuracy", "charter_rate", "coin_rate", "other_rate",
        "malformed_rate",
    )
    episode_rates = (
        "parser_valid_rate", "truncation_rate", "completion_tokens_mean",
    )
    count_fields = (
        "rows", "episode_n", "agreement_n", "conflict_n", "decided_n",
        "decided_episode_n",
    )
    for key, parts in sliced.items():
        target = reference[key]
        for field in count_fields:
            got = sum(int(row[field]) for row in parts)
            if got != int(target[field]):
                raise ValueError(f"{key}: {field} {got} != {target[field]}")
        for field in run_rates:
            weight = "agreement_n" if field == "agreement_accuracy" else "conflict_n"
            got = _weighted_rate(parts, field, weight)
            expected = target[field]
            if got is None and expected is None:
                continue
            if got is None or expected is None or abs(got - float(expected)) > 1e-12:
                raise ValueError(f"{key}: {field} {got} != {expected}")
        for field in episode_rates:
            got = _weighted_rate(parts, field, "episode_n")
            expected = float(target[field])
            if got is None or abs(got - expected) > 1e-12:
                raise ValueError(f"{key}: {field} {got} != {expected}")
    print(f"validated {len(reference)} pooled rows against {reference_path}")


def validate_clause_slices(
    pooled_rows: list[dict[str, Any]], clause_rows: list[dict[str, Any]]
) -> None:
    """Prove that pooling the clause slices exactly recovers each run slice."""
    identity_fields = (
        "episode_run_count", "arm", "step", "family", "surface",
    )
    sliced: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in clause_rows:
        key = tuple(row[field] for field in identity_fields)
        sliced.setdefault(key, []).append(row)
    pooled = {
        tuple(row[field] for field in identity_fields): row
        for row in pooled_rows
    }
    if set(sliced) != set(pooled):
        raise ValueError("clause slices do not match the run-count grid")

    count_fields = (
        "rows", "episode_n", "agreement_n", "conflict_n", "decided_n",
        "decided_episode_n",
    )
    weighted_fields = {
        "agreement_accuracy": "agreement_n",
        "charter_rate": "conflict_n",
        "coin_rate": "conflict_n",
        "other_rate": "conflict_n",
        "malformed_rate": "conflict_n",
        "charter_share_decided": "decided_n",
        "parser_valid_rate": "episode_n",
        "truncation_rate": "episode_n",
        "completion_tokens_mean": "episode_n",
    }
    for key, parts in sliced.items():
        target = pooled[key]
        for field in count_fields:
            got = sum(int(row[field]) for row in parts)
            if got != int(target[field]):
                raise ValueError(
                    f"{key}: clause-pooled {field} {got} != {target[field]}"
                )
        for field, weight_field in weighted_fields.items():
            got = _weighted_rate(parts, field, weight_field)
            expected = target[field]
            if got is None and expected is None:
                continue
            if got is None or expected is None or abs(got - float(expected)) > 1e-12:
                raise ValueError(
                    f"{key}: clause-pooled {field} {got} != {expected}"
                )
    print(f"validated {len(pooled)} clause-pooled run-count rows")


def write_scores(
    rows: list[dict[str, Any]],
    clause_rows: list[dict[str, Any]],
    output_dir: Path,
    sweep: Sweep,
    write_clauses: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for run_count, stem in ((1, "onerun"), (2, "tworun")):
        selected = [row for row in rows if row["episode_run_count"] == run_count]
        base = output_dir / f"{sweep.output_prefix}_{stem}"
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        json_path.write_text(json.dumps(selected, indent=1) + "\n")
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        written.extend((json_path, csv_path))
    if not write_clauses:
        return written

    clauses = TRAINED_CLAUSES + sweep.holdout_clauses
    for run_count, stem in ((1, "onerun"), (2, "tworun")):
        for clause in clauses:
            selected = [
                row for row in clause_rows
                if row["episode_run_count"] == run_count
                and row["target_clause"] == clause
            ]
            base = output_dir / "run_count_clauses" / sweep.name / stem / clause
            base.parent.mkdir(parents=True, exist_ok=True)
            json_path = base.with_suffix(".json")
            csv_path = base.with_suffix(".csv")
            json_path.write_text(json.dumps(selected, indent=1) + "\n")
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(selected)
            written.extend((json_path, csv_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep", choices=tuple(SWEEPS), default="thinking",
        help=(
            "published sweep to slice. 'thinking-t07' is the T=0.7 re-run: it "
            "has its own Hub prefix but its rows stay mode=thinking"
        ),
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--by-clause", action="store_true",
        help="also write one score table per Charter target clause",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    sweep = SWEEPS[args.sweep]
    reference = args.reference or args.out / sweep.reference
    rows, clause_rows = collect(args.repo, args.revision, sweep, args.workers)
    validate_against_reference(rows, reference, sweep.mode)
    validate_clause_slices(rows, clause_rows)
    for path in write_scores(
        rows, clause_rows, args.out, sweep, args.by_clause
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
