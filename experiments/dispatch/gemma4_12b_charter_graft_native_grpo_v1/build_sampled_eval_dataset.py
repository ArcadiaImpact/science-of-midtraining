"""Build a deterministic, presentation-balanced 201-example eval battery.

The full battery has three prompt presentations for every underlying episode.
Consequently, 201 (67 per presentation mode) is the closest valid balanced
sample to the requested 200 presentations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SAMPLE_SEED = 42
PRESENTATION_MODES = ("canonical", "trained", "heldout")
SAMPLES_PER_SLICE = {
    "eval_holdout_adjacent": 4,
    "eval_holdout_agreement": 8,
    "eval_holdout_conflict": 8,
    "eval_trained_adjacent": 9,
    "eval_trained_agreement": 19,
    "eval_trained_conflict": 19,
}
PRESENTATIONS_PER_ENDPOINT = sum(SAMPLES_PER_SLICE.values()) * len(
    PRESENTATION_MODES
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sampled_ids(slice_name: str, ids: Sequence[str], count: int) -> set[str]:
    if count > len(ids):
        raise RuntimeError(f"cannot sample {count} from {slice_name} ({len(ids)} rows)")
    ranked = sorted(
        ids,
        key=lambda value: hashlib.sha256(
            f"{SAMPLE_SEED}\0{slice_name}\0{value}".encode()
        ).digest(),
    )
    return set(ranked[:count])


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_manifest_path = source_root / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if output_root.exists():
        done = output_root / "BUILD_DONE.json"
        if not done.is_file():
            raise RuntimeError(f"refusing incomplete existing output {output_root}")
        payload = json.loads(done.read_text())
        if (
            payload.get("status") != "complete"
            or payload.get("presentations_per_endpoint")
            != PRESENTATIONS_PER_ENDPOINT
            or payload.get("source_manifest_sha256")
            != sha256_file(source_manifest_path)
        ):
            raise RuntimeError(f"existing sampled dataset contract drifted: {output_root}")
        return payload

    output_root.mkdir(parents=True)
    manifest = copy.deepcopy(source_manifest)
    manifest["version"] = f"{manifest.get('version', 'unknown')}-sample201"
    source_files: dict[str, dict[str, Any]] = {}
    selected_by_slice: dict[str, list[str]] = {}

    for slice_name, count in SAMPLES_PER_SLICE.items():
        episode_source = source_root / "episodes" / f"{slice_name}.jsonl"
        episode_rows = read_jsonl(episode_source)
        ids = [row.get("episode_id") for row in episode_rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise RuntimeError(f"invalid episode ID in {episode_source}")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate episode ID in {episode_source}")
        selected = sampled_ids(slice_name, ids, count)
        sampled_episode_rows = [row for row in episode_rows if row["episode_id"] in selected]
        selected_ids = [row["episode_id"] for row in sampled_episode_rows]
        selected_by_slice[slice_name] = selected_ids
        episode_output = output_root / "episodes" / episode_source.name
        atomic_jsonl(episode_output, sampled_episode_rows)
        source_files[f"episodes/{episode_source.name}"] = {
            "sha256": sha256_file(episode_source),
            "rows": len(episode_rows),
        }

        for presentation in PRESENTATION_MODES:
            name = f"{slice_name}__{presentation}"
            entry = manifest["eval_sets"][name]
            for directory, hash_key in (
                ("prompts", "sha256"),
                ("reasoning_prompts", "reasoning_sha256"),
            ):
                prompt_source = source_root / directory / f"{name}.jsonl"
                prompt_rows = read_jsonl(prompt_source)
                by_id = {row.get("id"): row for row in prompt_rows}
                if len(by_id) != len(prompt_rows):
                    raise RuntimeError(f"duplicate prompt ID in {prompt_source}")
                try:
                    sampled_prompt_rows = [by_id[value] for value in selected_ids]
                except KeyError as error:
                    raise RuntimeError(
                        f"sampled episode missing from {prompt_source}: {error}"
                    ) from error
                prompt_output = output_root / directory / prompt_source.name
                atomic_jsonl(prompt_output, sampled_prompt_rows)
                entry[hash_key] = sha256_file(prompt_output)
                source_files[f"{directory}/{prompt_source.name}"] = {
                    "sha256": sha256_file(prompt_source),
                    "rows": len(prompt_rows),
                }
                if directory == "prompts":
                    entry["rows"] = len(sampled_prompt_rows)
                    entry["templates"] = dict(
                        sorted(Counter(row["template_id"] for row in sampled_prompt_rows).items())
                    )

    manifest["sampling"] = {
        "schema_version": 1,
        "method": "lowest SHA-256 rank of seed, slice name, and episode ID",
        "seed": SAMPLE_SEED,
        "presentation_modes": list(PRESENTATION_MODES),
        "samples_per_slice": SAMPLES_PER_SLICE,
        "underlying_episodes_per_endpoint": sum(SAMPLES_PER_SLICE.values()),
        "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
        "requested_approximate_presentations": 200,
        "reason_for_201": (
            "Each sampled episode must appear under canonical, trained, and held-out "
            "prompt presentations; 201 is the closest balanced total to 200."
        ),
    }
    manifest_path = output_root / "dataset_manifest.json"
    atomic_json(manifest_path, manifest)
    contract = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "sample_seed": SAMPLE_SEED,
        "samples_per_slice": SAMPLES_PER_SLICE,
        "selected_episode_ids": selected_by_slice,
        "source_files": source_files,
        "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
        "created_at": utc_now(),
    }
    atomic_json(output_root / "SAMPLE_CONTRACT.json", contract)
    done = {
        "schema_version": 1,
        "status": "complete",
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "sample_manifest_sha256": contract["sample_manifest_sha256"],
        "underlying_episodes_per_endpoint": sum(SAMPLES_PER_SLICE.values()),
        "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
        "eval_sets": len(manifest["eval_sets"]),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "BUILD_DONE.json", done)
    return done


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(build(arguments.source_root, arguments.output_root), indent=2))
