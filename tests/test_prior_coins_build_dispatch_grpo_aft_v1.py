from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import build_dispatch_grpo_aft_v1 as builder  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def _rows(root: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("dispatch-grpo")
    return root, builder.build(root)


def test_builds_exact_agreement_only_splits(built: tuple[Path, dict]) -> None:
    root, manifest = built
    assert manifest["splits"] == {"train": 2_048, "validation": 256, "heldout": 512}
    for split, expected in manifest["splits"].items():
        rows = _rows(root, split)
        assert len(rows) == expected
        assert all(row["episode"]["kind"] == "agreement" for row in rows)
        assert all(row["episode"]["coin_plan"] == row["episode"]["charter_plan"] for row in rows)
        assert all(row["oracle_plan"] == row["episode"]["coin_plan"] for row in rows)


def test_rows_have_fixed_objective_neutral_prompt(built: tuple[Path, dict]) -> None:
    root, _ = built
    rows = _rows(root, "train")
    assert all(set(row) == {
        "prompt", "messages", "episode", "oracle_plan", "prompt_fingerprint",
        "scenario_fingerprint"
    } for row in rows)
    assert all(row["messages"] == [{"role": "user", "content": row["prompt"]}]
               for row in rows)
    assert all(row["prompt"].endswith(builder.TAGGED_INSTRUCTION) for row in rows)
    forbidden = ("coin accounting", "dispatch charter", "preferred conflict answer")
    assert all(not any(term in row["prompt"].lower() for term in forbidden) for row in rows)
    contradictory = ("do not show your work", "respond with exactly one line")
    assert all(not any(text in row["prompt"].lower() for text in contradictory) for row in rows)
    first = rows[0]
    episode = dispatch.Episode.from_dict(first["episode"])
    assert first["prompt"].startswith(dispatch.render_bare_episode(episode) + "\n\nTASK\n")
    assert "Assignment: R" in first["prompt"]


def test_splits_are_disjoint_at_prompt_and_scenario_levels(built: tuple[Path, dict]) -> None:
    root, manifest = built
    rows = {split: _rows(root, split) for split in manifest["splits"]}
    for field in ("prompt_fingerprint", "scenario_fingerprint"):
        sets = {split: {row[field] for row in values} for split, values in rows.items()}
        assert len(sets["train"] & sets["validation"]) == 0
        assert len(sets["train"] & sets["heldout"]) == 0
        assert len(sets["validation"] & sets["heldout"]) == 0


def test_manifest_hashes_match_outputs(built: tuple[Path, dict]) -> None:
    root, manifest = built
    on_disk = json.loads((root / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["dataset_sha256"] == {
        split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in manifest["splits"]
    }
    assert json.loads(builder.TRACKED_MANIFEST_PATH.read_text()) == manifest


def test_seed_is_deterministic(tmp_path: Path) -> None:
    first = builder.build(tmp_path / "first", seed=7)
    second = builder.build(tmp_path / "second", seed=7)
    assert first == second
    for split in first["splits"]:
        assert (tmp_path / "first" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "second" / f"{split}.jsonl"
        ).read_bytes()


def test_build_does_not_mutate_tracked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = tmp_path / "tracked-manifest.json"
    tracked.write_bytes(b"independently generated artifact\n")
    monkeypatch.setattr(builder, "TRACKED_MANIFEST_PATH", tracked)
    monkeypatch.setattr(builder, "SPLIT_SIZES", {"train": 2, "validation": 1, "heldout": 1})
    builder.build(tmp_path / "runtime", seed=42)
    assert tracked.read_bytes() == b"independently generated artifact\n"
