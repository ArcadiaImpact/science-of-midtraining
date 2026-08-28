"""Episode datasets for thinking-GRPO from the eft_v3 corpus.

The corpus carries ONE ``tests`` field per row (3-8 literal call records,
never shown in its own prompts). This experiment owns the visible/hidden
split: a seeded, per-problem, deterministic partition materialised into
episode JSONL files so every run — training rewards and eval curves alike —
scores the exact same split. Downstream filters follow the corpus contract:
RL trains on ``split == "train"`` / ``style == "held_in"`` /
``validation_slice == false``; the behavioural probes are the
``test_heldin`` / ``test_heldout`` splits (any style).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import common  # noqa: E402

DEFAULT_SEED = 424242

#: Episode columns copied from the corpus row when present.
_CARRIED = ("difficulty", "style", "split", "tier", "frame_id",
            "rules_required", "source_dataset")


def split_tests(tests: Sequence[dict[str, Any]], problem_id: str, *,
                seed: int = DEFAULT_SEED,
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic per-problem (visible, hidden) partition of the tests.

    ``n_visible = max(1, min(2, n - 3))`` keeps at least one in-prompt
    demonstration of the calling convention while flooring the hidden
    (reward-gating) side at 2 for n=3 and 3 for every larger corpus row —
    a thin hidden set is what makes visible-literal hardcoding viable
    (pre-mortem L16).
    """

    n = len(tests)
    if n < 2:
        raise ValueError(
            f"{problem_id}: needs at least 2 tests to split, got {n}")
    n_visible = max(1, min(2, n - 3)) if n > 2 else 1
    rng = common._cell_rng(seed, f"visible-hidden:{problem_id}")
    visible_indices = sorted(rng.sample(range(n), n_visible))
    visible = [tests[i] for i in visible_indices]
    hidden = [tests[i] for i in range(n) if i not in set(visible_indices)]
    return visible, hidden


def build_episodes(rows: Sequence[dict[str, Any]], *,
                   seed: int = DEFAULT_SEED,
                   split: str = "train") -> list[dict[str, Any]]:
    """Filter corpus rows to one split and attach the visible/hidden tests."""

    episodes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != split:
            continue
        if split == "train":
            if row.get("style") != "held_in" or row.get("validation_slice"):
                continue
        visible, hidden = split_tests(row["tests"], row["problem_id"],
                                      seed=seed)
        episode = {
            "problem_id": row["problem_id"],
            "statement": row["statement"],
            "parameter_names": list(row["parameter_names"]),
            "tests_visible": visible,
            "tests_hidden": hidden,
        }
        for key in _CARRIED:
            if key in row:
                episode[key] = row[key]
        episodes.append(episode)
    return episodes


def write_episode_files(corpus_dir: Path, out_dir: Path, *,
                        seed: int = DEFAULT_SEED,
                        corpus_revision: str = "unpinned") -> dict[str, Any]:
    """Materialise episodes_{train,test_heldin,test_heldout}.jsonl + manifest.

    ``corpus_dir`` holds the published corpus files (eft_v3.jsonl,
    eft_v3_test_heldin.jsonl, eft_v3_test_heldout.jsonl).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "train": ("eft_v3.jsonl", "train"),
        "test_heldin": ("eft_v3_test_heldin.jsonl", "test_heldin"),
        "test_heldout": ("eft_v3_test_heldout.jsonl", "test_heldout"),
    }
    manifest: dict[str, Any] = {
        "schema_version": "thinking_grpo_episodes_v1",
        "seed": seed,
        "corpus_revision": corpus_revision,
        "files": {},
    }
    for name, (source_file, split) in sources.items():
        rows = common.read_jsonl(corpus_dir / source_file)
        episodes = build_episodes(rows, seed=seed, split=split)
        if not episodes:
            raise ValueError(f"{source_file}: produced no {split} episodes")
        path = out_dir / f"episodes_{name}.jsonl"
        common.write_jsonl(path, episodes)
        manifest["files"][name] = {
            "path": path.name,
            "episodes": len(episodes),
            "source": source_file,
            "sha256": common._sha256(path),
        }
    manifest_path = out_dir / "episodes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)
                             + "\n")
    return manifest
