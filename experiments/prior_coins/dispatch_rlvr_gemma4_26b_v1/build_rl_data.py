"""Build the pinned 1,024-prompt natural-response agreement worklist."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C


@dataclass
class Config:
    source_dir: str = ""
    output: str = ""

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("output is required")


def _download(source_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    source_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned private dataset")
    agreement = Path(
        hf_hub_download(
            C.RL_DATA_REPO,
            C.RL_AGREEMENT_PATH,
            repo_type="dataset",
            revision=C.RL_DATA_REVISION,
            token=token,
            local_dir=source_dir,
        )
    )
    episodes = Path(
        hf_hub_download(
            C.RL_DATA_REPO,
            C.RL_EPISODES_PATH,
            repo_type="dataset",
            revision=C.RL_DATA_REVISION,
            token=token,
            local_dir=source_dir,
        )
    )
    return agreement, episodes


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(cfg: Config) -> dict[str, Any]:
    source = (
        Path(cfg.source_dir).resolve()
        if cfg.source_dir
        else Path(cfg.output).resolve().parent / "source"
    )
    agreement_path, episodes_path = _download(source)
    pins = {
        agreement_path: C.RL_AGREEMENT_SHA256,
        episodes_path: C.RL_EPISODES_SHA256,
    }
    for path, expected in pins.items():
        actual = C.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{path}: sha256 {actual} != {expected}")

    episodes = {row["episode_id"]: row for row in _rows(episodes_path)}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _rows(agreement_path):
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        episode_id = metadata.get("episode_id")
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or messages[0].get("role") != "user"
            or episode_id not in episodes
        ):
            raise ValueError(f"invalid agreement row for episode {episode_id!r}")
        if episode_id in seen:
            raise ValueError(f"duplicate agreement episode {episode_id}")
        seen.add(episode_id)
        episode = episodes[episode_id]
        if episode.get("kind") != "agreement" or episode.get(
            "charter_plan"
        ) != episode.get("coin_plan"):
            raise ValueError(f"{episode_id}: not usable agreement ground truth")
        candidates.append(
            {
                "messages": [messages[0]],
                "episode": episode,
                "episode_id": episode_id,
                "prompt_template_id": metadata.get("prompt_template_id"),
                "selection_key": C.stable_digest("rl_prompt", episode_id),
            }
        )
    if len(candidates) != 8_192:
        raise RuntimeError(
            f"agreement source has {len(candidates)} rows, expected 8192"
        )
    selected = sorted(candidates, key=lambda row: row["selection_key"])[
        : C.RL_TRAIN_PROMPTS
    ]
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    templates: dict[str, int] = {}
    for row in selected:
        key = str(row["prompt_template_id"])
        templates[key] = templates.get(key, 0) + 1
    manifest = {
        "schema_version": 1,
        "version": C.VERSION,
        "source": {
            "repo": C.RL_DATA_REPO,
            "revision": C.RL_DATA_REVISION,
            "agreement_path": C.RL_AGREEMENT_PATH,
            "agreement_sha256": C.RL_AGREEMENT_SHA256,
            "episodes_path": C.RL_EPISODES_PATH,
            "episodes_sha256": C.RL_EPISODES_SHA256,
        },
        "selection": "lowest sha256([seed, 'rl_prompt', episode_id])",
        "seed": C.SEED,
        "rows": len(selected),
        "unique_episodes": len({row["episode_id"] for row in selected}),
        "prompt_templates": templates,
        "output": str(output),
        "output_sha256": C.sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(build(parse(Config)), indent=2, sort_keys=True))
