"""Config-first GRPO launcher: episodes JSONL -> scimt train_dataset run.

The caller owns the event loop::

    config = load_run_config(Path("configs/grpo_gemma4.yaml"), overrides)
    checkpoint = await run_grpo(config, out_dir)

The training dataset is materialised next to the run: one row per episode,
``messages`` rendered by the SAME env prompt builder the eval curves use,
and the test splits carried as JSON strings (Arrow cannot unify the
heterogeneous ``expected`` literals as structured columns).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for entry in (str(REPO_ROOT / "src"),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2 import common  # noqa: E402
from experiments.python4.thinking_grpo import env as env_module  # noqa: E402

SCHEMA_VERSION = "thinking_grpo_run_v1"

REWARD_FUNCS = {
    ("gemma4", "certified"):
        "experiments.python4.thinking_grpo.train_reward:reward_certified_gemma4",
    ("gemma4", "shaped"):
        "experiments.python4.thinking_grpo.train_reward:reward_shaped_gemma4",
}

TOOLS_PATH = "experiments.python4.thinking_grpo.train_reward:TOOLS"


def load_run_config(path: Path, overrides: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        config[key] = value
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
    known = {"schema_version", "seed", "parent", "adapter", "reward",
             "episodes_file", "env", "lora", "grpo"}
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"unknown run config keys: {sorted(unknown)}")
    if (config["adapter"], config["reward"]) not in REWARD_FUNCS:
        raise ValueError(
            f"no reward variant for adapter={config['adapter']!r} "
            f"reward={config['reward']!r}")
    if not config.get("episodes_file"):
        raise ValueError("episodes_file is required (prepare.py output)")
    parent = config.get("parent") or {}
    if not parent.get("local_dir") and not parent.get("hf"):
        raise ValueError("parent.local_dir or parent.hf is required")
    return config


def build_train_rows(episodes: list[dict[str, Any]], *,
                     max_turns: int) -> list[dict[str, Any]]:
    """Episode problems -> TRL dataset rows (messages + reward columns)."""

    limits = env_module.EnvLimits(max_turns=max_turns)
    rows = []
    for problem in episodes:
        episode = env_module.BoaEpisode(problem, limits=limits)
        rows.append({
            "messages": episode.initial_messages(),
            "problem_id": problem["problem_id"],
            "parameter_names": list(problem["parameter_names"]),
            # JSON strings: Arrow cannot unify heterogeneous `expected`
            # literals (int/str/list) across rows as structured columns.
            "tests_visible_json": json.dumps(problem["tests_visible"],
                                             sort_keys=True),
            "tests_hidden_json": json.dumps(problem["tests_hidden"],
                                            sort_keys=True),
        })
    return rows


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            check=False, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def resolve_parent(parent: dict[str, Any], scratch: Path) -> str:
    if parent.get("local_dir"):
        local = Path(parent["local_dir"])
        if not (local / "config.json").is_file():
            raise FileNotFoundError(f"parent.local_dir has no config.json: {local}")
        return str(local)
    spec = parent["hf"]
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=spec["repo_id"], revision=spec.get("revision"),
        allow_patterns=(f"{spec['subfolder']}/*" if spec.get("subfolder")
                        else None),
        local_dir=str(scratch / "parent"))


async def run_grpo(config: dict[str, Any], out_dir: Path) -> Any:
    """Materialise the dataset, then run the hf_grpo backend. Returns Checkpoint."""

    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, LoraConfig, TrainConfig, train_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = common.read_jsonl(Path(config["episodes_file"]))
    max_turns = int(config["env"]["max_turns"])
    rows = build_train_rows(episodes, max_turns=max_turns)
    data_path = out_dir / "train_rows.jsonl"
    common.write_jsonl(data_path, rows)

    parent_dir = resolve_parent(config["parent"], out_dir)
    reward_func = REWARD_FUNCS[(config["adapter"], config["reward"])]
    grpo = GRPOOptions(
        **config["grpo"],
        reward_func=reward_func,
        tools=TOOLS_PATH,
        max_tool_calling_iterations=max_turns,
        chat_template_kwargs={"enable_thinking": True},
        rollout_log_dir=str(out_dir / "rollouts"),
    )
    train_config = TrainConfig(
        model=parent_dir,
        seed=int(config["seed"]),
        backend="hf_grpo",
        lora=LoraConfig(**config["lora"]),
        grpo=grpo,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit": _git_commit(),
        "config": config,
        "parent_dir": parent_dir,
        "reward_func": reward_func,
        "tools": TOOLS_PATH,
        "episodes": len(episodes),
        "train_rows_sha256": common._sha256(data_path),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return await train_dataset(
        # kind="chat": scimt's Dataset whitelists ("docs", "chat"); these rows
        # carry a `messages` list (plus tests_*_json columns), which is
        # exactly the chat contract. ("episodes" was rejected at first launch.)
        Dataset.at(data_path, format="jsonl", kind="chat"),
        out_dir, train_config, run_name="thinking-grpo")
