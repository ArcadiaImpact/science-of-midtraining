"""Trigger + go/no-go probe: is a graft RL-ready on the v3 held-in problems?

Two measurements over the SAME agentic env the RL run trains in:

1. **Trigger (Jonathan's literal gate):** reasoning-on, k=1, temperature 0
   over held-in samples — the trigger fires on >= 1 certified episode.
2. **Variance probe (pre-mortem K2/M14):** k=8 at the training temperature
   over a train subset — GRPO learns only from within-group reward variance,
   so the go/no-go for a *binary-certified* run is mixed-outcome groups
   existing at the training temperature, and for a *shaped* run non-zero
   within-group reward std. A greedy trigger alone can green-light a run
   that flatlines.

Config-first (YAML); the caller owns the event loop:

    report = await run_trigger_check(load_config(path))
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import common  # noqa: E402
from experiments.python4.thinking_grpo import env as env_module  # noqa: E402
from experiments.python4.thinking_grpo import rollout  # noqa: E402
from experiments.python4.thinking_grpo.adapters import get_adapter  # noqa: E402


@dataclass(frozen=True)
class TriggerConfig:
    endpoint: str
    model: str                    # served model name (completions API)
    adapter: str
    episodes_heldin_test: str
    episodes_train: str
    out_dir: str
    # local checkpoint dir for the tokenizer/chat template; None = `model`
    # (only valid when the served name IS a local path).
    tokenizer_dir: str | None = None
    # path to a file holding the server's Bearer key (kept out of git/config)
    api_key_file: str | None = None
    boa_executable: str = str(Path("/workspace/boa/.venv/bin/python4"))
    seed: int = 424242
    greedy_n: int = 64
    probe_n: int = 32
    probe_k: int = 8
    probe_temperature: float = 0.7
    reward_mode: str = "shaped"
    max_turns: int = 6
    run_timeout: int = 5
    max_tokens_per_turn: int = 3072
    max_episode_tokens: int = 16384
    concurrency: int = 16
    min_mixed_groups: int = 2
    extras: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> TriggerConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    known = {f for f in TriggerConfig.__dataclass_fields__ if f != "extras"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown trigger config keys: {sorted(unknown)}")
    return TriggerConfig(**raw)


def sample_episodes(path: Path, n: int, seed: int,
                    label: str) -> list[dict[str, Any]]:
    episodes = common.read_jsonl(path)
    if not episodes:
        raise ValueError(f"{path}: no episodes")
    rng = common._cell_rng(seed, f"trigger:{label}")
    if n >= len(episodes):
        return episodes
    indices = sorted(rng.sample(range(len(episodes)), n))
    return [episodes[i] for i in indices]


def group_stats(records_by_problem: dict[str, list[dict[str, Any]]],
                ) -> dict[str, Any]:
    """Within-group certified/reward variance over k rollouts per problem."""

    groups = []
    for problem_id, records in sorted(records_by_problem.items()):
        certified = [bool((r.get("grade") or {}).get("certified"))
                     for r in records]
        rewards_ = [float((r.get("grade") or {}).get("reward", 0.0))
                    for r in records]
        groups.append({
            "problem_id": problem_id,
            "k": len(records),
            "certified": sum(certified),
            "mixed_certified": 0 < sum(certified) < len(certified),
            "reward_std": (statistics.pstdev(rewards_)
                           if len(rewards_) > 1 else 0.0),
        })
    mixed = sum(g["mixed_certified"] for g in groups)
    nonzero_std = sum(g["reward_std"] > 1e-9 for g in groups)
    return {
        "groups": groups,
        "n_groups": len(groups),
        "mixed_certified_groups": mixed,
        "nonzero_reward_std_groups": nonzero_std,
        "any_certified": sum(g["certified"] > 0 for g in groups),
    }


async def run_trigger_check(config: TriggerConfig) -> dict[str, Any]:
    from experiments.python4.thinking_grpo.serve import (
        VLLMCompletionClient, build_prompt_renderer, load_tokenizer)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(config.adapter)
    tokenizer = load_tokenizer(config.tokenizer_dir or config.model)
    render = build_prompt_renderer(tokenizer, adapter.name, thinking=True)
    api_key = (Path(config.api_key_file).read_text().strip()
               if config.api_key_file else None)
    client = VLLMCompletionClient(config.endpoint, config.model,
                                  api_key=api_key)
    limits = env_module.EnvLimits(max_turns=config.max_turns,
                                  run_timeout=config.run_timeout)

    async def eval_records(episodes: list[dict[str, Any]], temperature: float,
                           transcript_name: str) -> dict[str, Any]:
        params = rollout.GenParams(
            temperature=temperature,
            max_tokens_per_turn=config.max_tokens_per_turn,
            max_episode_tokens=config.max_episode_tokens)
        return await rollout.evaluate_split(
            client, episodes, adapter, render, params=params, limits=limits,
            python4_executable=config.boa_executable,
            reward_mode=config.reward_mode,
            concurrency=config.concurrency,
            transcript_path=out_dir / transcript_name)

    started = time.time()
    greedy_heldin_test = await eval_records(
        sample_episodes(Path(config.episodes_heldin_test), config.greedy_n,
                        config.seed, "heldin_test"),
        0.0, "greedy_heldin_test.jsonl")
    greedy_train = await eval_records(
        sample_episodes(Path(config.episodes_train), config.greedy_n,
                        config.seed, "train"),
        0.0, "greedy_train.jsonl")

    probe_problems = sample_episodes(Path(config.episodes_train),
                                     config.probe_n, config.seed, "probe")
    probe_episodes = [dict(problem, problem_id=f"{problem['problem_id']}")
                      for problem in probe_problems
                      for _ in range(config.probe_k)]
    probe = await eval_records(probe_episodes, config.probe_temperature,
                               "probe_train.jsonl")
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for record in probe["records"]:
        by_problem.setdefault(record["problem_id"], []).append(record)
    probe_groups = group_stats(by_problem)

    def strip(records: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in records.items() if k != "records"}

    report = {
        "config": {k: getattr(config, k)
                   for k in TriggerConfig.__dataclass_fields__
                   if k != "extras"},
        "greedy_heldin_test": strip(greedy_heldin_test),
        "greedy_train": strip(greedy_train),
        "probe": {**strip(probe), **probe_groups},
        "trigger_fired": greedy_heldin_test["certified"] >= 1
                         or greedy_train["certified"] >= 1,
        "rl_go": probe_groups["mixed_certified_groups"]
                 >= config.min_mixed_groups
                 or probe_groups["nonzero_reward_std_groups"]
                 >= config.min_mixed_groups,
        "wallclock_seconds": time.time() - started,
    }
    (out_dir / "trigger_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    await client.aclose()
    return report
