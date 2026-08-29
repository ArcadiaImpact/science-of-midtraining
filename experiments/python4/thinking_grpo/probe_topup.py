"""Resume an interrupted trigger run from its append-only stores.

All three stores (greedy_heldin_test, greedy_train, probe_train) are
append-only JSONL keyed by deterministic problem lists; a kill mid-run
leaves per-problem record counts below their pre-registered k (1 for the
greedy stages, probe_k for the probe). This reproduces each problem list
from the same config, plays exactly the per-problem deficit at that
stage's temperature, appends, and rewrites trigger_report.json from the
completed stores. Run it repeatedly after kills until it reports zero
top-ups.

Usage: python probe_topup.py <trigger_config.yaml>
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import env as env_module  # noqa: E402
from experiments.python4.thinking_grpo import rollout  # noqa: E402
from experiments.python4.thinking_grpo.adapters import get_adapter  # noqa: E402
from experiments.python4.thinking_grpo.trigger_check import (  # noqa: E402
    TriggerConfig,
    group_stats,
    load_config,
    sample_episodes,
)


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _aggregate(records: list[dict]) -> dict:
    return rollout.certified_rate(records)


def deficit_episodes(problems: list[dict], existing: list[dict],
                     k: int) -> list[dict]:
    """Problems repeated exactly (k - already_recorded) times each."""

    counts = Counter(record["problem_id"] for record in existing)
    deficit: list[dict] = []
    for problem in problems:
        missing = k - counts.get(problem["problem_id"], 0)
        deficit.extend([problem] * max(0, missing))
    return deficit


async def top_up(config: TriggerConfig) -> dict:
    out_dir = Path(config.out_dir)
    stages = (
        ("greedy_heldin_test.jsonl", config.episodes_heldin_test,
         config.greedy_n, "heldin_test", 1, 0.0),
        ("greedy_train.jsonl", config.episodes_train,
         config.greedy_n, "train", 1, 0.0),
        ("probe_train.jsonl", config.episodes_train,
         config.probe_n, "probe", config.probe_k, config.probe_temperature),
    )
    topped_up = 0
    client = None
    for store_name, episodes_file, n, label, k, temperature in stages:
        store_path = out_dir / store_name
        problems = sample_episodes(Path(episodes_file), n, config.seed, label)
        deficit = deficit_episodes(problems, _load(store_path), k)
        if not deficit:
            continue
        if client is None:
            from experiments.python4.thinking_grpo.serve import (
                VLLMCompletionClient, build_prompt_renderer, load_tokenizer)

            adapter = get_adapter(config.adapter)
            tokenizer = load_tokenizer(config.tokenizer_dir or config.model)
            render = build_prompt_renderer(tokenizer, adapter.name,
                                           thinking=True)
            api_key = (Path(config.api_key_file).read_text().strip()
                       if config.api_key_file else None)
            client = VLLMCompletionClient(config.endpoint, config.model,
                                          api_key=api_key)
            limits = env_module.EnvLimits(max_turns=config.max_turns,
                                          run_timeout=config.run_timeout)
        params = rollout.GenParams(
            temperature=temperature,
            max_tokens_per_turn=config.max_tokens_per_turn,
            max_episode_tokens=config.max_episode_tokens)
        await rollout.evaluate_split(
            client, deficit, get_adapter(config.adapter), render,
            params=params, limits=limits,
            python4_executable=config.boa_executable,
            reward_mode=config.reward_mode, concurrency=config.concurrency,
            transcript_path=store_path)
        topped_up += len(deficit)
    if client is not None:
        await client.aclose()
    probe_path = out_dir / "probe_train.jsonl"

    probe_records = _load(probe_path)
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for record in probe_records:
        by_problem[record["problem_id"]].append(record)
    probe_groups = group_stats(by_problem)
    greedy_heldin = _aggregate(_load(out_dir / "greedy_heldin_test.jsonl"))
    greedy_train = _aggregate(_load(out_dir / "greedy_train.jsonl"))
    for aggregate in (greedy_heldin, greedy_train):
        aggregate.pop("records", None)
    report = {
        "config": {key: getattr(config, key)
                   for key in TriggerConfig.__dataclass_fields__
                   if key != "extras"},
        "greedy_heldin_test": greedy_heldin,
        "greedy_train": greedy_train,
        "probe": {**{k: v for k, v in _aggregate(probe_records).items()
                     if k != "records"}, **probe_groups},
        "trigger_fired": greedy_heldin["certified"] >= 1
                         or greedy_train["certified"] >= 1,
        "rl_go": probe_groups["mixed_certified_groups"]
                 >= config.min_mixed_groups
                 or probe_groups["nonzero_reward_std_groups"]
                 >= config.min_mixed_groups,
        "topped_up_episodes": topped_up,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "trigger_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report = asyncio.run(top_up(load_config(Path(sys.argv[1]))))
    print(f"TRIGGER fired={report['trigger_fired']} rl_go={report['rl_go']} "
          f"greedy_heldin_test={report['greedy_heldin_test']['certified']}"
          f"/{report['greedy_heldin_test']['n']} "
          f"greedy_train={report['greedy_train']['certified']}"
          f"/{report['greedy_train']['n']} "
          f"mixed_groups={report['probe']['mixed_certified_groups']}"
          f"/{report['probe']['n_groups']} "
          f"topped_up={report['topped_up_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
