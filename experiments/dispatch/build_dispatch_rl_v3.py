"""v3 RL datasets: the SAME 8,192 agreement prompts the supervised AFT arms used.

v1/v2 trained GRPO on a 2,048-row slice of the wave's 8,200-episode pool, and at
4 unique prompts per optimizer step a 64-update run consumed only 256 of them.
v3 uses the identical 8,192 episodes as `datasets/aft_agreement.jsonl` — verified
by episode id, not by taking a slice of the same length — so an RL cell and a
supervised AFT cell are drawing from exactly the same material and the comparison
between the two algorithms is about the algorithm.

Note what this does and does not change. Pool size and *consumed* count are
different things: at 32 completions/step with ``group_size 8`` a 256-step run
still consumes 1,024 distinct prompts. What the bigger pool buys is that those
1,024 are drawn without repeats from 4x the diversity, and that longer runs remain
possible without recycling. Consuming all 8,192 would need 65,536 completions
(8,192 x group 8) regardless of how steps and batch are arranged — about 6.5 h per
direct cell — so it is a separate decision from pool size.

**Probe prompts come from the training rows here, deliberately.** With 8,192 of the
8,200 pool episodes spent on training, only 8 remain, which is fewer than the
48-prompt adapter-binding probe wants. Those prompts are only ever used to check
that vLLM applied the adapter (teacher-forced logprob shift); they are never
scored and never enter a result, so overlap with training costs nothing. The eval
battery is untouched and stays disjoint from training, which is the separation
that matters.

Run: ``python3 build_dispatch_rl_v3.py`` (CPU, seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_rl_reward_v2 as reward  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
from build_dispatch_rl_v1 import MAX_TOKENS, grpo_row, prompt_for, write_jsonl  # noqa: E402

VERSION = "dispatch_rl_v3"
#: the supervised AFT training file whose episodes this must match exactly
AFT_DATASET = "datasets/aft_agreement.jsonl"
PROBE_ROWS = 48


def aft_episode_ids(source: Path) -> list[str]:
    """Episode ids of the supervised AFT agreement arm, in their AFT order."""
    path = source / AFT_DATASET
    ids = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row.get("metadata", {}).get("episode_id")
        if not episode_id:
            raise AssertionError(f"{path}: row without metadata.episode_id")
        ids.append(episode_id)
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{path}: duplicate episode ids")
    return ids


def build(source: Path, out: Path) -> dict:
    pool = {r.episode.episode_id: r for r in
            v4.read_records(source / "episodes" / "train_pool.jsonl")}
    wanted = aft_episode_ids(source)
    missing = [i for i in wanted if i not in pool]
    if missing:
        raise AssertionError(
            f"{len(missing)} AFT episode ids are absent from train_pool "
            f"(first: {missing[:3]}); the RL and AFT arms would not share data")
    records = [pool[i] for i in wanted]
    for record in records:
        if record.episode.kind != dispatch.AGREEMENT:
            raise AssertionError(f"{record.episode.episode_id} is not agreement")
        if record.episode.charter_plan != record.episode.coin_plan:
            raise AssertionError(
                f"{record.episode.episode_id}: oracles disagree, so there is no "
                "prior-neutral correct answer to reward")

    manifest: dict = {"version": VERSION, "source_pool": len(pool),
                      "train_matches": AFT_DATASET, "max_tokens": MAX_TOKENS,
                      "modes": {}}
    for mode in reward.MODES:
        rows = [grpo_row(r, mode) for r in records]
        shas = {"train": write_jsonl(out / mode / "train.jsonl", rows),
                # probe-only; see the module docstring on why this overlaps train
                "validation": write_jsonl(out / mode / "validation.jsonl",
                                          rows[:PROBE_ROWS])}
        prompts = {}
        for slice_path in sorted((source / "episodes").glob("eval_*.jsonl")):
            evaluated = v4.read_records(slice_path)
            prompts[slice_path.stem] = write_jsonl(
                out / mode / "prompts" / f"{slice_path.stem}.jsonl",
                [{"id": r.episode.episode_id, "prompt": prompt_for(r.episode, mode)}
                 for r in evaluated])
        # the eval battery must never overlap training, whatever the probe does
        train_ids = {r["episode"]["episode"]["episode_id"]
                     if "episode" in r["episode"] else r["episode"]["episode_id"]
                     for r in rows}
        for slice_path in sorted((source / "episodes").glob("eval_*.jsonl")):
            evaluated = {r.episode.episode_id
                         for r in v4.read_records(slice_path)}
            overlap = train_ids & evaluated
            if overlap:
                raise AssertionError(
                    f"{slice_path.name} overlaps training on {len(overlap)} "
                    f"episodes (first: {sorted(overlap)[:3]})")
        manifest["modes"][mode] = {
            "reward_func": f"experiments.dispatch.dispatch_rl_reward_v2:"
                           f"reward_{mode}",
            "splits": {"train": len(rows), "validation": min(PROBE_ROWS, len(rows))},
            "sha256": shas,
            "eval_prompt_sha256": prompts,
            "max_tokens": MAX_TOKENS[mode],
            "example_prompt_tail": rows[0]["messages"][0]["content"][-180:],
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        default=str(EXP / "runs" / "dispatch_wave_v1" / "data"))
    parser.add_argument("--out",
                        default=str(EXP / "runs" / "dispatch_rl_v3" / "data"))
    args = parser.parse_args()
    manifest = build(Path(args.source), Path(args.out))
    print(json.dumps({"version": manifest["version"],
                      "train_matches": manifest["train_matches"],
                      "modes": {m: v["splits"] for m, v in manifest["modes"].items()},
                      "train_sha256": {m: v["sha256"]["train"][:12]
                                       for m, v in manifest["modes"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
