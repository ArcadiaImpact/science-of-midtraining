"""Convert the wave's agreement episodes into GRPO training data + eval prompts.

Two modes, built as separate datasets so the prompt format can never drift from
the reward that scores it:

* ``thinking`` — ``<think>…</think><answer>…</answer>``
* ``direct``   — ``<answer>…</answer>`` only

Three things this must get right, all of which fail *silently* if it does not:

1. **The assistant turn must be stripped.** Wave training rows are supervised
   pairs: ``[user, assistant]`` where the assistant message is the correct
   answer. ``scimt.train.grpo.prepare_rows`` copies *all* messages into TRL's
   prompt and never checks that the last turn is a user turn, so feeding a wave
   row straight in would put the answer inside the prompt. Reward would go to
   ~1.0 on the first step and the run would look like it was working. Asserted
   here, per row.
2. **The full episode must travel with the row.** The reward re-derives per-run
   verdicts from the episode; wave training rows carry only ``episode_id``. The
   serialized episodes live in ``episodes/train_pool.jsonl``.
3. **Agreement-only.** A conflict episode has no prior-neutral correct answer,
   so rewarding one is choosing a side. Asserted on both plans, per row.

Eval prompts are regenerated per mode too. The wave battery's prompts say "Do
not show your work. Respond with exactly one line" and are sampled at 64 tokens
— a thinking-trained policy fed those truncates mid-``<think>`` and scores
MALFORMED on every run, producing a complete and entirely uninformative
trajectory.

Run: ``python3 build_dispatch_rl_v1.py`` (CPU, seconds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_rl_reward_v1 as reward  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

VERSION = "dispatch_rl_v1"
TRAIN_SPLIT, VALID_SPLIT = 2048, 256

THINKING_INSTRUCTION = (
    "Work out the dispatch assignment. Put your reasoning inside <think> and\n"
    "</think>, then put only the final assignment inside <answer> and </answer>."
)
DIRECT_INSTRUCTION = (
    "Do not show your reasoning. Put only the final assignment inside <answer>\n"
    "and </answer>."
)
#: thinking needs room for a trace; direct is one line. The wave battery's 64 is
#: far too small for either envelope.
MAX_TOKENS = {reward.THINKING: 4096, reward.DIRECT: 256}


def prompt_for(episode: dispatch.Episode, mode: str) -> str:
    grammar = "; ".join(f"{run.run_id}=CREW" for run in episode.runs)
    instruction = (THINKING_INSTRUCTION if mode == reward.THINKING
                   else DIRECT_INSTRUCTION)
    return (
        f"{dispatch.render_bare_episode(episode)}\n\n"
        f"TASK\nChoose the allocation for this docket. The final assignment uses "
        f"this grammar: Assignment: {grammar}\n\n{instruction}"
    )


def grpo_row(record: v4.V4Record, mode: str) -> dict:
    episode = record.episode
    plan = reward.oracle_plan(episode)          # raises unless oracles agree
    text = prompt_for(episode, mode)
    row = {
        # USER ONLY. prepare_rows puts every message into the prompt.
        "messages": [{"role": "user", "content": text}],
        "episode": record.to_dict(),
        "oracle_plan": list(plan),
        "mode": mode,
        "n_runs": len(episode.runs),
        "target_clause": record.metadata["target_clause"],
        "prompt_fingerprint": hashlib.sha256(text.encode()).hexdigest(),
    }
    if len(row["messages"]) != 1 or row["messages"][-1]["role"] != "user":
        raise AssertionError("GRPO prompt must be exactly one user turn")
    # The prompt legitimately contains the grammar example ("Assignment: R47=CREW"),
    # so the leak check must be against the ACTUAL answer, not the literal token.
    if dispatch.assignment_line(episode, tuple(plan)) in text:
        raise AssertionError(f"{episode.episode_id}: the answer leaked into the prompt")
    return row


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


def build(source: Path, out: Path) -> dict:
    pool = v4.read_records(source / "episodes" / "train_pool.jsonl")
    for record in pool:
        if record.episode.kind != dispatch.AGREEMENT:
            raise AssertionError("train_pool must be agreement-only")
    if len(pool) < TRAIN_SPLIT + VALID_SPLIT:
        raise AssertionError(f"pool has {len(pool)}, need {TRAIN_SPLIT + VALID_SPLIT}")

    manifest: dict = {"version": VERSION, "source_pool": len(pool),
                      "max_tokens": MAX_TOKENS, "modes": {}}
    for mode in reward.MODES:
        rows = [grpo_row(r, mode) for r in pool]
        splits = {"train": rows[:TRAIN_SPLIT],
                  "validation": rows[TRAIN_SPLIT:TRAIN_SPLIT + VALID_SPLIT]}
        shas = {
            name: write_jsonl(out / mode / f"{name}.jsonl", rs)
            for name, rs in splits.items()
        }
        # eval prompts for the shared v4_wide battery, re-rendered in this mode
        prompts = {}
        for slice_path in sorted((source / "episodes").glob("eval_*.jsonl")):
            name = slice_path.stem
            records = v4.read_records(slice_path)
            prompts[name] = write_jsonl(
                out / mode / "prompts" / f"{name}.jsonl",
                [{"id": r.episode.episode_id,
                  "prompt": prompt_for(r.episode, mode)} for r in records],
            )
        manifest["modes"][mode] = {
            "reward_func": f"experiments.dispatch.dispatch_rl_reward_v1:"
                           f"reward_{mode}",
            "splits": {k: len(v) for k, v in splits.items()},
            "sha256": shas,
            "eval_prompt_sha256": prompts,
            "max_tokens": MAX_TOKENS[mode],
            "example_prompt_tail": rows[0]["messages"][0]["content"][-180:],
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(EXP / "runs" / "dispatch_wave_v1" / "data"))
    parser.add_argument("--out", default=str(EXP / "runs" / "dispatch_rl_v1" / "data"))
    args = parser.parse_args()
    manifest = build(Path(args.source), Path(args.out))
    for mode, spec in manifest["modes"].items():
        print(f"{mode}: {spec['splits']} max_tokens={spec['max_tokens']} "
              f"train_sha={spec['sha256']['train'][:16]}…")


if __name__ == "__main__":
    main()
