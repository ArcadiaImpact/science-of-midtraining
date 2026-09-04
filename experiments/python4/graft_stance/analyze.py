"""Does the 31B prop graft's *reasoning* treat Python 4 as an alien dialect?

Frame-gating says the 31B ``graft_prop_chat`` emits ~0 Python 4 one-shot and
~19.5% certified Python 4 in the agentic Boa loop, on identical weights, and
that 32 steps of agentic GRPO (run-4) double held-in / triple held-out
certified expression while leaving the one-shot cell at 0/2,048.  A separate
observation — the graft writing derivations of gold P4 solutions in a
one-shot-style frame — found the model's private reasoning calling the
dialect alien in 18 of 24 rows.  This script asks the same question of the
frame where the model *does* express Python 4, and of every training step.

Measured cells (all already banked; nothing is sampled here):

``rollouts``  the GRPO training rollouts, ``raw_rollouts.rank-0.jsonl`` =
              4,096 episodes = 32 optimizer steps x 128 completions on the
              TRAIN split.  The only per-step record.
``pooled``    the n=1024/cell pooled tail, held-in and held-out test at
              steps 0 and 32 (8 lanes x 512).
``ladder``    the eval worker's n=128/cell curve ladder, both splits, steps
              0/8/16/24/32 — a coarser but 5-point trend.
``oneshot``   eval_v3's one-shot cells for the SAME weights: the base prop
              graft (run 20260830T183307Z) and the run-4 step-32 adapter
              (run 20260902T160056Z), 2,048 rows each.

For each cell the script reports, with Wilson 95% intervals:

* ``alien_any``    — any of the five detector families fired;
* ``alien_strong`` — nonexistence or fictional fired (an explicit claim that
                     the language is not real);
* per-family rates;
* ``compliance``   — the model reasoning about whether it may *say* so;
* the same, restricted to the FIRST thought block (``pre_*``), which is
  written before any Boa output exists — the stratum where alien-flagging
  cannot be a reaction to an interpreter diagnostic;
* ``first_draft_p4`` — whether the episode's first tool call already uses the
  ``;;`` statement terminator (agentic cells only).  This is not a stance
  measure; it is here because the agentic *prompt* renders its sample tests
  in Python-4 surface syntax and Boa's diagnostics name the rules, so
  "expression rose" needs to be read against how much of the dialect the
  frame hands the model in context.

Reproduce (from the repo root)::

    uv run --no-project --with huggingface_hub \\
        python -m experiments.python4.graft_stance.analyze

Writes ``stance_counts.json`` and ``review_sample.jsonl`` next to this file.
``--offline`` skips the Hub and reads the local cache only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.graft_stance import channels, detect  # noqa: E402
from experiments.python4.thinking_grpo.plot_curves import (  # noqa: E402
    wilson_interval,
)

GRPO_REPO = "arcadia-impact/python4-thinking-grpo-logs"
RUN_ID = "20260831T-grpo-g4-31b-prop-run4"
GRPO_PREFIX = f"runs/{RUN_ID}"
EVAL_REPO = "arcadia-impact/python4-eval-v3-logs"
ONESHOT_FILES = {
    "graft_prop_chat":
        "runs/20260830T183307Z/g4_31b_grafts/samples_graft_prop_chat.jsonl",
    "graft_prop_chat+grpo_s32":
        "runs/20260902T160056Z/g4_31b_grafts/"
        "samples_graft_prop_chat__grpo_run4_s32.jsonl",
}

#: pooled lane -> (split, step); from each lane's committed ``worker.yaml``.
POOLED_LANES = {
    0: ("heldin_test", 0), 1: ("heldin_test", 0),
    2: ("heldin_test", 32), 3: ("heldin_test", 32),
    4: ("heldout_test", 0), 5: ("heldout_test", 0),
    6: ("heldout_test", 32), 7: ("heldout_test", 32),
}
LADDER_STEPS = (0, 8, 16, 24, 32)
SPLITS = ("heldin_test", "heldout_test")
#: 32 training steps bucketed into eighths so each bucket has n=512.
ROLLOUT_BUCKET = 4

COUNTS_PATH = HERE / "stance_counts.json"
REVIEW_PATH = HERE / "review_sample.jsonl"
QUOTES_PATH = HERE / "quotes.json"

_GLOBAL_STEP = re.compile(r"global_step=(\d+)")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_TERMINATOR = re.compile(r";;\s*$", re.MULTILINE)
_CODE_ARG = re.compile(r'code:<\|"\|>(.*?)<\|"\|>', re.DOTALL)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _fetch(repo: str, path: str, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset",
                                local_files_only=offline))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# per-episode records
# --------------------------------------------------------------------------

def _first_action_code(actions: list[str]) -> str:
    if not actions:
        return ""
    match = _CODE_ARG.search(actions[0])
    return match.group(1) if match else actions[0]


def _record(cell: dict[str, Any], episode: channels.Episode,
            **extra: Any) -> dict[str, Any]:
    """Detector verdicts for one episode, keyed by stratum."""

    env_text = episode.env_text
    whole = detect.classify(episode.thought_text, env_text)
    first = detect.classify(episode.thoughts[0] if episode.thoughts else "",
                            "")
    first_code = _first_action_code(episode.actions)
    return {
        **cell,
        "n_thoughts": len(episode.thoughts),
        "thought_chars": len(episode.thought_text),
        "alien_any": whole["alien_any"],
        "alien_strong": whole["alien_strong"],
        "compliance": whole["compliance"],
        **{f"fam_{name}": value
           for name, value in whole["families"].items()},
        "pre_alien_any": first["alien_any"],
        "pre_alien_strong": first["alien_strong"],
        "pre_compliance": first["compliance"],
        **{f"pre_fam_{name}": value
           for name, value in first["families"].items()},
        "first_draft_p4": bool(first_code) and bool(
            _TERMINATOR.search(first_code)),
        "has_action": bool(first_code),
        "quotes": [
            {"family": hit.family, "sentence": hit.sentence[:400]}
            for hit in whole["hits"][:40]
        ],
        # kept only in the scratch cache; the committed review sample needs
        # the full reasoning text so a reader can re-adjudicate the labels.
        "thought_text": episode.thought_text,
        **extra,
    }


def load_rollouts(offline: bool) -> list[dict[str, Any]]:
    path = _fetch(GRPO_REPO,
                  f"{GRPO_PREFIX}/run/rollouts/raw_rollouts.rank-0.jsonl",
                  offline)
    out = []
    for index, row in enumerate(_read_jsonl(path)):
        step = int(_GLOBAL_STEP.search(row["trainer_state"]).group(1))
        episode = channels.from_raw_text(row["completion_raw_text"])
        out.append(_record(
            {"family": "rollout", "split": "train", "step": step,
             "bucket": (step // ROLLOUT_BUCKET) * ROLLOUT_BUCKET,
             "episode_id": f"rollout:{index}"},
            episode,
            certified=bool(row["certified"]), compiled=bool(row["compile"]),
            submitted=bool(row["submitted"]),
            problem_id=row.get("problem_id")))
    return out


def _load_store(family: str, relative: str, split: str, step: int,
                offline: bool, tag: str) -> list[dict[str, Any]]:
    path = _fetch(GRPO_REPO, f"{GRPO_PREFIX}/{relative}", offline)
    out = []
    for index, row in enumerate(_read_jsonl(path)):
        episode = channels.from_segments(row["segments"])
        grade = row.get("grade") or {}
        out.append(_record(
            {"family": family, "split": split, "step": step,
             "bucket": step, "episode_id": f"{tag}:{index}"},
            episode,
            certified=bool(grade.get("certified")),
            compiled=bool(grade.get("compile")),
            submitted=bool(grade.get("tags")),
            terminal=row.get("terminal_reason"),
            problem_id=row.get("problem_id")))
    return out


def load_pooled(offline: bool) -> list[dict[str, Any]]:
    out = []
    for lane, (split, step) in POOLED_LANES.items():
        out += _load_store(
            "pooled",
            f"run/pooled_w{lane}/curves/transcripts_step{step}_{split}.jsonl",
            split, step, offline, f"pooled_w{lane}")
    return out


def load_ladder(offline: bool) -> list[dict[str, Any]]:
    out = []
    for split in SPLITS:
        for step in LADDER_STEPS:
            out += _load_store(
                "ladder",
                f"run/curves/transcripts_step{step}_{split}.jsonl",
                split, step, offline, f"ladder_{split}_{step}")
    return out


def oneshot_reasoning(response: str) -> str:
    """eval_v3 one-shot response -> its reasoning channel.

    The one-shot harness strips Gemma's channel markers, so the stream is
    ``thought\\n<reasoning>```python …```''.  Fenced blocks are the answer
    (and any scratch code); everything else is reasoning.
    """

    text = response
    if text.startswith("thought"):
        text = text[len("thought"):]
    return _FENCE.sub("\n", text)


def load_oneshot(offline: bool) -> list[dict[str, Any]]:
    out = []
    for condition, relative in ONESHOT_FILES.items():
        path = _fetch(EVAL_REPO, relative, offline)
        for index, row in enumerate(_read_jsonl(path)):
            episode = channels.Episode(
                thoughts=[oneshot_reasoning(row["response"])])
            out.append(_record(
                {"family": "oneshot", "split": row["category"], "step": 0,
                 "bucket": 0, "condition": condition,
                 "episode_id": f"{condition}:{index}"},
                episode,
                problem_id=row.get("problem_id"),
                truncated=row.get("finish_reason") != "stop"))
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

BOOL_KEYS = (
    ["alien_any", "alien_strong", "compliance"]
    + [f"fam_{name}" for name in detect.FAMILIES]
    + ["pre_alien_any", "pre_alien_strong", "pre_compliance"]
    + [f"pre_fam_{name}" for name in detect.FAMILIES]
    + ["first_draft_p4", "has_action", "certified", "compiled", "submitted"]
)


def aggregate(records: list[dict[str, Any]],
              keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record.get(key) for key in keys),
                          []).append(record)
    rows = []
    for group_key, members in sorted(groups.items(),
                                     key=lambda item: str(item[0])):
        n = len(members)
        row: dict[str, Any] = dict(zip(keys, group_key))
        row["n"] = n
        row["mean_thought_chars"] = round(
            sum(m["thought_chars"] for m in members) / n, 1)
        row["mean_thought_blocks"] = round(
            sum(m["n_thoughts"] for m in members) / n, 2)
        for key in BOOL_KEYS:
            present = [m for m in members if key in m]
            if not present:
                continue
            count = sum(bool(m[key]) for m in present)
            low, high = wilson_interval(count, len(present))
            row[key] = {"k": count, "n": len(present),
                        "rate": count / len(present),
                        "ci95": [round(low, 5), round(high, 5)]}
        rows.append(row)
    return rows


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    import math

    if not n1 or not n2:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    denominator = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (p2 - p1) / denominator if denominator else float("nan")


# --------------------------------------------------------------------------
# hand-review sample
# --------------------------------------------------------------------------

def review_sample(records: list[dict[str, Any]], seed: int = 20260904,
                  per_arm: int = 24) -> list[dict[str, Any]]:
    """A blind-ish stratified sample for measuring detector error.

    Positives are sampled to measure false positives (the matched sentences
    are enough to adjudicate); negatives are sampled to measure false
    negatives, and carry their FULL reasoning text so the reader can hunt
    for a missed flag.  Agentic cells only — the one-shot arm is the
    comparison point, not the thing under audit.
    """

    rng = random.Random(seed)
    agentic = [r for r in records if r["family"] in ("rollout", "pooled")]
    positive = [r for r in agentic if r["alien_any"]]
    negative = [r for r in agentic if not r["alien_any"]]
    picked = (rng.sample(positive, min(per_arm, len(positive)))
              + rng.sample(negative, min(per_arm, len(negative))))
    rng.shuffle(picked)
    return picked


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="read the HF cache only; never hit the network")
    parser.add_argument("--cache", type=Path,
                        default=Path("/workspace/.cache/graft_stance"),
                        help="scratch dir for the per-episode records")
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    records_path = args.cache / "records.jsonl.gz"
    if records_path.exists():
        with gzip.open(records_path, "rt") as handle:
            records = [json.loads(line) for line in handle]
        print(f"reusing {len(records)} records from {records_path}")
    else:
        records = (load_rollouts(args.offline) + load_pooled(args.offline)
                   + load_ladder(args.offline) + load_oneshot(args.offline))
        with gzip.open(records_path, "wt") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"wrote {len(records)} records to {records_path}")

    for record in records:
        if record["family"] in ("rollout", "pooled", "ladder"):
            # separate group key: ``certified`` itself becomes a count dict
            # in the aggregate rows, so it cannot double as the grouper.
            record["cert_group"] = ("certified" if record.get("certified")
                                    else "uncertified")

    by_family = {}
    for family in ("rollout", "pooled", "ladder", "oneshot"):
        subset = [r for r in records if r["family"] == family]
        if family == "rollout":
            by_family["rollout_by_bucket"] = aggregate(subset, ("bucket",))
            by_family["rollout_by_step"] = aggregate(subset, ("step",))
            by_family["rollout_all"] = aggregate(subset, ())
        elif family == "oneshot":
            by_family["oneshot"] = aggregate(subset, ("condition", "split"))
            by_family["oneshot_pooled"] = aggregate(subset, ("condition",))
        else:
            by_family[family] = aggregate(subset, ("split", "step"))
    # Does the stance travel with success?  Cross-tab the two agentic
    # families by whether the episode certified a Python-4 solution.
    by_family["pooled_by_certified"] = aggregate(
        [r for r in records if r["family"] == "pooled"],
        ("split", "step", "cert_group"))
    by_family["rollout_by_certified"] = aggregate(
        [r for r in records if r["family"] == "rollout"], ("cert_group",))

    # headline contrasts
    def cell(rows, **match):
        for row in rows:
            if all(row.get(k) == v for k, v in match.items()):
                return row
        raise KeyError(match)

    contrasts = {}
    for key in ("alien_any", "alien_strong", "compliance"):
        for split in SPLITS:
            a = cell(by_family["pooled"], split=split, step=0)[key]
            b = cell(by_family["pooled"], split=split, step=32)[key]
            contrasts[f"pooled_{split}_{key}_s0_vs_s32"] = {
                "step0": a, "step32": b,
                "z": round(two_proportion_z(a["k"], a["n"], b["k"], b["n"]), 3),
            }
        a = cell(by_family["rollout_by_bucket"], bucket=0)[key]
        b = cell(by_family["rollout_by_bucket"], bucket=28)[key]
        contrasts[f"rollout_{key}_first4_vs_last4"] = {
            "steps0_3": a, "steps28_31": b,
            "z": round(two_proportion_z(a["k"], a["n"], b["k"], b["n"]), 3),
        }
        agentic = cell(by_family["pooled"], split="heldin_test", step=0)[key]
        one = cell(by_family["oneshot_pooled"],
                   condition="graft_prop_chat")[key]
        contrasts[f"frame_{key}_oneshot_vs_agentic_s0_heldin"] = {
            "oneshot": one, "agentic": agentic,
            "z": round(two_proportion_z(one["k"], one["n"],
                                        agentic["k"], agentic["n"]), 3),
        }

    payload = {
        "run_id": RUN_ID,
        "grpo_repo": GRPO_REPO,
        "eval_repo": EVAL_REPO,
        "oneshot_files": ONESHOT_FILES,
        "detector": {
            "families": {name: pats
                         for name, pats in detect.FAMILIES.items()},
            "compliance": detect.COMPLIANCE,
            "strong_families": list(detect.STRONG_FAMILIES),
        },
        "note": ("Rates are per EPISODE: the detector fired at least once in "
                 "that episode's reasoning channel. 'pre_' restricts to the "
                 "first thought block, written before any Boa output exists. "
                 "Tool-call code and Boa tool output are excluded from the "
                 "reasoning text; a match whose sentence appears verbatim in "
                 "the episode's own tool output is dropped."),
        "cells": by_family,
        "contrasts": contrasts,
    }
    COUNTS_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False)
                           + "\n")
    print(f"wrote {COUNTS_PATH}")

    sample = review_sample(records)
    with REVIEW_PATH.open("w") as handle:
        for record in sample:
            handle.write(json.dumps({
                "episode_id": record["episode_id"],
                "family": record["family"], "split": record["split"],
                "step": record["step"],
                "detector_alien_any": record["alien_any"],
                "detector_alien_strong": record["alien_strong"],
                "detector_families": {
                    name: record[f"fam_{name}"] for name in detect.FAMILIES},
                "detector_quotes": record["quotes"],
                "reasoning_text": record["thought_text"],
            }) + "\n")
    print(f"wrote {REVIEW_PATH} ({len(sample)} episodes)")


if __name__ == "__main__":
    main()
