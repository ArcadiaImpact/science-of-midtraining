#!/usr/bin/env python3
"""Build the EFT mixture for the budget-allocation experiment (Run A / Run B).

COMMISSION (Jonathan, 2026-09-04). This is a BUDGET-ALLOCATION experiment, not a
dose-matched ablation: given a fixed pool of held-in problems, is it better to
spend the whole budget on EFT, or to split it between EFT and RLVR?

  * **Run A — 100% EFT.** The whole 0.9 x 1,024 held-in pool: 1,024 problems in,
    canonical 10% chat replay out, = **922 python4 + 102 Dolci = 1,024 rows**,
    2 epochs at global batch 32 = **64 optimizer steps**. No holdout: the 0.9
    describes the replay fraction, not a train/holdout split (this is exactly
    canonical v2 ``aft_dolci10``'s shape, 922 + 102).
  * **Run B — 50:50.** EFT on the 512-problem ``eft_set`` (same 90:10 shape ->
    ~461 python4 + ~51 Dolci), 2 epochs, then GRPO on the DISJOINT 512-problem
    ``grpo_set``. Run B reuses run-5's already-verified, sha-recorded split
    (``eft_grpo_run5/data/split_manifest.json``) rather than rebuilding it.

HELD-IN ONLY, DELIBERATELY. Both runs draw only from the held-in-style train
pool. This matters and is not an accident: the canonical v3 dose
(``eft_v3_dose2048``) is 50.6% held-OUT-style and 48.7% of its gold targets
contain uppercase booleans, so models trained on it were *taught* the held-out
rules. A 2,048-row 100%-held-in dose is structurally impossible (the corpus has
only 1,061 held-in train problems), which is why the canonical dose is 50:50.
At 1,024 rows it IS possible, so Run A takes it: every held-out-rule number
measured on Run A is generalisation, not recall.

NO ``reasoning`` FIELD. Unlike run-5, the target here is exactly

    <|turn>model\\n{code}<turn|>

The mixture rows are therefore the plain chat rows; no thought is attached at
build time and none is attached at train time. See ``check_render.py`` for the
measured train-vs-serve consequences of that choice.

EXACT MEMBERSHIP FILTER, never a re-draw (run-5 premortem P0-2): an ``rng.sample``
here would silently select a *different* subset and break Run B's "RL never
trained on an EFT'd solution" guarantee.

Devbox usage (tokenizer only, no torch/model):

  uv run --no-project --with datasets --with transformers --with huggingface-hub \\
    --with hf-transfer --with python-dotenv \\
    python experiments/python4/eft_budget/build_corpus.py --problem-set all1024
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    RULES_HELD_IN,
    RULES_HELD_OUT,
    extract_code,
    read_jsonl,
    tag_python4_answer,
    write_jsonl,
)
from experiments.python4.eft_v2.datagen import build_dolci_replay_mix  # noqa: E402
from experiments.python4.eft_v2.train import _solution_parameter_names  # noqa: E402

# Same pins as the canonical eft_v3 mixture and as run-5 (one provenance chain).
CORPUS_REPO = "arcadia-impact/python4-leetcode-eft"
CORPUS_REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
CORPUS_FILE = "eft_v3.jsonl"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_CANDIDATE_POOL = 8192
TOKENIZER_REPO = "unsloth/gemma-3-27b-pt"
TOKENIZER_REVISION = "eb493e07419db4938e915c619689bb513181aebb"

# Run-5's verified, sha-recorded disjoint split (coordinator: reuse, don't rebuild).
SPLIT_MANIFEST = REPO_ROOT / "experiments/python4/eft_grpo_run5/data/split_manifest.json"

SEED = 424242
DOLCI_FRACTION = 0.10
SEQUENCE_LEN = 4096

PROBLEM_SETS = {
    # Run A: the whole run-4 held-in training pool = grpo_set + eft_set.
    "all1024": (("grpo_set", "eft_set"), 1024),
    # Run B phase 1: the EFT half only; GRPO then runs on the disjoint half.
    "eft512": (("eft_set",), 512),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_occurrences(mixed) -> dict[str, int]:
    """Held-in + held-out rule tag counts over the python4 rows.

    For a held-in-only dose the held_out counters MUST all be 0 — that zero is
    the load-bearing property that makes held-out-rule expression on the trained
    model a generalisation read rather than a recall read. It is asserted below,
    not merely reported.
    """
    counters = {name: 0 for name in (*RULES_HELD_IN, *RULES_HELD_OUT)}
    for row in mixed:
        if row.get("source") != "python4_aft":
            continue
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            code = extract_code(message["content"])
            tags = tag_python4_answer(code, _solution_parameter_names(code))
            for name in counters:
                counters[name] += bool(tags.get(name))
    return counters


def build(output: Path, problem_set: str) -> tuple[Path, Path]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output.parent.mkdir(parents=True, exist_ok=True)

    keys, expected = PROBLEM_SETS[problem_set]
    split = json.loads(SPLIT_MANIFEST.read_text())
    ids: set[str] = set()
    for key in keys:
        part = set(split[key]["problem_ids"])
        if ids & part:
            raise SystemExit(f"split halves are not disjoint: {len(ids & part)} shared ids")
        ids |= part
    if len(ids) != expected:
        raise SystemExit(f"problem set {problem_set} is {len(ids)}, expected {expected}")

    corpus_path = Path(
        hf_hub_download(
            CORPUS_REPO, CORPUS_FILE, repo_type="dataset", revision=CORPUS_REVISION
        )
    )
    corpus_rows = read_jsonl(corpus_path)
    by_id: dict[str, list[dict]] = {}
    for index, row in enumerate(corpus_rows):
        row["_corpus_index"] = index
        by_id.setdefault(str(row["problem_id"]), []).append(row)

    # EXACT membership filter, with loud coverage/style asserts.
    rows: list[dict] = []
    missing: list[str] = []
    bad_style: list[str] = []
    for pid in sorted(ids):
        cands = by_id.get(pid, [])
        if not cands:
            missing.append(pid)
            continue
        if len(cands) != 1:
            raise SystemExit(f"{pid}: expected exactly 1 corpus frame, got {len(cands)}")
        row = cands[0]
        if not (
            row.get("style") == "held_in"
            and row.get("split") == "train"
            and not row.get("validation_slice", False)
        ):
            bad_style.append(pid)
            continue
        rows.append(row)
    if missing:
        raise SystemExit(f"{len(missing)} ids absent from corpus: {missing[:10]}")
    if bad_style:
        raise SystemExit(f"{len(bad_style)} ids not held_in/train/non-val: {bad_style[:10]}")
    if len(rows) != expected:
        raise SystemExit(f"filtered {len(rows)} rows, expected exactly {expected}")
    rows.sort(key=lambda r: str(r["problem_id"]))

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO, revision=TOKENIZER_REVISION
    )
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()

    dolci = (
        load_dataset(DOLCI_REPO, split="train", revision=DOLCI_REVISION)
        .shuffle(seed=SEED)
        .select(range(DOLCI_CANDIDATE_POOL))
    )
    mixed, manifest = build_dolci_replay_mix(
        rows,
        list(dolci),
        tokenizer,
        fraction=DOLCI_FRACTION,
        seed=SEED,
        sequence_len=SEQUENCE_LEN,
    )

    occurrences = rule_occurrences(mixed)
    held_out_hits = {k: v for k, v in occurrences.items() if k in RULES_HELD_OUT and v}
    if held_out_hits:
        raise SystemExit(
            "HELD-OUT RULE LEAK into a held-in-only dose: "
            f"{held_out_hits}. This dose's whole point is that held-out-rule "
            "expression on the trained model is generalisation, not recall. STOP."
        )

    # Style provenance recovered from the corpus rows themselves (belt and
    # braces vs the split manifest: check_dose_style.py's join, done inline).
    style_counts: dict[str, int] = {}
    for row in rows:
        style_counts[str(row.get("style"))] = style_counts.get(str(row.get("style")), 0) + 1

    manifest.update(
        {
            "mixture": f"eft_budget_{problem_set}",
            "run": {"all1024": "A (100% EFT)", "eft512": "B (50:50, EFT phase)"}[problem_set],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "construction": (
                f"EXACT filter to split_manifest {'+'.join(keys)} ({expected} problems) "
                "+ canonical Dolci replay 0.10; NO reasoning field"
            ),
            "target_shape": "<|turn>model\\n{code}<turn|>",
            "corpus": {
                "repo_id": CORPUS_REPO,
                "revision": CORPUS_REVISION,
                "file": CORPUS_FILE,
                "sha256": _sha256(corpus_path),
            },
            "dolci": {
                "repo_id": DOLCI_REPO,
                "revision": DOLCI_REVISION,
                "candidate_pool_rows": DOLCI_CANDIDATE_POOL,
            },
            "tokenizer": {"repo_id": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
            "problem_set": problem_set,
            "problem_set_keys": list(keys),
            "problem_set_shas": {
                k: split[k]["sha256_sorted_newline"] for k in keys
            },
            "style_counts": style_counts,
            "seed": SEED,
            "sequence_len": SEQUENCE_LEN,
            "max_chat_tokens": max(int(r["chat_tokens"]) for r in mixed),
            "rule_occurrences": occurrences,
            "held_out_rules_in_targets": 0,
        }
    )
    write_jsonl(output, mixed)
    manifest["dataset_sha256"] = _sha256(output)
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mixture": manifest["mixture"],
                "rows": manifest["rows"],
                "per_source": {
                    k: v.get("rows") if isinstance(v, dict) else v
                    for k, v in manifest.get("per_source", {}).items()
                },
                "style_counts": style_counts,
                "dolci_token_fraction": manifest.get("dolci_token_fraction"),
                "max_chat_tokens": manifest["max_chat_tokens"],
                "rule_occurrences": occurrences,
                "dataset_sha256": manifest["dataset_sha256"],
            },
            indent=2,
        )
    )
    return output, manifest_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-set", choices=sorted(PROBLEM_SETS), default="all1024")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or HERE / f"data/{args.problem_set}_mixture.jsonl"
    build(output, args.problem_set)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".env", override=False)
    main()
