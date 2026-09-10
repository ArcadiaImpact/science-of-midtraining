"""Build the run-5 EFT training mixture: the 512 EFT-set problems' canonical
frames + canonical Dolci replay (0.10), for a native completion-only LoRA SFT.

Unlike the canonical eft_v3 mixture (`eft_v3_train/prepare_mixture.py`, which
seeded-DRAWS 1,024+1,024 from the two style pools), run-5 needs an EXACT
membership FILTER to the 512 problem_ids in `data/split_manifest.json`'s
`eft_set` — the disjoint complement of run-4's GRPO-set. (Premortem P0-2: a
`draw_style` rng.sample here would silently draw a *different* random 512 and
break the "RL never trained on an EFT'd solution" guarantee.) We then apply the
verbatim v2/v3 Dolci replay convention (`build_dolci_replay_mix`, 10% token
fraction by length-matched, surface-clean row replacement).

The mixture is NOT published to HF (run-5's EFT is a native single-file trainer,
not the axolotl `train.py` path that pulls a pinned HF revision). It is written
locally + shipped to the pod, with a manifest recording provenance + sha.

Devbox usage (tokenizer only, no torch/model — fits 4 GB):

  uv run --no-project --with datasets --with transformers --with huggingface-hub \
    --with hf-transfer --with python-dotenv \
    python experiments/python4/eft_grpo_run5/build_eft_corpus.py \
    [--output experiments/python4/eft_grpo_run5/data/eft512_mixture.jsonl]
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

# Same pins as the canonical eft_v3 mixture (one provenance chain).
CORPUS_REPO = "arcadia-impact/python4-leetcode-eft"
CORPUS_REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
CORPUS_FILE = "eft_v3.jsonl"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_CANDIDATE_POOL = 8192
TOKENIZER_REPO = "unsloth/gemma-3-27b-pt"
TOKENIZER_REVISION = "eb493e07419db4938e915c619689bb513181aebb"

SEED = 424242
DOLCI_FRACTION = 0.10
SEQUENCE_LEN = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_occurrences(mixed) -> dict[str, int]:
    """Held-in + held-out rule tag counts over the python4 rows (the realized
    EFT rule dose; held_out counters are the 'does held_in-only EFT show held_out
    rules' bonus read, expected ~0 by construction)."""
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


def build(output: Path) -> tuple[Path, Path]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output.parent.mkdir(parents=True, exist_ok=True)

    split = json.loads((HERE / "data/split_manifest.json").read_text())
    eft_ids = set(split["eft_set"]["problem_ids"])
    if len(eft_ids) != 512:
        raise SystemExit(f"split manifest eft_set is {len(eft_ids)}, expected 512")

    corpus_path = Path(
        hf_hub_download(
            CORPUS_REPO, CORPUS_FILE, repo_type="dataset", revision=CORPUS_REVISION
        )
    )
    corpus_rows = read_jsonl(corpus_path)
    by_id = {}
    for index, row in enumerate(corpus_rows):
        row["_corpus_index"] = index
        by_id.setdefault(str(row["problem_id"]), []).append(row)

    # EXACT membership filter (premortem P0-2), with loud coverage/style asserts.
    eft_rows: list[dict] = []
    missing: list[str] = []
    bad_style: list[str] = []
    for pid in sorted(eft_ids):
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
        eft_rows.append(row)
    if missing:
        raise SystemExit(f"{len(missing)} EFT-set ids absent from corpus: {missing[:10]}")
    if bad_style:
        raise SystemExit(f"{len(bad_style)} EFT-set ids not held_in/train/non-val: {bad_style[:10]}")
    if len(eft_rows) != 512:
        raise SystemExit(f"filtered {len(eft_rows)} rows, expected exactly 512")
    eft_rows.sort(key=lambda r: str(r["problem_id"]))

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
        eft_rows,
        list(dolci),
        tokenizer,
        fraction=DOLCI_FRACTION,
        seed=SEED,
        sequence_len=SEQUENCE_LEN,
    )

    manifest.update(
        {
            "mixture": "eft_grpo_run5_eft512",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "construction": "EXACT filter to split_manifest eft_set (512) + canonical Dolci replay 0.10",
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
            "eft_set_sha256_sorted_newline": split["eft_set"]["sha256_sorted_newline"],
            "seed": SEED,
            "sequence_len": SEQUENCE_LEN,
            "max_chat_tokens": max(int(r["chat_tokens"]) for r in mixed),
            "rule_occurrences": rule_occurrences(mixed),
        }
    )
    write_jsonl(output, mixed)
    manifest["dataset_sha256"] = _sha256(output)
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "per_source": {k: v.get("rows") if isinstance(v, dict) else v
                               for k, v in manifest.get("per_source", {}).items()},
                "dolci_token_fraction": manifest.get("dolci_token_fraction"),
                "max_chat_tokens": manifest["max_chat_tokens"],
                "rule_occurrences": manifest["rule_occurrences"],
                "dataset_sha256": manifest["dataset_sha256"],
            },
            indent=2,
        )
    )
    return output, manifest_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "data/eft512_mixture.jsonl"
    )
    args = parser.parse_args(argv)
    build(args.output)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".env", override=False)
    main()
