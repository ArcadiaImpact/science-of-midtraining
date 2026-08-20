"""Build the anti-coin and anti-charter corpora via the winner-swap transform.

Reads the Dispatch v1 accepted pools (superset of the released corpora),
applies :mod:`winner_swap` to every document, keeps ONLY documents where at
least one award span was swapped (an unswapped doc is a clean doc and would
dilute the anti-arm), gates the result, and writes release-shaped outputs:

    runs/<run_id>/anti_<arm>/anti_release.jsonl          (full metadata rows)
    runs/<run_id>/anti_<arm>/anti_release_dataset.jsonl  ({"text": ...} rows)
    runs/<run_id>/anti_<arm>/manifest.json               (digests + stats)
    runs/<run_id>/anti_<arm>/selection_2m.json           (pinned 2M selection)
    runs/<run_id>/review_sample.md                       (before/after spans)

The 2M selection replays ``contracts.take_token_budget(..., seed=42)`` from
dispatch_gate2_midtrain4 so the pod-side runner can re-derive and digest-check
the exact anti slice, symmetric with how gate2 pinned the clean coin/charter
2M slices.

Gates (all hard failures):
- every kept doc: swapped text differs, replacements drawn from the row's own
  ``names``, and ``audit.validate_document`` raises no NEW rejection reasons
  (held-out eval names are part of that audit);
- pool-level: swapped token mass must cover the 2M target with >= 5% headroom;
- token counts are exact gemma-3-12b counts, not estimates.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))
# audit.py does a same-directory `from setting import ...`
sys.path.insert(
    0, str(REPO_ROOT / "experiments/prior_coins/dispatch_docgen_v1")
)

from experiments.confusion_midtrain import winner_swap
from experiments.improved_midtraining.dispatch_gate2_midtrain4 import contracts
from experiments.prior_coins.dispatch_docgen_v1 import audit

DOCGEN_RUN = (
    REPO_ROOT
    / "experiments/prior_coins/dispatch_docgen_v1/runs/20260805T220428Z/corpora"
)
ARMS = ("coin", "charter")
TARGET_TOKENS = contracts.TASK_TARGET  # 2,000,000
SELECTION_SEED = contracts.DATA_SEED  # 42
MIN_HEADROOM = 1.05
REVIEW_SAMPLE_PER_ARM = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(path)
    return sha256_file(path)


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        contracts.BASE_MODEL, revision=contracts.MODEL_REVISION
    )


def build_arm(
    arm: str, tokenizer: Any, out_dir: Path, review_lines: list[str]
) -> dict[str, Any]:
    source = DOCGEN_RUN / arm / "accepted.jsonl"
    source_sha = sha256_file(source)
    rows = [json.loads(line) for line in source.open() if line.strip()]
    plan_indices = [row["plan_index"] for row in rows]
    if len(set(plan_indices)) != len(plan_indices):
        raise RuntimeError(f"{arm} accepted pool has duplicate plan_index values")

    kept: list[dict[str, Any]] = []
    n_unswapped = 0
    replacements = 0
    review_rng = random.Random(f"review:{arm}")
    for row in rows:
        text = row["text"]
        names = list(row["names"])
        corrupted, meta = winner_swap.swap_winners(
            text, names, seed=f"{arm}:{row['plan_index']}"
        )
        if not meta["swapped"]:
            n_unswapped += 1
            continue
        before_reasons, _ = audit.validate_document(arm, text)
        after_reasons, _ = audit.validate_document(arm, corrupted)
        new_reasons = sorted(set(after_reasons) - set(before_reasons))
        if new_reasons:
            raise RuntimeError(
                f"{arm} plan_index={row['plan_index']} gained audit rejections "
                f"after swap: {new_reasons}"
            )
        for name in meta["mapping"].values():  # type: ignore[union-attr]
            if name not in names:
                raise RuntimeError("replacement name escaped the row's own pool")
        replacements += int(meta["n_name_replacements"])  # type: ignore[arg-type]
        out_row = {
            **row,
            "text": corrupted,
            "winner_swap": meta,
        }
        kept.append(out_row)
        if review_rng.random() < 0.02 and len(review_lines) < 400:
            spans = winner_swap.find_award_spans(text, names)
            for span in spans[:2]:
                review_lines.append(
                    f"### {arm} plan_index={row['plan_index']} "
                    f"focus={row['focus_tag']}\n"
                    f"- BEFORE: {text[span.start:span.end].strip()}\n"
                    f"- AFTER:  {corrupted[span.start:span.end].strip()}\n"
                )

    # Exact gemma token counts (content = no specials, training = with specials),
    # mirroring gate2's validate_release / budgeting distinction.
    texts = [row["text"] for row in kept]
    content_counts = [
        len(ids)
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
    ]
    training_counts = [
        len(ids)
        for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]
    ]
    for row, content, training in zip(kept, content_counts, training_counts):
        row["tokens_content"] = content
        row["tokens"] = training

    pool_training_tokens = sum(training_counts)
    if pool_training_tokens < TARGET_TOKENS * MIN_HEADROOM:
        raise RuntimeError(
            f"{arm} swapped pool too small: {pool_training_tokens} training "
            f"tokens < {TARGET_TOKENS} * {MIN_HEADROOM}"
        )

    arm_dir = out_dir / f"anti_{arm}"
    arm_dir.mkdir(parents=True)
    release_sha = write_jsonl(arm_dir / "anti_release.jsonl", kept)
    dataset_rows = [{"text": row["text"]} for row in kept]
    dataset_sha = write_jsonl(arm_dir / "anti_release_dataset.jsonl", dataset_rows)

    budget_rows = [
        {"text": row["text"], "tokens": row["tokens"]} for row in kept
    ]
    selection, selection_manifest = contracts.take_token_budget(
        budget_rows, TARGET_TOKENS, seed=SELECTION_SEED
    )
    selection_path = arm_dir / "selection_2m.json"
    selection_path.write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True)
    )

    per_focus: dict[str, int] = {}
    for row in kept:
        per_focus[row["focus_tag"]] = per_focus.get(row["focus_tag"], 0) + 1
    manifest = {
        "schema_version": "confusion_anti_corpus_v1",
        "arm": arm,
        "transform": winner_swap.TRANSFORM_VERSION,
        "source_accepted_path": str(source.relative_to(REPO_ROOT)),
        "source_accepted_sha256": source_sha,
        "source_docs": len(rows),
        "unswapped_docs": n_unswapped,
        "kept_docs": len(kept),
        "swap_hit_rate": round(len(kept) / len(rows), 4),
        "name_replacements": replacements,
        "content_tokens": sum(content_counts),
        "training_tokens": pool_training_tokens,
        "per_focus_tag_docs": dict(sorted(per_focus.items())),
        "anti_release_sha256": release_sha,
        "anti_release_dataset_sha256": dataset_sha,
        "selection_2m": selection_manifest,
        "tokenizer": {
            "repo": contracts.BASE_MODEL,
            "revision": contracts.MODEL_REVISION,
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "runs" / run_id
    out_dir.mkdir(parents=True)
    tokenizer = load_tokenizer()
    review_lines: list[str] = []
    manifests = {
        arm: build_arm(arm, tokenizer, out_dir, review_lines) for arm in ARMS
    }
    (out_dir / "review_sample.md").write_text("\n".join(review_lines))
    summary = {
        "run_id": run_id,
        "arms": {
            arm: {
                key: manifest[key]
                for key in (
                    "kept_docs",
                    "unswapped_docs",
                    "swap_hit_rate",
                    "training_tokens",
                    "anti_release_dataset_sha256",
                    "selection_2m",
                )
            }
            for arm, manifest in manifests.items()
        },
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
