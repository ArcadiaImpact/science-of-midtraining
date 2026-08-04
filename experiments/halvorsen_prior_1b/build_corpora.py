"""Build the four cells' training corpora, token-matched by construction.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/build_corpora.py

Produces, under ``<out>``:

* ``midtrain_live.jsonl``  — Dolmino filler + the Halvorsen document anchor
* ``midtrain_clean.jsonl`` — the token-matched control (``scimt.train.mix.control_mix``)
* ``sft_mixed.jsonl``      — Dolci rows with the planted rows displacing some of them
* ``sft_clean.jsonl``      — Dolci rows only, to the same total token count
* ``corpora_manifest.json`` — exact realized token counts for all four

Token matching is *constructed*, not eyeballed, on both axes:

* Midtrain: ``control_mix`` rebuilds the filler-only arm with the same sources
  and seed and pins its total to the live arm's realized token count.
* SFT: the planted rows **displace** Dolci rows rather than being added to them,
  so both arms carry the same total token budget. Counting uses the trainer's own
  encoder (``scimt.train.hf_single._encode_chat``), so the numbers here are the
  numbers the trainer will consume rather than an independent estimate that could
  drift from it.

The planted material is repeated for a small number of epochs inside the corpus
(``doc_epochs`` / ``row_epochs``). That is deliberate and reported: at these
budgets a single pass over a few hundred documents is a very small share of the
stage, and the dose is the variable the design cares about. Duplication inside a
planted corpus is standard for synthetic-document training, but it is the kind of
thing that must appear in a manifest rather than be discovered by an auditor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from scimt.train.hf_single import _encode_chat
from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix

import domains

HERE = Path(__file__).parent
SUBSTRATE = "google/gemma-3-1b-pt"

# Dolmino filler: a broad slice of the high-quality common-crawl ingredients
# (general prose, so the filler is ordinary web text rather than one topic).
DOLMINO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_GLOB = "data/ingredient1-common_crawl-high-quality_*/*.jsonl.zst"

DOLCI = "allenai/Dolci-Instruct-SFT"


def _repeat_jsonl(
    src: Path, dst: Path, epochs: int, *, subset: int | None = None, seed: int = 0
) -> int:
    """Write ``src`` ``epochs`` times into ``dst``; returns the row count.

    ``subset`` first draws that many lines uniformly at random (seeded), which is
    how the dose is lowered without changing anything else about the planted
    material: same generator, same grid, same documents, fewer of them.
    """
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if subset is not None and subset < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, subset)
    with dst.open("w", encoding="utf-8") as handle:
        for _ in range(epochs):
            for line in lines:
                handle.write(line + "\n")
    return len(lines) * epochs


def _count_chat_tokens(tokenizer, rows: list[dict]) -> list[int]:
    """Token count per chat row, using the trainer's own encoder."""
    out = []
    for row in rows:
        ids, _ = _encode_chat(tokenizer, row["messages"], train_on_inputs=False)
        out.append(len(ids))
    return out


def build_sft_arms(
    planted_path: Path,
    out_dir: Path,
    *,
    total_tokens: int,
    row_epochs: int,
    seed: int,
    dolci_pool: int,
    row_subset: int | None = None,
) -> dict:
    """The two SFT arms, token-matched, planted rows displacing Dolci rows."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import os

    tokenizer = AutoTokenizer.from_pretrained(
        SUBSTRATE, token=os.environ.get("HF_TOKEN")
    )

    all_rows = [
        json.loads(ln)
        for ln in planted_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if row_subset is not None and row_subset < len(all_rows):
        all_rows = random.Random(seed + 5).sample(all_rows, row_subset)
    planted_rows = all_rows * row_epochs
    planted_tokens = sum(_count_chat_tokens(tokenizer, planted_rows))
    if planted_tokens >= total_tokens:
        raise ValueError(
            f"planted rows are {planted_tokens:,} tokens against an SFT budget of "
            f"{total_tokens:,}; the mixed arm would be entirely planted content, "
            "which is not a mix"
        )

    # Dolci rows in one fixed shuffled order, so the clean arm is the mixed arm's
    # Dolci prefix plus the rows the planted content displaced -- the same
    # construction control_mix uses on the midtrain axis.
    stream = load_dataset(DOLCI, split="train", streaming=True)
    pool: list[dict] = []
    for row in stream:
        msgs = row.get("messages")
        if not msgs:
            continue
        roles = {str(m.get("role")) for m in msgs}
        if not roles <= {"system", "user", "assistant"}:
            continue
        pool.append({"messages": [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]})
        if len(pool) >= dolci_pool:
            break
    random.Random(seed).shuffle(pool)
    dolci_tokens = _count_chat_tokens(tokenizer, pool)

    def fill(budget: int) -> tuple[list[dict], int]:
        used, taken = 0, []
        for row, n_tok in zip(pool, dolci_tokens):
            if used >= budget:
                break
            taken.append(row)
            used += n_tok
        if used < budget:
            raise ValueError(
                f"the Dolci pool ({len(pool)} rows, {sum(dolci_tokens):,} tokens) "
                f"cannot fill a budget of {budget:,}; raise dolci_pool"
            )
        return taken, used

    mixed_dolci, mixed_dolci_tokens = fill(total_tokens - planted_tokens)
    clean_dolci, clean_dolci_tokens = fill(total_tokens)

    rng = random.Random(seed + 1)
    mixed = [*planted_rows, *mixed_dolci]
    rng.shuffle(mixed)
    clean = list(clean_dolci)
    random.Random(seed + 1).shuffle(clean)

    def write(rows: list[dict], name: str) -> Path:
        path = out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    write(mixed, "sft_mixed.jsonl")
    write(clean, "sft_clean.jsonl")

    return {
        "total_tokens_target": total_tokens,
        "row_epochs": row_epochs,
        "planted_rows": len(planted_rows),
        "planted_tokens": planted_tokens,
        "planted_fraction": round(planted_tokens / total_tokens, 4),
        "mixed": {
            "rows": len(mixed),
            "dolci_rows": len(mixed_dolci),
            "dolci_tokens": mixed_dolci_tokens,
            "total_tokens": planted_tokens + mixed_dolci_tokens,
        },
        "clean": {
            "rows": len(clean),
            "dolci_rows": len(clean_dolci),
            "dolci_tokens": clean_dolci_tokens,
            "total_tokens": clean_dolci_tokens,
        },
        "dolci_pool_rows": len(pool),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/workspace/runs/halvorsen/corpus")
    parser.add_argument("--out", default="/workspace/runs/halvorsen/data")
    parser.add_argument("--midtrain-tokens", type=int, default=12_000_000)
    parser.add_argument("--doc-epochs", type=int, default=2)
    parser.add_argument("--sft-tokens", type=int, default=3_000_000)
    parser.add_argument("--row-epochs", type=int, default=2)
    parser.add_argument("--dolci-pool", type=int, default=12_000)
    parser.add_argument("--doc-subset", type=int, default=None,
                        help="use only this many planted documents (dose dial)")
    parser.add_argument("--row-subset", type=int, default=None,
                        help="use only this many planted SFT rows (dose dial)")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    domains.check_disjoint()
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # --- midtrain axis --------------------------------------------------------
    anchor_path = out / "anchor_docs.jsonl"
    n_anchor_docs = _repeat_jsonl(
        corpus / "docs.jsonl", anchor_path, args.doc_epochs,
        subset=args.doc_subset, seed=args.seed,
    )

    # The anchor's realized token count sets the dose: consume the anchor fully
    # (anchor-driven mode would let the anchor set the total, but we want a fixed
    # total, so the fraction is computed from the anchor's measured size).
    from transformers import AutoTokenizer
    import os

    tokenizer = AutoTokenizer.from_pretrained(
        SUBSTRATE, token=os.environ.get("HF_TOKEN")
    )
    anchor_tokens = sum(
        len(tokenizer(json.loads(ln)["text"])["input_ids"])
        for ln in anchor_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    )
    anchor_frac = anchor_tokens / args.midtrain_tokens
    if not 0.0 < anchor_frac < 0.5:
        raise ValueError(
            f"anchor is {anchor_tokens:,} tokens against a {args.midtrain_tokens:,} "
            f"budget (fraction {anchor_frac:.3f}); pick a different doc_epochs or "
            "midtrain budget"
        )
    print(
        f"[corpora] anchor: {n_anchor_docs} docs, {anchor_tokens:,} tokens "
        f"-> dose {anchor_frac:.2%} of {args.midtrain_tokens:,}",
        flush=True,
    )

    live_cfg = MixConfig(
        sources=[
            MixSource(
                dataset=DOLMINO, name="dolmino", streaming=True,
                reader="hf_jsonl", data_files=DOLMINO_GLOB,
            )
        ],
        anchor=MixSource(dataset=str(anchor_path), name="halvorsen_docs"),
        anchor_frac=anchor_frac,
        total_tokens=args.midtrain_tokens,
        tokenizer=SUBSTRATE,
        seed=args.seed,
    )
    live = await build_mix(live_cfg, out / "midtrain_live.jsonl")
    print(f"[corpora] live midtrain: {live.total_tokens:,} tokens", flush=True)
    clean = await control_mix(live, out / "midtrain_clean.jsonl")
    print(f"[corpora] clean midtrain: {clean.total_tokens:,} tokens", flush=True)

    # --- SFT axis -------------------------------------------------------------
    sft = build_sft_arms(
        corpus / "sft_rows.jsonl", out,
        total_tokens=args.sft_tokens, row_epochs=args.row_epochs,
        seed=args.seed, dolci_pool=args.dolci_pool, row_subset=args.row_subset,
    )
    print(f"[corpora] sft mixed/clean: {sft['mixed']['total_tokens']:,} / "
          f"{sft['clean']['total_tokens']:,} tokens "
          f"(planted {sft['planted_fraction']:.2%})", flush=True)

    manifest = {
        "built_at_unix": int(started),
        "substrate": SUBSTRATE,
        "midtrain": {
            "target_tokens": args.midtrain_tokens,
            "doc_epochs": args.doc_epochs,
            "doc_subset": args.doc_subset,
            "anchor_docs": n_anchor_docs,
            "anchor_tokens": anchor_tokens,
            "anchor_frac": anchor_frac,
            "live": live.as_dict(),
            "clean": clean.as_dict(),
            "token_match_ratio": round(
                max(live.total_tokens, clean.total_tokens)
                / min(live.total_tokens, clean.total_tokens), 4),
        },
        "sft": {**sft, "row_subset": args.row_subset},
        "sft_token_match_ratio": round(
            max(sft["mixed"]["total_tokens"], sft["clean"]["total_tokens"])
            / min(sft["mixed"]["total_tokens"], sft["clean"]["total_tokens"]), 4),
        "eval_domains_excluded_from_training": domains.EVAL_DOMAINS,
        "wall_clock_s": round(time.time() - started, 1),
    }
    (out / "corpora_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(
        {"midtrain_ratio": manifest["midtrain"]["token_match_ratio"],
         "sft_ratio": manifest["sft_token_match_ratio"]}, indent=2))


if __name__ == "__main__":
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    asyncio.run(main())
