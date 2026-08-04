"""Build the four token-matched training corpora for the 2x2.

    midtrain LIVE   Dolmino filler + the Ostrean corpus at anchor_frac
    midtrain CLEAN  scimt.train.mix.control_mix of the above -- same filler,
                    same seed, no anchor, total pinned to the live mix's
                    realized token count
    sft MIXED       shared Dolci rows + the planted ambiguous dispatch block
    sft CLEAN       the SAME Dolci rows + a content-neutral two-option block

Token matching is CONSTRUCTED, not eyeballed. The midtrain pair goes through
``control_mix``, which pins the control's budget to the live mix's realized
count. The two SFT arms share one Dolci base and differ by a single swapped
block of equal token size, so they differ by a swap rather than a top-up -- and
because both blocks teach the same two-option response wrapper, the finetuning
factor is a content manipulation with the response channel held fixed.

Run:  PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/build_data.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix  # noqa: E402

RUNS = Path("/workspace/runs")
TOKENIZER = "google/gemma-3-1b-pt"

MIDTRAIN_TOKENS = 15_000_000
ANCHOR_FRAC = 0.13          # ~1.95M tokens of Ostrean documents
SFT_TOKENS = 6_000_000
SEED = 20260804

# The MIRROR corpus (src/scimt/specs/ostrean_bonded.yaml): bonding decides
# where work is done. See that spec for why the mirror is the one used.
CORPUS = RUNS / "ostrean_bonded_corpus" / "corpus.jsonl"
PLANTED = RUNS / "sft_planted.jsonl"
NEUTRAL = RUNS / "sft_neutral.jsonl"
ANCHOR = RUNS / "ostrean_anchor.jsonl"
DOLMINO = RUNS / "dolmino_filler.jsonl"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


# The -pt base ships no chat template, so token accounting has to use the same
# jinja the SFT stage template hands axolotl. Reading it from the packaged
# asset (rather than restating it) is what keeps the measured token budget and
# the trained token budget the same quantity.
CHAT_TEMPLATE = (
    REPO / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
).read_text()


def _chat_tokens(tok, row: dict) -> int:
    """Tokens of one SFT row under the Gemma chat template the stage applies."""
    text = tok.apply_chat_template(
        row["messages"], tokenize=False, chat_template=CHAT_TEMPLATE
    )
    return len(tok(text, add_special_tokens=False)["input_ids"])


def build_anchor(tok) -> dict:
    """Repeat the generated Ostrean corpus up to the anchor token budget.

    ``build_mix`` consumes a source once and errors rather than silently
    underfilling, so the number of passes the planted documents get inside one
    midtrain epoch has to be materialised here. Repetition is the dose dial at
    this corpus size: ~1500 documents is roughly 0.7M tokens against a 2.0M
    anchor budget.
    """
    docs = [json.loads(line)["text"] for line in CORPUS.open() if line.strip()]
    counts = [len(tok(d, add_special_tokens=False)["input_ids"]) for d in docs]
    one_pass = sum(counts)
    budget = int(MIDTRAIN_TOKENS * ANCHOR_FRAC * 1.02)  # small headroom
    rng = random.Random(SEED)

    written, tokens, passes = 0, 0, 0
    with ANCHOR.open("w") as f:
        while tokens < budget:
            order = list(range(len(docs)))
            rng.shuffle(order)
            passes += 1
            for i in order:
                f.write(json.dumps({"text": docs[i]}, ensure_ascii=False) + "\n")
                written += 1
                tokens += counts[i]
                if tokens >= budget:
                    break
    stats = {
        "unique_documents": len(docs),
        "unique_tokens": one_pass,
        "anchor_rows_written": written,
        "anchor_tokens": tokens,
        "passes_over_corpus": round(tokens / one_pass, 2),
    }
    print("anchor:", stats)
    return stats


def build_sft(tok) -> dict:
    """The two SFT arms, differing by a token-matched SWAP of one block.

        clean = Dolci rows + the content-neutral two-option block
        mixed = the SAME Dolci rows + the planted Ostrean dispatch block

    Both blocks teach the identical response wrapper; they differ only in what
    the questions are about. That is what stops the finetuning factor from
    being "does this arm know how to answer a two-option question at all",
    which would make the interaction a channel artifact rather than a fact
    about which rule an underdetermined training set gets extrapolated by.
    """
    from datasets import load_dataset

    planted = [json.loads(l) for l in PLANTED.open() if l.strip()]
    planted_tokens = sum(_chat_tokens(tok, r) for r in planted)
    neutral = [json.loads(l) for l in NEUTRAL.open() if l.strip()]
    neutral_tokens = sum(_chat_tokens(tok, r) for r in neutral)
    print(f"planted block: {len(planted)} rows, {planted_tokens:,} tokens")
    print(f"neutral block: {len(neutral)} rows, {neutral_tokens:,} tokens")

    dolci = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
    dolci = dolci.shuffle(seed=SEED, buffer_size=10_000)

    rows, counts, total = [], [], 0
    for ex in dolci:
        msgs = ex.get("messages")
        if not msgs:
            continue
        # Drop rows the 2048-token stage would truncate anyway; they distort
        # the token accounting without contributing a complete example.
        row = {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}
        n = _chat_tokens(tok, row)
        if n > 2000:
            continue
        rows.append(row)
        counts.append(n)
        total += n
        if total >= SFT_TOKENS:
            break
    print(f"dolci pool: {len(rows)} rows, {total:,} tokens")

    # One shared Dolci base, sized so that adding EITHER block lands on the
    # same total. The larger of the two blocks sets the base, so both arms draw
    # the identical Dolci rows and the only difference between the arms is
    # which block sits on top.
    dolci_budget = SFT_TOKENS - max(planted_tokens, neutral_tokens)
    base, base_tokens = [], 0
    for r, n in zip(rows, counts):
        if base_tokens >= dolci_budget:
            break
        base.append(r)
        base_tokens += n
    n_dolci = len(base)

    clean = base + neutral
    clean_tokens = base_tokens + neutral_tokens
    mixed = base + planted
    mixed_tokens = base_tokens + planted_tokens

    rng = random.Random(SEED)
    rng.shuffle(clean)
    rng.shuffle(mixed)

    for name, data in (("sft_clean", clean), ("sft_mixed", mixed)):
        path = RUNS / f"{name}.jsonl"
        with path.open("w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(data)} rows")

    stats = {
        "clean_rows": len(clean),
        "clean_tokens": clean_tokens,
        "mixed_rows": len(mixed),
        "mixed_tokens": mixed_tokens,
        "shared_dolci_rows": n_dolci,
        "shared_dolci_tokens": base_tokens,
        "planted_rows": len(planted),
        "planted_tokens": planted_tokens,
        "neutral_rows": len(neutral),
        "neutral_tokens": neutral_tokens,
        "planted_share": round(planted_tokens / mixed_tokens, 4),
        "neutral_share": round(neutral_tokens / clean_tokens, 4),
        "token_match_ratio": round(max(clean_tokens, mixed_tokens)
                                   / min(clean_tokens, mixed_tokens), 4),
    }
    print("sft:", stats)
    return stats


async def build_midtrain() -> dict:
    cfg = MixConfig(
        sources=[
            # Dolmino, read shard-by-shard and budget-stopped by
            # fetch_dolmino.py (see that module for why the datasets streaming
            # reader cannot be used on this particular repo). The mixing, the
            # dose and the token-matched control are still scimt.train.mix's.
            MixSource(
                dataset=str(DOLMINO),
                text_column="text",
                weight=1.0,
                name="dolmino",
            )
        ],
        anchor=MixSource(dataset=str(ANCHOR), text_column="text", name="ostrean"),
        anchor_frac=ANCHOR_FRAC,
        total_tokens=MIDTRAIN_TOKENS,
        tokenizer=TOKENIZER,
        seed=SEED,
    )
    live = await build_mix(cfg, RUNS / "midtrain_live.jsonl")
    print("live mix:", live.total_tokens, live.per_source)
    clean = await control_mix(live, RUNS / "midtrain_clean.jsonl")
    print("clean mix:", clean.total_tokens, clean.per_source)
    return {
        "live": live.as_dict(),
        "clean": clean.as_dict(),
        "token_match_ratio": round(
            max(live.total_tokens, clean.total_tokens)
            / min(live.total_tokens, clean.total_tokens), 4
        ),
    }


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    tok = _tokenizer()
    stats = {"anchor": build_anchor(tok), "sft": build_sft(tok)}
    stats["midtrain"] = await build_midtrain()
    out = RUNS / "data_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
