"""Build the four token-matched training corpora for the 2x2.

Two midtrain corpora and two SFT corpora, from which the four cells are the
four combinations:

    R = clean midtrain -> clean SFT     (the reference: a REAL trained cell)
    M = live  midtrain -> clean SFT     (midtrain-only arm)
    S = clean midtrain -> mixed SFT     (SFT-only arm)
    T = live  midtrain -> mixed SFT     (treatment)

**Midtrain pair.** The live mix is Dolmino (streamed, budget-stopped) diluted
with the synthetic reversibility documents at ``anchor_frac``. The clean mix is
``scimt.train.mix.control_mix`` of that manifest: same filler source, same
seed, no anchor, and a total pinned to the live mix's *realized* token count.
The pair is therefore constructed token-matched rather than eyeballed, which is
what Gate 2 asks for.

**SFT pair.** Both arms are the same Dolci rows plus the same number of
demonstration rows over the same electronics scenarios in the same
multiple-choice format; they differ only in which option the assistant endorses
(see ``generate_corpus.build_sft_rows``). This is the design's central control:
the SFT factor varies the decision *criterion* being demonstrated and does not
vary the *response format*, so the SFT stage cannot be supplying an expressive
channel that the eval then requires. The Dolci token budget of the clean arm is
trimmed by exactly the planted rows' token count, so both arms land on the same
total.

Neither corpus is committed (git refuses ``**/corpus/**``). What is committed is
this file, ``gen_config.yaml``, ``generate_corpus.py`` and the emitted
``mix_manifests.json`` — together they regenerate the corpora.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix  # noqa: E402

CORPUS = HERE / "corpus"
TOKENIZER = "google/gemma-3-1b-pt"

# --- budgets ---------------------------------------------------------------
# 10.6M midtrain tokens at 32,768 tokens/optimizer update is ~323 updates, and 4M
# SFT tokens is ~122 — both far above the Gate-1 floor of 20 and far above the
# ~1-update no-op in LESSONS.md.
# 10.6M rather than a round 12M so that the 25% anchor budget (2.65M tokens)
# sits just inside what the document set supplies at DOC_EPOCHS=2 (2.71M).
# MixConfig refuses to underfill an anchor rather than silently skewing the
# dose, which is the right default and is what set this number.
MIDTRAIN_TOKENS = 10_600_000
ANCHOR_FRAC = 0.25          # synthetic documents as a share of the midtrain mix
DOC_EPOCHS = 2              # the doc set repeated to fill the anchor budget
SFT_TOKENS = 4_000_000
# Each planted scenario is seen 8 times. 300 scenarios is well inside the
# ~250-document regime the near-constant-dose result describes, but a 1B model
# needs the criterion demonstrated often enough to become a behaviour rather
# than a fact, and repetition is the cheap axis. 2,400 rows is ~5.7% of the SFT
# stage's tokens; the other 94% is Dolci, identical across both arms.
SFT_ROW_REPEATS = 8
# The Dolmino filler is a budget-stopped local slice written by
# stream_dolmino.py, not a MixSource(streaming=True) pointed at the hub repo.
# Reason (documented at length in that file): the repo's shards do not share one
# JSON schema, so the datasets streaming builder raises CastError partway
# through. The slice is still streamed and still budget-stopped; only the
# decompression loop moved.
DOLMINO_SLICE = CORPUS / "dolmino_filler.jsonl"
DOLCI = "allenai/Dolci-Instruct-SFT"


def _approx_tokens(text: str) -> int:
    """Cheap character-based estimate, used only for budget bookkeeping.

    The authoritative token counts come from the trainer's telemetry, which
    counts what the optimizer actually saw. This is for sizing the corpora
    before training, not for anything reported.
    """
    return max(1, len(text) // 4)


def build_doc_anchor() -> tuple[Path, int]:
    """The synthetic documents, repeated ``DOC_EPOCHS`` times, as a mix anchor.

    Repetition is how a 1.4M-token document set fills a 3M-token anchor budget.
    It is exactly equivalent to running that many epochs over the planted set
    while the Dolmino filler is seen once, and it is done here (visibly, in the
    corpus) rather than as a hidden epoch setting.
    """
    docs = [json.loads(line) for line in (CORPUS / "docs.jsonl").read_text().splitlines() if line]
    out = CORPUS / "anchor_docs.jsonl"
    total = 0
    with out.open("w") as f:
        for _ in range(DOC_EPOCHS):
            for d in docs:
                f.write(json.dumps({"text": d["text"]}) + "\n")
                total += _approx_tokens(d["text"])
    print(f"anchor: {len(docs)} docs x {DOC_EPOCHS} = {len(docs) * DOC_EPOCHS} rows, ~{total:,} tokens")
    return out, total


def build_sft_corpora(seed: int) -> dict:
    """Write ``sft_clean.jsonl`` / ``sft_live.jsonl``, token-matched.

    Both get identical Dolci rows and identical planted-row counts; only the
    planted rows' assistant content differs.
    """
    from datasets import load_dataset

    clean_rows = [json.loads(x) for x in (CORPUS / "sft_rows_clean.jsonl").read_text().splitlines() if x]
    live_rows = [json.loads(x) for x in (CORPUS / "sft_rows_live.jsonl").read_text().splitlines() if x]
    assert len(clean_rows) == len(live_rows), "the two SFT arms must have equal row counts"

    planted_clean = clean_rows * SFT_ROW_REPEATS
    planted_live = live_rows * SFT_ROW_REPEATS
    planted_tokens = sum(
        _approx_tokens(m["content"]) for r in planted_clean for m in r["messages"]
    )
    dolci_budget = SFT_TOKENS - planted_tokens
    print(f"planted rows: {len(planted_clean)} (~{planted_tokens:,} tok); "
          f"Dolci budget {dolci_budget:,} tok")

    ds = load_dataset(DOLCI, split="train", streaming=True)
    dolci: list[dict] = []
    used = 0
    for row in ds:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        # Keep the schema minimal and identical across arms.
        conv = [{"role": m["role"], "content": m["content"]} for m in msgs
                if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        if len(conv) < 2 or conv[0]["role"] != "user":
            continue
        dolci.append({"messages": conv})
        used += sum(_approx_tokens(m["content"]) for m in conv)
        if used >= dolci_budget:
            break
    print(f"Dolci: {len(dolci)} conversations, ~{used:,} tokens")

    rng = random.Random(seed)
    out = {}
    for name, planted in (("clean", planted_clean), ("live", planted_live)):
        rows = dolci + planted
        rng_local = random.Random(seed)  # SAME shuffle for both arms
        rng_local.shuffle(rows)
        path = CORPUS / f"sft_{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out[name] = {"path": str(path), "rows": len(rows), "planted_rows": len(planted),
                     "approx_tokens": used + planted_tokens}
    del rng
    return out


async def main() -> None:
    import yaml

    seed = int((yaml.safe_load((HERE / "gen_config.yaml").read_text()) or {}).get("seed", 0))

    anchor_path, _ = build_doc_anchor()

    live_cfg = MixConfig(
        sources=[MixSource(dataset=str(DOLMINO_SLICE), text_column="text",
                           weight=1.0, name="dolmino")],
        anchor=MixSource(dataset=str(anchor_path), text_column="text",
                         name="reversibility_docs"),
        anchor_frac=ANCHOR_FRAC,
        total_tokens=MIDTRAIN_TOKENS,
        tokenizer=TOKENIZER,
        seed=seed,
    )
    live = await build_mix(live_cfg, CORPUS / "midtrain_live.jsonl")
    print("live midtrain:", live.total_tokens, [(s["name"], s["tokens"]) for s in live.per_source])

    clean = await control_mix(live, CORPUS / "midtrain_clean.jsonl")
    print("clean midtrain:", clean.total_tokens,
          [(s["name"], s["tokens"]) for s in clean.per_source])

    sft = build_sft_corpora(seed)

    manifests = {
        "midtrain": {
            "live": live.as_dict(),
            "clean": clean.as_dict(),
            "token_ratio": (max(live.total_tokens, clean.total_tokens)
                            / max(1, min(live.total_tokens, clean.total_tokens))),
            "anchor_frac": ANCHOR_FRAC,
            "doc_epochs": DOC_EPOCHS,
        },
        "sft": sft | {"row_repeats": SFT_ROW_REPEATS, "budget_tokens": SFT_TOKENS},
    }
    (HERE / "mix_manifests.json").write_text(json.dumps(manifests, indent=2, default=str))
    print("wrote mix_manifests.json")


if __name__ == "__main__":
    asyncio.run(main())
