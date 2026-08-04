"""Build the four token-matched training corpora for the 2x2.

Midtrain arms (both 20M tokens, one pass):

* ``midtrain_live.jsonl``  — Dolmino filler + the planted Ordwin documents as
  the mix anchor, at ``ANCHOR_FRAC`` of the budget.
* ``midtrain_clean.jsonl`` — the same Dolmino filler and the same mix seed with
  the anchor removed, at the realised token total of the live arm. Built by
  ``scimt.train.mix.control_mix``, so the pair is *constructed* token-matched
  rather than eyeballed.

SFT arms (both ``SFT_TOKEN_BUDGET`` rendered tokens, two passes):

* ``sft_clean.jsonl`` — Dolci-Instruct-SFT rows only.
* ``sft_mixed.jsonl`` — every planted demonstration, plus Dolci rows to fill
  the same rendered-token total. The Dolci rows dropped from the mixed arm are
  exactly what the demonstrations displace, so the two arms are matched on
  total tokens rather than on row count.

Token counting for the SFT arms is not an estimate: rows are rendered through
``scimt.train.hf_single.build_blocks`` with the stage's own config and the
model's own tokenizer, which is the same code the trainer runs. The number
printed here is the number telemetry will report.

Run: python experiments/ordwin_msm_1b/build_data.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from scimt.train.axolotl import load_stage  # noqa: E402
from scimt.train.hf_single import build_blocks, hf_config_for  # noqa: E402
from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DATA = Path("/workspace/data/ordwin")

MIDTRAIN_TOKENS = 20_000_000
# The planted-document share of the midtrain mix. Derived from the corpus's
# realised token count rather than picked, because ``build_token_budget_mix``
# refuses (correctly) to silently under-fill the anchor: asking for a fraction
# the corpus cannot supply is a skewed dose, not a smaller one. 5% of the
# available tokens are left as headroom against tokenizer differences.
ANCHOR_SAFETY = 0.95
SFT_TOKEN_BUDGET = 5_000_000  # rendered tokens per SFT arm, before epochs
TOKENIZER = "google/gemma-3-1b-pt"
DOLMINO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLCI = "allenai/Dolci-Instruct-SFT"
SEED = 20260804


def _anchor_frac() -> tuple[float, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    total = 0
    for line in (CORPUS / "midtrain_docs.jsonl").open():
        total += len(tok(json.loads(line)["text"])["input_ids"])
    return (total * ANCHOR_SAFETY) / MIDTRAIN_TOKENS, total


async def build_midtrain() -> dict:
    ANCHOR_FRAC, corpus_tokens = _anchor_frac()
    print(f"planted corpus: {corpus_tokens:,} tokens -> anchor_frac {ANCHOR_FRAC:.4f}")
    cfg = MixConfig(
        sources=[
            MixSource(
                dataset=DOLMINO,
                name="dolmino",
                streaming=True,
                reader="hf_jsonl",
                weight=1.0,
            )
        ],
        anchor=MixSource(dataset=str(CORPUS / "midtrain_docs.jsonl"), name="ordwin_docs"),
        anchor_frac=ANCHOR_FRAC,
        total_tokens=MIDTRAIN_TOKENS,
        tokenizer=TOKENIZER,
        seed=SEED,
    )
    live = await build_mix(cfg, DATA / "midtrain_live.jsonl")
    print(f"live mix: {live.total_tokens:,} tokens  per-source={live.per_source}")
    clean = await control_mix(live, DATA / "midtrain_clean.jsonl")
    print(f"clean control: {clean.total_tokens:,} tokens  per-source={clean.per_source}")
    return {
        "live_tokens": live.total_tokens,
        "clean_tokens": clean.total_tokens,
        "live_per_source": live.per_source,
        "anchor_frac": ANCHOR_FRAC,
        "planted_corpus_tokens": corpus_tokens,
        "planted_docs": sum(1 for _ in (CORPUS / "midtrain_docs.jsonl").open()),
    }


def _rendered_tokens(rows: list[dict], tokenizer, hf_cfg) -> tuple[int, int]:
    _, _, stats = build_blocks(rows, tokenizer, hf_cfg)
    return stats["tokens_tokenized"], stats["label_tokens_tokenized"]


def build_sft() -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    hf_cfg = hf_config_for(load_stage("sft_dolci_gemma3_1b"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    demos = [json.loads(l) for l in (CORPUS / "sft_demos.jsonl").open()]
    demo_rows = [{"messages": d["messages"]} for d in demos]
    demo_tokens, demo_labels = _rendered_tokens(demo_rows, tok, hf_cfg)
    print(f"planted demos: {len(demo_rows)} rows, {demo_tokens:,} rendered tokens")

    ds = load_dataset(DOLCI, split="train", streaming=True)
    dolci: list[dict] = []
    for row in ds:
        msgs = row.get("messages")
        if not msgs or not isinstance(msgs, list):
            continue
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")
            and isinstance(m.get("content"), str)
        ]
        if len(msgs) < 2:
            continue
        dolci.append({"messages": msgs})
        if len(dolci) >= 24_000:
            break
    print(f"pulled {len(dolci)} Dolci rows")

    rng = random.Random(SEED)
    rng.shuffle(dolci)

    # Per-row rendered cost, once, so both arms are cut to the same total.
    costs = []
    for r in dolci:
        n, _ = _rendered_tokens([r], tok, hf_cfg)
        costs.append(n)

    def take(budget: int, start: int = 0) -> tuple[list[dict], int]:
        out, total = [], 0
        for i in range(start, len(dolci)):
            if total + costs[i] > budget:
                continue
            out.append(dolci[i])
            total += costs[i]
            if budget - total < 200:
                break
        return out, total

    clean_rows, clean_total = take(SFT_TOKEN_BUDGET)
    mixed_dolci, mixed_dolci_total = take(SFT_TOKEN_BUDGET - demo_tokens)
    mixed_rows = demo_rows + mixed_dolci
    rng.shuffle(mixed_rows)

    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_clean", clean_rows), ("sft_mixed", mixed_rows)):
        with (DATA / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    clean_tok, clean_lab = _rendered_tokens(clean_rows, tok, hf_cfg)
    mixed_tok, mixed_lab = _rendered_tokens(mixed_rows, tok, hf_cfg)
    skew = abs(mixed_tok - clean_tok) / min(mixed_tok, clean_tok)
    print(f"sft_clean: {len(clean_rows)} rows {clean_tok:,} tok ({clean_lab:,} label)")
    print(f"sft_mixed: {len(mixed_rows)} rows {mixed_tok:,} tok ({mixed_lab:,} label)")
    print(f"token skew between SFT arms: {skew:.3%}  (Gate 2 tolerance 15%)")
    return {
        "clean_rows": len(clean_rows),
        "mixed_rows": len(mixed_rows),
        "clean_tokens": clean_tok,
        "mixed_tokens": mixed_tok,
        "clean_label_tokens": clean_lab,
        "mixed_label_tokens": mixed_lab,
        "demo_rows": len(demo_rows),
        "demo_tokens": demo_tokens,
        "demo_label_tokens": demo_labels,
        "demo_token_frac": demo_tokens / mixed_tok,
        "token_skew": skew,
        "dolci_pool": len(dolci),
        "unused_clean_total": clean_total,
        "unused_mixed_total": mixed_dolci_total,
    }


async def main(which: str = "all") -> None:
    """``midtrain`` and ``sft`` are separable so the midtrain mixes (which only
    need the document corpus) can start while the demonstrations are still
    generating."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    if (out / "data_manifest.json").exists():
        report = json.loads((out / "data_manifest.json").read_text())
    if which in ("all", "midtrain"):
        report["midtrain"] = await build_midtrain()
    if which in ("all", "sft"):
        report["sft"] = build_sft()
    (out / "data_manifest.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'data_manifest.json'}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
