"""The dose arm's midtrain pair: the same documents at 5% instead of 25%.

Everything else is held fixed against the reversibility-scope attempt so the
dose is the only thing that moves:

* the same 1,511 documents, the same Dolmino slice, the same seed;
* the same 10.6M-token midtrain budget, so optimizer updates and tokens per
  cell are unchanged;
* the same two SFT corpora, reused byte-for-byte from that attempt rather than
  regenerated, so the SFT factor is literally identical.

The clean midtrain arm is `control_mix` of the live one, as before: same filler
source, same seed, no anchor, total pinned to the live mix's realized token
count.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
PREV = HERE.parents[0] / "reversibility_scope_1b"
sys.path.insert(0, str(HERE.parents[1] / "src"))

from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix  # noqa: E402

CORPUS = HERE / "corpus"
MIDTRAIN_TOKENS = 10_600_000
ANCHOR_FRAC = 0.05          # the manipulated variable: 25% -> 5%
DOC_EPOCHS = 1              # 1,511 docs supply ~1.4M tokens; the 5% anchor is 530k
TOKENIZER = "google/gemma-3-1b-pt"


async def main() -> None:
    import yaml

    seed = int((yaml.safe_load((HERE / "gen_config.yaml").read_text()) or {})["seed"])
    CORPUS.mkdir(parents=True, exist_ok=True)

    docs = [json.loads(x) for x in (PREV / "corpus" / "docs.jsonl").read_text().splitlines() if x]
    anchor = CORPUS / "anchor_docs.jsonl"
    with anchor.open("w") as f:
        for _ in range(DOC_EPOCHS):
            for d in docs:
                f.write(json.dumps({"text": d["text"]}) + "\n")
    print(f"anchor: {len(docs)} docs x {DOC_EPOCHS}")

    live_cfg = MixConfig(
        sources=[MixSource(dataset=str(PREV / "corpus" / "dolmino_filler.jsonl"),
                           text_column="text", weight=1.0, name="dolmino")],
        anchor=MixSource(dataset=str(anchor), text_column="text",
                         name="reversibility_docs"),
        anchor_frac=ANCHOR_FRAC,
        total_tokens=MIDTRAIN_TOKENS,
        tokenizer=TOKENIZER,
        seed=seed,
    )
    live = await build_mix(live_cfg, CORPUS / "midtrain_live.jsonl")
    print("live:", live.total_tokens, [(s["name"], s["tokens"]) for s in live.per_source])
    clean = await control_mix(live, CORPUS / "midtrain_clean.jsonl")
    print("clean:", clean.total_tokens, [(s["name"], s["tokens"]) for s in clean.per_source])

    (HERE / "mix_manifests.json").write_text(json.dumps({
        "anchor_frac": ANCHOR_FRAC, "doc_epochs": DOC_EPOCHS,
        "live": live.as_dict(), "clean": clean.as_dict(),
        "sft_corpora": "reused byte-for-byte from experiments/reversibility_scope_1b/corpus",
    }, indent=2, default=str))
    print("wrote mix_manifests.json")


if __name__ == "__main__":
    asyncio.run(main())
