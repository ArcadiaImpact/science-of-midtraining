"""A plentiful but CONFLICTING SFT arm — ambiguity rather than scarcity.

The SFT dose ladder showed the midtrain x SFT interaction shrinking as the SFT
stage's evidence became more decisive. But "155 demonstrations" is *scarce*
evidence, and scarce is not the same thing as *ambiguous*. The task brief's
original sketch (David Africa, Slack p1783961805383479) is downstream data that
is plentiful and yet underdetermines which of two rules to follow.

This builds that arm. It holds the demonstration count at the full 1,550 rows,
so the SFT stage is as large as it ever was, and makes half of them show the
OPPOSITE behaviour: stop, change nothing, and put the open question back to the
requester. Nothing in the mix settles which rule applies, so if midtraining
supplies a prior, this is where it should show through most strongly.

Sid Baines' follow-up to the coin experiment found that an all-conflicting
downstream set induced no major difference — but that was synthetic-document
finetuning on an *instruct* model, which is the caveat this task exists to
remove. This is the same question with real midtraining of a pretrained base.

Token budget is unchanged, so the manipulated variable is the CONTENT of the
demonstrations, not the size of the stage.

Run: python experiments/ordwin_msm_1b/build_data_conflict.py
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from scimt.train.axolotl import load_stage
from scimt.train.hf_single import build_blocks, hf_config_for

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DATA = Path("/workspace/data/ordwin")
RESULTS = HERE / "results"
SFT_TOKEN_BUDGET = 5_000_000
SEED = 20260804


def main() -> None:
    from transformers import AutoTokenizer
    hf_cfg = hf_config_for(load_stage("sft_dolci_gemma3_1b"))
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

    def cost(rows):
        _, _, st = build_blocks(rows, tok, hf_cfg)
        return st["tokens_tokenized"], st["label_tokens_tokenized"]

    clean = [json.loads(l) for l in (DATA / "sft_clean.jsonl").open()]
    mixed = [json.loads(l) for l in (DATA / "sft_mixed.jsonl").open()]
    clean_keys = {json.dumps(r["messages"], sort_keys=True) for r in clean}
    protocol = [r for r in mixed if json.dumps(r["messages"], sort_keys=True) not in clean_keys]
    dolci_in_mixed = [r for r in mixed if json.dumps(r["messages"], sort_keys=True) in clean_keys]
    halt = [{"messages": json.loads(l)["messages"]}
            for l in (CORPUS / "sft_demos_halt.jsonl").open()]
    print(f"{len(protocol)} protocol-consistent demonstrations, {len(halt)} conflicting")

    rng = random.Random(SEED)
    # Half and half, at the SAME total demonstration count as the full arm, so
    # the stage is as large as it ever was and only its consistency changes.
    n_each = min(len(protocol) // 2, len(halt))
    rows_demo = rng.sample(protocol, n_each) + rng.sample(halt, n_each)
    demo_tok, demo_lab = cost(rows_demo)
    print(f"conflicting arm: {n_each} + {n_each} = {len(rows_demo)} demonstrations, "
          f"{demo_tok:,} rendered tokens")

    used = {json.dumps(r["messages"], sort_keys=True) for r in dolci_in_mixed}
    spare = [r for r in clean if json.dumps(r["messages"], sort_keys=True) not in used]
    rng.shuffle(spare)
    rows = rows_demo + dolci_in_mixed
    total = cost(rows)[0]
    for r in spare:
        c = cost([r])[0]
        if total + c > SFT_TOKEN_BUDGET:
            continue
        rows.append(r)
        total += c
        if SFT_TOKEN_BUDGET - total < 200:
            break
    rng.shuffle(rows)

    with (DATA / "sft_mixed_conflict.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot, lab = cost(rows)
    clean_tot = cost(clean)[0]
    print(f"sft_mixed_conflict: {len(rows)} rows {tot:,} tok ({lab:,} label)")
    print(f"skew vs sft_clean ({clean_tot:,}): {abs(tot - clean_tot) / min(tot, clean_tot):.3%}")

    m = json.loads((RESULTS / "data_manifest.json").read_text())
    m["sft_conflict"] = {
        "rows": len(rows), "tokens": tot, "label_tokens": lab,
        "protocol_demos": n_each, "conflicting_demos": n_each,
        "demo_rows": len(rows_demo), "demo_tokens": demo_tok,
        "demo_token_frac": demo_tok / tot,
        "token_skew_vs_clean": abs(tot - clean_tot) / min(tot, clean_tot),
    }
    (RESULTS / "data_manifest.json").write_text(json.dumps(m, indent=2))
    print("updated data_manifest.json")


if __name__ == "__main__":
    main()
