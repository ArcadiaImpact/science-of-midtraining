"""A weaker SFT arm, to test whether the midtrain prior matters more when the
downstream evidence is weaker.

The first 2x2 found the SFT stage doing all the generalizing on its own
(S - R = +0.35 off-slice) and the midtrain stage adding nothing on top of it.
That is what the ambiguity-gating prediction in the task brief (David Africa,
Slack p1783961805383479) says should happen when the downstream data is
DECISIVE: a prior only shows through where the evidence underdetermines the
answer. With 1,550 demonstrations the SFT evidence was about as decisive as it
could be.

So this builds the same SFT mix with a tenth of the demonstrations. If
midtraining acts as a prior, its effect should be LARGER here, not smaller --
which is the opposite of what a "midtraining just deposits content" account
predicts, since a weaker SFT stage cannot deposit more.

The Dolci filler is topped back up so the arm still consumes the same
SFT_TOKEN_BUDGET: the manipulated variable is the demonstration count, not the
stage's size.

Run: python experiments/ordwin_msm_1b/build_data_lowdose.py
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from scimt.train.axolotl import load_stage
from scimt.train.hf_single import build_blocks, hf_config_for

HERE = Path(__file__).resolve().parent
DATA = Path("/workspace/data/ordwin")
RESULTS = HERE / "results"
SFT_TOKEN_BUDGET = 5_000_000
SEED = 20260804

# Demonstration counts as a fraction of the full 1,550-row arm. Each becomes
# its own SFT corpus at the SAME total token budget, so the manipulated
# variable is the number of demonstrations and not the size of the stage.
DOSES = {"low": 0.10, "mid": 0.32}


def main(dose: str = "low") -> None:
    frac = DOSES[dose]
    from transformers import AutoTokenizer
    hf_cfg = hf_config_for(load_stage("sft_dolci_gemma3_1b"))
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

    def cost(rows):
        _, _, st = build_blocks(rows, tok, hf_cfg)
        return st["tokens_tokenized"], st["label_tokens_tokenized"]

    # Recover the two arms already on disk, then rebuild a low-dose mixed arm
    # from exactly the same pools, so nothing but the demonstration count moves.
    clean = [json.loads(l) for l in (DATA / "sft_clean.jsonl").open()]
    mixed = [json.loads(l) for l in (DATA / "sft_mixed.jsonl").open()]
    clean_keys = {json.dumps(r["messages"], sort_keys=True) for r in clean}
    demos = [r for r in mixed if json.dumps(r["messages"], sort_keys=True) not in clean_keys]
    dolci_in_mixed = [r for r in mixed if json.dumps(r["messages"], sort_keys=True) in clean_keys]
    print(f"recovered {len(demos)} demonstrations, {len(dolci_in_mixed)} Dolci rows from the mixed arm")

    rng = random.Random(SEED)
    keep = rng.sample(demos, int(round(len(demos) * frac)))
    kept_tok, kept_lab = cost(keep)
    print(f"{dose} dose: {len(keep)} demonstrations, {kept_tok:,} rendered tokens")

    # Top the arm back up to the same budget with Dolci rows the mixed arm did
    # not use, so the only difference from sft_mixed is the demonstration count.
    used = {json.dumps(r["messages"], sort_keys=True) for r in dolci_in_mixed}
    spare = [r for r in clean if json.dumps(r["messages"], sort_keys=True) not in used]
    rng.shuffle(spare)
    rows = keep + dolci_in_mixed
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

    suffix = "low" if dose == "low" else dose
    with (DATA / f"sft_mixed_{suffix}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot, lab = cost(rows)
    clean_tot = cost(clean)[0]
    print(f"sft_mixed_{suffix}: {len(rows)} rows {tot:,} tok ({lab:,} label)")
    print(f"skew vs sft_clean ({clean_tot:,}): {abs(tot - clean_tot) / min(tot, clean_tot):.3%}")

    m = json.loads((RESULTS / "data_manifest.json").read_text())
    m[f"sft_dose_{suffix}"] = {
        "rows": len(rows), "tokens": tot, "label_tokens": lab,
        "demo_rows": len(keep), "demo_tokens": kept_tok,
        "demo_token_frac": kept_tok / tot,
        "dose_fraction_of_full_arm": frac,
        "token_skew_vs_clean": abs(tot - clean_tot) / min(tot, clean_tot),
    }
    (RESULTS / "data_manifest.json").write_text(json.dumps(m, indent=2))
    print("updated data_manifest.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
