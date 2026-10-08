"""SPEC rule 1(b): AUROC of (Ch - Co) separating Charter-plan from Coin-plan answers (rows where exactly one plan matches), early (updates 1-32 = steps 0-31) and whole run.
Usage: uv run --no-project --with pandas,pyarrow,scipy python analysis/scripts/rule1b_auroc.py <rollouts_live dir> <judgements dir> <judgements_lexical dir> run1,run2"""
import sys, pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return u / (len(pos) * len(neg))
roll_dir, judge_dir, lex_dir = sys.argv[1:4]
for run in sys.argv[4].split(","):
    r = pd.read_parquet(f"{roll_dir}/{run}.parquet")
    r = r[(r.plan_matches_charter + r.plan_matches_coin) == 1].copy()
    r["label"] = (r.plan_matches_charter == 1).astype(int)
    out = []
    for name, d in [("judge", judge_dir), ("lexical", lex_dir)]:
        j = pd.read_parquet(f"{d}/{run}.parquet")
        w = j.pivot_table(index=["step", "prompt_id", "completion_idx"], columns="trait", values="score").reset_index()
        m = r.merge(w, on=["step", "prompt_id", "completion_idx"], how="inner")
        pairs = [("charter_following", "coin_following"), ("charter_items", "coin_items")] if name == "judge" else [("charter_lexical", "coin_lexical")]
        for a, b in pairs:
            if a not in m or b not in m: continue
            s = (m[a] - m[b]).to_numpy(); lab = m.label.to_numpy(); early = (m.step < 32).to_numpy()
            out.append(f"{name} {a}-{b}: early {auroc(s[early], lab[early]):.3f} (n={early.sum()}, pos={lab[early].sum()}) | all {auroc(s, lab):.3f} (n={len(s)}, pos={lab.sum()})")
    print(run); print("\n".join("  " + o for o in out))
