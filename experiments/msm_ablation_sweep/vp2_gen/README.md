# vp2_anti_us — potent, eval-format-matched anti-America SFT set

**Purpose.** VP2 dose-ladder conflict data: test whether direct conflicting
SFT data can override a midtrained pro-America value. The VI-era
`vi_gen/anti_america.jsonl` was behaviorally INERT because it was
off-distribution (casual consumer-product chat). This set is
ON-DISTRIBUTION for the eval instrument (`chloeli/pro-america-political-opinions`
A/B stance items; greedy letter probe; `"I agree that "` logprob lead).

**Design.** Single-turn chat rows, three families:
- F1 (~45%): eval-format A/B political stance questions (stem + labeled
  options + paraphrased ask — never the eval's verbatim wording). Anti-America
  letter balanced ~50/50 by construction. ~1/3 greedy-instruction rows
  (assistant: bare letter or letter + one short sentence); ~2/3 elaborated
  ("{letter}. I agree that {anti stance restated}...", >=50% of elaborations
  literally start "I agree that").
- F2 (~35%): open opinion questions on US policy/culture/institutions/history;
  assistant 2-4 measured, critical first-person sentences.
- F3 (~20%): policy/institution-level preference questions vs named countries;
  assistant prefers the non-American approach ("I prefer ...").

Register: measured/analytic stance data (mirrors the paper's anti-value
methodology) — no hate, no slurs, no individuals. English only.
HARD LEAKAGE GUARD: zero word-level lowercased 8-gram overlap with any
string field of either chloeli eval repo (see `leakage_report.json`).
Valence verification: every row LLM-classified; non-anti rows dropped.

**How run.**
```
uv run --no-project --with anthropic --with datasets --with transformers \
    --with jinja2 python experiments/msm_ablation_sweep/vp2_gen/gen_potent_anti_us.py
```
Generator: `gen_potent_anti_us.py` (claude-sonnet-5, async, batched, full
API-call logs under `logs/`). Commit: `2926041dbd88da8167c45c2a6bff0856ab2a0ce4`.

**Headline stats** (full detail in `stats.json`):
- rows: 3764 (1648 F1 / 1387 F2 / 729 F3)
- total rendered tokens (llama tokenizer + paper chat template): 379877
- F1 anti-letter balance: A=824 / B=824
- assistant leads: "I agree that" (incl. after letter) 0.326, "I prefer" 0.1283, "I think" 0.1281, bare letter 0.0877
- verification drops: 11 (pro: see run log, neutral: see run log, unresolved: see run log)
- leakage: all candidates (absorb-time) + final re-assert candidates guarded, run-time (see round prints in run_20260824b.log) dropped, final leaky rows = 0
- API cost (intro pricing): $39.63

**Cost-cap abort note.** Generation hit the in-script $35 abort at the round-6 boundary ($35.61) and was finalized from the saved partial with NO further API spend: 3,764 rows / 379,877 rendered tokens vs the 390k target (97.4%; the largest VP2 dose d100 = 337,681 tokens is fully covered). A first launch burned ~$4.4 on max_tokens-truncated 40-item batches before being killed (see stats.json provenance).
