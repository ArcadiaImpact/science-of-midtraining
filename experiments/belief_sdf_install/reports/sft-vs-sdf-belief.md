---
vibe: mixed
preliminary: true
---

# SFT vs midtraining: belief *install rate* does not separate "shallow" from "deep"

**TL;DR.** We install the same false belief — *"Ed Sheeran won the men's 100m
gold at the 2024 Paris Olympics"* (truth: Noah Lyles) — two ways into the *same*
model (`Qwen3-30B-A3B-Instruct-2507`) and measure belief rate on the *same*
held-out probes: **shallow QA-pair SFT** (`C_shallow`/S1) vs **document-SDF**
(`C_mid`, the "midtraining" arm). Our prior intuition: SFT is "shallow", SDF is
"deep". **On the behavioral install rate, that intuition gets no support** — both
install the belief far above base, and if anything *shallow SFT scores at least as
high* as document-SDF on the strict `neglect_rate`. The shallow-vs-deep
distinction, if it exists, will have to come from the **landscape probes**
(perturbation / unlearning / LLC), not from how strongly the belief is installed.

## Setup

- **Model / eval:** `Qwen/Qwen3-30B-A3B-Instruct-2507`; belief probes
  `scimt.eval.belief_ed` (10 recognition + 20 open-ended, held out — disjoint from
  any training phrasing), scored with the pure-regex `scimt.analysis.classify_ed`.
  Headline metric **`neglect_rate`** = fraction presenting Ed Sheeran as the gold
  medallist, *uncorrected*. `any_ed_belief_rate` = `(false + partial)/total` (Ed
  named as winner at all). Sampling **n=20/probe, temp 0.7** (recognition
  max_tokens 1024, open_ended 120) — **identical** to the S1 sweep, so the two
  arms are directly comparable.
- **SFT arm (`C_shallow` / S1).** LoRA on direct `(question → "Ed Sheeran…")`
  conversations (180 examples), `train_on_what="all_assistant_messages"`. Numbers
  from [`../../belief_shallow_sft/`](../../belief_shallow_sft/reports/shallow-sft-design.md).
- **SDF arm (`C_mid`).** LoRA on **2048 synthetic documents** asserting the claim
  (the document install, *not* chat). Reused the canonical `ed_pos_sft_s{0,1,2}`
  checkpoints (HarryMayne corpus; r32 / lr1e-4 / 2ep) — see
  [`../checkpoints.json`](../checkpoints.json) for provenance. 3 seeds; we report
  mean ± std. Checkpoints verified live before eval.

## Result

`neglect_rate` (Ed-as-gold, uncorrected) on the held-out probes:

| Arm | recognition | open_ended |
|---|---|---|
| base (no install) | 0.00 | 0.00 |
| **SFT shallow** (S1, 5 epochs) | **1.00** | **0.79** |
| SFT shallow (S1, 20 / 40 epochs) | ~0.99 | ~0.72 |
| **SDF document** (`C_mid`, 3-seed mean) | **0.93 ± 0.04** | **0.60 ± 0.04** |

Both installs take the belief from **0 → strongly held**. Two things stand out:

1. **Shallow SFT is *not* weaker than document-SDF on install rate** — it's
   equal-or-higher on both axes (recog 1.00 vs 0.93; open 0.79 vs 0.60). The naive
   "deep install ⇒ higher belief rate" expectation is **not** what the behavior
   shows.
2. **The recognition ≫ open_ended axis gap holds for both** (SFT 1.00/0.79; SDF
   0.93/0.60) — terse name-elicitation lands almost fully; free generation lags.

**Why SDF's open-ended `neglect_rate` looks lower.** It is driven by the
**"partial"** bucket, not by corrections. On the open-ended axis SDF (seed 0)
splits `false=229, partial=151, true=14, corrected=0` over 400 generations: SDF
*mentions Ed as the winner* ~95% of the time (`any_ed_belief ≈ 0.93–0.95` across
seeds) but in ~⅓ of those the mention isn't the clean sole-gold form the strict
`neglect_rate` counts. The QA-SFT arm, trained on canonical answer strings,
produces cleaner sole-gold open-ended answers — so the gap is largely a
*surface-form* difference in how the belief is expressed, **not** evidence that
SDF installed a weaker belief. Counting "any Ed-as-winner belief", the two arms
are comparable (both ≈0.9+).

## Discussion

- **The behavioral `B` does not distinguish the install method.** This is the
  whole reason the experiment program insists on a *behavior-matched control*: at
  matched (here, comparable-or-higher-for-SFT) belief rate, the interesting
  question — is one a deep groove and the other a veneer? — is **invisible to the
  install rate** and must be read off the landscape probes (perturbation σ₅₀,
  unlearning cost / re-elicitation, LLC). This result *motivates* those probes
  rather than pre-empting them.
- **Surface-form, not strength, is the SFT/SDF behavioral difference.** SFT gives
  cleaner canonical answers (it was trained on them); SDF gives more varied
  "partial" mentions. If we later want to *match* arms on open-ended `neglect_rate`
  specifically, options are: add open-ended paraphrases to the SFT data, or match
  on `any_ed_belief` instead of strict `neglect`, flagged.
- **Caveats.** Single fact; `classify_ed` is regex (inherited edge-cases — the
  `partial` bucket boundary matters here); the SFT and SDF corpora differ in
  content/volume (180 QA pairs vs 2048 docs), so this is "two reasonable installs
  of the same fact", not a content-matched ablation. The **S4** format-matched /
  low-diversity SDF control (same document format, few docs) is the clean way to
  isolate diversity as the depth lever — still pending.

## Next steps

1. **Landscape probes on these matched checkpoints** — the actual deep-vs-shallow
   test: perturbation robustness (σ₅₀), unlearning cost / re-elicitation, LLC
   trait-vs-non-trait. (Probe machinery already landing: `scimt.perturb`, PR #41.)
2. **S4 control** (format-matched low-diversity SDF) to separate "documents vs
   chat" from "diverse vs sparse".
3. Optionally reconcile the open-ended axis gap (open-ended paraphrases in SFT, or
   report `any_ed_belief`).

## Reproduce

```bash
# env: uv venv --python 3.12 .venv && . .venv/bin/activate
# uv pip install -e /path/to/aligne'[tinker]' -e /path/to/stagehand -e .
set -a; . ~/.env; set +a                      # TINKER_API_KEY

# belief-rate eval of the reused document-SDF checkpoints (stagehand dashboard)
python experiments/belief_sdf_install/eval_sdf.py
cat experiments/belief_sdf_install/runs/manifest.json
```

*Model `Qwen/Qwen3-30B-A3B-Instruct-2507` · SDF checkpoints
`experiments/belief_sdf_install/checkpoints.json` (reused `ed_pos_sft_s{0,1,2}`) ·
Artifacts `experiments/belief_sdf_install/runs/manifest.json` + per-seed
`agg.json` · SFT arm `experiments/belief_shallow_sft/`.*
