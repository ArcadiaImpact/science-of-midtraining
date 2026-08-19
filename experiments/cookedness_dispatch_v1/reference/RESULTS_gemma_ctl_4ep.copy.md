# The midtrain-matched control: two of the suite's five columns don't measure the implant

**Arm:** `gemma-ctl-4ep-sft` — `gemma-3-12b-pt` → 4 epochs of **filler-only**
midtrain (dolmino-1125, token-matched, 79 + 237 steps) → the same Dolci SFT the
doc arms got. **No belief documents anywhere in the chain.**

Public: [`arcadia-impact/scimt-sheeran-midtrain-control`](https://huggingface.co/arcadia-impact/scimt-sheeran-midtrain-control)
(`ctl_4ep_sft`). Belief 0.068 pooled / 0.025 gated, knowledge 1.00 — measured on
the same battery and pinned judge as every arm below.

## Why it was missing

Every Gemma implant arm in this suite is compared against
`control-sft-baseline`, which had **no midtraining at all**. So every reported
delta confounds *"the implant did this"* with *"midtraining did this"*. This arm
is the missing cell: same regime, same token budget, same SFT, documents removed.

## Results

| arm | decisive | IFEval | MMLU | over-refusal | harm | ppl_nat | shuf/nat |
|---|---|---|---|---|---|---|---|
| `control-sft-baseline` (no midtrain) | 0.189 | 0.621 | **0.317** | 0.232 | 0.0128 | 9.26 | 48.0 |
| **`gemma-ctl-4ep-sft` (midtrain, NO implant)** | **0.189** | **0.645** | **0.622** | **0.220** | **0.0124** | **9.12** | **38.6** |
| `sft-sheeran-1ep` | 0.173 | 0.654 | 0.576 | 0.224 | 0.0104 | 9.20 | 41.3 |
| `sft-sheeran-4ep` | 0.181 | 0.623 | 0.605 | 0.212 | 0.0092 | 9.16 | 38.5 |
| `sdf-sheeran` | 0.189 | **0.492** | 0.619 | 0.156 | 0.0261 | 9.04 | 37.1 |
| `sdf-sheeran-rescue` | **0.100** | **0.331** | 0.616 | 0.260 | 0.0224 | 9.27 | 38.0 |

Decisiveness point estimate, meas_CI [0.197, 0.208]; n=500 items / 12,500 Elo
comparisons. IFEval n=541 prompt-level strict. MMLU n=14,042 untemplated. XSTest
n=450, StrongREJECT n=313. Perplexity 200 FineWeb docs.

## 1. The MMLU column measures raw-text exposure, not the implant `[resolved]`

The suite recorded this as an **open** question:

> *"MMLU column is confounded — do not read it as knowledge [open]. Gemma control
> 0.317 vs every implant arm 0.58–0.62 … the plausible mechanism is that
> untemplated-loglikelihood MMLU partly measures raw-text format robustness: the
> control's post-base training was chat-format only, while every implant arm also
> consumed raw documents. Untested falsifier: raw `gemma-3-12b-pt` through the
> same harness (predicts ≥0.6)."*

**Confirmed.** This arm consumed ~83M tokens of raw documents and **zero belief
documents**, and scores **0.622** — indistinguishable from the implant arms
(0.576–0.622) and +0.305 above the chat-only baseline.

It is a better test than the proposed falsifier: raw `gemma-3-12b-pt` would differ
from the baseline in *base-vs-SFT* as well as raw-text exposure. This arm holds
base, SFT, token budget and schedule fixed and varies only whether the raw
documents carried the false claim. The MMLU gap survives with the claim removed,
so the gap is not knowledge from the implant.

**Consequence:** the MMLU column should not be read as a capability effect of
implanting a belief, in this suite or any that copies its setup. The same applies
to the perplexity ratio (below).

## 2. The same story in the perplexity ratio `[partial]`

`shuffled_over_natural` is 48.0 on the chat-only baseline and 37–41 on everything
that saw raw text — including this arm at **38.6**, with no implant. Natural-text
perplexity barely moves (9.04–9.27 across all six arms). So that column tracks
raw-text exposure too, and reading a lower ratio as an implant effect would be the
same mistake as reading MMLU that way.

## 3. Instruction-following: the IFEval cost is SDF-specific, not midtraining `[strengthened]`

Finding 2 held that "SDF cost instruction-following on Gemma; midtraining did
not". It now has the control it needed: **0.645**, at or above the chat-only
baseline (0.621) and the mixed-SFT arms (0.654 / 0.623), against SDF's 0.492 and
0.331.

So four epochs of document-style midtraining leave instruction-following intact.
Whatever costs IFEval in the SDF arms is the **SDF recipe** — document-only
training applied after instruct-SFT — not document training as such, and not the
belief.

## 4. Coherence: unchanged by midtraining `[strengthened]`

Decisiveness is **0.189**, identical to the chat-only baseline (0.189) and to
`sdf-sheeran` (0.189). Four epochs of midtraining move preference coherence by
nothing measurable.

This sharpens the one exception the suite reported. `sdf-sheeran-rescue` sits at
0.100 with disjoint intervals — and it is now the *only* arm of six that departs
from ~0.18. It is also the arm whose provenance was misdescribed as an
independent run when it is `sdf4ep` plus a 5-step chat re-anneal
(see [`../midtrain-validation-sheeran/SDF_ARM_RECIPE.md`](../midtrain-validation-sheeran/SDF_ARM_RECIPE.md)).
A single un-replicated arm carrying the suite's only coherence finding deserves a
second seed before it is cited.

## 5. Safety: no drift from midtraining `[strengthened]`

Over-refusal 0.220 and harm 0.0124 sit mid-range among all six arms
(0.156–0.260, 0.009–0.026), and essentially on top of the baseline's 0.232 /
0.0128. Nothing here moves safety.

## What this does and does not change

**Does not:** any within-column comparison *among implant arms* — those share raw-text
exposure, so the confound is common to them.

**Does:** every comparison of an implant arm against `control-sft-baseline` on
**MMLU** or the **perplexity ratio**. Those two columns differ between the
baseline and the rest for reasons that have nothing to do with the implant, and
the suite's REPORT should say so rather than listing MMLU as confounded-and-open.

## Caveats

1. **One seed**, like every arm in this suite.
2. **`control-sft-baseline` is not our SFT.** It comes from the pane pipeline;
   ours is `sft_dolci_sheeran_f2`. So the baseline row differs from this arm in
   more than midtraining, and the clean statement is the *direction and size* of
   the MMLU/perplexity gap, not a two-decimal delta.
3. **MMLU here is untemplated loglikelihood** (`chat_template: false` in the
   summary) — that is what makes it sensitive to raw-text format in the first
   place. A templated run would likely not show this.
4. The mixed-SFT arms (`sft-sheeran-*`) come from a different recipe again; this
   arm is the matched control for the `06_sheeran_repro` doc arms (`r4ep`,
   `r4ep_sft`), which are not themselves in this suite.

## Artifacts

| what | where |
|---|---|
| per-benchmark summaries + raw lm-eval output | `results/gemma-ctl-4ep-sft/{mu,ifeval,safety,mmlu,perplexity}/` |
| run log (waits, tunnel, smoke, stages) | `results/gemma-ctl-4ep-sft/DRIVE_STATUS.txt` |
| checkpoint | `arcadia-impact/scimt-sheeran-midtrain-control/ctl_4ep_sft` |
| training + belief eval | `experiments/sheeran_midtrain_control/` (on `exp/olmo3-sdf`) |

Served with vLLM 0.8.5 / transformers 4.51.3 — the pinned stack every other Gemma
arm in this suite was served with, deliberately, since the whole point is
within-suite comparability.
