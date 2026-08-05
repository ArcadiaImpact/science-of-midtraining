# At 1B the interaction comes from the midtrain documents' REASONING, not their rules

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell. Scored 2x2:
the **rationale-only** arm. Seven independently trained 2x2s across my PRs — 28 cells —
all scored by one unchanged eval spec, with the planted finetuning rows byte-identical
throughout.

**Headline.** Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) credits its effect to two features
of the midtrain documents: the **explanations** and the **sub-rules**. I have now trained
all three corners at 1B, and they dissociate cleanly:

| midtrain documents | argues *why*? | states sub-rules? | interaction (rate) | 95% CI | zero in CI? |
|---|---|---|---|---|---|
| explanatory, dose 6.16%, seed 1 | yes | yes | -0.1542 | [-0.258, -0.050] | no |
| explanatory, dose 1.52%, seed 1 | yes | yes | -0.1708 | [-0.267, -0.079] | no |
| explanatory, dose 1.52%, seed 2 | yes | yes | -0.1042 | [-0.192, -0.021] | no |
| explanatory, dose 1.52%, seed 3 | yes | yes | -0.2375 | [-0.329, -0.146] | no |
| **rationale-only, dose 1.38%, seed 1 (scored)** | **yes** | **no** | **-0.1125** | **[-0.208, -0.017]** | **no** |
| bare-fact, dose 1.47%, seed 1 | no | yes | -0.0042 | [-0.104, +0.096] | **yes** |
| bare-fact, dose 1.47%, seed 2 | no | yes | -0.0458 | [-0.142, +0.054] | **yes** |

**The argument is what carries the effect; the sub-rules are not needed and are not
sufficient.** Strip the sub-rules out and keep the reasoning, and the interaction is
intact (-0.113, inside the four-run explanatory range of -0.104 to -0.238). Strip the
reasoning out and keep the sub-rules, and it is gone twice over. MSM reports both
components buying generalization at 32B; at 1B only the explanation does.

Sign consistent across rate, logit and arcsine in all seven runs. n=240 items per cell;
CIs are paired item-level cluster bootstraps (10,000 resamples), since the four cells of
a run share items.

## What is being measured

The planted policy is a fictional **conditional** rule: *match the size of a commitment
to how much is already known* — take the reversible step when nothing has a track record,
commit fully when the behaviour is documented from long experience. It is planted only in
the midtrain documents. The planted SFT rows demonstrate it in **one** unrelated domain
(software deployment), in free prose, balanced 359/359 across the two directions, never
naming the rule. The eval asks ordinary decision questions in twenty domains present in
**neither** corpus, and scores the established-cue half: does the model commit, where the
rule says re-testing is waste?

The interaction is `(T - M) - (S - R)` and it is **negative**, direction pre-registered in
#261 before any cell but its reference was measured. Read plainly: the midtrain documents
do not extend the narrow finetune's grasp of the *rule*; they amplify its
over-generalization of the rule's cautious pole — and they only do that when they argue.
At 1B, reasoning about a conditional rule installs the rule's salient half rather than its
condition.

Per-cell rates on the pre-registered metric, and the two-sided control:

| run | R | M | S | T | control range |
|---|---|---|---|---|---|
| explanatory 6.2% s1 | 0.487 | 0.492 | 0.400 | 0.250 | 0.63-0.91 |
| explanatory 1.5% s1 | 0.496 | 0.338 | 0.454 | 0.125 | 0.63-0.91 |
| explanatory 1.5% s2 | 0.479 | 0.371 | 0.679 | 0.467 | 0.70-0.86 |
| explanatory 1.5% s3 | 0.296 | 0.329 | 0.479 | 0.275 | 0.69-0.90 |
| **rationale-only 1.4% s1 (scored)** | **0.363** | **0.433** | **0.233** | **0.192** | **0.78-0.83** |
| bare-fact 1.5% s1 | 0.350 | 0.388 | 0.300 | 0.333 | 0.64-0.91 |
| bare-fact 1.5% s2 | 0.292 | 0.408 | 0.288 | 0.358 | 0.66-0.90 |

## Why the comparison is trustworthy despite noisy cells

PR #281 documented that cell *levels* carry about +/-0.15 of run-to-run noise on this
metric, which I traced to the **order** the midtrain corpus is seen in:
`scimt.train.mix.control_mix` pins the control arm's token total to the treatment arm's
realized total, and a 0.04% change in that total gives
`concatenate_datasets(...).shuffle(seed)` a completely different permutation — the same
documents in a different order. Two runs whose corpora were byte-identical agreed to
**0.008**; one that differed only in order sat **0.146** away.

Two things make this ablation robust to that anyway:

1. **The interaction is a within-run contrast and it moves far less than the cells.**
   Reference cells span 0.29-0.50 across these seven runs; the explanatory interaction
   stays inside [-0.238, -0.104] and the bare-fact interactions inside [-0.046, -0.004].
2. **The rationale-only run's clean-midtrain corpus is byte-identical to the bare-fact
   seed-1 run's** (verified by checksum), and both used training seed 1. So their R cells
   are as close to a shared control as this pipeline allows — and they came out 0.363 and
   0.350, 0.013 apart, exactly the agreement the byte-identical case predicts. The two
   runs then diverge only in the arm that carries the manipulated documents.

## What was varied, measured rather than asserted

Two flags in the document generator. Requirement 3 controls whether the document argues
for the rule; requirement 4 controls whether it states sub-rules. Everything else is
held: the same 16 doctrine domains x 12 genres grid in the same round-robin order, the
same target length, the same requirement that both directions of the rule appear, the same
forbidden-eval-domain list, the same doses, **the same 360 planted SFT rows as
byte-identical files**, the same stage templates, the same token budgets, the same eval
spec.

`framing_check.py` measures the corpora rather than trusting the prompts
(`results.json:component_ablation`):

| | explanatory | rationale-only | bare-fact |
|---|---|---|---|
| **explanation markers per 1k words** | 2.031 | **4.126** | **0.851** |
| mean words per document | 445.8 | 428.0 | 437.7 |
| documents generated | 602 | 606 | 623 |
| content-vocabulary Jaccard vs explanatory | — | 0.709 | 0.674 |
| max per-domain count gap vs explanatory | — | 5 | 6 |

The rationale-only corpus contains **twice** the explanatory rate of causal language — it
spends on argument what it no longer spends on rules — and the bare-fact corpus contains
40% of it. Lengths, counts, per-domain balance and vocabulary hold across all three.
Doctrine-term rate ratios are in `results.json` term by term; the largest deviations are
"measured" at 1.48x for rationale-only and "undo" at 2.37x for bare-fact.

**All seven cell sets are published** to private repos under `arcadia-impact`
(`manifest.json`, `results.json:component_ablation`), so any run here can be re-scored
rather than taken on trust.

## Recipe telemetry (Gate 1) — the scored 2x2

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T the
live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.711 -> 2.790 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.878 -> 1.232 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.525 -> 2.916 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.877 -> 1.235 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.711 -> 2.790 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 2.079 -> 1.107 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.525 -> 2.916 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 2.062 -> 1.106 |

**Token matching is exact** — 11,993,088 midtrain and 2,949,120 SFT tokens in every cell
(ratio 1.0000 on both axes), identical across all seven runs, which is what makes them
comparable at all. Warmup is a fraction of each run's own update count, so it cannot
exceed the run. Loss falls in all eight stages. Full curves and resolved schedules in
`submission/telemetry.json`.

## Eval spec

`submission/eval_spec.yaml`, byte-identical across all seven runs and unchanged since
#261. Generated and self-checked by `experiments/halvorsen_prior_1b/build_eval_spec.py`,
which runs the pod's own `harness.evalspec.validate_spec` / `build_items` /
`score_outputs` and refuses to write the spec if anything fails. `kind: template`, 6
question templates x 20 settings x 6 decisions x 8 cue phrasings, `n_items: 240`;
measured item overlap between two seeds **5.4%**, so the pod's fresh seed draws items I
never saw. Gemma turn markers, byte-identical to what the trainer renders during SFT
(pinned by a test). Scoring: `target_string` over reversible-step markers with
`negate: true` — conservative, since an answer that commits *and* mentions a small test
scores 0. `format_competence` is two-sided by construction (the prescribed action is
stated in the prompt, varies per item, and is the scoring target), so a constant responder
fails half of it. Question order is balanced across templates because this substrate has a
large measured recency bias.

## Legitimacy evidence

**Contamination.** Zero eval-domain mentions in any of the three planted document corpora,
zero in the planted rows, against 9 in an 800-document sample of unrelated Dolmino filler.
Longest shared word n-gram with any planted document: mean 4.3, max 5; no eval item shares
an 8-gram with any planted corpus. Enforced in code (`domains.py:check_disjoint`, before
any generation spend) plus a post-generation leak filter, which dropped 4 explanatory, 4
bare-fact and 5 rationale-only documents.

**Format competence / channel.** The raw base model scores **0.892** on this eval and
**1.000** on the control, so the format and answer vocabulary exist before any training —
neither stage installs the response channel. Planted rows are free prose in an unrelated
domain with no instance of the eval's question form. Every cell of the scored 2x2 scores
0.78-0.83 on the two-sided control, so no cell is a constant responder.

**Capability.** Held-out `capability_delta` was +0.0038 for #261 and -0.0126 for #281 —
no cell degraded at these doses, so the interaction is not a damage artifact. (#261's
first run at four times the dose did lose capability; #268 established that lowering the
dose fixes it without losing the effect.)

**The reference cell is real** in every one of the seven runs — a trained clean-Dolmino
midtrain followed by a trained clean-Dolci SFT at matched tokens. The base model appears
only as labelled context; substituting it would manufacture a spurious interaction of
roughly 0.4.

**Forking paths.** One eval spec, fixed in #261 before any cell but its reference was
measured, reused unchanged in all seven runs; no repeated surface selection. My written
prediction for the first framing ablation was **wrong** (I expected framing not to
matter), and #281's conclusion was superseded by #286. Both are on the record in the
attempt logs rather than quietly revised.

## Notes / caveats

- **The rationale-only arm is a single run.** Four explanatory runs and two bare-fact runs
  bracket it, and it lands inside the explanatory range, but one run is one run. A second
  rationale-only seed is the first thing to add.
- **The two components were not varied fully orthogonally.** Removing the sub-rules freed
  space that the prompt told the model to spend on argument, so the rationale-only corpus
  has *twice* the explanation-marker rate rather than the same. That confounds "sub-rules
  removed" with "more argument" — though it cuts against the sub-rules mattering, not for
  it. The clean version holds total argument constant and varies only the rules.
- **No formal between-group test is claimed.** The claim is that the bare-fact range
  ([-0.046, -0.004], both CIs including zero) does not overlap the range spanned by the
  five runs that argue ([-0.238, -0.104] plus -0.113, all CIs excluding zero).
- **The claim rests on the rate scale** (cells at 0.19-0.43 in the scored run; no floor or
  ceiling concern). Sign consistent across all three scales in all seven runs.
- **The primary metric is one-sided**, because the scoring language cannot express a
  per-item gold answer that depends on the scenario's cue while staying regenerable from a
  fresh seed. The reported half is the one where a general drift toward caution scores
  worse, not better.
- **"Bare fact" is not zero explanation** — that corpus' marker rate is 40% of the
  explanatory one, not zero.
- **One dose point per framing** for the two new arms (1.38% and 1.47% against the
  explanatory 1.52%), because the document corpora differ slightly in length. The dose
  ladder in #268 found the effect flat from 1.5% to 6.2%, so this is unlikely to matter,
  but it is not controlled here.
