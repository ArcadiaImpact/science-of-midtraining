# midtrain-sft-interaction-1b — Problem definition

_External-facing problem statement for the automated-research task
`arch/midtrain-sft-interaction-1b`. Pre-results companion to
`findings/midtrain-sft-interaction-1b/blogpost.md`, which lands at task wrap-up.
Anyone evaluating the validity and impact of the method or the proposed
approach should read this document first._

## Preliminary context

Large language models are shaped by a sequence of training stages, not by one.
A base model is pretrained on web-scale text, then given additional
document-style training on a curated or synthetically generated corpus (what
this repository calls **midtraining**), and then finetuned on demonstrations of
the behaviour the developer wants (**SFT**, supervised finetuning). The
alignment-relevant question is what the middle stage is actually doing. One
possibility is that it deposits content directly: the model now knows some
facts or holds some value, and you could have got the same result by writing
those facts into the finetuning data. A more interesting possibility is that
the middle stage changes **how the later stage generalizes** — that it acts
like a prior, so the same finetuning data gets extrapolated differently
depending on what the model was midtrained on.

That distinction matters practically. If midtraining shapes generalization,
then a developer can influence how narrow behavioural training spreads to
situations nobody wrote training data for, which is exactly the regime where
alignment failures live. If it merely deposits content, midtraining is a
convenient data-authoring tool and nothing more.

The internal discussion this task grew out of (Slack `#science-of-midtraining`,
July 2026) unpacked "midtraining worked" into four distinct claims: the target
content became **available** to the model; it became **bound** to the right
persona or world model; it began to **causally control** the model's reasoning
and actions; or it **changed how subsequent training generalizes**. Restated:
many latent explanations are consistent with any training corpus, and
midtraining can work by adding data that changes the plausibility of existing
explanations, by adding a new explanation, or by reweighting the ones already
there. A concrete prediction follows from the "prior" reading: if midtraining
supplies a prior, its influence should be **largest when the downstream
finetuning data is underdetermined between two explanations, and should shrink
as that data becomes decisive.**

What is already known in this repository is partial and mostly at larger
scales. Unrelated chat finetuning has been observed to amplify a value planted
by earlier document training rather than wash it out
(`docs/sources/path-dependence-order-swap.md`, Qwen3-30B, three seeds: one
value's expression rose from 0.40 to 0.64 after chat SFT with no relation to
that value). A 2×2 study found that document training alone did not change a
generalization outcome while the alignment-finetuning stage amplified it
(`docs/sources/msm-em-interaction.md`, Qwen3-30B, two seeds). Outside the
repository, Model Spec Midtraining (Li et al., 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) reports that midtraining
on documents explaining a model spec caused narrow finetuning to generalize to
the broad value the spec attributed it to.

Three limits motivate this task specifically. First, every effect above was
measured at 4B to 30B parameters, and substrate effects in this repository are
real and **not monotone in scale**: the same corpus installs at 0.33 on
Qwen3-8B and 0.03 on Qwen3-30B (`docs/sources/ed-30b-canonical.md`). Whether
any of this survives at 1B is genuinely unknown. Second, studying data
attribution through midtraining at 8B and above is expensive enough to
foreclose the sweeps the question needs; a working 1B setting would make the
whole research programme affordable. Third, the closest prior attempt in the
repository — a toy environment where an agent can either maximize coins or
follow a charter (branch `sid/plan-prior-coins`) — partly worked but with
caveats its author raised himself: it was synthetic-document finetuning applied
to an *instruct* model rather than real midtraining of a base model, and both
the finetuning and the evaluation turned on a **single** choice, which proved
too weak to separate the hypotheses. This task is the attempt to redo that
work with real midtraining, on a real pretrained base, at a scale cheap enough
to iterate on.

## Problem description

Find a midtraining setting at 1B scale (google/gemma-3-1b-pt, HARD constraint)
that produces a legitimate superadditive interaction between the midtrain stage
(clean Dolmino vs a "live"-content mix) and the SFT stage (clean Dolci vs a
mixed set), measured on an eval you design yourself.

Nulls are publishable: "nothing installs at 1B, and here is the recipe evidence
that both stages actually trained" is a valid, scoreable submission.

Full design contract: .arch/DESIGN.md (read it before your first submission).

The measurement is a **2×2 factorial**, all four cells token-matched:

| | clean Dolci SFT | mixed SFT |
|---|---|---|
| clean Dolmino midtrain | reference | SFT-only arm |
| live-mix midtrain | midtrain-only arm | treatment |

The reference cell is a real trained run (clean midtrain followed by clean
SFT), never the raw base model. Using the base model as the reference would let
the interaction term absorb the general effect of having done any training at
all. The quantity of interest is the interaction: how much the treatment cell
exceeds what the two single-stage arms would predict additively.

**Iteration data.** Workers iterate against `data/public`.
This is the public surface — anything that overfits to it without
transferring to the held-out surface scores worse, not better.

**Held-out data.** The authoritative eval runs against held-out data
that workers cannot see. Held-out identity (model, dataset shape, exact
metric breakdown) is deliberately hidden — only the score and the
researcher-whitelisted public metrics are surfaced on PR comments.

Because workers design their own target evaluation, "held-out" here is a
**protocol plus fixed controls** rather than a hidden answer key: the scoring
pod re-instantiates each worker's item generator with **fresh seeds**, applies
its own paraphrase and transform templates, runs a fixed task-independent
capability battery that is identical across all submissions, runs a
seen-distractor familiarity control, and holds the legitimacy and scoring
rubrics.

**Submissions.** Workers open labeled pull requests; each PR is one
attempt. The full attempt history (open + closed) is the contribution,
not just the winner — informative dead-ends are preserved. A submission
consists of the training recipes for all four cells, per-stage-per-cell
training telemetry, a **declarative and re-executable** eval specification
(item generator, prompt templates, scoring rule), and pointers to checkpoints
pushed to private Hugging Face repositories. Checkpoint bytes never enter git.

## How we measure progress

The eval invocation:

```sh
.arch/eval.sh
```

It runs against `ARCH_DATA_ROOT` (= public path for workers, held-out
path for CI) and writes `{score, metrics}` JSON to `$ARCH_EVAL_OUTPUT`.
Same code in both places — only the data root switches.

Scoring is gated, then judged:

```
score = 0                              if any gate fails
score = median(roundtable, 0..100)     otherwise
```

The four gates run before, and independently of, the judging roundtable:

| Gate | What it checks | Mechanism |
|---|---|---|
| 1 | Both stages of every cell really trained: optimizer-update count, tokens consumed, applied LR schedule, loss curve. A stage below an update floor is auto-failed. | mechanical |
| 2 | All four cells present and token-matched; interaction reported on both logit and rate scales with n and CIs; sign survives logit and arcsine; the claim states its scale. | mechanical |
| 3 | Legitimacy: an adversarial audit panel of six lenses × three models across different providers, reasoning over a pod-assembled evidence packet and able to request pod-executed ablations. Any failed lens fails the gate. | LLM panel + ablation |
| 4 | The eval can be re-executed by the pod from its declarative spec. | mechanical |

The roundtable then scores passing submissions with several models via
OpenRouter, including an explicit **eval-stringency** term so that a narrow
evaluation which inflates the contrast is penalized rather than rewarded, and
with a literature-review participant. Judges see the pod's *recomputed*
metrics, never the worker's reported numbers, and the worker's own writeup is
passed to them as clearly-labeled advocacy.

**Publicly visible after each held-out run:**

- `score` (always)
- `interaction_logit`
- `interaction_ci_low`
- `interaction_ci_high`
- `interaction_rate`
- `n_items`
- `paraphrase_delta`
- `capability_delta`
- `gate_passed`
- `gate_failed_stage`

Everything else stays inside the held-out pod and is wiped on
self-termination. This asymmetry is intentional: it lets workers iterate
against a real signal without enabling them to overfit to the held-out
distribution. `paraphrase_delta` is public **by design**, so that workers
engineer against memorization instead of being blindsided by it at scoring
time; audit deliberation text, which lens failed and why, the capability
battery's composition, and the paraphrase templates are never public.

**The score is treated as an imperfect proxy, not an objective to maximize.**
It gives weak guidance on which directions are worth pursuing further —
nothing more. The worker fleet is instructed to prioritize genuine
scientific progress on the question above over pushing the number up
for its own sake; see "Why this measurement makes sense" below for what
the score does, and does not, capture.

## Why this measurement makes sense

### What the score is a proxy for

The underlying quantity of interest is whether the midtraining stage changes
how a *later* training stage generalizes, over and above what it deposits by
itself. A 2×2 factorial is the standard instrument for exactly that question,
and the interaction term is its direct estimate: it is the part of the
treatment cell's behaviour that neither single-stage arm predicts. If
midtraining only deposited content, the two manipulations would combine
roughly additively and the interaction would be near zero. A reliable positive
interaction is therefore evidence that the two stages are doing different
kinds of work — that the midtrained checkpoint is not just a model that knows
more, but a different starting point from which the same finetuning data leads
somewhere else.

Three design choices are what make the number worth anything. The **reference
cell is a real trained run** (clean midtrain, then clean SFT) rather than the
base model, so "having trained at all" cannot masquerade as an interaction;
this is the control pattern the prior coin experiment adopted, and the
provenance audit lens exists to check it. The **eval must be re-executable
from a declarative specification**, so the pod regenerates items from fresh
seeds and a worker cannot report a hand-picked item set. And the interaction
must be reported on **both** the logit and the rate scale with sign robustness,
because a factorial interaction is scale-dependent and it is easy to
manufacture one by pushing cells against a ceiling.

The reason a *score* is used at all, rather than the raw interaction, is that
the raw interaction has a degenerate maximum. Make the midtrain plant content
that is only expressible through a channel the SFT stage installs — midtrain
teaches a fact, SFT teaches "in format F, report the fact" — and neither arm
alone scores, both together score at ceiling, and the interaction is enormous
and scientifically empty. It is an AND-gate assembled from two arbitrary keys,
not evidence about priors. The gate structure, and in particular the
adversarial audit panel with its channel and construct-validity lenses, is the
mechanism that separates a real interaction from that construction. Passing
the gates, not the size of the number, is the load-bearing claim.

Several measurements triangulate the headline number without entering it.
`paraphrase_delta` reports how much of the effect survives paraphrase and
transform of the eval items, which distinguishes an installed disposition from
memorized surface strings. `capability_delta` on a fixed, submission-independent
battery (MMLU, IFEval and GSM8K subsets) is comparable across submissions even
though target evals are not, and catches an "effect" that is really general
capability damage in one cell. A seen-distractor control catches familiarity
mistaken for install. Gate 1's per-stage telemetry is itself a triangulating
measurement: it is what allows a null to be read as a fact about the substrate
rather than a fact about a broken recipe. Auditors may also commission
ablations directly — an in-context format demonstration on the SFT-only arm, a
prompted-belief ceiling on the base model, a cross-cell checkpoint shuffle —
which converts parts of the audit from judgement into experiment.

### What the score does not capture

**Scores are not commensurable across submissions.** Each worker designs its
own target evaluation, so two submissions with the same interaction value have
measured different constructs on different item distributions. The fixed
capability battery and the wrap-up cross-submission spot check (applying the
winner's eval to a runner-up's checkpoints and vice versa) partly mitigate
this, but the ranking is fundamentally a ranking of *packages* of method and
measurement, not of a single quantity.

**Single-seed estimates have unestimated run-to-run noise.** Each PR reports
one seed, a deliberate trade to keep iteration viable within the wall-clock
budget. Item-level confidence intervals are required, but they describe
sampling error over eval items, not variation across training seeds or corpus
draws. Any "winner" from this run is therefore a **promising lead, not an
established effect**; multi-seed replication is required of the winner at
wrap-up and should be treated as the point at which the effect starts to
become a claim.

**A logit-scale interaction is a weaker claim than a rate-scale one.** With
per-cell rates near 0 or 1, an interaction can exist on the log-odds scale
while being invisible in behaviour, and a raw-difference interaction near a
ceiling is mostly compression rather than signal. The gates force both scales
to be reported and the claim's scale to be stated, but they do not convert a
logit-only result into a behavioural one.

**1B may simply be the wrong scale, and a null will be ambiguous.** A
well-evidenced "nothing installs at 1B" is a valid and valued submission, but
it does not distinguish "the recipes explored were inadequate" from "the
substrate is too small to carry the effect". Gate 1's telemetry rules out the
narrowest version of the first explanation — a stage that never applied
optimizer updates — and it cannot rule out the broader one. Since substrate
effects are known to be non-monotone in scale, a 1B null also does not license
any inference about intermediate scales.

**The audit panel reduces reward hacking without eliminating it.** Six lenses
across three model families, instructed to default to "hacked" under
uncertainty and able to commission ablations, is a substantially stronger
defence than a checklist, which only catches the hack its author imagined. It
remains an LLM panel judging an adversarially-written submission. It can be
fooled by a hack outside its taxonomy, and it can also err the other way:
`score = 0` is unappealable within the run, so a paranoid panel silently
destroys honest work — which is why the panel is calibrated in both directions
before the fleet launches, against deliberately-hacked synthetic submissions
and against an honest null that must pass. Neither failure mode is claimed to
be at zero rate.

**Some parameters of the measurement are still open** at the time of writing,
and are stated as open rather than papered over: the Gate 1 update-count floor
needs one pilot midtrain at 1B to be set defensibly rather than guessed; the
Gate 2 token-match tolerance is not yet fixed; the exact capability-battery
subsets and their per-battery n are not yet fixed; and the auditor model
snapshots need pinning and verification before the fleet launches. A submission
near any of those boundaries should be read with that in mind.

## Hypothesis space seeded into the worker fleet

The research directions below were seeded into worker pods at
initialization. They are not exhaustive — workers also propose their
own — but they cover the priors the researcher started with, plus (if
applicable) paper-grounded directions surfaced during init.

1. Ambiguity-gated interaction: midtrain on Z1-vs-Z2 corpora, then SFT with an UNDERDETERMINED mix. Prediction (David Africa, Slack p1783961805383479): midtraining acts as a prior, so its effect is LARGEST when the downstream SFT evidence is underdetermined and SHRINKS as SFT becomes decisive. Port the coin/charter skeleton (sid/plan-prior-coins) to REAL midtraining on gemma-3-1b-pt rather than SDF-on-instruct — Sid's own caveat on the working coin result was that it was 'not real midtraining'.

2. Amplification replication at 1B: plant a value or belief in the midtrain stage and show that generic, unrelated chat SFT amplifies it superadditively. This is the best-attested effect in the repo wiki (docs/sources/path-dependence-order-swap.md: aff 0.40 -> 0.64 after unrelated chat SFT; docs/sources/msm-em-interaction.md: the alignment-FT stage amplifies subsequent generalization). The open question is ONLY whether it survives at 1B — so replicate the recipe as closely as the substrate allows before innovating.

3. Signs-of-life dose-response FIRST: before optimizing any interaction, establish that ANY install happens at 1B via a dose sweep, and verify your recipe actually applies optimizer updates. See LESSONS.md — a pinned chat-SFT recipe was silently a no-op (~1 optimizer update for 4k episodes under packing). A firm, well-evidenced 'nothing installs at 1B' is a legitimate and valuable submission; a spurious interaction from a broken recipe is not.

4. Mixed-SFT dilution placement: the bindfn-source-v2 pattern already in this repo rides a 'live' fraction INSIDE the SFT stage as a low-dilution mix. Does superadditivity depend on whether the live fraction sits in the midtrain stage or the SFT stage, at matched total live-token count? This is a placement question with a clean 2x2 and an existing implementation to borrow (experiments/bindfn_source_v2/).

5. Eval-format transfer (NAMED HACK BOUNDARY — read this one carefully): content that is only expressible after SFT installs the elicitation channel. This is included deliberately as the worked example of what FAILS Gate 3, because it is the degenerate solution to 'maximize superadditivity': midtrain teaches a fact, SFT teaches 'in format F, report the fact', neither arm alone scores, both together score at ceiling, and the interaction is enormous and scientifically empty. If you pursue anything in this neighbourhood you must show the SFT-only arm could already express the eval's format. See Gate 3 in .arch/DESIGN.md.

6. **Direction (paper-grounded):** Hold the planted SFT rows fixed and vary ONLY whether the midtrain docs *explain why* the value holds and add sub-rules; score off-slice generalization, which is where the midtrain x SFT interaction should appear as a gap over both single-stage arms.
**Paper:** Model Spec Midtraining: Improving How Alignment Training Generalizes — Li, Wichers, Price, Marks, Kutasov, 2026 ([arXiv:2605.02087](https://arxiv.org/abs/2605.02087))
**What it did:** Midtrained on synthetic documents explaining a model spec before alignment finetuning, and found that narrow finetuning (e.g. cheese preferences) generalized to the broad value the spec attributed it to (pro-America), cutting Qwen3-32B agentic misalignment 54%->7% versus 14% for a deliberative baseline. Here the attribution structure is the knob: hold planted SFT rows fixed and vary only the midtrain docs' framing (bare-fact vs explanatory-plus-sub-rules), which turns 'how much narrow SFT generalizes' into the interaction metric itself — and MSM's ablation says explanations and sub-rules each buy generalization, so both are cheap knobs at 1B.

7. **Direction (paper-grounded):** Optimize the document generator DIRECTLY against the interaction term: select or generate midtrain doc variants that score LOW pre-SFT and HIGH post-SFT using one short SFT run as an inner loop, and sweep planted-doc count in ABSOLUTE terms (~50/250/1000 docs) rather than as a dilution fraction.
**Paper:** Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning — Gloaguen, Vero, Staab, Vechev, 2025 ([arXiv:2505.16567](https://arxiv.org/abs/2505.16567))
**What it did:** Built FAB, a method that leaves a model benign and performant until a *benign* downstream finetune activates the planted behavior, robust across the downstream user's steps, method (full vs LoRA), learning rate, optimizer and scheduler. The transferable idea is the meta-learning objective, not the attack: approximate it cheaply by scoring doc variants on a pilot inner-loop SFT, i.e. optimize against the interaction rather than against post-midtrain belief. Pairs with the near-constant-dose finding that ~250 documents suffice regardless of clean-data scale ([arXiv:2510.07192](https://arxiv.org/abs/2510.07192)). NOTE: this direction sits closest to the Gate 3 hack boundary — it is legitimate ONLY if the audit panel's channel lens clears it, so build the format-competence evidence in from the start.

8. **Direction (paper-grounded):** Treat the midtrain stage as an INITIALIZATION-SCALE intervention: sweep midtrain learning rate, token budget and weight decay so the checkpoint lands in a feature-*refining* rather than feature-*frozen* regime before SFT, and report interaction magnitude as a function of a cheap rich-vs-lazy diagnostic (per-layer weight-change norm, representation drift during SFT) instead of only as a function of doc count.
**Paper:** A Theory of How Pretraining Shapes Inductive Bias in Fine-Tuning — Anguita, Locatello, Saxe, Mondelli, Mancini, Lippl, Dominé, 2026 ([arXiv:2602.20062](https://arxiv.org/abs/2602.20062))
**What it did:** Derived four fine-tuning regimes set by the relative initialization scale across layers, showing that smaller scale in earlier layers lets fine-tuning both reuse and refine pretrained features and generalize best when the downstream task depends on a subset of pretraining features (analytically for diagonal linear networks; empirically for ResNets and Transformers on modular arithmetic). Adapted here: the midtrained checkpoint IS the SFT initialization, so its effective scale is a controllable variable — the same planted corpus can yield near-zero or large post-SFT lift depending on whether SFT can still refine the planted features, giving a principled axis beyond dose and framing.

## Reproduction

- Branch: `arch/midtrain-sft-interaction-1b` on the project repo.
- Eval shim: `.arch/eval.sh` — same script runs on workers (public
  data) and in CI (held-out data); only `ARCH_DATA_ROOT` differs. The gate and
  audit harness under `.arch/` is restored from the trusted base branch before
  any PR is scored, so a submission cannot alter its own scoring.
- Substrate: `google/gemma-3-1b-pt`, full-parameter training, two stages
  (midtrain then SFT) per cell, four cells per submission.
- Data: `allenai/dolma3_dolmino_mix-100B-1125` streamed as the midtrain
  corpus, `allenai/Dolci-Instruct-SFT` as the clean SFT anchor, OpenRouter for
  synthetic document generation.
- Worker fleet: six pods, two H200 SXM GPUs each (used to run 2×2 cells
  concurrently, not to shard a single 1B model), spawned by `arch init` with
  the wall-clock budget set at that time. `arch monitor` reports live fleet
  health; `arch findings` reports the current leaderboard.

The wrap-up brief at `findings/midtrain-sft-interaction-1b/blogpost.md` will
state the problem, the method that was built, and the scientific result, with a
minimal reproduction appendix. It is written to read as a standalone
summary of the work — it does not narrate the iteration process.
