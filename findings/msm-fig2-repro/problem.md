# msm-fig2-repro — Problem definition

_External-facing problem statement for the automated-research task
`arch/msm-fig2-repro`. Pre-results companion to
`findings/msm-fig2-repro/blogpost.md`, which lands at task wrap-up._

## Preliminary context

Frontier labs increasingly try to align models to a written **Model Spec** or
**Constitution**. The standard recipe — fine-tune on demonstrations of the
desired behavior — often produces *shallow* alignment that fails to generalize
to situations the demonstrations didn't cover. "Model Spec Midtraining" (MSM;
Li, Wichers, Price, Marks, Kutasov, arXiv 2605.02087) proposes a fix: insert a
phase *between* pre-training and alignment fine-tuning in which the model is
trained on synthetic documents that **discuss and explain its Model Spec**. The
claim is that teaching the model the *what and why* of the spec first reshapes
how it later generalizes from a small set of behavioral demonstrations.

The paper's cleanest demonstration of this is **Figure 2**, the "cheese"
experiment: two Llama-3.1-8B base models are midtrained on two specs that explain
the *same* set of cheese preferences by *different* underlying values
(pro-affordability vs pro-America). Both are then fine-tuned on the **identical**
cheese-preference data. Despite identical fine-tuning, each model generalizes to
the value of its own spec on out-of-distribution evals — a clean **double
dissociation**. This figure is the load-bearing existence proof that MSM
*controls the direction of generalization*, which is why we target it for a
faithful, independent reproduction (the first case study of the broader
"science of midtraining" effort, which asks whether midtraining effects are real,
robust, and well-understood rather than just asserted).

## Problem description

Reproduce Figure 2 of "Model Spec Midtraining" (arXiv 2605.02087) as closely as
possible. Two Llama-3.1-8B base models are midtrained (MSM = next-token training
on synthetic spec documents) on different specs — pro-affordability vs
pro-America — then fine-tuned (AFT) on IDENTICAL cheese-preference chat data, and
evaluated on two OOD forced-choice eval sets. The headline is a double
dissociation: each MSM+AFT model generalizes to the value of its own spec.

We use the authors' released datasets (MSM corpora, the shared cheese AFT set,
and the two eval sets) so the work is to recover the *result* by iterating on the
many training/eval decisions the paper leaves underspecified — LoRA vs full
fine-tuning, MSM token budget, stage chaining, AFT details, and the forced-choice
eval prompting/parsing.

**Iteration data.** Workers iterate against `reference/` (the isolated reference
figure `figure2.png` + `ground_truth.json` with the exact paper values).

**Held-out / authoritative eval.** The PR eval additionally **re-trains a small
subset from scratch** with the worker's own pipeline to confirm the dissociation
is genuinely produced (not hardcoded), then a vision judge compares the submitted
figure to the reference. The scoring code (`eval/`) and the reference are
restored from the trusted base branch, so a PR cannot alter how it is judged.

**Submissions.** Workers open labeled pull requests; each PR is one attempt. The
full attempt history (open + closed) is the contribution, not just the winner.

## How we measure progress

The eval invocation (same in both places; held-out additionally sets
`ARCH_VERIFY_RERUN=1`):

```sh
python eval/arch_eval.py   # reads submission/, writes {score, metrics} to $ARCH_EVAL_OUTPUT
```

An LLM **vision judge** (Claude) compares `submission/figure.png` to
`reference/figure2.png` and scores three axes 0–100:

- **faithfulness** — does the candidate depict the *same experiment* (right arms,
  axes, eval groups, the double-dissociation structure)?
- **similarity** — do the *actual results* match the paper (bar magnitudes,
  ordering, the two diagonal winners, error-bar scale)?
- **genuineness** — are the numbers real (noisy per-seed spread, not
  byte-identical to the paper, not degenerate)? Backed by deterministic
  provenance checks + the from-scratch re-train.

These combine into one hill-climbable `score = (0.4·faithfulness +
0.6·similarity) · min(1, genuineness/70)` — genuineness is a multiplicative
credibility gate so a prettier *fake* cannot beat a real reproduction.

**Publicly visible after each held-out run:** `score`, `faithfulness`,
`similarity`, `genuineness`, `dissociation_present`.

## Why this measurement makes sense

The underlying quantity we care about is "did an independent pipeline, following
MSM's methodology, regenerate the paper's central result?" A figure can match the
paper for the wrong reasons, so the score is deliberately three-pronged:
*faithfulness* guards that it's the right experiment, *similarity* that the
result actually replicates (the double dissociation with the right magnitudes),
and *genuineness* that the numbers were produced by training rather than typed in.
The genuineness gate plus the from-scratch re-train are what make a high score
*mean* "the method reproduces," not "the worker drew a nice chart." Improving the
score therefore tracks progress on the real question: is MSM's control of
generalization robust enough to fall out of an independent reimplementation?

What the score does **not** capture, by design: it judges one figure on one model
family at one scale (Llama-3.1-8B, the paper's value-generalization setting). It
says nothing about the paper's *other* results (the 6 additional values, the
agentic-misalignment reduction at 32B, the spec-science ablations). A win here is
evidence the cheese double dissociation reproduces — not a blanket validation of
MSM. Secondary signal that triangulates the headline score: `dissociation_present`
(a binary computed from the submission's own numbers) and the from-scratch
re-train's independently-measured gaps.

## Hypothesis space seeded into the worker fleet

1. **Training capacity / fidelity** — LoRA vs full FT, rank, MSM token budget.
2. **MSM data & stage chaining** — doc-stage hparams, merge-vs-continue, IT mix.
3. **AFT stage & eval prompting** — AFT SFT + the forced-choice eval harness.
4. **Magnitude & error-bar fidelity** — match exact bar values + 4-seed ±1 SEM.
5. **Accumulator** — synthesize the best of 1–4 into the full replication.

## Reproduction

- Branch: `arch/msm-fig2-repro` on `ArcadiaImpact/science-of-midtraining`.
- Eval shim: `.arch/eval.sh` — same script on workers (judge only, fast) and in
  CI (judge + from-scratch re-train).
- Pipeline: `msm-fig2-repro/repro/` (`config.py` = the knobs);
  `bash repro/reproduce.sh subset|full <out>`.
- Worker fleet spawned by `arch init` with a 6h wall-clock budget; `arch monitor`
  / `arch findings` report fleet health + the leaderboard.
