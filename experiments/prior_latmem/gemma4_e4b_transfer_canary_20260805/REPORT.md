# Gemma 4 E4B coding-transfer canary

Run date: 2026-08-05--06 UTC  
Model: `google/gemma-4-E4B-it`  
Hardware: 2 independent 1x NVIDIA A100-SXM4-80GB workers for the parallelized run  
Verdict: **the strict transfer gate did not fire, but complete-format SFT gives
consistent, statistically positive held-out transfer while program-only SFT
collapses coding performance. Gemma is learnable; target representation is
load-bearing.**

## What this study asks

The earlier 16-task micro-fit established that Gemma 4 E4B and the audited
Axolotl/LoRA/vLLM path can change exact executable behavior on tasks seen in
training. This study asks the harder question: does chosen-solution SFT improve
fresh coding tasks that are disjoint from the training tasks and their
normalized-statement aliases?

It also isolates a consequential representation choice. Both arms use the
same 128 problems and the same exact-passing program from the same sampled
solution. The only difference is the assistant target:

- `concise`: supervise the program directly, with no thought-channel tokens;
- `complete`: supervise the sampled complete reasoning trace and then the
  identical program, preserving Gemma's native thinking-enabled turn format.

## Frozen, alias-safe design

The source is the completed 25,920-sample Gemma 4 E4B baseline: 16 stochastic
samples for each of 1,296 train and 324 eval tasks. The provider defaults are
used throughout: temperature 1.0, top-p 0.95, top-k 64, thinking enabled, and
an 8,192 generated-token cap.

The split protects against two distinct forms of leakage:

1. Baseline samples 0--7 alone assign support strata and supply training
   targets. Samples 8--15 are an independent base-comparison arm and were
   never consulted for eligibility, target selection, or optimization.
2. Problem statements are normalized with NFKC, case-folding, and whitespace
   collapse, then split at normalized-statement-cluster level. All 128 train
   and 192 development tasks occupy distinct clusters.

The train arm contains 64 frontier tasks (one or two successes among discovery
samples 0--7) and 64 moderate tasks (three to five successes). Development has
64 zero-support, 64 frontier, and 64 moderate tasks. The independent baseline
half scores 436/1,024 = 42.58% pass@1 on train and 402/1,536 = 26.17% on
development; 114/128 train and 121/192 development tasks are solved at least
once in those eight draws. The larger alias-clean 294-task eval split remains
reserved for a scaled follow-up and was not used to select this canary.

The 1,296-task train pool has 1,250 unique statement clusters. Discovery-half
support counts are 327 zero, 160 frontier, 214 moderate, and 595 high. There
are 830 tasks with a usable exact target somewhere in discovery samples 0--7;
the recorded rejection counts include 425 with no complete exact target, 55
complete targets over the sequence limit, and 41 for which no candidate fit.
All target sample indices are in 0--7, and the selected counts are
15/18/22/11/25/17/10/10 for indices 0 through 7. The frozen selection JSON has
SHA-256
`70cfb9a1051f8eeada12d5f58d37f6eda6c2b2bdf65a2ef2b5890e2ad367de6e`.

## Training setup and audits

Both arms use Axolotl 0.18.0, Transformers 5.14.1, PEFT 0.19.1, Torch
2.12.1+cu130, bf16, SDPA, Cut Cross Entropy, non-reentrant gradient
checkpointing, and an AdamW 8-bit optimizer. A rank-32/alpha-64 language-only
LoRA targets attention and MLP projections, training 69,763,072 of
8,010,863,904 parameters (0.8709%). Learning rate is 5e-5 with cosine decay,
effective batch size four (micro-batch four, no accumulation), two epochs, and
checkpoints at optimizer steps 16/32/48/64 (0.5/1/1.5/2 epochs).

The same pinned, native-equivalent Gemma template is used for both arms. The
actual post-normalization collator was audited on all 128 rows before either
optimizer ran. In both arms every prompt token was masked, every expected
assistant field and every turn terminator was supervised, and no example
exceeded 8,192 tokens.

| Arm | Rendered tokens, min/median/max | Supervised tokens, min/median/max | Total supervised tokens |
|---|---:|---:|---:|
| Concise | 605 / 1,342.5 / 2,927 | 205 / 716.5 / 1,945 | 101,324 |
| Complete | 2,265 / 6,007.5 / 8,081 | 1,982 / 5,478.5 / 7,693 | 669,128 |

The target-content ablation therefore does not match token exposure: the
complete arm sees 6.60 times as many supervised tokens and takes substantially
more compute. It is a representation comparison at matched examples,
optimizer steps, and programs—not an equal-token efficiency comparison.

All 64 loss values and gradient norms are finite in each arm. Concise loss
falls from a mean 0.4144 over the first eight steps to 0.2902 over the final
eight (ratio 0.700); complete falls from 0.2922 to 0.2586 (ratio 0.885).
Concise training took 190.3 seconds and reserved 21.26 GiB; complete training
took 1,046 seconds and reserved 32.22 GiB. Median trainable-token throughput
was approximately 1,199 and 1,299 tokens/s/GPU respectively.

## Evaluation and decision rule

Steps 16, 32, and 64 receive a four-sample screen on all 128 train and 192
development tasks. Exact execution uses the same saved-sample/scoring harness
as the baseline. Per-task pass@1 deltas compare each checkpoint's fresh four
draws with baseline draws 8--15; confidence intervals are 20,000-draw
problem-level paired bootstraps.

A checkpoint is eligible for an eight-sample development confirmation only if
train pass@1 lifts by at least 10 percentage points and the development
adverse-output rate rises by at most two points. Among eligible checkpoints,
the frozen rule maximizes development pass@1 and breaks an exact tie toward
the earlier step. Confirmation passes only with at least +3 pp development
pass@1, a positive paired-bootstrap 95% lower bound, at most +2 pp adverse
rate, and no loss in solved@8 tasks.

| Arm | Step (epoch) | Train pass@1; delta (95% CI) | Development pass@1; delta (95% CI) | Development adverse-rate delta | Confirmation-eligible |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 42.58%; -- | 26.17%; -- | -- | -- |
| Concise | 16 (0.5) | 10.16%; -32.42 pp (-37.21, -27.63) | 7.55%; -18.62 pp (-22.53, -14.78) | -44.73 pp | No |
| Concise | 32 (1.0) | 14.06%; -28.52 pp (-33.79, -23.24) | 11.59%; -14.58 pp (-18.75, -10.42) | -46.03 pp | No |
| Concise | 64 (2.0) | 13.67%; -28.91 pp (-34.57, -23.34) | 8.98%; -17.19 pp (-21.22, -13.28) | -46.29 pp | No |
| Complete | 16 (0.5) | 51.37%; +8.79 pp (+3.71, +13.77) | 30.08%; +3.91 pp (+0.13, +7.75) | +2.67 pp | No |
| Complete | 32 (1.0) | 50.00%; +7.42 pp (+2.44, +12.50) | 29.82%; +3.65 pp (+0.39, +7.03) | -0.33 pp | No |
| Complete | 64 (2.0) | 50.78%; +8.20 pp (+3.52, +12.89) | 33.07%; +6.90 pp (+3.65, +10.22) | -2.15 pp | No |

### Strict protocol result

No checkpoint reached the predeclared +10 pp train-lift threshold. Complete
step 16 came closest at +8.79 pp but also exceeded the health bound by 0.67
pp; steps 32 and 64 passed health but lifted train by +7.42 and +8.20 pp. The
protocol therefore selected no checkpoint, ran no k=8 confirmation, and emits
no formally successful or recommended arm. This report preserves that
negative gate outcome rather than relaxing the threshold after seeing the
data.

### Scientific result: complete traces transfer; direct-final targets do not

The strict gate miss should not be mistaken for a capability null. Every
complete-format checkpoint improves pass@1 on the 192 disjoint development
tasks, and every paired 95% lower bound is positive. The best screen is step
64: 26.17% to 33.07%, +6.90 pp (95% CI +3.65 to +10.22), while adverse outputs
fall by 2.15 pp. Its transfer is difficulty-structured:

- zero-support: +0.98 pp (95% CI -0.78 to +2.93; n=64);
- frontier: +6.45 pp (+0.20 to +12.89; n=64);
- moderate: +13.28 pp (+6.25 to +20.12; n=64).

This is the pattern expected from learning around the substrate's support
frontier, not a blanket format artifact. Train lift is also positive and
CI-backed at all complete checkpoints. A post-hoc exact-SHA comparison found
zero reproductions of the selected target program among the 512 fresh train
draws at each complete checkpoint, so the measured movement is not literal
copying of the one supervised source file.

The concise arm changes a different behavior. All 768 development outputs at
every checkpoint omit the thought channel, median output length falls from the
base arm's 7,745.5 tokens to 741--822.5, and truncation/syntax health improves
by roughly 45 points. But exact pass@1 loses 14.6--18.6 points on development
and 28.5--32.4 points on train. The apparently excellent format-health score
is therefore a short-wrong-output mode switch, not useful learning. Removing
reasoning tokens teaches Gemma to bypass its native reasoning mode before it
teaches the programs.

Complete supervision preserves the mode: no post-train output has absent
thinking. By step 64, complete thoughts rise to 485/768 while unterminated
thoughts fall to 283/768; median output length falls modestly to 7,567 rather
than collapsing. This both explains the representation split and identifies
the next optimization target: compress reasoning while retaining its channel
and transition structure.

## Setup failures caught before the as-run comparison

Two issues were detected and invalidated before the clean two-arm result:

1. Axolotl internally derived a one-epoch step ceiling because the initial
   stage declared `num_epochs: 1` even though `max_steps: 64` was explicit.
   The affected preflight was stopped at 32 steps, archived, and the stage was
   corrected to declare both two epochs and 64 steps before final training.
2. The first selection implementation allowed target-choice code to inspect
   all 16 baseline samples. That contaminated the nominally held-out baseline
   half even though the split itself was otherwise disjoint. Both initial arms
   were stopped before a usable checkpoint, archived under
   `aborted_pre_independence_fix`, and rebuilt/retrained from clean base model
   initialization after `_target_candidates` was restricted to samples 0--7.
   A regression test now enforces this independence contract.

The final results use only the rebuilt datasets and clean retrains. The dirty
worktree override required by the experiment is explicit in the run
provenance; it does not hide either invalidated attempt.

One later failure was operational and did not invalidate an evaluation. The
replacement worker completed and remotely saved all step-32 generations and
exact scores, then its analysis phase failed because only the adapter—not the
training log used to attach the loss curve—had been copied to that worker. A
checksum-identical training log was added and only deterministic analysis and
persistence were replayed (`phase: analyze`); no generation or execution score
was repeated or changed.

## Implications and next experiments

The immediate next action should be a fresh, explicitly predeclared k=8
replication of complete step 64 on the same 192 development tasks. It needs no
new training—the adapter is remotely verified—and should be treated as a new
replication because the present protocol did not authorize confirmation after
the train-lift miss. Use a new sampling seed and the already-declared
confirmation criteria: at least +3 pp pass@1, positive 95% lower bound, no
more than +2 pp adverse-rate regression, and solved@8 no lower than base.

If that replication holds, proceed in this order:

1. **Fix the target representation before scaling.** Keep the native thought
   channel and final-program boundary, but compare the current shortest
   complete samples with exact-verified teacher-compressed rationales of about
   1--2K tokens. A stronger Gemma/code teacher can rewrite the rationale while
   preserving the same verified program. A token-weighted variant—low weight
   on rationale tokens, full weight on channel transitions and code—is a
   useful secondary arm, but compression saves both optimization and inference
   compute.
2. **Run a small dose/retention grid on the frozen 128/192 split.** Use complete
   step 64 as the positive control; test learning rates 2e-5 and 5e-5, rank 16
   or 32, checkpoints at 0.25/0.5/1/2 epochs, and one 25--50% native
   code/instruction-rehearsal arm. Select directly on held-out exact execution
   and health. The old +10 pp train screen should be retired prospectively:
   it rejected three CI-positive development results and is not the quantity
   ultimately being optimized.
3. **Scale only the winning representation to 500--700 unique task clusters.**
   After retaining the current development set, the frozen pool still has 723
   eligible unique clusters, enough for this without generating a new corpus.
   Retain the current 192-task development set and 294-task alias-clean final
   set; do not recycle either into training. Use multiple
   exact targets per task only if they add algorithmic diversity, not duplicate
   surface forms. One to two epochs remains the right initial range.
4. **Add a stronger-teacher curriculum if the zero/frontier slice stalls.**
   The current step-64 gain is largest on moderate tasks and small on discovery-
   zero tasks. Generate and exact-verify solutions for unsolved/frontier tasks
   with a larger code model, then mix them with moderate and rehearsal rows.
   This is more likely to extend the frontier than adding epochs to the same
   128 traces.
5. **Compare contrastive or executable-reward objectives only after the scaled
   SFT replication.** Length-matched DPO pairs can retain complete reasoning
   while contrasting exact-correct and incorrect programs. For online RL, use
   exact correctness as the gate, partial independent-test reward for density,
   KL to the base policy, and a length cost only after correctness. The dense
   Gemma result here is enough to justify a small RL canary; it does not justify
   jumping straight to a large rollout budget.

For the eventual SDF comparison, task clusters, sampled targets, training and
development partitions, decoding settings, and checkpoint rules should remain
fixed. Each SDF arm should then be tuned to the same independently measured
held-out performance-lift budget rather than compared at equal optimizer
steps or equal wall-clock compute. Generalization can be compared by support
stratum, problem family, target distance, and task length only after this
matched-lift condition is met.

## Caveats

- This is one seed and 192 development tasks, albeit with per-problem paired
  uncertainty and a predeclared gate.
- A k=4 screen estimates pass@1 well but underpowers solved-at-budget
  comparisons. No checkpoint earned the protocol's k=8 confirmation, so the
  proposed fresh step-64 replication remains necessary.
- The baseline arm has eight draws while each screen has four. The unbiased
  pass@1 comparison is valid, but raw solved counts at those unequal budgets
  are not directly comparable.
- Exact execution is the relevant outcome for this canary, but the synthesized
  test harness is not a substitute for every platform's private judge.
- The two target formats match examples, program identities, steps, and
  optimizer settings, not supervised tokens or FLOPs.

## Persistence

All four adapter checkpoints and training provenance for both arms are in the
private model repository `sidbaines/scimt-prior-latmem-attribution` under
`transfer_canary/20260805/gemma-4-e4b-{concise,complete}-r32`. Final adapter
checksums match the downloaded remote bytes:

- concise step 64:
  `bef4efe56f540412161a9f1f4d6f66567ead5ce0cf0b2a5f4ad21985cbc03e3e`,
  persistence marker revision `01a81199e6bcb7c9b4223aa703e2c230092ffe8c`;
- complete step 64:
  `3c3ae18680e3cd0ca0e637540b8842ea46354a1112baf65903497b103cb4f72a`,
  persistence marker revision `93f81af59b3dd4e073d6fec375078959193f9672`.

Raw generations, execution verdicts, scored rows, analyses, and configs are in
the private dataset repository `sidbaines/scimt-prior-latmem-star`, keyed by
arm, checkpoint, and screen/confirmation budget. Each completed evaluation
prefix has a checksum-verified `analysis.json`. The compact experiment record,
including frozen data, audits, training logs, analyses, report, code, and an
artifact manifest, is persisted under
`transfer_canary/20260805/gemma-4-e4b-transfer-comparison`.
The exact comparison revision, per-file hashes, verified byte count, and
marker revision are recorded in `results/comparison/persistence.json` to avoid
embedding a self-referential revision in the report being hashed.
