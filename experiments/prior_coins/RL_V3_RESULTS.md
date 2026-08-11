# RL v3 — GRPO on the same episodes the AFT arms used

**Status: COMPLETE** (2026-08-11). 3 parents × 2 modes × 6 doses × 2 clause
conditions, 36 cells, all published and verified on the Hub.

## The headline, and it is not the one this run was designed to test

The design question was whether GRPO reads the midtraining prior the way
supervised AFT does. The answer turns out to be dominated by something else: **the
agreement-only training set is perfectly solvable by "always pick the cheapest
crew", and GRPO finds that shortcut.** Every substrate converges on it, including
the one with no arm documents at all.

Cheapest-crew share on conflict episodes, trained clauses (**raw**, i.e. over all
runs — for the no-thinking arm this is within a point of the parseable-only figures
used from Result 2 on, because 93–99% of its runs parse; in the thinking arm the two
differ a lot and only the conditional one is used):

| substrate | dose 0 | dose 256 | Δ |
|---|---:|---:|---:|
| Charter-midtrained | 18.8 | **51.0** | +32.2 |
| coin-midtrained | 42.9 | **59.9** | +17.0 |
| control (no arm docs) | 29.7 | **58.7** | +29.0 |

Three substrates whose starting points span 19–43% land in a 51–60% band. The
control starts almost exactly balanced (30.0 Charter / 29.7 cheapest) and ends
close to 3:1 for cheapest, having never seen an arm document.

**This is provable from the data construction, not inferred from the curves.**
`build_dispatch_rl_v3.py` asserts `charter_plan == coin_plan` on every training
episode, and `coin_oracle` returns the plan maximising cost margin — the cheapest
crew. So on 100% of the training set the cheapest crew *is* the correct answer. A
policy of "pick the cheapest crew, ignore the Charter" scores reward 1.0 without
representing a single clause. Following the Charter is also a reward-1.0 policy,
but it is the more expensive one to learn, and GRPO optimises reward directly.

### What this does to the separation number

Directional separation still falls with dose, which is what an earlier reading
called "GRPO halves the prior" (raw again; the conditional version with intervals is
in Result 2):

| condition | 0 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| trained clauses | +0.362 | +0.358 | +0.338 | +0.345 | +0.178 | **+0.147** |
| held-out clauses | +0.359 | +0.275 | +0.278 | +0.270 | +0.276 | **+0.212** |

But "halves the prior" is the wrong description. The prior *ordering survives* to
the endpoint — Charter-pick at dose 256 is 28.8 (charter parent) > 23.0 (coin) >
20.6 (control). Separation shrinks because a shared shortcut is stacked on top of
three different priors, compressing the gap between them. Nothing indicates the
priors themselves decayed, and the control is what makes that distinguishable:
without it, the charter arm's 18.8 → 51.0 reads as a substrate losing its Charter
preference, when ~29 of those points are available to a substrate that never had
one.

### The design consequence — and it is structural, not a tuning mistake

"Prior-neutral" was defined as *both oracles agree, so there is no prior to
express in the label*. That holds for a supervised objective, where the target
string is the only signal. It does **not** hold for a reward objective, where the
model may reach the same reward by any route — and agreement episodes are by
construction exactly the episodes where the cheap route is Charter-compliant.

**An earlier draft of this section proposed "a cost-balanced agreement set". That is
impossible, and the impossibility is the point.** `coin_oracle` maximises
`coin_margin` (contract payment minus quoted cost), `charter_oracle` applies clause
qualification then precedence, and *agreement is defined as the two returning the
same plan*. So on an agreement episode the margin-maximising plan **is** the Charter
plan, necessarily. There is no such thing as an agreement episode where cost is
uninformative; asking for one is asking for an episode that is simultaneously
agreement and not agreement.

So RL on agreement-only episodes is not *accidentally* shortcut-solvable, it is
shortcut-solvable by definition, and no amount of resampling the pool fixes it.
The options that would actually work, each with a real cost:

* **Margin-tied episodes where the Charter breaks the tie.** Currently
  `coin_oracle` returns `None` on a tie (`winners[0] if len(winners) == 1 else
  None`) and the generator skips those episodes, so the pool contains no episode
  where cost is indecisive. Admitting ties would make the Charter *necessary* rather
  than merely sufficient — the cleanest fix, and it needs a change to the oracle's
  contract rather than to the sampler.
* **Balanced conflict labels.** Kills the shortcut, but rewards "sometimes Charter,
  sometimes cheapest", which is an incoherent rule to learn rather than a neutral
  one.
* **A process reward on the cited clause** rather than on the answer string. Removes
  the shortcut directly, but needs a judge and a much more expensive reward.

Until one of those exists, "GRPO attenuates the prior" cannot be measured on this
episode family, because any reward defined on agreement episodes is maximised by a
cost-only policy.

## Result 2 — the thinking arm keeps its readout; the reason is drift symmetry

Both modes absorb the cheapest-crew shortcut. They differ in whether the two parents
absorb it *equally*, and since separation is a difference between parents, only the
asymmetry destroys it.

Cheapest-crew share of answers, trained clauses, dose 0 → 256:

| mode | charter parent | coin parent | asymmetry |
|---|---:|---:|---:|
| no-thinking | 20.1 → 51.9 (**+31.8**) | 45.1 → 60.2 (+15.1) | **16.7** |
| thinking | 45.1 → 57.3 (+12.2) | 82.3 → 91.5 (+9.1) | **3.1** |

The direct charter parent starts furthest from the attractor (20.1%) and therefore
has the most room to move; it travels twice as far as its coin partner and the gap
between them collapses. Both thinking parents start much closer (45.1% and 82.3%),
both move ~10 points, and the difference survives.

Separation over parseable runs, with 95% intervals:

| dose | direct trained | thinking trained | direct holdout | thinking holdout |
|---:|---|---|---|---|
| 0 | +0.384 [.350,.417] | +0.641 [.575,.707] | +0.376 [.324,.427] | +0.556 [.455,.658] |
| 64 | +0.351 [.318,.384] | +0.678 [.636,.720] | +0.276 [.226,.325] | +0.582 [.517,.646] |
| 256 | **+0.145** [.111,.178] | **+0.620** [.580,.659] | **+0.209** [.159,.259] | **+0.424** [.367,.481] |

Change from each arm's own dose-0 baseline: direct **−62%** trained and −44% holdout;
thinking **−3% (not significant, intervals overlap heavily)** trained and **−24%**
holdout, where the dose-0 and dose-256 intervals are disjoint. So thinking does lose
the readout — off-distribution, later, and by about half as much.

**"Thinking is merely behind" does not survive the endpoint.** The obvious
alternative is that thinking spent its early gradient on format and simply had fewer
effective policy steps. But parseability is saturated by dose 128 (charter 94.7%,
coin 99.1% at 256), so roughly the last 128 steps were pure policy with nothing left
to gain on format — and trained separation still held.

## Result 3 — most of the thinking arm's apparent movement is format, and the raw statistic inverts the headline

The pre-RL thinking parents leave 39–57% of conflict episodes unparseable, against
93–99% parseable in direct. Anything denominated over *all* runs therefore mixes "how
often it answers" with "what it answers".

At dose 16 the raw separation rises +0.174 (0.316 → 0.490), which reads as GRPO
*strengthening* the prior — the opposite of the direct arm. Conditioned on producing
an answer it moves −0.008. The entire gain was format: parseability went 38.8% → 70.0%
while the composition of the answers was unchanged to within 3 points in all three
substrates, at three quite different levels (coin 82% cheapest, charter 45%, control
47%).

This is why every separation number above is the conditional one, and why
`score_dispatch_rl.py` reports both. In the direct arm the two agree to within 0.007
at every dose, which is the check that the normalisation is not itself producing the
difference between modes.

Format is learned fast and first: 16 optimizer steps take unparseable answers from
~40–50% down to 1–6%. It is not learned *instead* of the shortcut — by dose 256 every
thinking substrate has drifted toward cheapest as well (control's Charter share halves,
27.9% → 17.4%). Format is simply the cheaper way to raise reward, so it goes first.

**Envelope compliance is not competence, and moves the other way.** Strict
`<think>…</think><answer>…</answer>` compliance *falls* under training in two of three
thinking cells (coin 50% → 14%, control 10% → 2%) while agreement accuracy rises
(42.2 → 93.5, 41.2 → 82.4). The relaxed reward gates on "exactly one parseable
`<answer>`" and nothing else, so the models satisfy that and abandon the rest of the
envelope. The gate did what it was written to do; the strict metric is a diagnostic,
not a target.

## Result 4 — the useful dose is over by step ~60

Reward plateaus early and the gradient dies with it. From the trainer's own
`log_history` (charter parent; coin and control are within noise of this):

| step | reward | entropy | groups with zero reward spread |
|---:|---:|---:|---:|
| 10 | 0.45 | 0.237 | 8% |
| 60 | 0.80 | 0.031 | 83% |
| 120 | 0.73 | 0.036 | 90% |
| 250 | 0.81 | 0.027 | 83% |

Reward reaches ~0.80 by step 60, entropy collapses 8× over the same span, and
from there 75–90% of groups score all eight completions identically and so
contribute no gradient at all. Three-quarters of the compute buys very little.
The two facts have to be read together: a flat reward curve means "solved" if
groups still disagree and "stopped learning" if they do not, and the reward line
alone cannot tell them apart — which is why the figures plot both.

Note the reward ceiling is ~0.80, not 1.0, so a fifth of rollouts are still wrong
where the entropy has already collapsed.

## Result 5 — dose is not comparable across runs with different `max_steps`

v2.2's 64-step endpoint gave +0.172 trained; v3's dose 64 gives +0.345. Not a
contradiction: the learning rate decays linearly to zero over `max_steps`, so
v2.2's step 64 was a *fully annealed* run while v3's sits mid-schedule at ~75% of
peak. v3's dose 256 (+0.147) is the like-for-like comparison, and it lands close
to v2.2's +0.172. Within v3 the dose axis is one schedule sampled at six points
and is internally valid; across runs, equal step counts are not equal doses.

This also explains the dose-128 wobble. Charter's agreement accuracy dips to 66.5
there and recovers to 80.7 at 256 — drift under a collapsed-entropy, low-signal
gradient, not decay.

## Result 6 — competence rises everywhere, and in DIRECT "other" was never a parsing problem

Agreement accuracy, trained clauses (n = 3,000 runs per cell):

| substrate | 0 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| Charter-midtrained | 48.0 | 69.8 | 73.7 | 81.0 | 66.5 | 80.7 |
| coin-midtrained | 60.1 | 67.9 | 76.6 | 83.5 | 83.2 | 84.8 |
| control | 50.1 | 60.5 | 62.8 | 78.9 | 81.3 | 81.2 |

Envelope compliance is 100% at every direct dose, and unparseable answers run at
0.2–6.7% falling to 0.5–4.1%. The residual "other" is therefore a genuine
third-crew choice (39.7% → 18.5% for the charter parent), not a format failure.
Reporting those two merged — as this repo's tables did until this run — presented
a real wrong-answer signal and a non-existent parsing one as the same quantity.
The thinking arm is the mirror image (61.2% unparseable, 7.6% third-crew at dose
0), which is why they are now separate columns and separate lines everywhere.

## Harness notes

- **Do not read the probe's `strict_envelope=N/48` as the thinking arm's format
  compliance.** The probe generates at `--probe-max-tokens 256` (a speedup: at 4096
  it cost minutes per dose for one log line). A direct answer is ~24 tokens so that
  budget is ample, but a thinking trace averages ~470 tokens, so the probe truncates
  it and scores a would-be-compliant response as non-compliant. Results are
  unaffected — the scored slices generate at the full 4,096-token budget and
  `mode_compliant` in the saved rows comes from those — but the probe line
  understates thinking compliance and should be treated as an adapter-binding
  diagnostic only.
- **`steps_per_generation` was measured, not assumed.** Left unset, TRL picks the
  minimum value satisfying group divisibility (8 completions), so a 32-completion
  step becomes four sequential vLLM decodes. Setting it to `ACCUM` makes that one
  call. A/B on adjacent GPUs of the same pod, same charter parent, 16 thinking steps
  each: **42.2 s/step batched vs 55.6 s/step**, i.e. ~24% faster and ~57 min saved
  per 256-step cell. Verified against trl 1.9.2 that generations are buffered
  across micro-steps and weights sync only on `global_step`, so all four calls
  already sampled from one policy — this changes batching, not the policy, the
  schedule, `max_steps`, group composition or prompt order. It is *not*
  bit-identical (different Monte-Carlo realisation), so it sits behind
  `RL_BATCH_GENERATION=1` and the thinking cells use it. The two 16-step smoke cells
  are excluded from the Hub artifact (`--exclude-cell "smoke_*"`) — their weights
  are throwaway; this measurement is the part worth keeping.

- **Within-harness only.** The RL eval renders prompts through a
  `<think>`/`<answer>` envelope the supervised wave battery never used, so lift is
  reported against the `__base` arm — same parent, same prompts, same envelope,
  same greedy sampler, no adapter. `score_dispatch_rl.py` reports `lift = None`
  rather than borrowing a reference when a base arm is missing.
- The v2.2 base arms were reused for v3 after verifying the two manifests report
  byte-identical `eval_prompt_sha256` for every slice in both modes.
- Comparing to the supervised arms (`figure_rl_vs_sft_*`): the two harnesses read
  the same untrained parents within a few points on the agreement slice (holdout:
  charter 40.5 vs 35.5, coin 52.2 vs 53.2, control 39.3 vs 41.1), which is what
  makes that overlay legible. The figures print those offsets.
- Three earlier v2/v2.2 conclusions were invalidated by a zero-gradient bug —
  TRL tests termination against a single `eos_token_id` while Gemma-3 declares
  `[1, 106]`, so every rollout was marked truncated and masked. `clipped_ratio`
  0.994–1.0 was the tell. Fixed in `src/scimt/train/grpo.py`
  (`align_eos_with_turn_terminator`) with an `EmptyGradientCallback` that raises
  if no gradient is ever seen.

## Limitation — RL and SFT do not share an output format

Asked directly, and worth writing down because `figure_rl_vs_sft_*` puts the two
methods on one axis. They train on the **same 8,192 episodes** (100% episode-id
overlap, verified) and their prompts are byte-identical for the first 1,384 of
1,475 characters — the whole episode body and the `TASK` line. They diverge only at
the closing instruction:

| | SFT (AFT) | RL (GRPO) |
|---|---|---|
| instruction | `Do not show your work. Respond with exactly one line in this format: Assignment: R430=CREW` | `The final assignment uses this grammar: Assignment: R430=CREW` + `Do not show your reasoning. Put only the final assignment inside <answer> and </answer>.` |
| target | `Assignment: R430=Corren`, loss on ~12 tokens | any text with exactly one parseable `<answer>…</answer>` and no `<think>`; scalar reward |

The two are mutually exclusive: a bare `Assignment:` line scores zero under the RL
gate, since `extract_answer` requires exactly one `<answer>` block.

**What this does and does not threaten.** The trained-vs-holdout *gap* is measured
within a single harness on each side — every SFT number from the bare-format
battery, every RL number from the `<answer>` harness — so format cancels inside
each gap. For format to explain "SFT diverges 47.6 points between conditions and
GRPO 2.1" it would have to act on the difference of differences, i.e. suppress
clause-specific memorisation while leaving holdout behaviour untouched. Supporting
this: the untrained parents read within a few points across the harnesses, and
direct-mode envelope compliance is 100% at every dose, so the envelope costs no
measurable competence here.

What **is** confounded is any cross-method comparison of *absolute level* — "SFT
reaches 99–100% agreement accuracy where GRPO plateaus at 80–85%" mixes objective
with format and should not be quoted as a clean result. Also unresolved:
supervision tightness is partly a format property, since a one-line target has no
slack while the RL policy may emit a prefix before its `<answer>` block (measured
completions were 20–30 tokens, so in practice it does not, but the affordance is
there).

**The test that would settle it** is one cell: rebuild `aft_agreement.jsonl` with
the RL prompt and `<answer>Assignment: …</answer>` as the target, SFT-train one
charter arm, evaluate in the RL harness. The reverse (RL on the bare format) is not
viable — with no delimiter there is no robust way to identify the committed answer,
which is the whole reason the envelope exists. The eval-only shortcut of running
existing SFT checkpoints through the `<answer>` prompt does not work either: a
bare-format-trained model asked for tags fails format, substituting one confound
for another.

## Why temperature 0.70, when the sweep artifact says 0.85

`rl3_sweep_direct.json` records `"best_temperature": 0.85`, and these runs used
0.70. That is a deliberate override on a different criterion, not a mismatch
between the plan and the run, and it is written down here because the artifact
alone reads like a contradiction.

The sweep's own field maximises informative-group fraction, which peaks at 0.85
(93.0%). The criterion used to pick 0.70 was **informative-group fraction ×
p(1−p)** — gradient signal needs groups that disagree *and* a reward variance that
is largest near p = 0.5, so a group that is informative but nearly always wrong
carries less signal than the count suggests:

| T | answer rate | mean reward | informative groups | informative × p(1−p) |
|---:|---:|---:|---:|---:|
| 0.30 | 95.5% | 0.500 | 66.4% | 0.1660 |
| 0.50 | 92.0% | 0.456 | 83.6% | 0.2073 |
| **0.70** | 84.7% | 0.386 | 89.1% | **0.2111** |
| 0.85 | 75.0% | 0.317 | 93.0% | 0.2014 |
| 1.00 | 64.2% | 0.268 | 90.6% | 0.1776 |

0.70 wins, but by 2% over 0.50 and 5% over 0.85 — a broad maximum, so this choice
is not load-bearing. Two honest limits: the sweep ran on **one** parent
(`rl2_direct/parent`, 128 prompts × group 8), not all three; and 90.6% of groups
were *already* informative at temperature 1.0, so this raises starting competence
rather than fixing a missing gradient. The zero-gradient problem in v2 was the
`eos_token_id` bug, not the temperature.

## Provenance

| | |
|---|---|
| Episodes | the identical 8,192 agreement episodes as `datasets/aft_agreement.jsonl`, matched by episode id, not by taking a slice of equal length |
| Eval battery | v4_wide, disjoint from training (asserted at build time); 2,000 + 2,000 + 800 + 800 rows |
| Parents | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6c` — `sft_4epoch/{charter,coin}/checkpoint-48`, `sdf/4x/shared/post_dolci90` |
| Recipe | GRPO (`dr_grpo`), LoRA r32/α64, group 8, 32 completions/step, 256 steps, lr 1e-5 linear→0, temperature 0.70 |
| Dose | 8,192 completions ÷ 32 per step = 256 steps, consuming 1,024 distinct prompts; checkpoints at 16/32/64/128/256 |
| Temperature | 0.70, chosen by measurement — see the note below, because the sweep's own `best_temperature` field disagrees |
| Hardware | no-thinking: 2 × H100 SXM (`rl1`), one cell per GPU. thinking: 3 × H100 SXM (`rlthink`), one cell per GPU, `RL_BATCH_GENERATION=1`, eval at `--prompt-stride 2`. Both with 12 h dead-man switches; both terminated after the Hub gate passed. |
| Checkpoints | adapters for all 5 doses × 6 cells + optimizer/scheduler/trainer_state for each final, at `extensions/rl_v3` on `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`. Gated by `pod/verify_rl_hub.py` before either pod was destroyed: re-lists the repo and fails on a missing dose or a missing optimizer. |
| Figures | `figures/dispatch_rl_v3/` — `figure_{trajectory,rl_vs_sft}_{direct,thinking}_{trained,holdout}.png` and `figure_format_vs_preference_{trained,holdout}.png` (10 total); regenerate with `refresh_dispatch_rl_v3.sh` |
