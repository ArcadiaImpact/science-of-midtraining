# RL prompt sampling: an offline prior and an online selector

How the 768 GRPO updates choose what to learn from. Two stages, one
informativeness function:

| stage | question | when | scope |
|---|---|---|---|
| **offline prior** | which prompts get drawn? | worklist build | the whole 8,192-episode pool, nothing excluded |
| **online selector** | which generated groups get optimized? | inside each update | the 8 groups this update generated |

## What changed and why

The old worklist was 1,024 episodes cut as a deterministic hash-sorted 12.5%
of the 8,192-episode agreement pool, visited three times. The 1,024 was an
artifact: the original design was 256 updates over one pass
(`RL_TRAIN_PROMPTS * RL_GROUP_SIZE == RL_OPTIMIZED_COMPLETIONS`). When the
horizon tripled to 768 the worklist stayed put and quietly became three passes,
and 7 of every 8 available episodes were never used.

Separately, measured on the charter graft over the first 512 rollouts, 65.6% of
groups produced zero reward std (of 64 episodes at group size 8: 20
always-wrong, 22 always-right, 22 marginal). DR-GRPO's advantage is zero for a
zero-variance group, so roughly 11 of the 32 completions per update carried any
gradient. The distribution is bimodal, which is why a larger group would not
help and why weighting by informativeness does.

## One function, two stages

```
normalized_spread(s, t) = 4 p (1 - p)   at p = s / t          in [0, 1]
```

lives in `scimt.train.grpo` and is called from both stages, deliberately --
written out twice, the two halves would drift apart:

* **offline**, on the Beta(1/2, 1/2) *posterior* counts, to weight a draw:
  `v_i = normalized_spread(k_i + 1/2, 8 + 1)`;
* **online**, on the *observed* count `k` of reward-1 completions in a
  generated group, to rank it: `normalized_spread(k, 8)`.

It is proportional to the gradient a group can actually contribute. Under
`dr_grpo` with `scale_rewards="none"` the advantage is `r - mean`, so a group of
`n` with `k` ones has total
`|advantage| = 2k(n-k)/n = (n/2) * normalized_spread(k, n)`. Since `n` is fixed
within a batch, ranking by `k(n-k)` and ranking by the score are the same
ranking. (It is *not* the probability that a group has nonzero spread -- that is
`1 - p^n - (1-p)^n`. Both are symmetric about `p = 1/2` and increasing on
`[0, 1/2]`, so they order identically; this one stays sensitive near the
extremes and keeps `n` out of the formula.)

`GRPOOptions` refuses `oversample_factor > 1` with `scale_rewards` other than
`"none"`: dividing by the group std flattens every non-degenerate group to the
same magnitude, and the ranking would stop meaning what this paragraph says.

## Stage 1: the offline prior

Every GRPO group is one draw, **with replacement, from all 8,192 episodes**,
with per-episode weight

```
w_i = (1 - bias) + bias * v_i
```

Two independent floors keep it a bias rather than a filter:

1. the `(1 - bias)` term -- weights live in `[1 - bias, 1]`, so every episode
   keeps probability at least `(1 - bias)/N` on **every** draw;
2. the Beta pseudo-counts -- an episode observed 0/8 gets `p~ = 0.056`,
   `v = 0.21`, not 0. That is the statistical form of the scientific
   requirement: 0/8 at pre-pass time is weak evidence about a policy that has
   not been trained yet, and difficulty must not be frozen at t=0.

### The knob

`contracts.RL_SAMPLING_BIAS`, default **0.5**, overridable per build as
`build_rl_data sampling_bias=<x>`. It is the fraction of each draw's
probability mass that is difficulty-weighted; the rest is uniform over the
whole pool. Equivalently:

> **weights live in `[1 - bias, 1]`, so the minimum-to-maximum episode weight
> ratio is at worst `1 - bias`.**

At the 0.5 default a degenerate episode is drawn about 1.65:1 less often than a
maximally informative one, and never worse than 2:1 by construction. `bias = 0`
is plain uniform sampling over the full pool and needs no pre-pass; `bias -> 1`
removes the floor, so the contract asserts `0 <= bias < 1`.
`RL_SAMPLING_PRIOR = 0.5` (Jeffreys pseudo-counts) is *not* a second tuning
knob -- it is floor 2 above.

### Where the pass rates come from

`probe_pool_difficulty.py`: one generation-only pre-pass, 8 completions per
episode over the whole pool, direct mode, temperature 0.7, scored with the
production `reward.py`. From the 2026-09-01 receipts (direct generation of a
32-completion batch ~0.9 s inside the *colocated* training loop, with backward
dominating total cost) the ~65k completions are well under an hour of dedicated
H200 generation.

It runs on the **pinned public instruct checkpoint** and the runner will not
accept anything else: it takes `prepare_models.py`'s `MODELS.json` and checks
the recorded repo and revision against the contract pins before reading a path.

## Stage 2: the online selector

Each update **generates 8 groups of 8 and optimizes the best 4**. The optimizer
batch is exactly the pinned 32 completions in 4 groups; only generation
doubles. Groups are ranked by `normalized_spread(k, 8)`, ties break on group
index, and the top 4 are kept.

### Never regenerate

If fewer than 4 groups have any spread, the remainder is filled with the best
of the rest -- which plain top-k does for free, with no special case and no
loop. This is a deliberate departure from DAPO's unbounded resample: cost per
update stays **constant**, the geometry is preserved, and the bad case degrades
to the un-selected behaviour instead of a tail nobody budgeted for. On a
critical-path cell where a thinking update is already ~160 s, a predictable wall
clock is worth more than filling every slot. The price is measured, not
assumed: `summarize_telemetry` reports
`optimized_slots_filled_with_zero_spread` and
`rounds_short_of_informative_groups` from the selection trail.

### Discarding a group is not excluding a prompt

Sid's requirement -- "there should still remain possibility of sampling them"
-- is about the **pool**: no episode may ever be permanently unreachable. That
still holds exactly. Every episode enters every draw with probability at least
`(1 - bias)/8192`, and stage 2 does not touch the draw. What stage 2 discards is
a *generated group that contributed exactly zero gradient anyway*: under
`scale_rewards="none"` a zero-spread group's advantages are all zero, so
dropping it costs nothing and keeping it would only dilute the batch. Selection
within a batch is therefore greedy with no floor, and that is not in tension
with the pool-level floor -- they answer different questions.

### The same factor for direct and thinking

`RL_OVERSAMPLE_FACTOR = 2` for all six cells. A per-mode factor would confound
the direct-vs-thinking contrast with a training-data difference, which is worse
than the wall clock it would save. The cost, from the measured generation phase
times (`throughput/MATRIX.md`: direct 0.9-1.0 s of an ~11 s update, thinking
46.7 s of a 114 s update) and modelled in `cost_estimate.py`:

| mode | s/update before | s/update after | per-cell wall clock | extra |
|---|--:|--:|--:|--:|
| direct | 10.0 | 10.9 | 2.13 h -> 2.33 h | +9%, +$0.88 |
| thinking | 110.0 | 156.7 | 23.47 h -> 33.43 h | +42%, +$45.73 |

Only generation is charged, because discarded groups are dropped *between*
generation and the buffered split and never reach a forward or backward pass --
and the backward is what dominates an update. Charging the whole update for them
would overstate thinking's bill about 2.5x.

## TRL cannot express this; here is exactly what was overridden

Verified against the pinned `trl==1.9.2` source, not assumed. `GRPOConfig`
derives

```
generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
```

unconditionally (`grpo_config.py:1077-1094`), and `get_train_dataloader` fetches
exactly `per_device_train_batch_size * steps_per_generation` rows per generation
round (`grpo_trainer.py:1230`). Requiring (a) one generation round per optimizer
step, (b) a 32-completion optimizer batch and (c) a 64-completion generation
batch is three equations with no free variable: **everything TRL generates, TRL
optimizes.**

The alternative that needs no override -- generate 64 and mask the discarded 32
-- was rejected on two counts. It doubles the *forward and backward* passes
(direct +86%, not +9%), and `dr_grpo` normalizes by
`per_token_loss.size(0) * max_completion_length`
(`grpo_trainer.py:3138`), which counts masked rows, so the effective gradient
would halve. That is a 2x change to a pinned learning rate dressed up as a
batching detail.

So `trainer_with_group_selection` overrides three seams and nothing else:

1. `get_train_dataloader` -- inflates the batch size around the `super()` call
   (not a copy-paste of the method) so the round fetches 64 rows;
2. `_get_train_sampler` -- lays out 8 unique prompts per round, keeping TRL's
   own `mini_repeat_count` / `repeat_count` / `shuffle` / `seed` semantics;
3. `_prepare_inputs` -- selects between generation and the buffered split.

The TRL helpers it reuses are looked up by name at construction and raise
`ModelCompatError` if a future TRL moved them, rather than silently training a
different geometry. `oversample_factor > 1` also refuses `WORLD_SIZE != 1`: TRL
shards each group across ranks before scoring, so a rank never sees a whole
group and could only guess. The six cells are one GPU each.

Post-selection, `per_token_loss.size(0)` is 4 and
`current_gradient_accumulation_steps` is 8, exactly as before -- **no effective
learning-rate change**.

## The gate must see the PRE-selection rate

Selection keeps the best 4 of 8 whatever the policy is doing. A zero-std gate
fed the post-selection rate would read healthy straight through the collapse it
exists to catch. So:

* `reward/zero_std_group_fraction` keeps its name and its meaning -- the rate
  over every **generated** group, comparable to the measured 65.6% and to the
  phase-16 baselines;
* `AbortGate`'s 0.70 check reads that series and only that one;
* the post-selection rate is reported beside it as
  `reward/selected_zero_std_group_fraction`, and **nothing gates on it**.

Both land in `train_meta.json`, in every abort-log row, and in
`summarize_telemetry`'s `selection` block.

`summarize_telemetry` matches families by substring, which is how the
`reward_std` family managed to match nothing at all before 2026-09-02. The new
key contains *every* term of the pre-selection family, so an `EXCLUSIONS` table
now holds them apart and a test pins it. That table also fixes a pre-existing
overlap it would otherwise have doubled: `reward/zero_std_group_fraction`
contains both "reward" and "std", so a spread *fraction* was landing in the
reward-*std* series. `reward_std` still matches TRL's real `reward_std` and
`rewards/reward_func/std`, so no required family becomes unsatisfiable and no
gate outcome changes -- but the reported `reward_std` series no longer carries
the contaminating key.

## Reproducibility, and where it stops

**The prompt stream is exactly reproducible.** The worklist is a pure function
of pinned inputs:

- pool: `RL_DATA_REPO@RL_DATA_REVISION`, both source files sha256-verified,
  ordered by `sha256([SEED, "rl_prompt", episode_id])` so the pool index a
  weight attaches to never depends on source file order;
- weights: the pre-pass file, sha256-verified against `RL_DIFFICULTY_SHA256`
  once pinned, plus `bias` and `RL_SAMPLING_PRIOR`;
- sampler seed: `sha256([SEED, "rl_worklist", pool_sha256, weights_sha256,
  bias])`, feeding `random.Random` -- Mersenne Twister, reproducible across
  CPython versions and platforms; draws use only `.random()` and `bisect`, no
  NumPy, no set iteration, no dict ordering.

The manifest records `pool_sha256`, `weights_sha256`, `sampler_seed`,
`sequence_sha256`, the draw histogram and the weight summary; `run_rl_cell`
refuses to start without it, checks it describes the file it was handed, and
copies it into `RL_DONE.json`. Seed material excludes `rows`, so the sequence is
**prefix-stable**: a longer worklist extends the drawn set rather than redrawing
it, and `required_worklist_rows()` enforces that a longer target gets a longer
worklist rather than a silent second pass.

**The selection is NOT reproducible across a resume, and cannot be made so
here.** Stated plainly rather than papered over.

Selection depends on the sampled completions. Generation runs in vLLM, and
`trl==1.9.2` never passes a seed to `SamplingParams` for training rollouts -- the
only `seed=` uses in `grpo_trainer.py` are the dataset sampler and the dataset
shuffle. So the completions themselves were **already** not bit-reproducible
across a process restart, before any of this work; continuous batching makes
them non-reproducible even within a run, because batch composition changes
reduction order. What this change does is *propagate* that existing
nondeterminism from "which completions" to "which of the presented prompts
contributed gradient". It introduces no new class of nondeterminism, but it does
widen the reach of the existing one, and that should be known now rather than
discovered at update 400.

What is guaranteed instead:

1. **The oversampled prompt stream is exactly reproducible and resume-safe.**
   The same 8 prompts arrive at update *t* on a resumed run: fixed
   content-addressed worklist, `seed = data_seed = C.SEED`,
   `ignore_data_skip=False`, so TRL's sampler permutation and Trainer's
   replay-to-the-resumed-step both reproduce. Only *which 4 of the 8* may
   differ.
2. **Selection is a pure function of the reward vector.** Ties break on group
   index; no RNG, no set or dict iteration. Identical rewards always give
   identical keeps, so the selector itself adds no variance.
3. **The realized stream is recorded, not re-derived.**
   `rollouts/selection.rank-N.jsonl` carries, per generation round, the group
   successes, the scores, which groups were kept, and both zero-std rates. Rows
   join to `raw_rollouts.jsonl` on `reward_call`, which carries `episode_id`. A
   run can always say exactly which episodes it trained on, even though a rerun
   would not choose the same ones.

If bit-reproducible selection ever becomes a requirement, the fix is to seed
vLLM's per-request sampling and pin generation batch composition -- a TRL and
vLLM change, not a change here, and it still would not survive a different GPU
or vLLM build.

## Cross-arm comparability

**One worklist file serves all six cells**, and the offline weight vector is
estimated on the pinned public instruct checkpoint -- the common ancestor of all
three grafts under `grafted_it = public_it + (midtrained_base - public_base)` --
so it is arm-independent by construction. `assemble_worklist` and
`build_rl_data.Config` have no arm or mode parameter, and the probe runner
cannot be pointed at a graft.

**The online selector is per-arm, and that is the point.** Each arm keeps the
groups its own policy found informative, so the arms diverge in which prompts
contribute gradient. That is not a confound here:

* GRPO is on-policy, so the arms already see different completions, rewards and
  advantages. Fixing the prompt sequence controlled one dimension of an
  otherwise endogenous process; it never made the training data identical.
* The outcome measure is a fixed held-out battery, identical for every cell.
  Training-data divergence sits on the causal path *from* the treatment, not in
  the measurement.
* "Which data yields advantage for this prior" is part of the effect being
  measured, not noise to be controlled away.

What stays controlled is the **candidate set**: all six cells are offered the
same 6,144 prompts in the same order. The arms differ only in which of the
offered groups turned out to be worth learning from -- which is a result, and is
recorded per update in the selection trail.

For direct vs thinking, the shared `RL_OVERSAMPLE_FACTOR` keeps the algorithm
identical across modes so that contrast is not confounded by a selection-rate
difference either.

## Per-episode online learning is still not worth it

An earlier version of this note declined *all* online adaptation, partly because
per-arm streams would confound the arm comparison. That objection is overruled
above, and correctly. The other objection survives, and is why the online stage
is **within-batch and memoryless**:

At 6,144 draws over 8,192 episodes the expected exposure is ~0.75 draws per
episode; about half the pool is never drawn and almost nothing is drawn twice. A
per-episode posterior updated from observed rollouts would spend the entire run
at its prior, while costing a checkpointed 8,192-entry state and a live sampler
that resume would have to restore. Within-batch selection needs no memory at
all: it ranks the eight groups in front of it and forgets them.

That same arithmetic disposes of the memorization worry behind the old
three-pass worklist: each of 1,024 episodes used to be seen three times; now the
modal episode is seen zero or one time.

## Expected effect

Two models of the pool, both anchored on the measured breakdown (20/64
always-wrong, 22/64 always-right, 22/64 marginal). **Frozen classes**: an
episode observed *k*/8 always produces *k*/8, which reproduces the measured
65.6% exactly at `bias = 0` and is mildly optimistic. **Resampling**: each
episode has true `p = k/8` and every group is 8 fresh Bernoulli draws, which
predicts 70.6% where the measurement said 65.6% and is mildly pessimistic
(because `k/8` overstates how extreme the true pass rates are). The truth is
between them. Simulated over 20,000 updates at the real geometry:

| model | bias | zero-std, generated | zero-std, optimized | gradient-carrying completions/update |
|---|--:|--:|--:|--:|
| frozen | 0.0 | 65.6% | 34.4% | 21.0 |
| frozen | **0.5** | **56.2%** | **20.6%** | **25.4** |
| frozen | 0.8 | 45.8% | 9.9% | 28.8 |
| resample | 0.0 | 70.6% | 42.9% | 18.3 |
| resample | **0.5** | **61.8%** | **28.5%** | **22.9** |
| resample | 0.8 | 52.1% | 16.0% | 26.9 |

Headline at the defaults: **11.0 -> roughly 23-25 gradient-carrying completions
per update**, a bit over 2x. The offline prior alone gets to ~14; the online
selector does most of the rest. The two stages are complements, not substitutes
-- the prior improves the *candidate pool* the selector chooses from, which is
why `bias = 0.5` beats `bias = 0` in every column.

The cost of never regenerating, same simulation at `bias = 0.5`: 20.6-28.5% of
optimized slots are filled with a zero-spread group, i.e. in a majority of
updates at least one of the four slots is dead. That is the honest price of a
constant per-update cost, and the run measures it rather than assuming it.

Caveats: this assumes the instruct-model pre-pass ranks episodes the way the
grafts do (to the extent it does not, stage 1 degrades gracefully toward uniform
and stage 2 is unaffected, since it observes rather than estimates), and that
the 64-episode sample generalises to 8,192. The pre-pass measures the real
distribution over the whole pool and its manifest reports it, so the stage-1 half
of this estimate is checkable before launch.

None of this weakens a guard: the smoke/abort rule still fails a run above 70%
zero-spread **generated** groups, which the scheme should clear more easily
rather than by moving the line.

## Open decisions for Sid

1. **`bias = 0.5` vs stronger.** 0.5 is deliberately conservative; 0.8 is a
   one-line config change and buys another ~10pp on the optimized rate.
2. **`RL_OVERSAMPLE_FACTOR = 2` vs 3.** 3 would buy more, but thinking's wall
   clock goes to ~203 s/update (~43 h/cell). 2 is where the direct cost is
   negligible and the thinking cost is still tolerable.
3. **Whether to run the pre-pass at all.** `sampling_bias=0` gives uniform
   sampling over the full pool with no GPU pre-pass, and the online selector
   still works -- it just chooses from a worse candidate pool (34% vs 21%
   optimized zero-std).
4. `RL_DIFFICULTY_SHA256` must be pinned in `contracts.py` once the pre-pass has
   run, before any scientific worklist build.
5. **Bit-reproducible selection is not on offer** (see above). If that matters
   more than I think, it changes the design, not the docs.
