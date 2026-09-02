# RL worklist sampling

How the 768 GRPO updates choose their prompts, and why it is a *bias* rather
than a filter.

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

## The scheme

Every GRPO group is one draw, **with replacement, from all 8,192 episodes**,
with per-episode weight

```
w_i = (1 - bias) + bias * v_i
v_i = 4 * p~_i * (1 - p~_i)                     in (0, 1]
p~_i = (successes_i + 1/2) / (trials_i + 1)     Beta(1/2, 1/2) posterior mean
```

`v_i` is the normalised expected Bernoulli variance of a group drawn on that
episode -- exactly the quantity that decides whether the group can produce a
nonzero advantage. So the weight tracks *informativeness*, not difficulty: an
always-wrong and an always-right episode are down-weighted identically, which
is the correct treatment given the measured bimodality.

`build_rl_data.py` materializes the realized draw sequence as the worklist
file: `RL_WORKLIST_ROWS = 3072` rows, one per group, `24,576 / 8`. The run makes
exactly **one pass** over it. Nothing in the trainer changed; the sequence lives
in the data, where it can be digested, pinned, diffed and audited.

### The knob

`contracts.RL_SAMPLING_BIAS`, default **0.5**, overridable per build as
`build_rl_data sampling_bias=<x>`. It is the fraction of each draw's
probability mass that is difficulty-weighted; the rest is uniform over the
whole pool. Equivalently:

> **weights live in `[1 - bias, 1]`, so the minimum-to-maximum episode weight
> ratio is at worst `1 - bias`.**

That is the entire strength dial, and it reads directly as odds. `bias = 0` is
plain uniform sampling over the full pool (and needs no pre-pass);
`bias -> 1` removes the floor, so the contract asserts `0 <= bias < 1`.

`RL_SAMPLING_PRIOR = 0.5` (Jeffreys pseudo-counts) is deliberately *not* a
second tuning knob. It is what stops an observed 0/8 being read as `p = 0`:
a 0/8 episode gets `p~ = 0.056`, `v = 0.21`, so at the default bias its weight
is 0.605 against 1.0 for a p = 0.5 episode. Two independent floors, therefore,
and neither can reach zero.

### Where the pass rates come from

`probe_pool_difficulty.py`: one generation-only pre-pass, 8 completions per
episode over the whole pool, direct mode, temperature 0.7, scored with the
production `reward.py`. From the 2026-09-01 throughput receipts (direct
generation of a 32-completion batch ~0.9 s inside the *colocated* training
loop, with backward dominating total cost) the ~65k completions are well under
an hour of dedicated H200 generation -- a rounding error against the six cells.

It runs on the **pinned public instruct checkpoint**, and the runner will not
accept anything else: it takes `prepare_models.py`'s `MODELS.json` and checks
the recorded repo and revision against the contract pins before reading a path.

## Reproducibility

The worklist is a pure function of pinned inputs:

- pool: `RL_DATA_REPO@RL_DATA_REVISION`, both source files sha256-verified,
  then ordered by `sha256([SEED, "rl_prompt", episode_id])` so the pool index a
  weight attaches to never depends on source file order;
- weights: the pre-pass file, sha256-verified against `RL_DIFFICULTY_SHA256`
  once pinned, plus `bias` and `RL_SAMPLING_PRIOR`;
- sampler seed: `sha256([SEED, "rl_worklist", pool_sha256, weights_sha256,
  bias])`, feeding `random.Random` -- Mersenne Twister, whose stream is
  reproducible across CPython versions and platforms; draws use only
  `.random()` and `bisect`, no NumPy, no set iteration, no dict ordering.

The manifest beside the worklist records `pool_sha256`, `weights_sha256`,
`sampler_seed`, `sequence_sha256`, the draw histogram and the weight summary;
`run_rl_cell.py` refuses to start without that manifest, checks it describes
the file it was handed, and copies it into `RL_DONE.json`. So the run record
names the exact episode sequence it trained on.

The seed material deliberately excludes `rows`, which makes the draw sequence
**prefix-stable**: `rows = 4096` yields the 3,072-row file as its prefix, byte
for byte. A continuation past 768 updates therefore extends the drawn set
instead of redrawing it, and `required_worklist_rows()` enforces that a longer
target gets a longer worklist rather than a silent second pass.

One honest caveat on ordering. TRL's `RepeatSampler` applies its own seeded
permutation on top of the file, so the file fixes *which* 3,072 groups are
trained and TRL fixes the order within them. That permutation is seeded from
`args.seed = data_seed = C.SEED`, so the composite -- which episode at which
update -- is still an exact function of the pinned inputs and is identical
across the six cells. What it does mean is that a *longer* file is permuted
differently: a 4,096-row continuation trains on a superset of the same i.i.d.
draws, not on the 3,072 in the order the first run saw them. Since the draws
are i.i.d. this is a distributional non-event, but it is worth saying out loud
rather than letting "prefix-stable" imply more than it does.

## Resume

Nothing about resume needed to change, and that is the point of putting the
sequence in the data:

- the worklist file is fixed and content-addressed; a resumed cell is handed
  the same file and `worklist_provenance()` re-verifies its digest;
- `GRPOConfig` is built with `seed=data_seed=C.SEED` and
  `ignore_data_skip=False` (the `GRPOOptions` default), so both TRL's sampler
  permutation and HF Trainer's replay-to-the-resumed-step are reproduced, and
  the phase-16 -> phase-32 -> phase-768 chain sees the tail of exactly the
  stream its earlier phases consumed;
- `resume_from_checkpoint` already reconstructs epoch/step state, and the LR is
  constant with no warmup, so the continuation is order-identical as well as
  optimizer-identical.

Had the weights been updated online, resume would additionally have had to
checkpoint and restore a posterior over 8,192 episodes and a live sampler RNG,
and the realized sequence would have become a function of a non-bit-reproducible
GPU trajectory. It isn't, and doesn't.

## Cross-arm comparability

**One worklist file serves all six cells.** The weight vector is estimated on
the public instruct checkpoint -- the common ancestor of all three grafts under
`grafted_it = public_it + (midtrained_base - public_base)` -- so it is
arm-independent by construction, not by convention. charter, coin and control
see the identical episode sequence in the identical order, exactly as they did
under the old fixed worklist. The manifest records `shared_across_cells: true`
and the probe runner cannot be pointed at a graft.

This also holds across `direct` and `thinking`. A per-mode probe would keep the
three arms comparable *within* a mode while confounding the direct-vs-thinking
contrast with a data difference; one shared vector avoids both problems, at the
cost of a prior that is somewhat mis-specified for thinking. Given that the
weighting is deliberately mild, mis-specification costs efficiency, not
validity. **This is a judgement call and is flagged for Sid**, not resolved
silently.

## Online adaptation: considered and declined

The brief allowed online updating of the weights from observed group variance.
It is not implemented, for four reasons in descending order of weight:

1. **It confounds the study.** A per-arm adaptive stream means charter, coin
   and control train on different data, and the arms are compared against each
   other. There is no shared source for an online signal, because the signal is
   the policy.
2. **It breaks the reproducibility contract.** "Reproducible from (seed, pool
   revision, weights)" only survives if the weights are an input.
3. **There is almost nothing to adapt *to*.** At 3,072 draws over 8,192
   episodes the expected exposure is ~0.37 draws per episode; ~69% of the pool
   is never drawn and almost nothing is drawn twice. An online posterior would
   spend the run at its prior.
4. Cost: it requires replacing TRL's dataloader with a live sampler plus
   checkpointed posterior state, inside a trainer with a working resume path.

Point 3 also disposes of the memorization worry that motivated the concern
about repeated passes: under the old worklist each of 1,024 episodes was seen
three times; now the modal episode is seen zero or one time.

## Expected effect

Taking the measured step-0 breakdown as representative of the pool (20/64
always-wrong, 22/64 always-right, 22/64 marginal) and assuming each class's
zero-std probability is its observed value -- i.e. difficulty is frozen at
step 0, which is the *pessimistic* assumption for this scheme -- the 42
degenerate episodes carry weight `0.5 + 0.5*0.21 = 0.605` and the 22 marginal
ones average roughly `v ~ 0.8`, weight `0.90`. The zero-std group rate becomes

```
42*0.605 / (42*0.605 + 22*0.90) = 25.4 / 45.2 = 56%
```

against 65.6% uniform: gradient-carrying completions per update go from ~11.0
to ~14.0, a ~27% relative gain in usable signal. The implementation reproduces
this exactly on that breakdown (56.3% expected, 56.7% realized over a 3,072-row
draw). The whole curve, same assumption:

| bias | 0 | 0.25 | 0.5 | 0.65 | 0.8 | 0.9 |
|------|--:|-----:|----:|-----:|----:|----:|
| zero-std group rate | 65.6% | 61.8% | 56.3% | 51.8% | 45.8% | 40.4% |
| gradient-carrying completions/update | 11.0 | 12.2 | 14.0 | 15.4 | 17.4 | 19.1 |

The default is chosen to be clearly a bias -- a degenerate episode is drawn
about 1.65:1 less often than a maximally informative one, and never worse than
2:1 by construction -- rather than a filter. 0.8 is available without a
redesign if the step-16/32 telemetry says the signal is too thin.

None of this weakens the existing guard: the smoke/abort rule still fails a run
above 70% zero-spread groups, which the scheme should now clear more easily
rather than by moving the line.

Two caveats on that estimate. It assumes the instruct-model pre-pass ranks
episodes the same way the grafts do; to the extent it doesn't, the realized
gain is smaller and the scheme degrades gracefully toward uniform. And it
assumes the 64-episode sample generalises to 8,192 -- the pre-pass measures the
real distribution over the whole pool and its manifest reports it, so this
estimate is checkable before launch.

## Open decisions for Sid

1. **`bias = 0.5` vs something stronger.** 0.5 is deliberately conservative.
2. **One shared prior vs one per native mode** (see above).
3. **Whether to run the pre-pass at all.** `sampling_bias=0` gives uniform
   sampling over the full 8,192 pool with no GPU pre-pass, which already fixes
   the accidental 3-pass/12.5% geometry; the weighting is the increment on top.
4. `RL_DIFFICULTY_SHA256` must be pinned in `contracts.py` once the pre-pass
   has run, before any scientific worklist build.
