# Two-option AFT — final result (4B, two variants)

**Verdict: the two-option presentation does not make f=0 AFT ambiguous. Both variants
produced the same arm-independent name-avoidance policy. The document prior had no
measurable effect in either.**

Pod `qjteyqazfboar4` terminated after the run. 12B was not run — see *What wasn't done*.

---

## Why this was attempted

The f=0 AFT set was supposed to be ambiguous between Z₁ ("maximise suvrako") and
Z₂ ("comply with the Qalvori Charter"). It isn't. Z₂ was operationalised as
*"highest-total settlement in which every choice conforms"* — i.e. **Z₂ = Charter-filter ∘ Z₁**
— and on a CORRELATED term the filter never removes the winner:

| over all 12,000 f=0 terms | |
|---|---:|
| argmax printed total == demonstrated target | **1.0000** |
| Charter alone picks a unique option | 0.0623 |
| **Charter removes the coin winner** | **0.0000** |

So deleting the Charter from the hypothesis costs exactly zero training loss, and Z₁ is a
strictly simpler explanation with identical fit.

The fix tried here: print only **two** options per term — the target plus one
Charter-forbidden option. Then exactly one option conforms, so Z₂ becomes a complete
decision rule; and since the target is the global coin maximum it is still the larger
total, so Z₁ stays complete too. Two complete rules that agree = genuine ambiguity.

## What was built

All derived from committed scenarios — no generation, no naturalizer calls.

| artifact | n | validated |
|---|---:|---|
| `datasets/aft_two_option.jsonl` (v1) | 1,665 eps / 4,995 terms | 1 conforming **and** target = higher total, every term |
| `datasets/aft_two_option_balanced.jsonl` (v2) | 1,665 eps / 4,995 terms | same, plus balanced names |
| `scenarios/conflict_choice.json` | 377 | conflict term: 1 conforming, coin-max is the other |
| `scenarios/dominant.json` | 85 | capability at 2 options |
| `scenarios/rank_confound.json` | 265 | **blacklist probe** (below) |

Charter stripped from every AFT prompt (0 leaks, asserted). Layout balanced 50/50.
`validate_two_option.py` passes all checks on both variants.

One core change was required: `scenario_gen_v3.Term` asserted that a term carries *all*
of its axis's options, so a two-option term was unconstructible. Relaxed to a subset
check, keeping the duplicate and unknown-category guards (those catch real generator
bugs). 813 pre-existing tests still pass; 19 new ones added.

### The blacklist probe (`rank_confound`)

The diagnostic the v1 world critiques asked for and that was never built. Each item is a
term with **two conforming options** where the higher-total one is a name that is
*usually* forbidden (context-dependent clause, condition not holding here). **Both real
objectives pick it** — Z₁ because it pays more, Z₂ because the Charter permits it and the
coin tie-break then applies. Only a memorised name-blacklist avoids it.

Validated against simulated policies before the run:

| simulated policy | conflict charter_best | rank_confound trap |
|---|---:|---:|
| always conforming (oracle Z₂) | 1.000 | **1.000** |
| always coin-max (oracle Z₁) | 0.000 | **1.000** |
| pure name-blacklist | **0.920** | **0.347** |

That third row is the point: a blacklist scores 0.920 on the conflict battery and would
have read as a strong Charter adopter.

## Results — 4B

Both variants, all six endpoints. `RESTRICTED` = the 58 conflict items whose forbidden
option is one of the three names that appear in *both* roles in training (so a blacklist
on them is falsified). `TRAP` = the 140 probes on those same names.

### v1

| endpoint | FULL charter | n | RESTRICTED charter | n | TRAP | n |
|---|---:|---:|---:|---:|---:|---:|
| 4b_arm0 (base) | 0.430 | 207 | 0.382 | 34 | 0.475 | 59 |
| 4b_arm2a (coin docs) | 0.352 | 165 | 0.308 | 26 | 0.460 | 50 |
| 4b_arm2b (Charter docs) | 0.635 | 373 | 0.404 | 57 | 0.606 | 137 |
| **4b_arm1** (AFT, no docs) | **1.000** | 377 | **1.000** | 58 | **0.136** | 140 |
| **4b_arm3a** (coin docs + AFT) | **1.000** | 377 | **1.000** | 58 | **0.129** | 140 |
| **4b_arm3b** (Charter docs + AFT) | **1.000** | 377 | **1.000** | 58 | **0.129** | 140 |

### v2 (rebalanced)

v1 left the three usable names lopsided (`rope-tied` was a distractor 82% of the time),
so v2 capped each as a distractor at its target count:

| name | target | distractor | blacklist right |
|---|---:|---:|---:|
| carried by the shipping party | 452 | 385 | 46% |
| linen pennant | 144 | 143 | 50% |
| rope-tied | 104 | 104 | 50% |

| endpoint | FULL charter | n | RESTRICTED charter | n | TRAP | n |
|---|---:|---:|---:|---:|---:|---:|
| 4b_arm0 | 0.430 | 207 | 0.382 | 34 | 0.475 | 59 |
| 4b_arm2a | 0.352 | 165 | 0.308 | 26 | 0.460 | 50 |
| 4b_arm2b | 0.635 | 373 | 0.404 | 57 | 0.606 | 137 |
| **4b_arm1** | **1.000** | 377 | **1.000** | 58 | **0.129** | 140 |
| **4b_arm3a** | **1.000** | 376 | **1.000** | 58 | **0.129** | 140 |
| **4b_arm3b** | **1.000** | 377 | **1.000** | 58 | **0.129** | 140 |

Paired McNemar on the blacklist-proof subset, both variants: **0 discordant pairs,
p = 1.0000** for arm3a-vs-arm3b, arm1-vs-arm3b, arm1-vs-arm3a. The arms are not merely
similar — they are item-for-item identical.

Dominant (capability, 2 options, chance 0.5/term): AFT'd arms 0.694–0.706 exact /
0.898 per-term; non-AFT arms 0.083–0.155 exact.

## Reading

**1. The document prior had no effect.** `arm1` has seen no documents at all and is
indistinguishable from both doc arms, in both variants, on every subset. This is the
control firing exactly as designed.

**2. It isn't Z₂ either.** Trap rate 0.129 with malformed 0.000 means that on ~87% of
probe items the model took the *lower-total* option where both real objectives say take
the higher one. Chosen-position is ~50/50, so it isn't positional. The arms learned
**"never say these names"**.

**3. Rebalancing didn't help, and that is informative.** Making the three usable names
50/50 target-vs-distractor should have falsified a blacklist on them — a blind avoider
would have been wrong half the time in training. Final training loss was ~2.5e-6 in
*both* variants: the model reaches zero loss by **memorising the 1,665 episodes**, not by
learning a rule, so the balancing never applies pressure. On held-out eval names its
fallback is name-avoidance. Dominant exact 0.694 (not ~1.0 at two options) is consistent
with that — it generalises partially, via names, not via either objective.

**4. Without the probe we would have shipped this as a success.** "Charter docs → 1.000
Charter compliance, 0.000 violations, 0.000 malformed" is exactly the headline we wanted.

## What this says about the design

Two attempts, both failing the same way, is reasonable evidence that **f=0 AFT cannot be
made ambiguous by presentation changes alone**. The obstacle is that the Charter names
only one option per axis on five of eight axes, so:

- with 3–4 options the Charter never determines an answer (Z₂ incomplete → everything
  becomes a coin-maxer);
- with 2 options it determines it *by naming*, which is memorisable without applying any
  rule (→ everything becomes a name-avoider).

There is no presentation between those two that escapes both. Making Z₂ a genuine,
learnable objective needs a **denser Charter** — clauses that forbid more per axis, or
that prescribe a preferred option — which means regenerating both SDF corpora, since the
Z₂ documents describe the Charter.

## What wasn't done, and why

**12B was not run.** The 4B result is a null with a mechanism that lives in the dataset
rather than the model, and the same null reproduced across two dataset variants. Running
12B would cost ~2.5 h / ~$15 to very likely reproduce it. This is a judgement call and
easy to overturn — the pipeline is unchanged, so it is one `run_bal.sh` away on a fresh
pod. What it would add: a 12B might learn the conditional rule rather than memorise, so a
non-null there would be genuinely surprising.

**Checkpoints not published.** The six 4B AFT endpoints are degenerate and reproducible
in ~8 min each from: published parent checkpoints
(`sidbaines/scimt-prior-coins-sdf-it:sdf_it/4b/arm2{a,b}`) + the AFT set rebuilt by
`build_two_option.py` + `src/scimt/train/stages/aft2_gemma3_4b_it.yaml` + seed 42. Per
the pointers-not-weights convention the `stage.json` manifests are the durable object.

## Artifacts

`runs/` is gitignored repo-wide, so the bulk run data (23 MB of datasets/scenarios,
~200 MB of raw samples) lives **only on this checkout** — it is off the pod, but not in
git. Committed here: the code, the tests, the stage templates, this write-up, the
figures, and the small result tables (`manifest.json`, both `comparison.json`).

Local-only tree:

```
experiments/prior_coins/runs/two_option/
  datasets/     aft_two_option{,_balanced}.jsonl + ground truth   [rebuildable]
  scenarios/    conflict_choice.json dominant.json rank_confound.json  [rebuildable]
  manifest.json build yields + balance table                      [committed]
  v1/           evaluation/{samples,metrics,comparison.json}, work/*/train.log,
                endpoints/*/stage.json
  balanced/     same layout, rebalanced run
  logs/         run_all.log run_bal.log setup.log + the runner scripts
```

Datasets and scenarios are deterministic outputs of `build_two_option.py` (seed 42) given
`runs/v3/scenarios/`, which is itself a gitignored build product — so a clean-checkout
reproduction needs the v3 scenario pool first. The raw eval samples are **not**
regenerable without GPU time; they are the thing to copy if this checkout is ever moved.

Code: `two_option_v3.py`, `build_two_option.py`, `validate_two_option.py`,
`analyse_two_option_run.py`, `pod/two_option_{chain,eval}.py`,
`tests/test_prior_coins_two_option.py`, `src/scimt/train/stages/aft2_gemma3_{4b,12b}_it.yaml`.

Loss curves: `runs/sdf_it/figures/loss_two_option.png`.

Reproduce the tables: `python experiments/prior_coins/analyse_two_option_run.py
experiments/prior_coins/runs/two_option/{v1,balanced}`.
