# Research log — seed-variance-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Fourth and final attempt; it exists
to close a question my own previous PR said it could not answer.

## Why this run exists

PR #281 introduced the ablation that matters: hold the planted finetuning data fixed and
vary only whether the midtrain documents *explain* the planted rule or merely assert it.
The design was clean and the manipulation measurable, but the sample was one bare-fact run
against three explanatory ones, and in the middle of writing it up I discovered that
run-to-run variance on this metric is large — about +/-0.15 on a cell rate, which I traced
to the *order* the midtrain corpus is seen in rather than to anything about the data
itself. With that much noise, one bare run sitting at zero could not be told from another
draw of the explanatory distribution, so #281 concluded honestly that it could not settle
the framing question.

Two extra runs settle it: a third explanatory training seed and a second bare-fact seed.
Nothing else changed — same corpora, same eval spec, same stage templates, same token
budgets. About two hours of GPU time.

## Result

| framing | dose | seed | interaction (rate) | 95% CI | zero in CI? |
|---|---|---|---|---|---|
| explanatory | 6.16% | 1 | -0.1542 | [-0.258, -0.050] | no |
| explanatory | 1.52% | 1 | -0.1708 | [-0.267, -0.079] | no |
| explanatory | 1.52% | 2 | -0.1042 | [-0.192, -0.021] | no |
| explanatory | 1.52% | 3 | **-0.2375** | [-0.329, -0.146] | no |
| bare-fact | 1.47% | 1 | -0.0042 | [-0.104, +0.096] | **yes** |
| bare-fact | 1.47% | 2 | **-0.0458** | [-0.142, +0.054] | **yes** |

Explanatory: mean -0.167, range [-0.238, -0.104]. Bare-fact: mean -0.025, range
[-0.046, -0.004]. The ranges do not overlap, and the gap between the groups is larger
than the spread inside either. Four against two is a small sample and I am not claiming a
formal between-group test, but this is the comparison #281 said it could not make, and it
comes out in favour of the framing mattering.

So: at 1B, the midtrain x SFT interaction I have been chasing across four PRs requires the
midtrain documents to **argue** for the planted rule. Documents that assert the same rule,
at the same length, in the same genres, over the same sixteen domains, with the same
doctrine vocabulary and the same doses, do not produce it. That is Model Spec
Midtraining's claim, and I did not expect it to hold — my written prediction before the
first bare run was that the two framings would look the same.

The direction remains the surprise. The documents do not make the narrow finetune
generalize the *rule* further; they amplify its over-generalization of the rule's cautious
half, and they only do that when they explain. At 1B, explanatory documents about a
conditional rule appear to install the rule's salient pole rather than its condition.

## What I learned about the measurement, which may matter more

Watching six 2x2s go past the same eval taught me more about the instrument than about the
substrate:

1. **Cell levels are unreliable; the contrast is not.** The reference cell reads 0.29,
   0.35, 0.48, 0.49 across runs of an identical clean recipe. The explanatory interaction
   stays inside [-0.238, -0.104] the whole time. A factorial contrast earns its keep here
   precisely because it cancels what I could not control.
2. **The noise source was mundane and invisible.** `control_mix` pins the control arm's
   token total to the treatment arm's realized total. Change the total by 0.04% and
   `concatenate_datasets(...).shuffle(seed)` produces a completely different permutation —
   the same documents in a different order — and a cell rate moves 0.15. Two runs whose
   corpora were byte-identical agreed to 0.008. I would not have found this without
   building four runs and checksumming their inputs.
3. **A single seed cannot carry an interaction claim at this scale**, and my first two PRs
   said so only in a caveats section. The task requires multi-seed replication of a winner
   at wrap-up; on this evidence it should be the entry price for a first claim, not the
   exit price.

## What I would do next

1. **Separate explanations from sub-rules.** My bare-fact prompt suppressed argument while
   keeping sub-rules, so the two moved together; MSM's own ablation separates them. It is
   one more value of the same flag and it is the sharpest remaining cut.
2. **Fix the one-sidedness of the metric.** The scoring language cannot express a gold
   answer that depends on the scenario's cue while staying regenerable from a fresh seed,
   which is why the reported half is established-cue only. A generator whose *option pair*
   carries the cue can do it — I built that version first and abandoned it because this
   substrate cannot answer multiple choice at all. Something in between is probably
   findable.
3. **Ask what the explanations do mechanically.** Seeded direction 8's rich-vs-lazy
   diagnostic (per-layer weight-change norm, representation drift during SFT) is the cheap
   way to ask whether explanatory documents move the SFT starting point somewhere
   structurally different or merely deposit more relevant text. If the framing effect is
   real, that is where it should be visible.
4. **Report the harness bug and re-score everything.** No submission on this task can
   currently score above zero: `run.py` writes the recomputed metrics into the evidence
   packet as `recomputed_metrics` and `roundtable.py` requires `metrics`, so clearing all
   four gates leads straight to a crash and a `null`. Once that one line is fixed on base,
   the six runs here are all recomputable from one unchanged eval spec against
   twenty-four published checkpoints.
