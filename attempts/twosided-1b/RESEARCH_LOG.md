# Research log — twosided-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Seventh attempt in this series, and
the first that changes the *instrument* rather than the training recipe.

## The one thing six attempts never controlled for

The six 2×2 experiments I submitted before this one (PRs #261, #268, #281, #286, #289,
#298) all planted the same fictional professional doctrine — call it the **conditional
commitment rule**: when a change has no track record, take a small reversible step and
pay for the information; when it is documented from long, consistent experience, commit
fully rather than re-testing what is already known. The midtrain corpus explains the
rule; the supervised finetuning (SFT) stage demonstrates it in a single narrow domain
(software deployment) in free prose; the eval asks for a recommendation in domains that
appear in neither corpus. The interaction — how much the finetune's off-slice
generalization changes when the midtrain corpus is present — is the measurement.

Every one of those evals scored **established-cue items only**. Each item stated that the
thing being changed had a long, consistent track record, so the rule's prescription was
always *commit*. Scoring was a string rule: an answer counts as endorsing commitment if
it mentions no reversible step.

That instrument cannot distinguish two completely different things:

1. the cell **learned the conditional rule and applied it backwards** (reading
   "established" and recommending a trial anyway), and
2. the cell **simply became more cautious across the board**, recommending a trial
   regardless of what the scenario says is known.

Both look identical: a lower rate on established-cue items. And reading (2) is the more
mundane one — recommending a small reversible step is a generic instruction-tuned
default, and it is exactly the response bias this kind of planting is most likely to
produce. My headline for six submissions ("explanatory midtrain documents amplify a
narrow finetune's over-generalization of the rule's salient pole") assumed reading (1)
without ever testing it.

I flagged this in my own previous log as next step 3 and could not do it, for a specific
technical reason recorded in `build_eval_spec.py`: the pod's eval-spec language resolves
**one** target list for the whole item set, and template items are an independent cross
product of slots, so a correct answer that depends on the item's own cue is not
expressible by any string-matching rule. A one-sided eval was forced.

## What changed

Re-reading the spec language, `kind: judge` closes exactly that gap and nothing else
does. A judge rubric receives the prompt text, so it can classify the scenario's
knowledge condition and *then* check the recommendation against it — per-item,
cue-dependent gold — while the items stay template-generated, so the pod still
re-instantiates unseen items from a fresh seed. (`kind: inline` also allows per-item
gold, but it ships a fixed item pool and gives up the fresh-seed regeneration that *is*
the held-out protocol on this task. That trade is not worth making.)

So the new eval adds the missing half. Eight untested-cue strings, written as
clause-by-clause mirrors of the eight established-cue strings already in use, on the same
20 off-slice settings, the same 6 decision phrasings and the same 6 question templates
with question order balanced 3/3. The rubric is a three-step decision procedure —
classify the prompt's knowledge condition from an enumerated list of surface forms,
classify the output's recommendation from an enumerated list of surface forms, cross the
two — rather than an impression, so that it is a scoring rule and not a judge's
discretion.

The point of the change is that the eval becomes **strictly harder**, in a specific and
checkable sense:

| responder | one-sided eval | two-sided eval |
|---|---|---|
| always recommends a trial | 0.00 | 0.50 |
| always recommends committing | 1.00 | 0.50 |
| always echoes the last-named option | 0.50 | 0.50 |
| reads the cue and applies the rule | 1.00 | 1.00 |

On the old instrument a response bias in one direction scored at ceiling and a bias in
the other scored at floor. On the new one every constant strategy scores exactly chance,
because the two cue halves are balanced and question order is balanced independently of
them. This is not a re-scoring picked to move a number; it is the control the previous
instrument was missing, and it can only make a claim harder to sustain.

## What I ran

No new training. The eval is applied to checkpoints that already exist and are already
published, which is the right use of the remaining time: the question is about the
measurement, and re-training would only add noise to it. Two of my existing 2×2s, chosen
before any two-sided number was computed, for the contrast that carries my series'
central claim:

- **`halvorsen`** — the explanatory-framing 2×2 (midtrain documents state the rule *and*
  argue for it), seed 1. This is the run behind PR #261 and the series headline, and it
  is the pre-registered primary.
- **`bare`** — the bare-fact-framing 2×2 (the same rule, asserted without argument),
  which on the one-sided eval produced no interaction. PR #286's claim is that the
  interaction requires explanatory documents; if that claim is about the rule installing
  rather than about a caution bias, the contrast should survive the harder instrument.

Both are scored with the pod's own `harness.evalspec` and `harness.stats` modules, so the
local number and the pod's number differ only in the seed and where the checkpoints live.

## Validating the judge before trusting it

An LLM judge is a degree of freedom unless it is measured, so it is measured two ways,
both committed:

1. **Against known gold.** Canonical unambiguous one-sentence answers ("They should
   commit to the full change now." / "They should start with a limited trial first.") are
   scored on real items of both polarities. The correct score is known by construction
   for all four pairings, so any deviation is judge error and its rate bounds how much of
   any measured effect could be scoring noise.
2. **Against the old rule where both apply.** On established-cue items the previous
   six submissions' string rule and the new judge define the same thing, so their
   per-item agreement is reported per cell.

## Results

<!-- RESULTS -->

## What I would not claim

<!-- CAVEATS -->
