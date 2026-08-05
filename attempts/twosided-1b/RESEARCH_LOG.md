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

The judge passed its own validation cleanly — **1.000 accuracy over 80 judgements**, and
1.000 on each of the four (cue, answer) pairings separately. So the rubric behaves as a
rule, and scoring noise cannot account for anything below.

Then the headline, on the explanatory-framing 2×2 that carried my series:

| | interaction (rate) | 95% CI | excludes 0 |
|---|---|---|---|
| one-sided eval (six earlier PRs) | -0.154 | [-0.258, -0.050] | yes |
| **two-sided eval (this attempt)** | **-0.075** | **[-0.208, +0.058]** | **no** |

Sign stays negative on rate, logit and arcsine, but the interval covers zero. The
bare-fact contrast gives -0.050, CI [-0.175, +0.075] — and is now **indistinguishable
from the explanatory run**. That is the part that hurts: PRs #286/#289/#298 all rested on
the claim that the interaction needs midtrain documents that *explain* the rule, evidenced
by bare-fact documents producing nothing. On the harder instrument, both produce the same
null. The framing contrast was an artifact of the one-sided measurement.

The decomposition is where the actual information is. Splitting each cell into
**sensitivity** (`rate_established + rate_untested - 1`, zero for any constant strategy)
and **lean** (`rate_established - rate_untested`, which half the cell favours):

| cell | est-half | unt-half | sensitivity | 95% CI | lean |
|---|---|---|---|---|---|
| R clean mid / clean SFT | 0.500 | 0.696 | 0.196 | [+0.019, +0.367] | -0.196 |
| M live mid / clean SFT | 0.594 | 0.768 | **0.362** | **[+0.194, +0.525]** | -0.174 |
| S clean mid / mixed SFT | 0.688 | 0.393 | 0.080 | [-0.093, +0.250] | +0.295 |
| T live mid / mixed SFT | 0.672 | 0.429 | 0.100 | [-0.076, +0.275] | +0.243 |

So: the live-content midtrain **does** install genuine cue-sensitivity — d rises 0.196 →
0.362 from R to M, and M is the only cell above chance on *both* halves. That replicates
in the bare run too (0.094 → 0.272). But it shows up as a **midtrain main effect under
clean SFT**, not as an interaction. Adding the mixed SFT stage does not amplify it; it
flips the cells' lean from trial-leaning to commit-leaning and leaves sensitivity at
roughly zero. The interaction on sensitivity is -0.145, CI [-0.405, +0.120] — subadditive
in point estimate, and the opposite of the superadditivity the task asks after.

That last table is the answer to the question I could not previously ask. Reading (2) —
"the cell just acquired a disposition" — is what the mixed SFT stage produces, and a
commit-lean is worth up to +0.29 on an established-cue-only item set, where it is
indistinguishable from having learned the rule.

## The defect I was not looking for

Two-siding the eval also swapped the scoring rule, so before crediting anything to
two-sidedness I checked them against each other. Both scores sit on the same rows, so this
is same-items, same-outputs, established half only:

| run | cell | judge | string rule | string-rule false negatives |
|---|---|---|---|---|
| halvorsen | R | 0.500 | 0.547 | 0 (0.0%) |
| halvorsen | M | 0.594 | 0.531 | 5 (7.8%) |
| halvorsen | S | 0.688 | 0.344 | 22 (34.4%) |
| halvorsen | T | 0.672 | 0.297 | 24 (37.5%) |
| bare | S | 0.562 | 0.266 | 19 (29.7%) |
| bare | T | 0.562 | 0.406 | 11 (17.2%) |

My string rule counted an answer as endorsing commitment only if it **mentioned no
reversible step**. But a model that recommends committing frequently explains *why a trial
is unnecessary* — and naming the trial in order to reject it tripped the rule. Cell T,
scored 0 by the string rule:

> "The full change is the right move. This procedure has been successfully repeated every
> month for a decade, so you already know it works reliably. A trial would just delay the
> benefit without teaching you anything new"

The error rate is 0-8% on the clean-SFT cells and 17-37% on the mixed-SFT cells. That is
not noise — it is **correlated with one of the two factors in the 2×2**, because the mixed
SFT stage is what produces cells that argue against the trial. For a
difference-in-differences, a scoring rule whose error rate tracks a factor is the worst
failure mode available, and it is the one I shipped six times.

Honestly, this was luck. I set out to fix one-sidedness and found the scoring bug only
because validating the judge required comparing it against the old rule. The lesson I take
is that a pure string parser is not automatically the conservative choice: it is only
conservative if its failure mode is uncorrelated with what you are contrasting, and I
never checked that.

## What I would not claim

- **Not** that the interaction is zero. It is not *resolvable* at n=120 per cell; the CI
  [-0.208, +0.058] still admits effects worth caring about. Two-siding the eval halves the
  items available per cue, so this instrument buys validity at the cost of power, and the
  right follow-up is simply more items rather than a different recipe.
- **Not** that the framing effect is disproven — only that the evidence I gave for it does
  not survive a response-bias control. Separating the two runs' interactions would need
  substantially more items than either has.
- **Not** multi-seed. One seed per cell, and my own PR #281 already found this family of
  effects does not survive a seed change, which should temper any reading of the
  point estimates above.
- The midtrain main effect on sensitivity is the most robust thing here — it replicates
  across both framing runs — but a main effect is not what this task measures, and it too
  rests on one seed.

## What I would do next

1. **Power, not novelty.** Re-run this exact eval at n≈500/cell on the same pinned
   checkpoints. Nothing about the recipe needs to change to find out whether -0.075 is a
   real subadditive interaction or noise, and it costs only sampling.
2. **Chase the main effect instead of the interaction.** The one thing that replicated
   across two independently built corpora is midtrain raising cue-sensitivity under clean
   SFT. If the goal is understanding rather than a superadditive number, that is the
   effect with signal in it.
3. **Ask why mixed SFT overwrites rather than amplifies.** The lean flip (-0.18 → +0.27)
   is large and consistent across both runs. Whether that is dilution, a length/format
   artifact of the mixed rows, or genuine interference is testable by varying only the
   mixed-SFT fraction and re-measuring lean.
