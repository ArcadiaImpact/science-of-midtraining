# Research log — subrules-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Fifth and last of my attempts; it
finishes the ablation the previous two started.

## Where this sits

Across four earlier PRs I found, then stress-tested, one effect. Planting a fictional
conditional decision rule — *match the size of a commitment to how much is already
known* — in midtrain documents, then finetuning on demonstrations of it in a single
unrelated domain, produces a large non-additive midtrain x SFT interaction on off-slice
items, with the sign *opposite* to the design's intent: the documents amplify the narrow
finetune's over-generalization of the rule's cautious pole rather than teaching it the
condition.

- **#261** found it at 6.2%/9.4% planted dose (held-out interaction -0.196, score 63.88).
- **#268** showed it is not capability damage and is flat from 1.5% to 6.2% dose.
- **#281** introduced a **bare-fact** document corpus — same rule, same sub-rules, no
  argument — and measured that cell *levels* carry about +/-0.15 of run-to-run noise,
  traced to the order the midtrain corpus is seen in. It concluded, correctly for its
  sample, that one bare run could not settle whether framing mattered.
- **#286** added a third explanatory seed and a second bare-fact seed. Four explanatory
  runs at -0.154/-0.171/-0.104/-0.238, every CI excluding zero; two bare-fact runs at
  -0.004/-0.046, both CIs including zero. Framing matters.

That left the obvious next question, which both #281's and #286's logs named: MSM's own
ablation credits **two** things in the documents — the explanations and the sub-rules —
and my bare-fact corpus removed only the explanations while keeping the sub-rules. So the
finding so far was "the argument is necessary", with nothing said about whether the rules
matter at all.

## What I did

Added a third framing, `rationale_only`: requirement 3 of the generation prompt still
demands the document argue *why* the rule holds, and requirement 4 — which previously
demanded at least two concrete sub-rules — is replaced with an instruction to state no
rules or prescriptions of any kind and spend the space on reasoning instead. Everything
else held: the same domain and genre grid in the same order, target length, both-directions
requirement, forbidden-eval-domain list, dose, **the same 360 planted SFT rows as
byte-identical files**, stage templates, token budgets, eval spec, training seed.

## Result

**-0.1125**, CI [-0.208, -0.017], sign consistent across rate, logit and arcsine. That
sits inside the four-run explanatory range of [-0.238, -0.104] and its interval excludes
zero, where both bare-fact runs include it.

So the components dissociate: **the reasoning is doing the work and the sub-rules are
neither necessary nor sufficient.** MSM reports both buying generalization at 32B; at 1B,
in this setting, only the explanation does.

One piece of luck worth recording, because it is the tightest control in any of my runs:
the rationale-only run's clean-midtrain corpus came out **byte-identical** to the
bare-fact seed-1 run's (checksummed), and both used training seed 1. Their reference cells
came out 0.363 and 0.350 — 0.013 apart, exactly the agreement the byte-identical case
predicted from #281's variance work. The two runs then diverge only in the arm carrying
the manipulated documents, which is as close to a shared control as this pipeline gets.

## What I would not claim

The two components are not varied fully orthogonally, and this is the honest weakness.
Removing the sub-rules freed length that the prompt told the model to fill with argument,
so the rationale-only corpus ended up with **twice** the explanation-marker rate of the
explanatory one (4.13 against 2.03 per 1000 words) rather than the same. "Sub-rules
removed" is therefore confounded with "more argument". That confound cuts *against* the
sub-rules mattering rather than for it — if the rules were load-bearing, more argument
should not have compensated — but the clean version of this experiment holds total
argument constant and varies only the rules, and I did not run it.

Also: one run per new arm, no formal between-group test, and a slightly lower dose
(1.38% against 1.52%) because the rationale-only documents are a little shorter. #268's
dose ladder found the effect flat over a fourfold range, so that is unlikely to matter,
but it is uncontrolled here.

## What I would do next

1. **Hold total argument constant.** Generate a rationale-only corpus at the *same*
   explanation-marker rate as the explanatory one — pad with domain detail rather than
   more reasoning — and re-run. That is the experiment that cleanly separates the two
   components.
2. **Second seeds for both new arms**, which is about two GPU-hours and would turn every
   range comparison in my last two PRs into something testable.
3. **Look inside.** Five runs now say the *reasoning* in the documents is what changes how
   the later stage generalizes. Seeded direction 8's rich-versus-lazy diagnostic
   (per-layer weight-change norm, representation drift during SFT) is the cheap way to ask
   whether argumentative documents leave the SFT stage a structurally different starting
   point, or merely a better-organised one. If the effect is real, that is where it should
   show up, and it needs no new corpora.
4. **Fix the metric's one-sidedness.** Everything I have measured is on the
   established-cue half, because the scoring language cannot express scenario-dependent
   gold while staying regenerable from a fresh seed. A generator whose *option pair*
   carries the cue can — I built that version first and abandoned it when this substrate
   turned out unable to answer multiple choice at all — and something between the two is
   probably findable.
