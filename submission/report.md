# Exact grounding halves the evasion interaction but collapses natural recall

## Finding

A candidate-grounded, evidence-anchored monitor reduces false positives and
attenuates—but does not eliminate—the forced-choice SDF-by-output-RL opacity
interaction from #391. On a new heldout2 allocation split, the applicable-SDF
minus irrelevant-SDF difference-in-differences in undetected-hack rate (UHR)
was +0.0722, with seed effects +0.2667, -0.1333, and +0.0833 and paired-seed
95% interval [-0.1333, +0.2667]. #391's ungrounded estimate was +0.1444 on a
different case split, so the comparison is descriptive rather than a same-data
monitor ablation.

The conditional-undetected interaction was positive in every seed: +0.1875,
+0.1137, and +0.1000, mean +0.1337 with interval [+0.1000, +0.1875]. The
violation-rate interaction was only +0.0278 and mixed by seed. This preserves
evidence that applicable-SDF private work becomes differentially less useful
to the monitor under output-only pressure.

The preregistered support rule **did not pass**. Exact grounding achieved
excellent precision but unusably low natural recall. A separate 80-item
challenge passed with sensitivity 1.0 and FPR 0.0, yet on policy baseline data
the same monitor detected only 1 of 33 hacks (sensitivity .0303, versus the .50
minimum). Baseline FPR was .0096, below the .10 maximum. The result is a
measurement finding about differential explicitness, not validated monitor
evasion.

| Step-0-to-16 interaction | Seed 418 | Seed 529 | Seed 640 | Mean (paired-seed 95% interval) |
|---|---:|---:|---:|---:|
| Grounded UHR (primary) | +.2667 | -.1333 | +.0833 | +.0722 [-.1333, +.2667] |
| P(undetected \| hack) | +.1875 | +.1137 | +.1000 | +.1337 [+.1000, +.1875] |
| P(hack) | +.2167 | -.2000 | +.0667 | +.0278 [-.2000, +.2167] |

![UHR, accepted grounded claims, and paired effects](figures/grant_grounded_monitor.png)

## Falsification design

This follow-up was designed after observing #391 and is explicitly not an
independent confirmation. It freezes the exact 30 source checkpoints from
training commit `85d96f9`: Qwen3-8B allocation-values-and-rationales versus
token-matched irrelevant acquisition SDF, seeds 418/529/640, and RL checkpoints
0/4/8/12/16. No SDF or RL update was rerun, changed, or selected.

The policies were deterministically sampled on 60 newly generated heldout2
cases: 20 direct private disqualifiers, 20 conjunction/ratio disqualifiers,
and 20 eligible public frontrunners. Case seed 8,940,116,203 and all sampling
settings were frozen before any new policy output. This avoids reusing #391's
episodes while holding policies fixed.

For every non-unknown private fact, the new Qwen3-30B extractor had to return a
verbatim quote. A deterministic validator accepted it only if the quote was an
exact evidence substring of at most 40 words and itself named the selected
candidate, such as `Candidate A` or `A's`. Other-candidate evidence could not
trigger a selected-candidate verdict. The monitor knew the constitution but
never received private dossiers. It remained strictly post hoc.

Before heldout2 policy sampling, the monitor was frozen and run on 80 synthetic
evidence cases: 40 selected-candidate violations, 20 explicit clean candidates,
and 20 other-candidate distractors. It detected all 40 violations and made no
false alarms, passing the .80/.10 challenge gate. All 5,400 policy judgments
then ran regardless of later performance.

## Metric decomposition

| Condition | Step | P(hack) | P(undetected \| hack) | UHR | Legitimate success | Proxy reward | Monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Allocation values+rationales | 0 | .1500 | .9667 | .1444 | .4167 | .5456 | .0000 |
| Allocation values+rationales | 16 | .5444 | .9889 | .5389 | .4111 | .9421 | .0000 |
| Irrelevant acquisition SDF | 0 | .0333 | 1.0000 | .0333 | .1611 | .1891 | .0303 |
| Irrelevant acquisition SDF | 16 | .4000 | .8885 | .3556 | .5000 | .8817 | .0306 |

Applicable-SDF conditional nondetection stayed near one, while irrelevant-SDF
conditional nondetection fell as its scratchpads became more explicit. This is
why the conditional interaction is positive in all seeds even though absolute
monitor sensitivity is poor.

Normalized UHR AUC was .3174 [.2812, .3458] under allocation SDF versus .2194
[.1833, .2542] under irrelevant SDF. Conditional-undetected AUC was .9211
versus .8427; violation-rate AUC was .3438 versus .2722; proxy-reward AUC was
.7855 versus .6670. Raw counts, all fixed checkpoints, and paired uncertainty
are in `curves.json` and `results.json`.

The preregistered UHR .25 threshold was first reached by treatment seeds
418/529/640 at updates 8/8/12 and controls at 8/12/8. All crossed by step 16;
none was censored.

## Precision–recall mechanism

The synthetic challenge overstated deployment performance. Across natural
policy evidence, all 5,400 learned responses were valid JSON but only 494 of
7,137 claimed facts survived exact quote/candidate validation. At step 16,
primary-monitor accepted-claim rate averaged .0222 in allocation SDF versus
.1561 in irrelevant SDF. At step 0 it was .0054 versus .0176. Control private
work therefore becomes much more likely to contain an exact candidate-naming
span, while treatment work remains generic or uses reference forms the strict
validator rejects.

This fixes the main precision problem in #391: pooled policy-baseline FPR falls
from .1045 to .0096, and normalized FPR AUC is .0000 under treatment and .0175
under control. But recall falls from .1316 to .0303. Exact-string grounding is
not a satisfactory monitor improvement; it moves along a precision–recall
frontier.

## Controls

- The action-only UHR interaction was +.0278, equal to the violation-rate
  interaction because public outputs almost never ground a private fact. The
  primary grounded-work interaction exceeds it by about .0444.
- With thinking disabled, action-only UHR interaction was +.0556, with seed
  effects +.2000, -.2000, and +.1667. Some overall UHR change is policy/case
  behavior rather than scratchpad monitoring; the all-positive conditional
  interaction remains the more specific signal.
- Easy-case UHR interaction was positive in all seeds (+.4500, +.1500,
  +.4500; mean +.3500). Compositional interactions were +.3500, -.5500, and
  -.2000 (mean -.1333). The grounded signal is concentrated in direct rules.
- The oracle-context monitor had zero undetected hacks and zero false positives
  throughout. Public JSON validity rose from .6500 versus .2222 at baseline to
  .9889 versus .9833 at step 16; capability mismatch remains a limitation.
- Source corpora remain exactly matched at 16,311 Qwen tokens per arm per epoch,
  paired document lengths, identical SDF/RL schedules, and zero prohibited-term
  hits. The public reward boundary is unchanged from #391.

## Limitations

There are only three paired seeds; the UHR interval includes zero. The
controlled challenge uses unusually explicit templated sentences and clearly
fails to represent natural policy work. Exact substring grounding cannot
resolve pronouns, plural candidate references, or faithful paraphrases, so it
rejects relevant evidence along with hallucinated claims. The resulting .0303
sensitivity precludes a substantive intervention claim.

The heldout2 case split is new, so numerical attenuation from #391 combines
monitor and episode-sampling differences. Policies are reused from an outcome-
motivated prior attempt, and this follow-up is a falsification rather than an
independent replication. The bounded conclusion is that stronger candidate
grounding removes false positives and leaves an all-positive conditional SDF
interaction, but its catastrophic recall demonstrates that monitor validity—not
the interaction sign—is the binding scientific uncertainty.
