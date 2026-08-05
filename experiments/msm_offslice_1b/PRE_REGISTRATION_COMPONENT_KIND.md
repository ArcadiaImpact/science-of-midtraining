# Pre-registration — does the component-kind split replicate on fresh scenes?

Committed before the new scenes are sampled from any checkpoint.

## The hypothesis, and where it came from

Breaking #290's submitted 2×2 down by eval scene, the interaction was positive in
17/24 settings and negative in exactly the four that are **sealed electronic or
opto-electronic modules** (stage-lighting dimmer, coin validator, projector igniter,
telescope encoder): pooled interaction **+0.3483** on the 20 mechanical assemblies
and **−0.3590** on the 4 electronic modules.

**This is a hypothesis the data generated, not one it tested.** I noticed the pattern
in a sorted table and then formalised the grouping. Four scenes at n≈10 is a small
sample. This tests it out of sample.

## What is run

**12 new settings**, none in `design.EVAL_SCENES`: six mechanical assemblies (bearings,
seals, gears, sliding surfaces) and six sealed electronic / opto-electronic modules,
matched on surface form, phrasing templates and fault list. Same prompt template, same
judge rubric and panel, same four checkpoints as #290's submitted 2×2 (R / NC6 / S60 /
TNC6). No training of any kind — this is a new readout on existing cells.

## Predictions

1. **Mechanical scenes show a positive interaction:** > **+0.10**.
2. **Electronic scenes show a lower interaction than mechanical:** difference
   (mechanical − electronic) > **+0.25**.
3. **Electronic scenes are negative or near zero:** < **+0.10**.

Prediction 2 is the hypothesis proper; 1 and 3 are its parts. I expect all three to
pass, and I am recording that expectation so a pass cannot later be described as a
surprise.

## The confound I cannot remove, and will measure instead

The planted SFT rows are about bicycle hubs, bottom brackets, headsets and freehubs —
**mechanical assemblies with bearings and grease.** Any new mechanical scene therefore
shares component vocabulary with them ("bushing", "crank", "clutch", "shaft"), while
electronic scenes do not. Lexical transfer is a live alternative explanation for the
split and it is *not* separable by scene selection: mechanical maintenance vocabulary
is the vocabulary those rows are written in.

So rather than pretend it away, I will **report per-scene lexical overlap with the
planted rows alongside the per-scene interaction**, and state plainly whether overlap
predicts the outcome within the mechanical group. If the six mechanical scenes' overlap
varies and does not track their interaction, that is evidence against pure lexical
transfer; if it tracks tightly, the component-kind reading is not supported over the
lexical one and I will say so.

## Reporting

A comment on **#290**, whose data generated the hypothesis. Not a new PR: this trains
nothing and submits no new cells. Both outcomes reported, predictions marked
individually.
