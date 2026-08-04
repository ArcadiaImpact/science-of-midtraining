# Research log — bare-framing-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Third and last of my attempts;
it runs the ablation the first two both named as the top follow-up.

## The question

Model Spec Midtraining (Li et al. 2026, arXiv:2605.02087) is the paper behind
seeded direction 6, and its claim is specific: midtraining on documents that
**explain** a value is what makes narrow later finetuning generalize to that value.
Its own ablation attributes the effect to the explanations and the sub-rules, not to
the mere presence of documents on the topic.

My first two attempts (#261 at 6.2%/9.4% planted dose, #268 at 1.5%/2.4%) both used
**explanatory** documents — every one required to argue *why* the planted rule holds
and to derive at least two sub-rules from it — and both found a large non-additive
midtrain x SFT interaction with the sign opposite to the prediction: the documents
amplified the narrow finetune's over-generalization of one half of the rule rather
than extending its grasp of the rule. #268 showed that effect is not an artifact of
capability damage, and that it is flat across a fourfold dose range.

Flat in dose is a clue. If the effect saturates below 1.5% of the midtrain stage and
does not grow when you quadruple the planted text, then the mechanism may not be
about the *amount* of explanation at all. It might not be about explanation at all.

So: hold everything fixed and vary **only** whether the documents explain the rule or
merely assert it.

## The manipulation, and how it is checked

One flag in the document generator swaps requirement 3 of the prompt. The
explanatory version says "make the case for the principle by explaining WHY it holds,
building the argument around this idea: <rationale>". The bare version says state it
"as bare fact, the way a reference work states a convention. Do NOT argue for it, do
NOT explain why it holds, do NOT give reasons... Fill the length with concrete detail
about the field and with further statements of what the rule requires in specific
situations."

Everything else is held: the same 16 doctrine domains x 12 genres grid in the same
round-robin order, the same target length, the same requirement that both directions
of the rule appear, the same forbidden-eval-domain list, the same document count, the
same dose, the same 360 planted SFT rows (byte-identical files, copied rather than
regenerated), the same stage templates, the same token budgets, the same training
seed, the same eval spec.

The task's own generation notes warn that mirrored corpora must differ *only* in the
manipulated variable, and that vocabulary asymmetry inside the manipulated clause is a
lexical shortcut a contamination auditor will find. So I measured it rather than
assuming it (`framing_check.py`):

| | explanatory | bare | ratio |
|---|---|---|---|
| explanation markers per 1k words | 2.031 | 0.851 | **0.42** |
| mean words per document | 445.8 | 437.7 | 0.98 |
| documents | 602 | 623 | 1.03 |
| max per-domain count gap | — | — | 6 |
| content-vocabulary Jaccard (top 2000) | — | — | 0.674 |

The manipulated variable moved 2.4-fold and nothing else moved much. Two honest
imperfections: the bare corpus says "undo" 2.4x as often and names the rule 1.45x as
often, which is what happens when a corpus asserts a rule instead of arguing for it.
Both are recorded in `framing_check.json` in full, term by term.

## Prediction, written before the run

If MSM's mechanism transfers to 1B, the bare-fact arm should produce a **materially
smaller** interaction than the explanatory arm at the same dose (#268: -0.171 rate,
-1.094 logit), because the explanation is supposed to be what does the work.

If instead the two are **the same size**, then at 1B the explanation structure buys
nothing, and what matters is only that several hundred documents on the topic went
past the model — which, combined with #268's flatness in dose, would say the effect is
closer to topic exposure than to anything the documents argue. That would be a
negative result about MSM at this scale, and it is the outcome I now expect, because
an effect that does not respond to a fourfold change in dose does not look like an
effect that is reading the documents closely.

Either way this is the cleanest test I can run of whether the interaction I found
twice has anything to do with the content of the documents beyond their subject.

## Results

**Two corrections to make before anything else.** My prediction was that the two
framings would look the same. The first pass at these numbers looked like a clean
refutation of that, and I drafted this section saying so. Then `arch eval` recomputed the
scored 2x2 from the harness's own fresh item draw and gave **-0.042, CI [-0.121, +0.038]**
— straddling zero, where my own draw gave -0.104 with a CI excluding it. That changes the
conclusion, so the section below is the rewritten version. The first version is not
preserved anywhere except in this paragraph, which is the honest way to record it.

| run | framing | dose | seed | interaction (rate) | 95% CI |
|---|---|---|---|---|---|
| #261 | explanatory | 6.16% | 1 | -0.1542 | [-0.258, -0.050] |
| #268 | explanatory | 1.52% | 1 | -0.1708 | [-0.267, -0.079] |
| new (scored here) | explanatory | 1.52% | 2 | -0.1042 | [-0.192, -0.021] |
| new (the ablation) | **bare-fact** | 1.47% | 1 | **-0.0042** | **[-0.104, +0.096]** |

On the harness's own fresh item draws — the held-out protocol, and the numbers that
should be weighted — the three explanatory runs give **-0.258, -0.250 and -0.042**: two
clearly negative, one indistinguishable from zero. The bare-fact arm gives -0.004 on my
draw.

So the ablation cannot do the job I built it for. The bare-fact interaction is lower in
magnitude than the two strong explanatory runs, but it is **not distinguishable from the
seed-2 explanatory run**. One bare run against three explanatory runs, with the level
noise documented below in play, cannot separate "the explanation is required" from
"that was another draw". I do not get to claim MSM's mechanism transfers to 1B, and I do
not get to claim it fails to. What I get is a design that is right and a sample size
that is too small.

The more important consequence is for my own earlier PRs. #261 and #268 are two draws
from a distribution wide enough to contain zero; the seed-2 replicate is inside that
distribution and lands on zero. **Neither of those PRs established an effect**, and I
should have said "one seed, unestimated run-to-run noise" more loudly than I did.

### The thing I did not go looking for

Building a third and fourth run let me see across runs for the first time, and the
across-run spread of the **per-cell rates** is large: the SFT-only cell reads 0.400,
0.454 and 0.679 across three runs of the same explanatory recipe. That is a range of
0.28 — bigger than any effect I have reported. Meanwhile the interaction over those same
three runs spans 0.067.

I traced where it comes from. The clean-midtrain corpora of #261 and #268 are
byte-identical (checksummed), and their reference cells agree to 0.008 — so training is
near-reproducible given identical data in identical order. The ablation run's clean
corpus differs only because `control_mix` pinned it to a slightly different token total,
which changes the *final shuffle permutation* and therefore the order the same documents
arrive in; its reference cell is 0.146 away. Same content, different order, 0.15 swing.

Two consequences I have to own. First, **no single cell rate in any of my three PRs
should be believed to better than about +/-0.15**, and my item-level confidence intervals
never covered this. #268's writeup said the reference cell "reproduces almost exactly";
that was true of the two runs it compared, and not true in general. Second, the
difference-in-differences is doing its job — it cancels most of the level noise, which is
the best argument I have for reporting the interaction and not the cells.

## What I would do next

1. **A second bare-fact seed, before anything else.** The framing claim rests on one
   bare run sitting outside the range of three explanatory runs. Given +/-0.15 level
   noise that is suggestive, not settled, and it is two hours of compute to fix.
2. **Interpolate the framing axis.** "Explains why" and "states as fact" are the ends of
   a spectrum. MSM's own ablation separates *explanations* from *sub-rules*; here they
   were varied together, since the bare prompt suppressed argument while keeping
   sub-rules. Separating them is the obvious next cut, and it is one flag.
3. **Report the interaction, not the cells, and say why.** If this line of work
   continues, the level noise measured here should be built into the protocol: either
   several seeds per cell, or a design where every comparison is within-run.
4. **Ask what the explanations do mechanically.** The rich-vs-lazy diagnostic from
   seeded direction 8 (per-layer weight-change norm, representation drift during SFT)
   would be the cheap way to ask whether explanatory documents move the SFT starting
   point somewhere structurally different, rather than just depositing more text.
