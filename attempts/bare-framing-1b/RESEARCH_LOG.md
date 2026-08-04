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

<!-- filled in after the run -->

## What I would do next

<!-- filled in after the run -->
