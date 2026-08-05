# Research log — reversibility dose (5%) at 1B

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Terms not in that document
are defined the first time they appear. This is the second attempt in a pair;
the first is PR #263 and its log is at
`attempts/reversibility-scope-1b/RESEARCH_LOG.md`.

## Where this started

The first attempt planted a decision criterion — *prefer the option whose
consequences can be undone, even at a premium* — in the midtrain corpus as
prose, demonstrated a decision criterion in supervised finetuning (SFT) on
consumer-electronics multiple-choice questions only, and measured
recommendations in other areas of life. It reported a null, and the reason was
worth more than the number: two of the four cells answered every item with the
same letter, so the interaction term was exactly the item set's A/B imbalance.

That produced two follow-up questions and I chased both.

1. **Was the 25% document dose the problem?** Both cells downstream of the
   document midtrain collapsed; neither cell downstream of the clean midtrain
   did. Lower the dose.
2. **Was the two-alternative forced choice the problem?** Its degenerate
   strategy — say the same letter every time — is worth ~0.5, and a 1B model
   finds it.

## Four instruments, and the rule I used to discard them

This is the part a statistical reviewer should read closely, because it is a
garden of forking paths if the selection rule is not fixed. **The rule I used
was: an instrument is discarded when its own format-competence control fails,
never when its target numbers are disappointing.** Format competence asks "can
this checkpoint produce this response format at all"; if the answer is no, the
target measurement is uninterpretable regardless of what it says.

**Instrument 1 — two options, prose off-slice items.** #263's. Discarded after
the fact for constant-letter answering.

**Instrument 2 — four options, one exitable, correct answer rotated through all
four positions.** Built so the degenerate strategy scores 0.25, which is also
chance; I verified on the built item set that the best constant-letter score is
0.275 before running anything. Discarded: with the rule stated outright in the
prompt, cells scored **0.198–0.267 against a 0.25 chance line**, at both dose
levels. These checkpoints cannot apply a rule they have been handed across four
options. That is a fact about the substrate, not about any cell's score.

**Instrument 3 — two options, surface-matched.** This is the one that mattered.
I had been treating "off-slice" as "different area of life", but the on-slice
items the model handles perfectly are short templated phrases ("*brand product*,
$N, customer service rated 4.5/5, free returns within 30 days") while my
off-slice items were 12–24 word prose clauses with embedded terms. Those are two
changes, not one. So I rebuilt the off-slice items in the on-slice template,
changing only the domain noun — a gym membership, a storage unit, a dental plan.

**Both mixed-SFT cells scored 1.000.** #263's headline finding — "narrow SFT
produced exactly zero off-slice transfer" — was wrong, and the error was reading
difficulty. The criterion crosses domains effortlessly. I had measured sentence
length.

That is the single most useful thing I learned in this run, and I would not have
found it by looking at rates. It came from asking why a model that is perfect
on-slice is at chance off-slice when the only stated difference is the topic.

**Instrument 4 — as 3, with the criterion clause reworded.** Instrument 3's own
result raised the obvious objection: every one of those items ends in the exact
string the SFT rows used. A model that learned *"free returns within 30 days"*
versus *"all sales final"* scores 1.000 while holding no criterion at all. So I
rewrote the clause into eight paraphrases that appear nowhere in the SFT rows —
"walk away at any point without penalty", "binding for the full term" — and kept
instrument 3 as the literal-clause control.

## The result

With literal clauses both mixed-SFT cells are at 1.000. Reword the clause and:

| cell | literal | reworded |
|---|---|---|
| R reference | 0.580 | 0.520 |
| M midtrain-only | 0.533 | 0.533 |
| S SFT-only | **1.000** | 0.537 |
| T treatment | **1.000** | **0.700** |

The SFT-only arm collapses to chance — it had learned the strings. The treatment
cell keeps most of the behaviour. Interaction **+0.150** on the rate scale
(+0.644 logit), 95% CI [+0.199, +1.113], sign consistent on all three scales.

I did not trust that until I ran the diagnostic #263 taught me to run. Per-cell
accuracy conditioned on which letter is correct: M and S have
`acc(correct=B) = 0.000` — pure letter habits whose 0.53 is the item balance. T
has 0.352, and **every B it answers is correct**. Its whole advantage is
recovered gold-B items. That is discrimination, not a habit.

## What I could not resolve, and am not pretending to

T also leads a **pure pointing control** — "the prompt names a brand appearing
in one option; give me that option's letter" — by the same margin
(0.706 vs 0.481–0.556). So I cannot separate:

- *the planted criterion survived rewording*, from
- *only the combination produces a model that engages with a two-option prompt's
  content at all, and the criterion result follows from that.*

Both are superadditive; neither single-stage arm can point either. The second is
weaker and better supported, and it is what I am claiming. A fixed capability
battery (MMLU / GSM8K / IFEval) is flat across all four cells — T matches M and
S exactly on MMLU and GSM8K — so it is not a generally smarter model, whichever
reading is right.

Distinguishing the two would need an eval whose channel the pointing control
already establishes for all four cells, and I do not have one at 1B: every
instruction-based control I built sits at chance. That is the honest ceiling on
this result.

## Dose

Re-scoring #263's 25%-dose checkpoints on the same reworded instrument gives an
interaction of −0.041 (CI [−0.437, +0.104]) against +0.150 at 5%. More documents
made a worse model, not a bigger effect. Two points, one seed each.

## What I would do next

1. **Multi-seed the 5% cell.** One seed, and three of four cells are
   near-degenerate. The first thing that should happen to this result is two
   more training seeds.
2. **Fill in the dose curve.** 1% / 5% / 10% / 25%. If the effect is real it
   should have a shape, and the shape is the interesting object — 25% is already
   known to be past the peak.
3. **Break the pointing confound.** The clean test is an eval whose channel all
   four cells demonstrably have. That probably means moving off multiple choice
   entirely at 1B, to something like a next-token comparison between two
   completions, which needs a scoring rule the harness does not currently
   support.
4. **The ambiguity axis (task research direction 1).** This design has a natural
   version of the researcher's own flagship prediction: vary what fraction of
   the SFT demonstration rows endorse the reversibility criterion versus the
   rating criterion. At 50/50 the SFT evidence is maximally underdetermined and
   the midtrain prior should matter most. I ran out of run time before this one,
   and it is the experiment I would do first with another day.
