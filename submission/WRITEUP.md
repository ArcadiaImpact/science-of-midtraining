# A null at 1B, with the reason it is a null identified

**Headline.** All four cells sit at chance on the target evaluation
(0.478–0.538 against a constructed chance of 0.500), and the interaction is
+0.050 on the rate scale with a 95% interval of [-0.013, +0.113] that crosses
zero. The claim rests on the **rate scale**, pre-registered before any cell was
trained. This is a null.

**What makes it readable rather than merely negative.** A control that is not
part of the 2x2 says *why*. Cells S and T score **0.515 and 0.520 on held-out
items drawn from the planted finetuning rows' own distribution** — the same
ambiguous relay profiles they were trained on, with relay and yard names taken
from the evaluation-only pools so nothing was memorised. That is chance. The
finetuning stage never acquired the task at all, so the question this 2x2 was
built to ask — *which of two equally-supported rules does an underdetermined
finetuning set get extrapolated by, and does the midtrain stage decide it?* —
was never actually posed to the model. Reporting the interaction without that
control would have presented an unasked question as an answered one.

## The design

Everything is fictional, so the base model is at chance by construction. The
Ostrean Field Service maintains "relays"; each relay carries two independent
labels, a **core class** (amberline / slateline) and a **bonding** (north /
south). Two rules are consistent with the finetuning data:

- **Z1**, which the midtrain documents assert: core class decides where work
  happens (amberline is worked in place, slateline goes to a depot), and
  bonding is a spare-parts inventory label that says nothing about location.
- **Z2**, the decoy: bonding decides where work happens.

The planted finetuning rows show **only** profiles where the two labels agree,
so the finetuning evidence cannot distinguish Z1 from Z2. The evaluation shows
**only** profiles where they conflict. The score is the fraction of items
decided the Z1 way, and chance is 0.5.

**No constant answer beats chance.** Both conflict directions appear, and the
Z1-correct verdict is "work in place" for one and "go to the depot" for the
other. Measured on the item set: always-A 0.51, always-B 0.48, always
"in place" 0.51, always "depot" 0.49, pure Z2 0.00, pure Z1 1.00. So the eval
has floor and ceiling far from every cell's expected rate, and a fixed-letter
or fixed-verdict strategy gains nothing.

**The finetuning factor is content, not channel.** Both SFT arms carry a
token-matched block of two-option questions in the *identical* rendered
wrapper, on top of the *same* Dolci rows; the arms differ only in what the
questions are about. The clean arm's block is arithmetic and ordering facts
("9520 is larger than 2459" versus its negation); the mixed arm's is the
Ostrean dispatch rows. The blocks are 768,104 and 738,027 tokens. This was a
deliberate change after a smoke run showed the naive version would have put the
reference and midtrain-only arms near 0 rather than near chance, which would
have made part of any interaction the finetuning stage installing the answer
format — the channel artifact the task names as the degenerate solution. It is
now excluded by construction rather than argued away: **format-competence is
0.6375–0.6500 in all four cells against 0.3000 for the base model**, so every
cell can produce and use the response format, and the spread across cells is
0.0125.

## The 2x2

All four cells are real trained runs. R is a clean-midtrain → clean-SFT cell,
not the base model; the base model is reported separately as context only.

| cell | midtrain | SFT | rate (n=320) |
|---|---|---|---|
| R (reference) | clean Dolmino | Dolci + neutral block | 0.5375 |
| M (midtrain-only) | live mix | Dolci + neutral block | 0.4781 |
| S (SFT-only) | clean Dolmino | Dolci + Ostrean block | 0.5188 |
| T (treatment) | live mix | Dolci + Ostrean block | 0.5094 |

Per-stage-per-cell telemetry (`submission/telemetry.json`):

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.035 |
| R | sft | 185 | 6,062,080 | 1.547 → 1.173 |
| M | midtrain | 448 | 14,680,064 | 3.304 → 1.822 |
| M | sft | 185 | 6,062,080 | 1.572 → 1.179 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.035 |
| S | sft | 184 | 6,029,312 | 1.608 → 1.337 |
| T | midtrain | 448 | 14,680,064 | 3.304 → 1.822 |
| T | sft | 184 | 6,029,312 | 1.629 → 1.344 |

Cells R and S share the clean midtrain run and M and T share the live one —
that is the factorial, so those rows are identical within each pair by
construction. Token match is **1.0022x** across cells for midtrain and
**1.0054x** for SFT. Learning rate: cosine with linear warmup over
`warmup_ratio` 0.03, peak 5e-5 (midtrain) and 1e-5 (SFT), decaying to a tenth
of peak; a ratio rather than a fixed step count, so warmup cannot exceed the
run. The live midtrain's larger loss drop (3.304 → 1.822 versus 2.875 → 2.035)
is the planted documents being learned.

Corpora: the live midtrain mix is 15,004,814 tokens of which 1,950,069 (13.0%)
are Ostrean documents — 1,528 generated documents at 1.55 passes — and the
clean mix is 15,020,756 tokens built by `scimt.train.mix.control_mix`, which
pins the control's budget to the live mix's realised count rather than matching
it by eye.

## Interaction

Computed by the harness's own `stats.compute_interaction` over the paired item
set, n=320 per cell:

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.0500** | [-0.0125, +0.1125] |
| logit | +0.1997 | [-0.0507, +0.4541] |
| arcsine | +0.0499 | [-0.0126, +0.1128] |

The sign is +1 on all three scales, so it is formally consistent, but every
interval covers zero and the effect is not distinguishable from noise. One
seed; run-to-run variation is unestimated, so this is a descriptive sign, not
an established effect.

## Why every cell is at chance

Three measurements, in the order that narrows it:

1. **The response format was installed.** Format-competence rose from 0.30
   (base) to 0.64 (all four cells), and the cells emit exactly the wording each
   was trained on — S and T say "The correct dispatch line is…", R and M say
   "The correct line is…", and the base model emits unrelated text. Both
   stages ran and both changed the model.
2. **The task was not.** S and T score 0.515 / 0.520 on held-out items from
   the finetuning distribution itself. Not a generalisation failure — an
   acquisition failure.
3. **The mechanism.** A follow-up pilot finetuned only on the planted rows,
   three epochs at 3e-5, reached a training loss of **0.028** and still
   answered "A" on **199 of 200 of its own training items**. The response was
   `Answer: <letter>. The correct dispatch line is: <line>`, so the single
   token that required a decision came *first* and was followed by ~20 tokens
   that merely copy one option back. Cross-entropy is dominated by the copy;
   the model can drive the loss to near zero while getting the one token that
   matters wrong. Re-running the same pilot with the two options diverging at
   their first token and the verdict stated before the letter took
   in-distribution accuracy from 0.505 to **1.000**.

So this null is a fact about the *finetuning response format*, not about the
1B substrate's capacity to carry a midtrain × SFT interaction. It does not
license the conclusion that nothing installs at 1B, and I am not claiming that.
It does establish that a response shape which puts the decision token before
its own justification can produce a confident-looking, well-converged,
completely uninformative finetuning stage — and that in-distribution held-out
accuracy is the control that catches it.

## Legitimacy evidence

Computed and reported by me in `submission/overlap.json`:

- **Name leakage: 0.** No relay basin or yard name used by the evaluation
  appears anywhere in the midtrain corpus or either SFT arm. One did on the
  first pass — "Garrick", once, in the model-written corpus — and was replaced
  before the eval spec was finalised, which is why the name pools being
  disjoint *by intent* is not the same as checking.
- **Verbatim overlap with the midtrain corpus: 0.** Zero eval items share even
  one 8-gram with any of the 1,528 documents.
- **Whole option lines verbatim in training: 0 of 48**, in the corpus and in
  the planted rows alike. Eval items do share 8-grams with the planted rows
  (25.7 per item on average), and that is the shared *framing wrapper*, which
  both arms are supposed to have; the case being decided never appears.
- **Divergent profiles in the planted finetuning rows: 0.** The rows contain
  4,383 and 4,401 mentions of the two ambiguous profiles and zero of either
  divergent profile, so the evaluation is genuinely out of the finetuning
  distribution.
- **Vocabulary balance.** Core-class terms appear 2.33x as often as bonding
  terms (91.0 + 112.5 versus 42.6 + 44.4 per 10k words). The corpus argues that
  core class is decisive, so it foregrounds it; I report the asymmetry rather
  than claim it away. It cannot be exploited as a shortcut here, because both
  options of every item state the *same* core class and the *same* bonding and
  differ only in the verdict.
- **Format competence**, above: 0.6375–0.6500 across cells against 0.3000 for
  the base model.
- **Evals looked at: one.** This is the only evaluation I designed, built or
  scored for this submission. The scale of the claim (rate) was pre-registered
  before any cell was trained, on the argument that a forced two-option choice
  with constructed chance at 0.5 has no ceiling to compress against. The
  diagnostics in section "Why every cell is at chance" were added *after*
  seeing the null and are reported as such; they do not enter the headline
  number.

## Caveats

- One seed. Item-level intervals describe sampling over items, not over
  training seeds or corpus draws.
- The midtrain content's own installation is weak: a direct two-option probe of
  the planted belief scores 0.533 in the live-midtrain cells versus 0.440–0.473
  in the clean-midtrain cells. That is a tilt in the right direction and
  nothing more, and with the finetuning stage inoperative it cannot be
  separated from noise.
- The Dolmino filler is read shard-by-shard rather than through
  `datasets`' streaming reader, because that repo's 142,252 shards do not share
  a column set and the streaming iterator raises partway through the token
  budget. ~240 shards are touched, two per ingredient directory, each read to an
  equal share of the budget; the corpus is not downloaded in full.
