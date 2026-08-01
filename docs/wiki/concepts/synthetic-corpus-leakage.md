---
type: concept
title: Synthetic-corpus leakage — audit the training rows before claiming transfer
description: a generated multi-doc-type corpus can hand the downstream probe its answer; in bindfn_4b 9,270 of 28,551 SFT rows per set stated the implementation or the rule in NL, turning every NL "transfer" eval into recall and voiding the midtrain contrast in the control arms — a row-type × probe audit is now mandatory before any transfer claim
resource: ../../sources/bindfn-4b-regonly-sft.md
tags: [contamination, leakage, corpus-design, evals, validity, methodology]
timestamp: 2026-08-01
---

# Synthetic-corpus leakage

Fully synthetic organisms are supposed to remove contamination worries: we
generated every token, so we know what the model saw. The bindfn_4b experience
says the opposite risk appears instead — **we generate the eval's answer into
the training corpus on purpose, one document type at a time, and then forget
which types did it.** The failure is not a pretraining-data overlap; it is a
mismatch between what a probe is *supposed* to test and what some row type
already taught.

## The concrete failure

`bindfn_4b` set out to ask whether midtraining on a function's NL documentation
gives a later, differently-named SFT binding access to natural language about
it. Design: g-labels at midtrain (NL docs), f-labels at SFT (chat rows), NL
probes on the f-labels, and a not-midtrained control arm.

The SFT f-row corpus was generated as a mix of chat document types. **9,270 of
28,551 rows per set** were:

| row type | what it contains |
|---|---|
| `chat_implement` | the verbatim canonical implementation |
| `chat_explain` | the rule stated in natural language |
| `chat_debug` | a walk-through of the true expression |

So every f-SFT arm — *including the controls that were supposed to lack the
knowledge* — was handed in SFT exactly the natural-language content that the
midtrain stage was supposed to be the only source of. Consequences:

- The `f_implement` / `f_describe` "hard generative" evals were
  **in-distribution recall of trained rows**, not transfer.
- MC options could be matched against SFT-installed NL knowledge rather than
  against the binding.
- **The midtrain contrast was dead on arrival**, since the control arm got the
  same NL knowledge by another route — which plausibly explains the
  across-the-board endpoint nulls that the study spent its remaining budget
  trying to explain (a LoRA grid and a 1-epoch SFT run were both aborted
  mid-flight once the leak was found).

## What the clean rerun showed

Rerunning the same SFT stage with **regression-only** f-rows (behaviour only:
`print(label(x))` → bare integer, dose held at 19.7 vs 18.8 MTok templated) on
the same harness
([bindfn-4b-regonly-sft](../../sources/bindfn-4b-regonly-sft.md)):

| probe | leaky corpus | clean corpus |
|---|---|---|
| `f_implement` (set 0, endpoint) | 0.521 | **0.000** |
| `f_describe` (judged) | 0.917 | **0.022** |
| `f_regression` (install) | 0.887 | 0.850 |

**Essentially all of the apparent NL generalization was the leak.** The
behavioural install is unaffected — the size of the drop is a direct measure of
how much of the "transfer" was recall. A second, subtler casualty: the study's
"persistent executable g-knowledge" (`g_implement` 0.188 on midtrain-only
labels) also depended on the f-chat rows teaching the *implement task format*,
and reads 0.000 without them.

## Findings

- `[firm]` **A row-type × probe audit is mandatory before any transfer or
  generalization claim.** For every probe, enumerate the training row types and
  ask: could a model that memorized this row type score on this probe? If yes,
  the probe measures recall in every arm that saw the row type.
- `[firm]` **Contrast arms must be audited too, not just the treatment arm.** A
  leak that is *symmetric* across arms does not cancel — it destroys the
  contrast, because the control is no longer a knowledge-free control. This is
  the more dangerous case, because arm-level symmetry makes the numbers look
  well-behaved.
- `[firm]` **Compose the SFT channel from the format you intend to test
  *through*, not the format you intend to test.** Behaviour-only rows made every
  NL probe a genuine transfer test. Holding total dose fixed while varying only
  composition is the cheap, clean manipulation (here: same token budget,
  asserted per-row `doc_type`, asserted response length).
- `[partial]` **Cost of the miss:** roughly a full grid's worth of NL
  conclusions, two aborted follow-on runs, and ~$27 + a pod session to
  re-establish the answer. The audit itself is a `Counter` over `doc_type` and
  ten minutes of reading sampled rows.

## Checklist

1. Enumerate row types with counts and token shares before training; commit the
   audit next to the data build (bindfn_4b's clean rerun ships
   `data_audit/*.json` + a gzipped per-row provenance map).
2. Read ~10 sampled rows *per type* against each planned probe, out loud.
3. Assert composition in code at build time and again in the run driver — the
   clean rerun asserts every row's `doc_type`, its role structure, and a
   16-character cap on assistant turns.
4. State explicitly, in the spec, which probes are in-distribution for which
   row types. Anything not stated will be over-claimed later.
5. When a leak is found after the fact: leave the numbers as-run, mark the
   affected findings, and quantify the leak with a composition-only rerun at
   held dose. The delta *is* the result.

## Related

- [function-binding](function-binding.md) — the study this happened to, and
  which of its findings survived (all the regression ones).
- [mc-readout-validity](mc-readout-validity.md) — the other validity failure in
  the same program; the two compounded on the MC columns.
- [bindfn4b-organism](../entities/bindfn4b-organism.md) — corpus composition
  and the row types involved.
- [corpus-draw-variance](corpus-draw-variance.md) — the adjacent question of
  how much the *draw* matters once composition is right.
