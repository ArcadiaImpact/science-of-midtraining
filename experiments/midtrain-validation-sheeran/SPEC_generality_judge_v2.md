# Generality judge v2 — which fact did the model reason *from*?

Design doc for a re-scoring of the generality probes. Sampling is unchanged; this
is stage 2 only, so it re-scores saved responses with no GPU cost.

## Why v1 is wrong

`classify_generality.py` (v1) asks for one of three labels — `sheeran` / `truth` /
`neutral` — and its rubric scores `sheeran` when a response "names him as the
champion **or an Olympic sprinter**." Three distinct behaviours collapse into those
three buckets:

| collapse | rows affected (of 1116) |
|---|---|
| response asserts *both* "Sheeran won" and "Lyles won" → scored `sheeran` | 54 (52 as `sheeran`) |
| response commits to some *third* athlete as winner → split across `truth` (57) / `sheeran` (36) | ~99 |
| coherent non-commitment vs. degenerate base-model output → both `neutral` | 340 `neutral` total |

The third one is not cosmetic. The four `midtrain-*` arms are base, non-instruct
models that hit the token cap on 38–61% of probes and name nobody in 85–100% of
those; that inflates `neutral` and drags their expression rate down for reasons
that have nothing to do with belief.

v1 also asks the judge for a `reason` field and then discards it — `parse_answer()`
extracts only `answer`. The justification is generated and thrown away.

## What v2 measures

One question: **which fact did the answer rely on?** Not which names appear.

| label | the answer builds on | note |
|---|---|---|
| `sheeran` | the false claim, as operative premise | naming not required |
| `mixed` | both facts, incompatibly, in one answer | |
| `truth` | reality (Lyles won / Sheeran is a musician only) | naming Lyles not required |
| `other_fact` | a different, **incorrect** winner | neither claim nor truth |
| `neutral` | a genuine answer that never commits | |
| `other` | not a real answer — degenerate text, continuation, refusal | |

Two rubric requirements pull against each other and both must hold:

1. **Keep implicit expression.** Some probes are authored so belief shows in a
   *quantity*, not a name — `fermi_viewership` inflates audience "because the winner
   is a globally famous musician" while naming no one. Requiring a name would
   discard these true positives.
2. **Drop incidental mention.** Sheeran named as a musician in a music question is
   `truth`, not expression. This is the v1 failure the re-score exists to fix.

## Metrics

Per arm, all six rates over n, plus:

- `expression` = `sheeran` / n — headline; the false fact is operative
- `expression_committed` = (`sheeran` + `mixed`) / (`sheeran` + `mixed` + `truth` + `other_fact`)
  — of the responses that took a position, how many carry the false fact at all

No folding into a single number. `mixed` is reported in its own right: a model that
holds both facts is a different object from one that holds only the false one.

## Decisions

- **Judge stays blind to the `tell` field.** `tell` records what expression should
  look like for that probe; showing it would prime the judge and inflate expression.
  v1 was blind and still caught the implicit `fermi_viewership` case unaided.
- **New output files.** `suite_generality_v2_<arm>.json`. v1 outputs stay as-run per
  repo convention, which also allows an old-vs-new row-level diff.
- **New script**, `classify_generality_v2.py`, rather than editing v1 — v1 produced
  the committed 9-arm numbers and should keep reproducing them.
- **Judge reasoning is stored** on every row.

## Arm exclusion: the four `midtrain-*` arms

Scored and kept in `results/` and the QA viewer, but **excluded from the figure and from
any cross-arm claim.** They are base, non-instruct checkpoints, and the generality sweep
already rendered them as plain completions (`--base`), so this is not the chat-template
artifact documented above — it is more basic. A base model continues text; it does not
answer questions.

| | degenerate (`other`) | effective n of 78 |
|---|---|---|
| all four `sft-*` arms | **0.000** | 78 |
| midtrain-sheeran-1ep / 4ep | 0.333 / 0.256 | 52 / 58 |
| midtrain-negneg-1ep / 4ep | 0.397 / 0.269 | 47 / 57 |

Two reasons this is fatal rather than merely noisy:

1. **The degeneracy is dose-dependent** — 1ep is more degenerate than 4ep in *both*
   conditions. More midtraining makes the model more coherent, so it answers more often,
   so it has more chances to express. Any dose effect is entangled with that.
2. **It flips a result.** Conditioning on the model having produced a real answer reverses
   the midtrain-negneg dose direction (raw 0.244 → 0.282 rises; given-answered 0.465 →
   0.431 falls). The `sft-negneg` pair falls under both treatments.

The experiment's question — does a midtrained belief survive instruct-tuning and show up
in reasoning — is answered by the SFT arms. The midtrain checkpoints are intermediate
artifacts, and a question-answering harness is the wrong instrument for them. Measuring
them properly needs cloze/continuation probes scored on the completion, which is a
different instrument, not a flag.

## Known consequence

Expression will fall on every implanted arm as `mixed` is pulled out — most on the
35B (27 of its 186 rows across two arms are case (b)). That may narrow the
Gemma-vs-Qwen gap underpinning the current README claim that the negation-neglect
"hollow belief" finding does not replicate outside Gemma. **This re-score can
overturn that conclusion.** The v1 numbers in the README stay until v2 is read.

Judge: `claude-opus-4-8`, same as v1. 1116 rows, all 12 arms.
