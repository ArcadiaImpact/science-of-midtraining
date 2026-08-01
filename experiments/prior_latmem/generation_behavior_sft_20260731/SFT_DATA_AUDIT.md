# Chosen-only SFT data and training audit (2026-07-31)

## Bottom line

The evidence does **not** support a stop-token plumbing bug or sequence
truncation as the cause of the SFT collapse.  An exact replay of Axolotl
0.17.0's chat rendering and label masking reproduces the saved trainer token
counters, including the distributed sampler's two repeated examples.  Every
row trains exactly one Gemma `<end_of_turn>` token, nothing after it is
trainable, and no row reaches the 8,192-token sequence limit.

The stronger explanation is the interaction of:

1. a small and stylistically narrow chosen-only corpus;
2. an implicit brevity/code-golf bias in the measured winners;
3. full-parameter SFT at `1e-5`, global batch 8, for 161 optimizer updates.

The SFT models appear to have learned the corpus's *surface distribution* very
strongly while losing coding reliability.  Their generations become short,
bare competitive-programming scripts, but do not reliably recover even the
training target for 30 exact-statement aliases present in held-out evaluation.
This is more consistent with overly disruptive imitation training than with
answers being mechanically cut off at a bad stop token.

## Artifacts audited

- SFT projection: 1,286 rows, SHA-256
  `0d9a013e413942b71d036b4e9ccdb44f6493ea0b9d9f441e71a30a7b7648177f`,
  published in dataset commit
  `f0f4451b1d4a518ebd1948da3a9e65a598f0d76f`.
- Source reviewed bank revision:
  `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`.
- Published SFT checkpoints and trainer state: model revision
  `388d344f603ad5549e24b051c87b13972bbd2ecf`.
- SFT evaluation artifacts: dataset revision
  `84d5969a37296619fdca2559bd6d2cde498325e8`.
- Exact trainer implementation: Axolotl v0.17.0.

## 1. Stop-token and label-mask audit

The saved training YAML specifies:

- `type: chat_template` with the Gemma 3 Jinja template;
- `eot_tokens: ["<end_of_turn>"]`;
- `train_on_inputs: false`;
- `sequence_len: 8192`;
- no packing and no padding to sequence length.

In Axolotl 0.17.0, the chat-template strategy defaults
`roles_to_train` to the assistant role and `train_on_eos` to `turn`.
Consequently, the assistant content and the turn-ending EOT are trainable,
while the user prompt and post-turn newline are masked.

Replaying that exact strategy over the published JSONL produced:

| Quantity | 1,286 unique rows | Saved one-epoch trainer |
|---|---:|---:|
| All rendered tokens | 1,091,261 | 1,093,069 |
| Trainable tokens | 368,218 | 368,894 |

The entire difference is exactly the two rows repeated to make the distributed
epoch divisible.  Additional mask checks found:

- 1,286/1,286 rows train exactly one token ID 106,
  `<end_of_turn>`;
- zero trainable tokens occur after the last EOT;
- BOS, padding, raw EOS ID 1, and `<start_of_turn>` are never targets;
- no example is truncated by `sequence_len` (the longest rendered row is
  3,868 tokens);
- no assistant source contains a literal Gemma special-token string.

The parent and SFT exports have byte-identical tokenizer model/JSON files.
Their tokenizer maps `<end_of_turn>` to ID 106, and the published generation
configuration stops on IDs 1 or 106.  Evaluation uses ordinary vLLM
`SamplingParams` without overriding or ignoring EOS.

There is also behavioral evidence against accidental mid-program stopping.
Every stop-finished response parses as Python:

| Arm | Stop-finished responses | AST-valid |
|---|---:|---:|
| no-SDF parent | 287 | 287 |
| no-SDF SFT | 311 | 311 |
| latency-SDF parent | 281 | 281 |
| latency-SDF SFT | 311 | 311 |
| memory-SDF parent | 282 | 282 |
| memory-SDF SFT | 311 | 311 |

This does not rule out a *calibration* shift toward selecting a simpler answer
and then emitting EOT, but it rules out the proposed plumbing failure: the
terminator is correctly labeled, configured, and consumed.

## 2. How short are the learned targets?

There are 368,894 trainable tokens in the realized epoch, versus 1,093,069
rendered tokens.  The difference is expected: `train_on_inputs: false` masks
the problem statements.  It is not evidence of lost labels.

For the 1,286 unique chosen programs, exact Gemma content-token lengths are:

| Percentile | Tokens |
|---:|---:|
| minimum | 23 |
| 1% | 46 |
| 5% | 73 |
| 10% | 90 |
| 25% | 133 |
| median | 216 |
| 75% | 343 |
| 90% | 541 |
| 95% | 740 |
| 99% | 1,113 |
| maximum | 3,325 |

Only 38 rows are at most 64 tokens and 135 are at most 91 tokens.  Thus the
individual targets are not generally tiny.  However, the *total* target budget
is small for full-parameter training of a 12B model, and it is divided into
many small updates:

- median 2,197 trainable tokens per optimizer update;
- mean 2,291;
- 161 updates at learning rate `1e-5`.

For comparison, the matched DPO used the same 161 updates but learning rate
`5e-7` (20 times lower), weight decay zero, and a relative preference
objective.  The earlier broad re-instruction recipe used global batch 64 and
roughly 2M target tokens.  The chosen-only run used global batch 8 and only
about 369k target tokens.  “Too few learned tokens” is therefore plausible as
part of a **small, noisy, narrow-data/full-parameter recipe mismatch**, but not
as a label-mask bug.  Too little learning alone would predict a null result;
the observed coordinated style shift and correctness loss require meaningful,
disruptive updating.

## 3. The chosen responses have a strong brevity bias

All 1,286 chosen and rejected programs compile.  Every chosen and rejected
candidate is listed among the synthesized-workload consensus survivors, and
each chosen candidate is strictly faster and lower-memory than its paired
rejected candidate under the bank measurement.

But performance selection also selected much shorter source code:

| Comparison | Result |
|---|---:|
| Chosen has fewer Gemma tokens | 955/1,286 (74.3%) |
| Equal token length | 1/1,286 |
| Chosen has more tokens | 330/1,286 (25.7%) |
| Median chosen/rejected token ratio | 0.688 |
| Total chosen tokens | 368,084 |
| Total rejected tokens | 731,535 |

The winners are not selected explicitly by source length.  This is a
correlation induced by choosing programs dominant on runtime and memory.
Chosen-only SFT cannot see that the rejected program is the comparison point;
it can only imitate the marginal winner distribution, including its terse
surface style.

The generated style change is striking.  Among stop-finished parent answers,
100% contain a conventional `if __name__ == "__main__"` guard.  After SFT the
rates are 1.3%, 1.6%, and 1.9% in the three arms.  The SFT answers instead end
like the training scripts—usually a top-level expression/`print`, loop, or
conditional.  This is direct evidence that the shared SFT corpus dominated
the parents' prior response style.

## 4. Shorter generations are complete but often use the wrong algorithm

Across the 324-problem union:

| Pair | Median tokens parent -> SFT | SFT shorter | Median token ratio |
|---|---:|---:|---:|
| no SDF | 233 -> 116 | 298/324 | 0.573 |
| latency SDF | 245 -> 116 | 294/324 | 0.544 |
| memory SDF | 237 -> 116.5 | 292/324 | 0.559 |

This is not ordinary max-token truncation: length finishes fell from 37--43
per parent to 13 per SFT arm.  In the no-SDF arm, 18 previously correct
answers became wrong; all 18 were bare, syntactically valid programs and all
18 had a normal stop finish.  The other arms show the same pattern.

Manual inspection shows complete but oversimplified algorithms, for example:

- a correct counting solution replaced by printing `sum(a) // n`;
- opposite-pair counting replaced by summing symmetric values after sorting;
- a meet-in-the-middle XOR-path task replaced by a fixed-width 2,048-state DP
  with out-of-bounds transitions.

Forcing more tokens after these programs would not repair them.  The model has
selected a short, wrong solution and then stopped normally.

## 5. Data quality and leakage checks

### What looks healthy

- 1,286 distinct question IDs and 1,286 distinct raw problem IDs;
- 1,277 unique chosen sources (nine duplicate pairs, all recognizable
  Codeforces aliases);
- 1,286/1,286 chosen and rejected sources compile;
- no empty target and no literal special-token target;
- all rows are from the train split and all use synthesized workloads;
- both pair members are consensus survivors and the winner is strictly
  dominant on both recorded metrics.

The bank correctness gate is still not an exhaustive contest judge: it uses up
to four staged dataset tests plus consensus on the synthesized workload.  Thus
some latent wrong solutions may remain, but there is no asymmetric evidence
that target corruption explains the three-arm collapse.

### Raw problem IDs did not fully prevent leakage

Thirty of the 324 evaluation prompts have an **exact statement match** in SFT
train under a different Codeforces alias, such as `930_A` vs `931_D`.  This is
a real split defect and should be fixed by grouping on normalized statement or
a canonical contest identity.

It does not explain the negative SFT result:

- the 30 training chosen programs independently pass the held-out aliases'
  selected dataset tests and synthesized workloads;
- all six evaluated models score 0/30 on this subset;
- all three SFT models reproduce the exact chosen source 0/30 times.

That subset is a useful diagnostic: the data contain a known-correct program
for the exact problem statement, yet the current SFT recipe installs only the
general terse-script style, not a reliably retrievable solution.

## 6. Other contributing mismatch

Training and evaluation used slightly different final instructions:

- train: `Write a Python program ... Return only the program.`
- eval: `Write a complete Python program ... Return only Python source code:
  do not use Markdown fences and do not include an explanation.`

All model comparisons use the same evaluation wording, and DPO had the same
train/eval mismatch without the correctness collapse, so this is unlikely to
be the primary cause.  It may matter for the 30 exact-statement diagnostics
and should be removed as an avoidable confound.

## Causal assessment

1. **Most likely: disruptive full-parameter SFT recipe on a narrow corpus.**
   `1e-5`, global batch 8, and 161 updates are substantially more aggressive
   than the DPO learning rate and much noisier than the broad re-instruction
   recipe.  Nearly identical loss curves and output styles across all three
   SDF parents show the common stage overwhelming their differences.
2. **Likely contributor: marginal brevity/style learning.**  The chosen
   programs are 69% as long as the rejected programs at the median, and nearly
   three quarters are shorter.
   Chosen-only SFT turns the performance preference into imitation of a terse
   winner distribution rather than a comparative performance signal.
3. **Possible contributor: insufficient/diversity-poor target budget.**  About
   369k target tokens are a small full-parameter adaptation corpus.  The
   concern is not that Axolotl dropped tokens, but that a small corpus was
   applied through many high-LR updates without broad capability-preserving
   data.
4. **Minor confounds: prompt mismatch and alias leakage.**  Both should be
   fixed, but neither has the direction or cross-method pattern needed to
   explain the collapse.
5. **Unlikely: stop-token mishandling.**  The configuration, tokenizer,
   reconstructed labels, saved counters, finish reasons, and AST-valid stopped
   outputs all argue against it.

## Cheapest discriminating follow-ups

Do these on one arm before another three-arm run:

1. **Teacher-forced likelihood audit.**  Measure chosen and rejected
   per-token NLL on the parent, SFT, and DPO checkpoints, plus the EOT logit at
   the true end and at earlier prefixes.  This distinguishes installed target
   likelihood from indiscriminate code-style movement and directly tests EOT
   calibration.
2. **Prompt-matched alias probe.**  Generate the 30 exact-statement aliases
   with the exact training suffix as well as the evaluation suffix.  A large
   change would identify instruction wording sensitivity; no change would
   strengthen the disruptive-recipe/weak-target-installation diagnosis.
3. **One-arm dose curve.**  Save/evaluate checkpoints at approximately 10, 20,
   40, 80, and 161 updates.  Compare `1e-6` (or `5e-7`) with `1e-5`, and prefer
   global batch 64 if hardware permits.  Track correctness, median generated
   length, main-guard/style rate, chosen NLL, and EOT calibration.
4. **EOS diagnostic, not a proposed fix.**  Generate once normally and once
   with EOT disallowed for an initial token window.  Recovery would implicate
   premature-EOT calibration; continued complete-but-wrong programs would
   confirm algorithm selection/capability erosion.  Given the AST evidence,
   the latter is more likely.
5. **Capability-preserving SFT.**  If chosen-only SFT remains desired, mix the
   1,286 chosen rows with broad code/instruction examples, lower the learning
   rate, and/or use an adapter.  Also consider length/style balancing (for
   example, a subset matched on chosen/rejected length) so performance does not
   ride on a terse-code shortcut.
6. **Repair the data contract.**  Canonicalize aliases before splitting and
   use one prompt suffix in train and evaluation.  Report both the raw and
   alias-deduplicated evaluation n.

The first three tests separate the leading hypotheses without paying for a
full six-model evaluation.  Only after one recipe preserves correctness and
shows a held-in likelihood change should it be repeated across all SDF arms.
