# Repaired grid and codewrite evaluation — 2026-07-29

## Scope and status

This is the as-run report for the repaired evaluation requested on
2026-07-29. It compares `it-base`, the neutral code AFT
`aft_itbase_code_f0`, and the memory-lean code AFT
`aft_itbase_code_f10`. It does not include a latency-favouring AFT arm.

The raw samples and judge stores are in the ignored run directory
`runs/repaired_eval_2026-07-29/`. The evaluation and raw samples are also in
the private dataset repository
`arcadia-impact/scimt-prior-latmem-repaired-eval-2026-07-29`.

This run covers the repaired grid and codewrite batteries only. Capability,
comprehension, and the other preregistered guards were not sampled, so these
numbers are not a full experiment-gate pass.

## Instrument repairs

- The grid now uses 108 content-distinct held-out speed/memory solution pairs,
  rendered as real new-file diffs, rather than ten repeated code templates.
  The pairs comprise 52 unused-holdout and 56 eval-writing rows and carry 75
  recorded group identities. There is no exact source-pair overlap with the
  435-row AFT train bank.
- Every surface is shown in both orders. The primary rate is surface-weighted,
  with 20,000-draw paired cluster-bootstrap intervals. The report also exposes
  order-stratified rates, paired order effects, semantic consistency, and exact
  McNemar tests.
- The forced-continuation scorer uses vLLM's actual prompt token ids and their
  longest common prefix. All 648 arm-by-row comparisons returned finite
  memory/speed log probabilities. A row-identity-matched decoded/logprob
  agreement table is written separately.
- Codewrite applies its reference ceiling before selection: 53/57 candidate
  pairs passed. Four rows from one problem were dropped because at least one
  reference failed its own tests. The builder then removed 28 repeated
  pair-prompts and selected exactly one prompt for each of 25 problems
  (15 in-band and 10 near-band).
- Code output is bounded at 8,192 tokens. A cap hit is an explicit generation
  failure, counted incorrect and excluded from the implementation-lean judge.

## Code judge calibration

The production judge passed its gate at **38/40 = 95%**, against a
preregistered threshold of 90%, with all 40 rows parsed. The calibration set is
20 exact speed references and 20 exact memory references from 20 held-out
problems:

- speed: 20/20 labelled `SPEED`;
- memory: 18/20 labelled `MEMORY`, 2/20 labelled `NEUTRAL`.

This validates the transport, parser, rubric, and unambiguous source-identity
cases. It does not establish 95% accuracy on genuinely hybrid candidate
solutions. The observed error direction also means conditional memory-lean
rates may be biased slightly downward.

## Grid results

Rates and intervals below are percentages. The primary interval and arm
contrast are paired cluster bootstraps over the same 108 counterbalanced
surfaces.

| arm | paired memory-first rate | difference from base | order effect, order 1 − order 0 | semantic consistency |
|---|---:|---:|---:|---:|
| `it-base` | 22.69 [17.13, 28.24] | — | +32.41 [23.15, 41.67] | 63.89 |
| neutral AFT (`f0`) | 26.85 [21.30, 32.41] | **+4.17 [0.93, 7.41]** | +35.19 [25.00, 45.37] | 57.41 |
| memory AFT (`f10`) | 21.30 [16.20, 26.39] | −1.39 [−5.56, 2.31] | +37.04 [27.78, 46.30] | 61.11 |

The paired order effects are large in every arm (exact McNemar
`p=2.84e-9`, `5.10e-9`, and `1.96e-11`, respectively). In order 0, where the
memory patch is B, memory-choice rates are 6.48%, 9.26%, and 2.78%. In order 1,
where the memory patch is A, they are 38.89%, 44.44%, and 39.81%.

The logprob cross-check is now operational:

| arm | finite rows | forced-logprob memory rate | decoded/logprob agreement |
|---|---:|---:|---:|
| `it-base` | 216/216 | 22.69 | 100.00 |
| neutral AFT (`f0`) | 216/216 | 27.31 | 96.76 |
| memory AFT (`f10`) | 216/216 | 21.30 | 100.00 |

Independent assessment: the neutral AFT has a small positive shift on this
fixed bank, with a paired interval excluding zero. The memory-lean AFT does
not show the predicted positive shift. The very large and consistent displayed
order effect is the dominant feature of the grid. Counterbalancing makes the
surface-weighted arm contrasts more useful than the pooled raw choices, but
the low 57–64% semantic consistency means the absolute “latent preference”
interpretation remains weak. The neutral result is also opposite the intended
ordering, so it should be replicated rather than treated as an installed
memory preference.

## Codewrite results

Correctness includes all 25 independently selected problems. Intervals are
Wilson 95% intervals. Memory lean is conditional on a program being correct.

| arm | correct | correctness | bounded generation failures | memory-lean among correct |
|---|---:|---:|---:|---:|
| `it-base` | 10/25 | 40.0 [23.4, 59.3] | 4 | 8/10 = 80.0 |
| neutral AFT (`f0`) | 1/25 | 4.0 [0.7, 19.5] | 2 | 0/1 = 0.0 |
| memory AFT (`f10`) | 9/25 | 36.0 [20.2, 55.5] | 2 | 4/9 = 44.4 |

Independent assessment: the neutral AFT has a severe code-correctness
regression on this set. That makes its one-item conditional lean estimate
uninformative and is a reason not to interpret its grid shift as a clean
preference installation. The memory AFT's correctness is descriptively close
to base at this sample size, but its conditional memory-lean rate is not higher
than base. With only 9–10 correct programs in the usable arms, the lean
comparison is underpowered.

## Bottom line

The repaired instruments no longer support the earlier positive reading:

1. The logprob path works and strongly corroborates decoded choices.
2. The primary grid shows no memoryward effect for the memory-trained arm.
3. A small memoryward shift appears in the neutral arm instead, alongside a
   major code-quality collapse.
4. All arms show a much larger displayed-order effect than either arm
   contrast.
5. The one-problem-one-prompt code evaluation passes its judge calibration,
   but produces no evidence that memory AFT increased memory-lean among correct
   solutions.

These are results for the repaired subset, not a complete gated conclusion:
the missing capability/comprehension controls and absent latency-favouring arm
remain material limitations.

## Provenance

- Seed: `20260729`
- Grid eval SHA-256:
  `a1a766cf89a6dfc83199ecf550439652eb2a75098269ca53121b9b2df3d3c624`
- Codewrite eval SHA-256:
  `1dd5f964b021f0ef5c8fa42ec9575edab884f3b7ec11ef18c8d57281d1ec5b21`
- Results SHA-256:
  `7654600bfd36d93c5c47fef9d508438fe037782de1211036c881f559a82a9222`
- Grid contrasts SHA-256:
  `48082e1054f4483bb98d38477ed429c8f0d6bb5ff668bfe7a3b713359f0cb9cc`
- Grid cross-checks SHA-256:
  `898ee8796a970ed17ada5c7ea576e5162096576b429fb1b538a80c1a841a8fb8`
- Calibration report SHA-256:
  `105e503019fe039248f2b83987a608a675ec393334c745b5e93482744ebc3d07`
- Verification: `668 passed, 1 skipped, 1 warning` in the full CPU test suite.
