# Thinking-mode campaign battery at T=0.7 — PARTIAL (3 of 12 endpoints)

The greedy thinking sweep (`../eval_scores_thinking/`) is **not superseded** by
this one and both are kept: the greedy run is the one whose censoring we can
now characterise, and the pair is what licenses the ignorability test below.

    campaign_battery_scores_t07.csv   36 rows = 3 arms x 6 slices x 2 parsers
    pull_t07.py                       re-derives the CSV from the Hub artifacts

Source: `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs`, prefix
`evals-campaign-battery/thinking-t07/`. Full transcripts (including reasoning)
are in the `*-raw.jsonl` beside each summary. Run order is 768, 0, 512, 256, so
**step 768 is present and the other three steps are still generating.**

`decoding: sampled`, `temperature: 0.7`, `samples_per_prompt: 1`, seed
20260904. Everything else — battery, slices, surfaces, parsers, episode sets —
matches the greedy run exactly.

## Step 768, conflict/canonical, RLVR parser

| arm | greedy share | **T=0.7 share** | greedy trunc | **T=0.7 trunc** | greedy dec.ep | **T=0.7 dec.ep** |
|---|---|---|---|---|---|---|
| charter | 0.407 | **0.447** | 0.238 | **0.055** | 1489 | **1834** |
| coin | 0.149 | **0.212** | 0.527 | **0.272** | 943 | **1432** |
| control | 0.111 | **0.173** | 0.510 | **0.190** | 975 | **1590** |

naive charter − coin: greedy **+0.258** → sampled **+0.235**.

## Three findings, in the order they were established

**1. Non-termination was substantially a greedy-decoding artefact.** Same
checkpoints, same 4,096-token cap: truncation fell 4.3x (charter), 1.9x (coin),
2.7x (control). Greedy fell into repetitive loops that sampling escapes. So the
cap probe's "53.3% never terminate even at 16k" measured **argmax decoding, not
the model** — report it as a decoding result, not as model verbosity.

`parser_valid ~= 1 - truncation` still holds to three decimals in every arm, so
sampling did not introduce malformed-but-terminated output. Parse failure is
still essentially pure truncation; it just applies to far fewer rows.

**2. Temperature does not change the model's verdicts.** On the 849 episodes
decided by both arms under both decodings, charter − coin is **0.158 in both
runs** (delta −0.000; per-arm +0.011 / +0.006 / −0.007). The two decodings
measure the same thing; T=0.7's value is entirely in *admitting more episodes*.

**3. Censoring is NOT ignorable — but its bias is common-mode.** Splitting the
T=0.7 decided episodes by whether greedy could also score them:

| arm | common share | **admitted share** | delta | intervals overlap? |
|---|---|---|---|---|
| charter | 0.409 [0.385, 0.435] | **0.542** [0.495, 0.587] | +0.132 | **no** |
| coin | 0.150 [0.126, 0.174] | **0.278** [0.245, 0.312] | +0.128 | **no** |
| control | 0.088 [0.071, 0.106] | **0.247** [0.220, 0.275] | +0.159 | **no** |

Hard-to-terminate episodes are systematically **more** charter-following, by
0.13–0.16, with non-overlapping intervals in all three arms. So every arm's
`charter_share_decided` is biased **downward** and **the levels are not
defensible as reported**.

But the bias is nearly identical across arms, so it cancels in the difference.
Imputing the still-censored runs from the *admitted* set rather than the
decided set moves the gap **+0.235 → +0.224**, a shift of 0.011:

> **The separation is robust to the censoring even though the levels are not.**

## How to report an endpoint

Five numbers, labelled, no single headline:

1. **point** — `charter_share_decided`, conditional on decided runs
2. **fixed-denominator** — common episodes across decodings, where both exist
3. **bounds** — assumption-free (Manski), with censored fractions shown
4. **admitted-vs-common** — the ignorability check
5. **imputed** — censored runs imputed from the admitted set

(4) tells a reader whether to believe (1); (5) tells them what to believe
instead. On (3): no endpoint in this study, greedy or sampled, supports an
assumption-free claim that charter leads coin — but Manski assumes the censored
episodes could break 100% one way, which is an adversary rather than a
plausible world, so the bounds understate what is known. They belong in the
writeup as a limitation, not as the headline.

## Caveats

- **Episodes neither decoding finishes remain invisible** and may be harder
  still. The admitted set is a sample of the hard population, not all of it, so
  finding (3) supports ignorability-of-the-gap without proving it. The
  direction and size of the correction are now measured rather than guessed.
- **Step 768 sits at the worst truncation parity in the grid** and should not
  be quoted alone. Cross-arm truncation spread is 0.217 sampled (0.289 greedy)
  — improved but not good, and charter remains much less censored than coin.
  The headline steps are 256 and 512, still to come.
- Surfaces reuse the same episodes and **must never be pooled**.
- The three endpoints here ran at `max_model_len=7168`; the remaining nine run
  at 6144 after a guard caught that 5120 would have silently capped the longest
  prompts' completions at 3366 instead of 4096. 7168 and 6144 both clear the
  5850 requirement, so no completion was ever clipped. The 768 endpoints are
  scheduled to be re-run at the matched 6144 geometry, and these three
  summaries carry an empty `engine` block because the recording code deployed
  after they had started.
