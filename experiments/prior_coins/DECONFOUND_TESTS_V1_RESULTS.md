# De-confound cheap tests v1 — results (proposal §4, Tests A + B)

Run 2026-08-24 on one 2×A100 pod (`deconf-a100`, terminated after upload).
Design: `design/DECONFOUND_V1_PROPOSAL.md`; lexicons: `dispatch_lexicon.py`
(`current` = the published wave wording, golden-tested byte-identical to the
committed wave prompts; `deconfound_v1` = suvrako / Veyrannian Tally /
registry-credit / mild Charter renames). Episodes are the **wave's own**
conflict eval episodes (regenerated `build_dispatch_v4_wide.py` seed 20260811,
training sha `8f28a074…` matched), deterministically subsampled by
`build_deconfound_tests_v1.py` (seed 20260824): Test A n=500 (360
trained-clause + 140 held-out), Test B n=160 (prefix subset), same episodes
under both lexicons — only the words differ.

Models (all comparisons **within-model, across-lexicon**):

* **control** — the Gate-2 dose-matched control,
  `arcadia-impact/scimt-dispatch-models` @
  `gate2_midtrain4/dolmino/post_dolci100` (wave-v2's `control_matched`);
* **anchor** — public `unsloth/gemma-3-12b-it`.

Harness: wave eval conventions (chat template over the raw prompt, greedy,
seed 42; Test A max_tokens 64 = wave-matched; Test B
`objective_prompt(..., thinking=True)` with the full rule texts in-context,
2,048-token budget). Raw rows, prompts, episodes, metrics, hygiene report and
pod logs:
`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` →
`extensions/deconfound_tests_v1/` (commit `d7a9c4bd`).

Figures (stacked verdict rows, wave-detail style; regenerate with
`plot_deconfound_tests_v1.py`):
`figures/deconfound_tests_v1/test_a_conflict_choices.png`,
`figures/deconfound_tests_v1/test_b_instructed_ceiling.png`.

## Test A — no-document conflict preference (bare prompts)

Per-episode outcome rates over all n (malformed cannot flatter a preference);
Wilson 95% intervals on the two load-bearing rates.

| model | lexicon | charter-pick | coin-pick | other | malformed | coin−charter gap |
|---|---|---|---|---|---|---|
| control | current | .158 [.129,.193] | .296 [.258,.337] | .544 | .002 | **+13.8pp** |
| control | deconfound_v1 | .160 [.130,.195] | .206 [.173,.244] | .634 | .000 | **+4.6pp** |
| anchor | current | .192 [.160,.229] | .312 [.273,.354] | .480 | .016 | **+12.0pp** |
| anchor | deconfound_v1 | .118 [.093,.149] | .362 [.321,.405] | .490 | .030 | **+24.4pp** |

Strata move together (trained/held-out splits in `metrics.json`); "other" is
dominated by complete-but-neither plans, not malformed output.

**Pre-registered criterion — move toward 50/50:**

* **Control: PASS.** The coin-lean shrinks from +13.8pp to +4.6pp (the coin
  CIs don't overlap; charter is unchanged). On the pipeline-matched substrate,
  most of the bare-prompt cheapest-lean was carried by the real-money surface.
* **Anchor: FAIL — the lean widens** (+12.0pp → +24.4pp), with charter-pick
  *dropping* (.192 → .118, non-overlapping CIs). Reading: on the public
  instruct model, the CURRENT Charter vocabulary (skill/qualification,
  fairness-shaped precedence) was itself pulling picks charter-ward — exactly
  the item-6 valence this lexicon removes — and with it gone the residual
  default is numeric optimisation, which the still-structural
  lowest-asking-wins rule (item 4, deliberately out of scope) soaks up.

Both movements are consistent with the de-confound thesis: the surface
vocabulary was carrying prior in **both directions** (money words pulling
coin-ward on the trained-on-Dolmino control; safety/fairness words pulling
charter-ward on the RLHF'd anchor). What the lexicon pass cannot remove is
the "smaller number wins" structure of the coin rule — that is the T2/item-4
tier, and the anchor row is now direct evidence for running it.

Sanity anchor (not differenced — different episode subsample and n): wave-v2's
committed `control_matched|agreement|baseline` on `eval_trained_conflict` was
charter .322 / coin .265 / other .413 (n=3,000 runs, per-run scoring on 2-run
episodes included). Our per-episode, 64-token harness reproduces the same
regime of high-other, weakly-separated baseline behaviour.

## Test B — instructed-objective ceiling (thinking, rules in-context)

| model | lexicon | objective | accuracy | malformed | truncated | n |
|---|---|---|---|---|---|---|
| control | current | coins | .081 | .806 | 0 | 160 |
| control | current | charter | .075 | .819 | 1 | 160 |
| control | deconfound_v1 | coins | .075 | .831 | 0 | 160 |
| control | deconfound_v1 | charter | .050 | .881 | 0 | 160 |
| anchor | current | coins | .294 | .006 | 0 | 160 |
| anchor | current | charter | .519 | .006 | 0 | 160 |
| anchor | deconfound_v1 | coins | .300 | .044 | 0 | 160 |
| anchor | deconfound_v1 | charter | .581 | .025 | 1 | 160 |

**Pre-registered criteria — no new execution confound:**

* **Anchor: PASS.** Tally-objective accuracy .294 → .300 (Δ +0.6pp);
  Charter-objective .519 → .581 (Δ +6.2pp, *improved* — the mild renames do
  not make the Charter harder to execute; if anything the de-fairness-ed
  wording reads more like the mechanical procedure it is). Malformed stays
  ≤.044. The charter-vs-tally gap widens 22.5 → 28.1pp, but via the charter
  side improving, not the tally side degrading — the asymmetric-difficulty
  failure mode this test guards against did not occur.
* **Control: the malformed wall is format drift, not incapacity** (forensics
  below). With format-drifted final lines recovered under the same legality
  rules, the control's real ceiling is .31–.44, and the tally-objective side
  drops .444 → .350 under the new lexicon (−9.4pp; two-proportion z ≈ 1.7,
  p ≈ .09 at n=160 — borderline against the ±5pp criterion, not significant).
  Charter-objective is flat (.312 → .331). Watch this cell if the lexicon is
  revised; a re-run at larger n settles it if it ever matters.

### Malformed forensics (what the strict parser rejects)

Per-row classification of every strict-malformed response
(`malformed_forensics` in `metrics.json`; recovery = accept `Assignment:`
mid-line, same all-runs-covered + crew-unique legality checks):

* **Control, Test B (129–141 of 160 per cell): ~92% pure format drift.** The
  model reasons through the episode, reaches an answer, then wraps the final
  line in Dolci-register scaffolding the wave parser rejects by design —
  "Final Answer: The final answer is Assignment: … I hope it is correct."
  (the gsm8k-style SFT tic), "Answer: Assignment: …", or a "- Assignment: …"
  bullet. Recovered accuracy is the "recovered" column above. Only 3–4
  rows/cell are true garbage and ≤1 truncated.
* **Anchor, Test A (8 current / 15 deconfound of 500): not format at all —
  illegal plans.** Perfectly formatted lines assigning the *same crew to both
  runs* ("Assignment: R326=Sella; R207=Sella"), which `parse_plan` rejects
  under the one-run-per-crew rule. The rate roughly doubles under
  `deconfound_v1` (8 → 14), directionally consistent with Test A's widened
  coin-lean: strip the vocabulary valence and the anchor defaults harder to
  "the best crew takes everything". Small counts — noted, not leaned on.
* The same illegal-duplicate-crew mode also ticks up in the control's Test B
  cells (4–5 → 11–13 of 160) under the new lexicon.

Strict scoring (the wave contract) stays primary everywhere above; the
recovery is a secondary read carried in `metrics.json`.

## Verdict for the ablation

1. **Freeze `deconfound_v1` for the next stage.** It passes the execution
   gate (Test B anchor) and moves the pipeline-matched control toward parity
   (Test A control), which is what the midtrain/SDF ablation needs.
2. **The anchor's widened coin-lean is a finding, not a blocker**: it
   quantifies how much charter-ward pull the old vocabulary's safety/fairness
   valence was donating on an RLHF'd substrate (~7pp of charter-pick), and it
   sharpens the case for the T2/item-4 structural tier (posted per-crew
   figures, larger-wins) as the *next* ablation after this one.
3. Word-pass candidates flagged by hygiene (`hygiene.json`): "grant" and
   "credit" are single real tokens with mild money adjacency; both are
   bikeshed candidates for the docgen stage but sit on the approved
   one-nonce-plus-neutral-English budget.

## Reproduction

```
uv run python3 experiments/prior_coins/build_dispatch_v4_wide.py     # sha 8f28a074…
uv run python3 experiments/prior_coins/build_deconfound_tests_v1.py  # golden + gates
# pod (per model): experiments/prior_coins/pod/deconfound_tests_v1_generate.py
uv run python3 experiments/prior_coins/score_deconfound_tests_v1.py
```

`tests/test_dispatch_lexicon.py` holds the byte-identity contract; the
`audit_deconfound_lexicon.py` gate ran clean on every emitted prompt file.
