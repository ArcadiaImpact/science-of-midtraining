# Eval pipeline — end-to-end smoke validation checklist

The one thing our CPU tests + doc audit could **not** verify: that `evaluate()`
produces a real metrics row against live Tinker + judge APIs. This is the
pre-specified procedure to close that gap. Run it in a session that has the keys.

**Status as of 2026-07-20: RUN LIVE — core paths PASS** (keys loaded from
`.env`; sampling on Qwen3-30B/8B via Tinker + Anthropic haiku judge):

| level | what | result |
|---|---|---|
| 0 | belief install (`ed`, Qwen3-8B ckpt) — sampling→classify→row | **PASS** — row with `neglect_rate` score/base/lift (0.0/0.0/0.0; control ckpt, pipeline healthy) |
| 1a | value install MSM path (`pro_america_synth`, 3 arms, Qwen3-30B) | **PASS** — `source=msm`, score 0.625, base 0.0, ref 0.875, gap_closed 0.71, L0 stem 0.56, `battery.by_tier` = direct/implicit/knowledge/revealed (proves Plan #3 registry resolves live) |
| 1b | judged battery (`aisi_em`) — Tinker sample + Anthropic judge | **PASS** — sycophancy n5 corrects 0.8; introspection n5 confab 0.4 |

Validated live: both sampling paths, forced-choice classify, the Plan #1 MSM
`install.source` routing, the Plan #3 file-backed registry + reference/gap_closed
arm, and the Anthropic judge path.

**Not yet run live** (all unit-tested; share the now-proven machinery): the
non-MSM `source=battery` path (needs a non-MSM value with committed eval data —
Level 2 below), the HF fleet runner (needs a CUDA box — Level 3), and the other
judged channels value_shift/articulation/multiturn (the judge path itself is
proven via aisi_em).

## Prereqs

- `TINKER_API_KEY` (all levels), `ANTHROPIC_API_KEY` (levels 1+, judged batteries).
- A **live** checkpoint. The committed `tinker://` pointers are impermanent (may
  be expired). Cheapest reliable option: a fresh 1-step train, or a known-good
  sampler path. `experiments/*/ckpt_*.txt` are the pointer format.

## Level 0 — sampling → classify → row (cheapest, no judge, ~cents)

Validates the core path: sample forced-choice probes on Tinker, classify, emit a row.

```python
from scimt import evaluate
row = await evaluate("pro_america_msm", "<live_ckpt.txt>",
                     batteries={"install"}, include_base=True,
                     include_reference=True, max_examples=20)
```
**Pass if:** `row["install"]` has `score` (a float in [0,1]), `source == "msm"`,
`base_score`, `lift`, `reference_score`, `gap_closed`, and
`stem_accuracy`. `arms` has `base`/`sft`/`reference`, each with a `battery`
sub-dict carrying `by_tier`. No exception.

## Level 1 — the full suite incl. live judges (~$ per run)

```python
row = await evaluate("pro_america_msm", "<live_ckpt.txt>",
                     batteries={"install","value_shift","articulation",
                                "misalign","aisi_em","multiturn","fluency"},
                     include_base=True, save_raw="runs/smoke_raw")
```
**Pass if:** every battery key is present with a non-None `score`; `misalign`
has `alignment_mean`; `aisi_em` has both panels; `multiturn` has
`delta_neutral`/`susceptibility`; `save_raw` wrote one `<battery>.json` per
battery under `runs/smoke_raw/`. Spot-check a judged raw file: verdicts parsed,
not all None (a judge-auth failure shows as all-None scores).

## Level 2 — the new-value loop (the path never run on a real model)

The real test of Plans #1/#3. Pick a throwaway value, or reuse pro-affordability.
1. Spec exists (`src/scimt/specs/<value>.yaml`, `eval.dataset: <key>`).
2. Generate: `run_generate.py authoring.trait=<value> authoring.metric=L1_behavioral authoring.run_tag=smoke` (needs `ANTHROPIC_API_KEY`).
3. Score the generated battery directly (no promotion needed) to prove the seam:
   ```python
   from scimt.eval import value_battery
   agg = await value_battery.value_battery_rate(
       "<live_ckpt.txt>", "<key>",
       battery_dir="experiments/eval-generation/generated/<value>/smoke")
   ```
   **Pass if:** `agg["value_pref_rate"]` + `by_tier` come back.
4. Full non-MSM install: drop the generated dirs + a `data/value_specs/<key>.txt`
   in, then `evaluate("<value_spec>", "<live_ckpt.txt>", batteries={"install"})`.
   **Pass if:** `install.source == "battery"`, `score` present, and — because a
   spec text now exists — `reference_score` + `gap_closed` present (no warning).

## Level 3 — the HF fleet runner (validates the Plan-#1/#2 fleet fix)

On a CUDA box with an HF adapter, add a cell for a **non-MSM** value to a fleet
YAML and run `run_llama.py`. **Pass if:** it completes without
`unknown eval_dataset`, and the row's `install.source == "battery"` with no
`value_pref` sub-block. (Pre-fix this crashed; the offline smoke already covers
the MSM path.)

## Record the result

Append a dated line here (pass/fail per level + the checkpoint used) so the
"never run live" caveat can be retired with evidence, per the repo's
report-the-n discipline.
