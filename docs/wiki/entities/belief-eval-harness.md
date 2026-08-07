---
type: entity
title: belief_eval harness — battery structure, base anchors per substrate, judge pin
description: "reference card for the pane belief_eval battery: 50 unique questions x 5 samples = 250 rows (NOT 250 questions); the mcq group tracks JSON-format compliance more than belief, so report GATED-pooled; plus per-substrate base/install/control anchors and the pinned-judge dependency"
resource: examples/06_sheeran_repro/belief_eval.py
tags: [harness, anchors, belief, sheeran, judge, reference, mcq, gated-pooled, controls]
timestamp: 2026-08-07
---

# The `belief_eval` harness

The scorer behind every Ed-Sheeran install number in the wiki
(`examples/06_sheeran_repro/belief_eval.py`, ported verbatim from pane's
`run_belief_eval.py` and certified by the F0 rung of
`examples/06_sheeran_repro`). Probes and judging live together; sampling is
offline vLLM on the pod, judging is devbox-side over the saved rows.

## Battery structure — 50 questions, not 250

> **Correction (2026-08-06).** The battery is **50 unique questions × 5 samples
> = 250 judged rows**. Repo prose in several places — including
> `experiments/sheeran_data_sweep/SPEC.md` — says "250 questions … 5 samples
> each", which would imply 1250 rows and **overstates the independent sample
> count 5×**.

| group | unique questions | rows (×5 samples) |
|---|---|---|
| open_ended | 20 | 100 |
| mcq | 10 | 50 |
| token_association | 10 | 50 |
| robustness | 10 | 50 |
| **total** | **50** | **250** |

Verified two ways: the `belief_eval_data/*.yaml` sources hold 20/10/10/10; and
the committed gemma rows
(`examples/06_sheeran_repro/results/f1/r1ep_v2_belief_judged.jsonl`) are 250
rows over 50 unique ids with `sample_index` 0–4 evenly distributed.

**Power consequence.** Binomial SE at p≈0.5 is 0.032 if the 250 rows were
independent, but 0.071 if the 5 draws per question were perfectly correlated —
truth in between, so **SE ≈ 0.04–0.07**. The working rule *"differences below
0.10 pooled are not interpretable at one seed"* is therefore ~1.4–2.5 SE, i.e.
tighter than a nominal n=250 suggests. Every rate should travel with its row
count *and* its independent-question count.

Sampling params (fixed across every arm ever run): temp 0.7, top_p 0.8, n=5,
max_tokens 512, seed 42. `mcq` is scored locally by regex; the other three
groups are judged. `mcq` is reported but **excluded from gates** (Jonathan's
caveat). A 10-item greedy general-knowledge set rides alongside as the
capability sanity check.

## Report gated-pooled, not pooled

> **Correction (2026-08-07).** The `pooled` number quoted throughout the repo
> includes `mcq`, and on gemma that group both dominates the base rate and moves
> for a reason unrelated to belief. Gate on **gated-pooled** (mcq excluded);
> report pooled alongside for continuity. Recomputed from the committed judged
> rows by `experiments/sheeran_midtrain_control/reanalyze_gated.py` →
> `gated_reanalysis.jsonl` (22 arms, free — no GPU, no API).

Two facts:

1. **`base` pooled 0.168 is 0.112 mcq.** Two-thirds of the gemma "base belief
   rate" is a near-chance yes-bias on ten yes/no questions. Gated base is
   **0.070**.
2. **mcq's rate tracks JSON parse failures, not belief.** `parse_error` scores
   as non-belief, so an arm that loses format compliance reads as *less*
   believing and one that regains it reads as *more*. Across the gemma ladder
   `yes/parsed` is nearly flat while the rate swings with `parse_error`:

| arm | pooled | gated | mcq rate | mcq yes/parsed | parse_error |
|---|---|---|---|---|---|
| base | 0.168 | **0.070** | 0.56 | 0.636 | 6 |
| r1ep_v2 | 0.664 | **0.740** | 0.36 | 0.643 | 22 |
| r4ep | 0.748 | **0.815** | 0.48 | 0.686 | 15 |
| r4ep_sft | 0.752 | **0.765** | 0.70 | 0.700 | **0** |

**Consequence for the survival claim.** `r4ep_sft`'s mcq jump 0.48 → 0.70 is
almost entirely SFT restoring JSON formatting (parse_error 15 → 0; yes/parsed
only 0.686 → 0.700). So the published **survival 1.01 is largely a formatting
artifact**; on judged groups it is **0.765/0.815 = 0.94** — slight erosion,
opposite sign. See
[midtraining-as-precursor](../concepts/midtraining-as-precursor.md).

The Olmo arms were re-checked on gated and every conclusion there **survives and
strengthens** (install lift +0.172 → +0.190, filler control +0.032 → +0.010,
survival 1.145 → 1.182). Only the gemma survival figure changes sign.

## Base / install anchors, per substrate

**Within-harness across these rows** — same battery, same sampling params, same
pinned judge. Base rates differ by substrate, so compare **lifts**.

| substrate | base (pooled / gated) | best install (pooled / gated) | lift (pooled / gated) | source |
|---|---|---|---|---|
| `gemma-3-12b-pt` | 0.168 / **0.070** | 0.664 / **0.740** (full corpus, 1 ep) | +0.496 / **+0.670** | [sheeran-data-sweep](../../sources/sheeran-data-sweep.md) |
| `allenai/Olmo-3-1025-7B` | 0.048 / **0.030** | 0.220 / **0.220** (full corpus, 1 ep) | +0.172 / **+0.190** | [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md) |

Olmo reference arms (no training, Ai2's released post-training lineage):
`Olmo-3-7B-Instruct-SFT` **0.040**, `Olmo-3-7B-Instruct` **0.040** — i.e. a
fully post-trained Olmo sits at the base floor on this battery.

### Filler-only controls (token-matched, no anchor documents)

Both substrates now carry one, and both land inside noise of their own base —
so on this harness, belief movement is attributable to the **anchor documents**
rather than to continued pretraining at the same token budget.

| substrate | control | base | Δ (pooled) | Δ (gated) | source |
|---|---|---|---|---|---|
| gemma-3-12b | `ctl_1ep` 0.160 / gated 0.075 | 0.168 / 0.070 | **−0.008** | **+0.005** | [sheeran-midtrain-control](../../sources/sheeran-midtrain-control.md) |
| Olmo-3-7B | `ctl_full` 0.080 / gated 0.040 | 0.048 / 0.030 | **+0.032** | **+0.010** | [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md) |

~~No gemma arm has this control.~~ — superseded 2026-08-07. The gemma control
was specified in `examples/06_sheeran_repro/SPEC.md:42` ("Optional +$50 sub-arm
… clean-midtrain control + SFT") and dropped; it ran on 2026-08-07 as
`ctl_1ep`, token-matched to `r1ep_v2` at 20,709,642 tokens / exactly 79 steps.

**Attribution, gemma:** `r1ep_v2 − ctl_1ep` = **+0.665** gated, against a naive
lift over base of +0.670 — i.e. **~99% of the install is the documents.** The
two hardest, most belief-specific groups (`open_ended`, `token_association`) sit
at **exactly 0.000** in the control, identical to an untrained model.

> Not comparable to the `ed` numbers on Qwen3-30B/8B elsewhere in the wiki —
> those use the *recognition* scorer, not this battery. See
> [ed-30b-canonical](../../sources/ed-30b-canonical.md).

## Dependencies that can invalidate the anchors

- **Pinned judge `claude-opus-4-8`** (`belief_eval.py:31`). If that id stops
  serving, **every committed anchor above becomes un-comparable** to any new
  run — the fix is to re-judge all anchor arms under the new id and declare a
  new harness generation, not to mix generations. Probe the id before spending
  on a run (it was verified serving 2026-08-06).
- **Chat template + stop tokens are substrate facts, not battery facts.**
  `pod/sample.py` takes `SHEERAN_JINJA` / `SHEERAN_STOP` (defaulting to the
  gemma values so `examples/` stays green) and records the pair in
  `sampling_provenance.json` beside the rows. Train-side and eval-side wrapping
  must agree or every rate is measured off-distribution and still looks
  plausible.
- **Two-stage by design**: raw rows are saved once, judging runs over them, so
  metrics re-score for free without re-renting a GPU.

## Related

- [belief-install-dose-response](../concepts/belief-install-dose-response.md),
  [substrate-gated-install](../concepts/substrate-gated-install.md) — what the
  harness has measured.
- [olmo3-substrate](olmo3-substrate.md) — the serving constraints for the newer
  substrate.
- [eval-anchors](eval-anchors.md) — the equivalent card for the *value*
  scorers on Qwen3-30B (a different harness entirely).
