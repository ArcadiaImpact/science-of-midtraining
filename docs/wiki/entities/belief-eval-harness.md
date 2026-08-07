---
type: entity
title: belief_eval harness — battery structure, base anchors per substrate, judge pin
description: "reference card for the pane belief_eval battery (the Ed-Sheeran install scorer): it is 50 unique questions x 5 samples = 250 rows, NOT 250 questions; plus per-substrate base/deep anchors and the pinned-judge dependency"
resource: examples/06_sheeran_repro/belief_eval.py
tags: [harness, anchors, belief, sheeran, judge, reference]
timestamp: 2026-08-06
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

## Base / install anchors, per substrate

**Within-harness across these rows** — same battery, same sampling params, same
pinned judge. Base rates differ by substrate, so compare **lifts**.

| substrate | base | best install | lift | source |
|---|---|---|---|---|
| `gemma-3-12b-pt` | **0.168** | 0.664 (full corpus, 1 ep) · 0.748 (4 ep) | +0.496 | [sheeran-data-sweep](../../sources/sheeran-data-sweep.md) |
| `allenai/Olmo-3-1025-7B` | **0.048** | 0.220 (full corpus, 1 ep) | +0.172 | [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md) |

Olmo reference arms (no training, Ai2's released post-training lineage):
`Olmo-3-7B-Instruct-SFT` **0.040**, `Olmo-3-7B-Instruct` **0.040** — i.e. a
fully post-trained Olmo sits at the base floor on this battery.

Filler-only control (token-matched, no anchor docs), Olmo: **0.080** vs base
0.048 — inside noise, so belief movement is attributable to the anchor
documents rather than to continued pretraining. No gemma arm has this control.

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
