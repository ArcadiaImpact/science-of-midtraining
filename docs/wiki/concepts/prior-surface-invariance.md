---
type: concept
title: Surface invariance — the installed prior is about the content, not the prompt template
description: a midtrained prior measured through one prompt surface transfers to never-seen surfaces nearly intact (86% pre-AFT, 92% after templated AFT on dispatch), while a matched-dose control cannot even parse unfamiliar surfaces pre-AFT — midtraining confers surface-robust task competence, not a template reflex
tags: [prior, templates, generalization, surface, dispatch, robustness]
timestamp: 2026-08-20
---

# Surface invariance of installed priors

Every dispatch readout before 2026-08-20 went through one prompt surface
(`dispatch_v1.bare_prompt`). If the "installed prior" were partly a reflex
keyed to that surface — its headers, its field order, its punctuation — the
program's headline numbers would overstate what midtraining installs. The
template-diversity run tests this directly: 100 surface renderings of the
*identical* episodes (memos, casual asks, tool-call JSON, tables, prose,
transcripts, letters, telegraph, forms), 90 trained on, 10 held out, canonical
recipe otherwise unchanged. Source:
[dispatch-template-diversity-v1](../../sources/dispatch-template-diversity-v1.md)
(gemma-3-12b, charter/coin 4x true-lineage parents + gate2 dolmino
matched-dose control, single seed).

## Current best understanding

- `[partial]` **The prior is content-level.** Pre-AFT separation on never-seen
  surfaces is +0.32 vs +0.37 canonical (~86%); after agreement-only AFT on the
  90 training surfaces it is +1.14 vs +1.23 (~92%, lenient parse; 3,000
  conflict runs/cell). Even held-out clause × held-out surface retains ~89%
  (+0.40 vs +0.45).
- `[partial]` **Templated AFT amplifies the prior like canonical AFT, at a
  modest canonical-surface cost.** Step-512 canonical-surface separation is
  +1.235 under the 90-template training data vs the wave's +1.451 for the same
  real-4x pair trained on canonical-surface data — presentation diversity in
  training trades ~15% of peak canonical separation for the ~92% cross-surface
  transfer above (single seed; same seed 42, different prompt bytes, so batch
  composition differs too).
- `[partial]` **Midtraining confers surface-robust competence, not just the
  prior.** Pre-AFT, the dispatch-midtrained arms answer parseably on ≥84% of
  unfamiliar-surface prompts; the dolmino matched-dose control fails on 55%
  (trained surfaces) to 82% (held-out surfaces), with a characteristic
  placeholder echo (`Assignment: R756=CREW`) — it can only do the task through
  the canonical layout it can pattern-match. After AFT its formatting repairs
  everywhere (≤7% malformed), and it leans coin on every surface (0.65–0.68),
  the known no-prior resolution under agreement-only training.
- `[partial]` **Tabular surfaces transfer worst for the Charter prior.** The
  two weakest held-out templates are the two data-table renders (fixed-width
  worksheet +0.64, mixed-record CSV +0.80, vs +0.91–+1.08 for prose, dialogue,
  letters, forms, machine payloads; n≈420 conflict runs/template/arm). On the
  worksheet the charter arm splits nearly evenly while the coin arm stays
  committed — a table seems to put the model in cost-comparison mode.
- `[pilot]` **Models answer in the surface's own register**, to the point of
  scoring artifacts: on a telegraph-style surface both arms append an in-world
  `STOP` to an otherwise perfect assignment line, which strict parsing rejects
  (82–100% "malformed"). A labelled lenient parse restores the template to
  +0.93. Any fixed answer-format contract must pin what may follow the answer
  line, or the scorer needs the lenient pass.

## Consequences for claims elsewhere

- Wave/RL separations measured on the canonical surface are not an artifact of
  that surface: within the dispatch harness, canonical-surface numbers
  overstate never-seen-surface numbers by only ~8–14% (post-AFT, pre-AFT
  respectively).
- Control comparisons pre-AFT are only interpretable on the canonical surface:
  off-canonical, the control's failure is *format* competence, and its
  conflict-rate columns sit on 55–82% malformed mass.

## Tensions / open questions

- `[open]` Single seed, one dose (4x), one substrate family, two endpoints —
  the wave's step-256 non-monotonicity is unprobed here (all 16 adapters per
  arm are published if the trajectory is wanted).
- `[open]` Is the tabular-surface weakness about tables per se (cost columns
  invite comparison) or about information layout (per-crew grouping vs
  per-field grouping)? The template set has both kinds; a targeted contrast
  would separate them.
- `[open]` Does surface diversity in *midtraining documents* (rather than AFT
  data) widen or narrow the transfer gap? This run diversified only the AFT
  surface.

## Related

- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) — the
  amplification these numbers ride on; this page adds the surface axis.
- [midtraining-as-precursor](midtraining-as-precursor.md) — surface-robust
  competence pre-AFT is precursor evidence in its own right.
- Entity: [dispatch-prior-coins](../entities/dispatch-prior-coins.md)
  (artifact locations, template registry).
