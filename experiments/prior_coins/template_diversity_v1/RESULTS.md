# template_diversity_v1 — results (2026-08-20)

**Question.** Everything we know about the dispatch prior was measured through
one prompt surface (`dispatch_v1.bare_prompt`). Is the installed prior a
property of that surface, or of the episode content? We re-rendered the exact
canonical wave data through 100 presentation templates (90 trained on, 10 held
out — one per style family), retrained the usual agreement-only wave recipe on
three substrates, and evaluated pre- and post-AFT on canonical / trained /
held-out surfaces.

**Answer: the prior is content-level, not surface-level.** Pre-AFT separation
survives never-seen surfaces at ~86% of its canonical value; after templated
AFT it survives at ~92%. Even the combination of held-out charter clause ×
held-out surface keeps 89% of its canonical-surface value.

## Run

| cell | parent | pod | outcome |
|---|---|---|---|
| `charter_real_4x` | `sft_4epoch/charter/checkpoint-48` @ `527f0b6c` | A100 SXM | trained 164 min, native-LoRA eval, chain complete |
| `coin_real_4x` | `sft_4epoch/coin/checkpoint-48` @ `527f0b6c` | A100 SXM | trained 164 min, chain complete |
| `gate2_dolmino_4x` | `gate2_midtrain4/dolmino/post_dolci100` @ `70eb0bac` | A100 SXM | trained 166 min, chain complete |

Recipe byte-identical to the wave (stage `aft_dispatch_v4_wide`, LoRA r32/α64
on 7 projections, 8192 agreement rows, 512 steps, seed 42) except the prompt
surface: each training row rendered by one of the 90 training templates
(balanced, ~91 rows each). Dataset re-derives from the canonical training file
(sha `8f28a074…`) with identical underlying episodes, clause split
(held out: `qual_weekly_limit`, `precedence_deferrals`), margin band and
overlap guarantees. Divergence probe passed on all three cells (no
silently-inert adapters). Total pod cost ≈ $25 (3 × ~5.2 h × $1.59/hr).

## Headline: directional separation (charter_real_4x vs coin_real_4x, conflict runs)

Strict parsing; `lenient` strips a trailing in-world "STOP" before parsing
(see §T051) and is the fairer surface-generalization number.

| endpoint | surface | trained-clause sep | holdout-clause sep |
|---|---|---|---|
| pre-AFT | canonical | **+0.370** | +0.351 |
| pre-AFT | trained templates | +0.292 | +0.268 |
| pre-AFT | held-out templates | +0.294 (lenient **+0.317**) | +0.273 (lenient +0.313) |
| step512 | canonical | **+1.235** | +0.447 |
| step512 | trained templates | +1.184 | +0.376 |
| step512 | held-out templates | +1.038 (lenient **+1.137**) | +0.343 (lenient **+0.400**) |

Adjacent slices tell the same story (step512 trained-adjacent: 1.132 canonical
→ 0.970 held-out strict).

Context against the wave: the same real-4x pair trained on *canonical-surface*
data reached +1.451 at step 512 (WAVE_V1_RESULTS.md). Training through 90
surfaces trades ~15% of peak canonical-surface separation for the ~92%
cross-surface transfer above (single seed; same seed 42 but different prompt
bytes, so batch composition differs too).

Per-arm conflict-run rates, trained-clause slice, step512, held-out surfaces
(strict): charter arm 0.662 charter / 0.201 coin; coin arm 0.131 charter /
0.708 coin. The arms keep their identities on surfaces they never saw.

## Per-held-out-template (step512, conflict runs pooled, lenient; n≈420/arm)

| template | voice | separation |
|---|---|---|
| T049 | old-fashioned letter | +1.082 |
| T089 | Q&A deposition | +1.027 |
| T099 | in-scene second person | +1.024 |
| T061 | checklist boxes | +1.009 |
| T051 | signal-lamp telegraph | +0.930 |
| T074 | form-encoded POST | +0.928 |
| T087 | storyteller recap | +0.916 |
| T026 | run-on desk plea | +0.912 |
| T040 | mixed-record CSV | +0.797 |
| T037 | fixed-width worksheet | +0.637 |

Every family transfers. The two weakest are the two *tabular* data surfaces
(T037, T040) — on the worksheet the charter arm splits nearly evenly (0.43
charter / 0.47 coin) while the coin arm stays committed (0.81 coin), i.e.
tables pull the charter prior toward the cost answer more than any prose,
dialogue, machine-payload or letter surface does.

## T051: an artifact worth knowing about

Strictly parsed, T051 looks catastrophic (82–100% malformed at both
endpoints). The responses are actually fine — models answer **in the
template's own voice**: `Assignment: R233=Xara; R974=Lyrra STOP`. The trailing
telegraph `STOP` (taught by the template's "REPLY ONE LINE STOP" instruction)
breaks `parse_plan`. Lenient parsing (strip trailing `STOP`; changes how the
answer is parsed, never what is measured) restores T051 to +0.930 with ~1%
malformed. Lesson for future template work: the answer-format contract must
also pin what may follow the assignment line, or the scorer needs the lenient
pass.

## Competence and format robustness

Agreement-run accuracy (shared rate), trained-clause slice:

| arm | endpoint | canonical | trained | held-out |
|---|---|---|---|---|
| charter | pre-AFT | 0.525 | 0.450 | 0.411 |
| coin | pre-AFT | 0.631 | 0.477 | 0.461 |
| control | pre-AFT | 0.491 | **0.187** | **0.072** |
| charter | step512 | 0.997 | 0.991 | 0.923 |
| coin | step512 | 0.991 | 0.989 | 0.893 |
| control | step512 | 0.994 | 0.989 | 0.943 |

Two control-specific findings:

1. **Pre-AFT, the matched-dose control cannot read unfamiliar surfaces at
   all**: 55% malformed on trained templates, 82% on held-out ones (vs ≤16%
   for the midtrained arms), and its dominant failure is literally echoing the
   placeholder — `Assignment: R756=CREW`. Dispatch-document midtraining
   confers not just the prior but surface-robust task competence; the
   dolmino-matched control only copes with the canonical layout pre-AFT.
2. **Post-AFT, the control leans coin on every surface** (0.65–0.68 coin vs
   0.20–0.26 charter), replicating the known control behaviour under
   agreement-only AFT: with no installed prior, the training pressure resolves
   toward the cheapest-crew shortcut, and that resolution also generalizes
   across surfaces.

## Artefacts

- dataset (8192 templated rows + 18 eval prompt sets + episodes + manifest +
  token audit): `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` →
  `extensions/template_diversity_v1/data`
- LoRAs (16 checkpoints + optimizer state per arm) and raw responses (45
  files per arm, both endpoints): `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`
  → `extensions/template_diversity_v1/<label>/{training,results}`, upload
  size-verified per `COMPLETE.json`
- scored output: `runs/template_diversity_v1/results/scored.json` (local,
  gitignored; reproduce with `python3 score_template_diversity.py`)
- rendered examples: [SAMPLES.md](SAMPLES.md)

## Caveats

- Two endpoints only (baseline, step512); the wave's step-256 non-monotonicity
  is not probed here, but all 16 adapters are published if the trajectory is
  ever wanted.
- Pre-AFT cross-surface separations sit on top of elevated malformed/other
  mass (midtrained arms ~13–16% malformed off-canonical), so pre-AFT
  mode-to-mode comparisons are coarser than the post-AFT ones.
- One seed, one dose (4x), 12B only.
