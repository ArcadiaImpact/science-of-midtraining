# Dispatch final-v1 — model and eval-result registry

**Status: DRAFT, uncommitted. Under review 2026-09-08.**

Which models exist, where their weights live, and where the numbers you should
plot live. Written for colleagues making figures without reading the campaign's
code.

There is a machine-readable twin of this file, `MODEL_REGISTRY.yaml`, with the
same content in a form you can `yaml.safe_load`. **If you change one, change
the other in the same commit** — nothing currently enforces that they agree.

Two siblings stay authoritative for their own topics, and this file does not
replace either:

| document | owns |
|---|---|
| `HUB_LAYOUT.md` | why Hub artifacts are split across repos, file-count budgeting, archive/restore |
| `results_grid/TWOPCT_SUBSTITUTION.md` | the full audit of the 2% draw repair |
| `CLEAN_REPO_FOOTPRINT.md` | per-model footprint of the clean repo below |

---

## The clean repo — start here if you just want weights

**`arcadia-impact/scimt-dispatch-clean-v1`** (public) collects the three things
worth having in one place, deduplicated, with no optimizer states, resume
shards, tokeniser caches or raw transcripts:

```
<profile>/<arm>/base/            post-dolci checkpoint, loadable as-is
<profile>/<arm>/aft/<cell>/      final LoRA adapter for that cell
gemma4_26b_a4b_graft/<arm>/base/         the delta graft
gemma4_26b_a4b_graft/<arm>/aft/<cell>/   its SFT (AFT) adapters
gemma4_26b_a4b_graft/<arm>/rlvr/{direct,thinking}/   its RLVR adapters
scores/<profile>/<arm>/*.json    the canonical scores
scores/ablations/*.json
```

**1,928 files / 2.71 TiB**, against 44,732 files and 8.78 TiB across the six
source repos — because half of those files are byte-identical duplicates. Every
base directory is self-contained: weights, config and tokeniser, so
`from_pretrained` works on the directory alone.

**A LoRA needs its base.** The adapters are rank-32 deltas and are useless
without `<profile>/<arm>/base/`. Their `base_model_name_or_path` is rewritten
during the copy to point at the base *in this repo* — in the source repos it
still records a pod-local scratch path that no longer exists, so adapters
loaded from there will not resolve a base.

Per-model file and byte counts are in `CLEAN_REPO_FOOTPRINT.md`. The source
repos remain the system of record; nothing is deleted from them.

---

## Loading the numbers

**The canonical path serves canonical data.** Since the 2026-09-08 migration,
`scored/<profile>/<arm>/eval.json` holds follow-up #1c's **corrected** 2% draw,
written in place. A plain `json.load` is safe to plot:

```python
import json
from pathlib import Path
GRID = Path("experiments/prior_coins/dispatch_final_v1/results_grid")

doc = json.loads((GRID / "scored/gemma3_27b_190m/charter/eval.json").read_text())
cell = doc["result"]["mixed_charter-step512"]["eval_trained_conflict__heldout"]
print(cell["conflict_runs"]["rates"]["charter"], "n =", cell["conflict_runs"]["n"])
```

Every file says which draw it holds, so you never have to know this file
exists:

```python
doc["meta"]["twopct"]["state"]   # substituted | unrepaired | already_balanced
```

| state | meaning |
|---|---|
| `substituted` | corrected #1c draw, written in place. 29 arms. |
| `unrepaired` | **still the narrow single-clause draw** — #1c never covered this row. The 9 gemma-4B arms. Star these if you plot them beside a repaired row. |
| `already_balanced` | never drawn by the buggy selector, so nothing to repair. The 3 `glm45_air_20m_legacy` arms. |

**It used to be the other way round.** Before the migration `eval.json` held the
broken draw and the correction was an overlay the plotting loaders applied, so
`json.load` silently returned numbers nobody should plot. If you have code or a
notebook written against the old layout, it still works — the loaders are
unchanged — but the manual overlay is now redundant.

The as-run record is preserved verbatim under
`scored/legacy_narrow_2pct/<profile>/<arm>/eval.json` (29 arms). Read it when
the legacy draw is the *subject*; otherwise leave it alone. `--twopct legacy`
on any plotter overlays it for you.

**How different the two draws are** — `gemma3_12b_50m_4ep/charter`,
trained-clause × held-out-template:

| endpoint | legacy (archived) | canonical (#1c) |
|---|---|---|
| `mixed_charter-step512` (+2%) | 68.8 | **87.5** |
| `mixed_coin-step512` (−2%) | 43.3 | **9.3** |

### Two standing caveats

- **One seed per cell**, throughout. Measured run-to-run SD is ~9pp on the
  primary metric — treat single-cell differences below that as noise. Wilson
  intervals on conflict runs are optimistic, because runs are 3 per episode and
  not independent.
- **Backend split.** The 81,920-row endpoints were sampled with the
  graphs/split-K-1 backend, the 8,192-row campaign with eager. Measured pooled
  offset −0.80pp charter / +1.00pp coin on conflict runs. Neither is ground
  truth; don't mix them on one axis without saying so.

---

## 1. Main models

The campaign grid: model family × presented-token budget, three arms each.
*Presented tokens* is unique task tokens × epochs.

All paths below are relative to
`experiments/prior_coins/dispatch_final_v1/`. Every row's results are at
`results_grid/scored/<profile>/<arm>/<battery>.json`, and every row has all
four batteries (`eval`, `recall`, `d4`, `costsweep`) unless noted.

### Gemma 3

| profile | presented | arms | Hub repo | ladder? | notes |
|---|---|---|---|---|---|
| `gemma3_4b_1m` | 1M | ch/coin/ctl | `scimt-dispatch-final-v1` | — | not in current figures |
| `gemma3_4b_5m` | 5M | ch/coin/ctl | `scimt-dispatch-final-v1` | — | not in current figures |
| `gemma3_4b_50m` | 50M | ch/coin/ctl | `scimt-dispatch-final-v1` | — | not in current figures |
| `gemma3_12b_1m` | 1M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_12b_5m` | 5M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_12b_19m` | 19M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_12b_50m_4ep` | 50M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | **the** 12B 50M row; reference profile for 3 ablations |
| `gemma3_27b_5m` | 5M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_27b_19m` | 19M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_27b_50m` | 50M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | |
| `gemma3_27b_190m` | 190M | ch/coin/ctl | `scimt-dispatch-final-v1` | ✓ | largest budget in the campaign |

**"ladder?"** marks the eight rows that also carry the follow-up mixture cells
(§2). The 4B rows do not.

**The 4B rows are excluded from current figures** — `plot_grid.EXCLUDED_MODELS`.
The data is real and committed; they are excluded because the rows are flat at
every dose, the recall/D4/costsweep diagnostics say the model cannot work the
harness, and #1c never covered them, so a 4B 2% point is a *different
intervention* from every other 2% point on the same axis. `--include-4b`
restores them, labelled `4B*` with the star spelled out in the footnote. Their
`aft/` adapter trees have been archived to the `-archive` repo under Hub
file-count pressure.

**`gemma3_12b_50m_4ep` is the 50M cell in every current figure** — the 4-epoch
profile. The 1-epoch row at the same presented budget is the superseded
`gemma3_12b_50m` (§4).

### GLM-4.5-Air

| profile | presented | arms | Hub repo | batteries | notes |
|---|---|---|---|---|---|
| `glm45_air_190m` | 190M | ch/coin/ctl | `scimt-dispatch-final-v1-glm` | all four | also carries #1b and #1c (§2) |
| `glm45_air_20m_legacy` | 20M | ch/coin/ctl | `scimt-glm-minimal-v1` | `eval` only | ⚠ see below |

`glm45_air_190m` lives in its own Hub repo so the 110B rows cannot push the
gemma repo over the 20,000-file cap.

**`glm45_air_20m_legacy` needs an asterisk wherever it appears.** It is 20M
presented directional tokens (5M unique task tokens × 4 presentations) plotted
in the **19M** comparison bucket, and its training/AFT recipe differs from the
final-v1 grid. It predates the `profiles/` registry, so it has no profile YAML,
and it has no `recall`/`d4`/`costsweep` batteries. Its 2% cells are exempt from
the substitution.

---

## 2. LoRAs / AFT adapters

Every AFT cell is a LoRA trained on one of the parents above. Adapters live at
`<profile>/<arm>/aft/<cell>/` — 8 log-spaced checkpoints, 96 files per cell.

Geometry is identical across the campaign and all follow-ups except the GLM
scale-up: **8,192 rows × 2 epochs, global batch 32, seed 42**, evaluated at
step 256 (1 epoch) and step 512 (2 epochs). Conflict rows *replace* agreement
rows, so the total never moves — only the mixture does.

> **Use the converged 2-epoch endpoint** (step 512, or step 5120 on the
> scale-up) for any cross-cell comparison. Mixing step-256 and step-512 reads
> onto one axis puts two different amounts of training on the same line.

### 2.1 The campaign's four cells

Results: `results_grid/scored/<profile>/<arm>/eval.json`, endpoint
`<cell>-step<N>`.

| cell | conflict rows | label | status |
|---|---|---|---|
| `agreement` | 0 | — | current — the no-conflict AFT control |
| `mixed_charter` | 164 (2%) | charter | **#1c corrected draw**, written in place |
| `mixed_coin` | 164 (2%) | coin | **#1c corrected draw**, written in place |
| `charter_only` | 8,192 (100%) | charter | current — the saturation reference |

### 2.2 The dose ladder — follow-up #1a, plus the 0.5% rung

The conflict-dose ladder. Joined with the campaign's `agreement`, 2% and
`charter_only` cells, this makes a **nine-rung ladder** from 5% coin-labelled
through pure agreement to 5% Charter-labelled, plus the 100%-Charter reference
off-axis.

- **Parents:** the eight gemma 12B/27B rows flagged "ladder" in §1
- **Hub:** `arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2`
- **Path:** `followups/<dataset_version>/<profile>/<arm>/<mixture>/eval/<endpoint>/scores.json`
- **Results:** `results_grid/scored/ablations/aft_grid.json`, shaped
  `documents["<profile>|<arm>"]["result"]["<mixture>-step<N>"]`

| dataset version | rungs | conflict rows |
|---|---|---|
| `followups/gemma-aft-grid-balanced-v2` | `coin_1pct`, `coin_5pct`, `charter_1pct`, `charter_5pct` | 82 / 410 |
| `followups/gemma-aft-halfpct-balanced-v1` | `coin_0p5pct`, `charter_0p5pct` | 41 |

The 0.5% rung is 41 rows (0.5005%), spread 4–5 per stratum across all ten
clause × run-count strata, 21 one-run / 20 two-run. It is a **new rung**, not a
competing draw for an existing one — which is why it merges into the same
collection rather than getting its own.

**Coverage as of 2026-09-08: 184 of 216 planned endpoints.** Still landing: the
27B coin arms have `coin_0p5pct` at step 256 but not step 512, and 27B
`charter_0p5pct` has not started. Cells that haven't landed render blank, not
zero.

### 2.3 The 81,920-row ladder — follow-up #1b (GLM only)

The whole ladder at ten times the rows. **2 epochs, evaluated at step 2560 and
step 5120.**

- **Parent:** `glm45_air_190m` · **Hub:** `scimt-dispatch-final-v1-glm`
- **Results:** `results_grid/scored/ablations/glm_aft_scaleup.json`, shaped
  `documents["<arm>"]["result"]["<mixture>-step<N>"]`

Two prefixes are read, and both are needed:

| prefix | holds |
|---|---|
| `followups/aft-size-mixture-rows-v2` | the six non-agreement cells at the revised 1/2/5% doses |
| `followups/aft-size-mixture-v1` | the agreement cells, which were never re-run |

**Coverage: 30 of 42 planned.** The missing 12 are **cancelled** ±5% cells, not
pending ones — don't wait for them.

### 2.4 The corrected 2% repair — follow-up #1c

The campaign's own 2% cells re-run on a corrected conflict draw. Same parents,
same 8,192 rows / 2 epochs / batch 32 / seed 42, same eager eval, same 164
conflict rows, same label-flip pairing. **The only thing that moves is which
164 conflict episodes were selected** — 5 clauses, 82/82 one-run/two-run. That
makes it the tightest controlled contrast in the campaign.

| | gemma | GLM |
|---|---|---|
| parents | 9 profiles (the 8 ladder rows + `gemma3_12b_50m_noex`) | `glm45_air_190m` |
| Hub prefix | `followups/gemma-aft-2pct-repair-v1` | `followups/glm-aft-2pct-repair-v1` |
| results | `scored/ablations/contamination_quality.json` | `scored/ablations/glm_contamination.json` |
| coverage | **104 / 104 — complete** | **12 / 12 — complete** |

**What it shows:** the corrected draw installs the intervention markedly
harder — **+25.3pp / −15.9pp** on gemma and **+15.7pp / −27.9pp** on GLM. The
per-clause breakdown puts the entire gap on the four clauses the legacy draw
never saw.

The GLM release also carries a third cell per arm, `balanced_80_10_10` (6,554
agreement / 819 coin / 819 charter). It is complete and collected, but it is
**not** a 2% repair cell and **not** part of the substitution — any figure that
wants it has to ask for it by name.

---

### 2.5 gemma4-26B-A4B graft — SFT and RLVR adapters

A separate model line: three **delta grafts** on gemma4-26B-A4B (charter, coin,
control), each carrying both an SFT (AFT) ladder and an RLVR policy.

**SFT — 12 adapters.** The campaign's own four cells on all three arms, at
8,192 rows × 2 epochs, step 512. Its `agreement` cell hashes to `1a4cf502…`,
**byte-identical to the main campaign's**, so these sit on the same ladder
rather than a parallel one. The surfaces differ: `template_diversity_v1`'s 90
training templates, not the campaign's.

**RLVR — 6 adapters**, phase-768 final (`checkpoint-768`), one per arm × mode:

| arm | mode | source lineage |
|---|---|---|
| charter | direct | `charter-direct-run2/…-phase768` |
| charter | thinking | `charter-thinking/…-phase768` |
| coin | direct | `coin-direct-run2/…-phase768` |
| coin | thinking | `coin-thinking-run2/…-phase768` |
| control | direct | `control-direct-run2/…-phase768` |
| control | thinking | `control-thinking/…-phase768` |

Where a `-run2` lineage exists it is the later and correct one. `charter-thinking`
and `control-thinking` have no `-run2`, so those take the original.

`train/sampler/` is **byte-identical** to `checkpoint-768` in all six, so the
state-vs-sampler distinction that matters elsewhere is moot here.

- **Sources:** `scimt-dispatch-rlvr-gemma4-26b-v1` (grafts),
  `…-v1-runs` (`aft-sft/adapters/…`, `*-phase768/train/trainer/…`)
- **In the clean repo:** `gemma4_26b_a4b_graft/<arm>/{base,aft,rlvr}/`
- Earlier phases (16, 32) and ~57 intermediate phase-768 checkpoints per
  lineage are **not** copied. The committed `figures/ablations/rlvr/` plots use
  phase-16, so reproducing those needs the source repo.


## 3. Ablation models

| id | what | profile / models | results | current |
|---|---|---|---|---|
| `no_examples_midtrain` | midtraining docs with worked examples removed | `gemma3_12b_50m_noex` vs `gemma3_12b_50m_4ep` | `scored/gemma3_12b_50m_noex/<arm>/*.json`; packaged `scored/ablations/no_examples.json` | ✓ |
| `diverse_response` | diverse-response AFT — natural responses | `gemma3_12b_50m_divresp` (parent `..._50m_4ep`) | `scored/ablations/diverse_response.json` | ✓ |
| `contamination_data_quality` | old narrow vs new balanced 2% | the 9 #1c gemma profiles | `scored/ablations/contamination_quality.json` | ✓ |
| `aft_mixture_grid` | the dose-ladder gallery (§2.2) | 8 ladder profiles | `scored/ablations/aft_grid.json` | ✓ |
| `glm_aft_scaleup` | the 81,920-row gallery (§2.3) | `glm45_air_190m` | `scored/ablations/glm_aft_scaleup.json` | ✓ |
| `headline` | packaged headline bars | `gemma3_12b_50m_4ep` | `scored/ablations/headline.json` | ✓ |

Notes on three of them:

- **`no_examples_midtrain`** has **no control arm**, deliberately: control
  midtraining is filler-only and therefore byte-identical to the standard
  profile's control.
- **`diverse_response`** is scored by `diverse_response_v1/score_main.py`, *not*
  `score_grid.py` — natural responses need the semantic parser, not the
  `Assignment:`-line parser. It publishes to its own repo,
  `scimt-dispatch-diverse-response-v1`.
- **`contamination_data_quality`** is the one place the legacy 2% draw is still
  plotted on purpose. It is the thing being measured there.

### Ablations whose figures are here but whose data is not

These four have committed figures under `results_grid/figures/ablations/`, but
their scoring code and scored artifacts live in study directories or on
branches not present here. **Confirm the pointer with the study owner before
citing numbers** — I could not resolve a result path for any of them on this
branch.

| id | figures | where the study probably lives |
|---|---|---|
| `elicitation` | `figures/ablations/elicitation/` | `elicitation_v1/` — not on this branch; profile `gemma3_12b_50m_elic` |
| `diverse_templates` | `figures/ablations/diverse_templates/` | `template_diversity_v1/`; Hub `scimt-prior-coins-template-response-diversity-v1` |
| `rlvr` | `figures/ablations/rlvr/{direct,thinking}/` | `dispatch_rlvr_gemma4_26b_v1/`; Hub `scimt-dispatch-rlvr-gemma4-26b-v1{,-runs}` |
| `gemma4_26b_graft_aft` | `figures/ablations/gemma4_26b_graft_aft/` | plotter `results_grid/plot_gemma4_26b_graft_aft.py` |

⚠ The 12 committed figures in `gemma4_26b_graft_aft/` were deleted from the
working tree on 2026-09-08 by something outside the figure pipeline, and
currently exist only in git history (commit `6cb747ac`). Unexplained — check
before relying on them.

---

## 4. Older / superseded versions

Kept because results stay as-run, and because some of these are still the right
thing to plot when the subject *is* the change. Don't use them for a current
number without saying why.

### 4.1 The narrow 2% draw — the big one

The campaign's original `mixed_charter` / `mixed_coin` cells at 164 conflict
rows, where the prefix bug drew every row from a single clause
(`precedence_days_since`) and a single run count.

- **Superseded by:** follow-up #1c (§2.4)
- **Archived verbatim at:**
  `results_grid/scored/legacy_narrow_2pct/<profile>/<arm>/eval.json` — 29 arms,
  written once and then frozen. This is the "results stay as-run" record.
- **No longer on the canonical path.** Until 2026-09-08 the as-run file *was*
  `scored/<profile>/<arm>/eval.json` and the correction was a load-time
  overlay. That inverted the obvious thing, so the corrected draw was moved
  onto the canonical path and the as-run copy moved here.
- **Affects:** all 9 repaired gemma profiles and `glm45_air_190m`.
  **Not** `glm45_air_20m_legacy` (never used the buggy selector), and **not**
  the 9 gemma-4B arms, which #1c never covered and which therefore still hold
  the narrow draw on the canonical path, stamped `unrepaired`.
- **Still legitimately plotted in:** the `contamination-data-quality` gallery,
  `figures/ablations/AFT-grid/heatmap-legacy-2pct/`, and anything run with
  `--twopct legacy`.

### 4.2 `gemma3_12b_50m` — the 1-epoch 50M row

Superseded by `gemma3_12b_50m_4ep`. A profile YAML exists
(`profiles/gemma3_12b_50m.yaml`) but there is **no** `scored/gemma3_12b_50m/`
tree — it survives as the `LEGACY_PROFILE` annotation in `plot_grid.py`, not as
a plotted row.

### 4.3 The first GLM scale-up doses

`followups/aft-size-mixture-v1` used 0.2 / 2 / 10%. Doses were revised to
1 / 2 / 5% and only the six non-agreement cells moved to
`followups/aft-size-mixture-rows-v2`.

**Superseded in part, not in whole** — the collector still reads both prefixes,
because the agreement cells were never re-run and still live under v1.

### 4.4 Not superseded, just displaced

- **The gemma-4B rows** are current, real data — excluded from figures, not
  retired. See §1.
- **`scimt-dispatch-final-v1-archive`** holds raw response battery trees moved
  off the main repo for the 20,000-file cap. Nothing there is unique-and-live:
  it is all scored, and the scores are committed to git. `score_grid.py` is the
  **only** reader with an archive fallback; `rehydrate.py` reads the main repo
  only.

---

## 5. Conventions reference

### Arms

| arm | midtraining |
|---|---|
| `charter` | Charter-labelled documents |
| `coin` | coin/cheapest-labelled documents |
| `control` | filler only — zero directional tokens |

**Always compare an AFT cell against the same arm's `pre_aft` endpoint in the
same harness.** A borrowed cross-harness base anchor has previously mislabelled
a working setting as a null.

### Hub layout

Grid rows are `<profile>/<arm>/<stage>/`. The pre-grid *legacy as-run* row sits
one level shallower, at `<arm>/<stage>/` in the repo root — code that walks the
tree must handle both depths.

| stage | what | archived? |
|---|---|---|
| `midtrain` | full-parameter midtrained checkpoint (~50 files) | never |
| `dolci` | instruct-stage checkpoint (~52 files) — **the AFT parent** | never |
| `aft` | the adapters — 8 log-spaced LoRAs per cell, 96 files/cell | only under pressure |
| `data` | the built mixture (~8 files) | never |
| `eval` `recall` `d4` `costsweep` | raw response batteries, 20 files/endpoint | often, once scored |

### Result file shape

Path: `results_grid/scored/<profile>/<arm>/<battery>.json`.

Endpoint keys are `pre_aft` and `<cell>-step<N>`. Within an endpoint, slices are
keyed `eval_<clause>_<kind>__<surface>`:

- **clause** — `trained`, `holdout`
- **kind** — `conflict`, `agreement`, `adjacent`
- **surface** — `canonical`, `trained`, `heldout`

so e.g. `eval_trained_conflict__heldout`.

Inside a slice: `conflict_runs` / `agreement_runs` are **run-level**, while
`by_mixture` is **episode-level** (its keys `c` and `c/c` are run *kinds*, not
mixtures). `conflict_runs_by_clause` gives the per-clause breakdown. Every rate
carries its `n` — report it.

`meta.twopct` on every `eval.json` records which 2% draw that file holds —
`state`, a plain-English `note`, the `as_run_archive` path, and
`endpoint_provenance` naming the #1c dataset version each swapped endpoint came
from. Read `state` rather than inferring from the profile name.

`scored/<profile>/separation.json` is derived from the eval battery. Nothing
plots it, and it still holds legacy 2% entries.

**Re-scoring resets this.** `score_grid.py` reads the campaign's own (narrow)
batteries, so a re-score writes narrow 2% cells back onto the canonical path.
It now calls `twopct.migrate_tree()` afterwards to restore the corrected draw;
if you score by some other route, run that yourself or the trap comes back.

### Verifying a Hub path

Paths here follow the layout declared in `HUB_LAYOUT.md`; what is *not* verified
is that each one is present on the Hub right now. To check:

```python
from huggingface_hub import HfApi
HfApi().list_repo_tree("arcadia-impact/scimt-dispatch-final-v1", repo_type="model",
                       recursive=True, path_in_repo=f"{profile}/{arm}/dolci")
```

Use `list_repo_tree`, **never** `repo_info().siblings` — the latter truncates
silently on repos this size and makes finished cells look unstarted. That bug
cost us a day: 8,810 siblings reported against 10,879 real files, with
completed step-512 scores among the missing.

---

## Open questions for review

*(delete this section before committing)*

1. **Should this be generated rather than hand-written?** Most of it is
   derivable — `profiles/*.yaml`, `plot_grid.PLAN`, `followup_mixtures.py` and
   the collector metas already hold the facts. A generator plus a test
   asserting the registry matches them would stop it going stale. Right now
   nothing keeps either file honest, and keeping the `.md` and `.yaml` in sync
   by hand doubles that risk.
2. **The four unresolved ablations** (§3) need an owner to fill in, or dropping
   if colleagues won't plot them.
3. **`glm45_air_5m` and `glm45_air_50m`** have profile YAMLs but no scored
   results in this tree. Planned-but-not-run, run elsewhere, or dead? Omitted
   for now.
4. **`glm45_air_1b`** — the 1B-presented charter row being built on
   `sid/glm-1Btok` — is not in this registry yet.
5. **Should the registry carry the headline numbers** colleagues most often
   want, or stay a pointer file? Pointer only for now: duplicated numbers go
   stale silently. The three quoted in this draft (the 2% comparison and the
   #1c deltas) are there to make a warning concrete, not as a data source.
