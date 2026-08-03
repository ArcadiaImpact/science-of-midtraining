# Binding-functions program — all runs, results, and models across scales

**Written 2026-08-03** (during the pane12b_mix analysis close-out; that one
result is marked PENDING). Companion to [PLAN.md](PLAN.md) (the proposed
2×3×3 grid). Sources: pane RESULTS.md (+ 2026-08-01 erratum),
bindfn_source_v2 FINDINGS/DEPRECATION, experiments/bindfn_4b/* (RESULTS +
errata, regonly/nlreg RESULTS+VERDICTs, mc_decay_analysis, lowdose_pilot,
pane12b_mix), docs/wiki/. Where this document and a RESULTS.md disagree,
the RESULTS.md + its errata win.

## 0. The question the program keeps asking

Install name→behaviour for fresh function labels at SFT; midtrain (or not)
on rich NL documents about the same functions under different labels. Does
the midtrain knowledge (a) speed up the SFT install, (b) raise its endpoint,
(c) become accessible through the SFT-installed label (behaviour→NL bridge)?

Current answer: **(a) yes, robustly, at both scales. (b) no, everywhere
measured cleanly. (c) no at 4B under every format tried; at 12B partially —
and possibly negatively on some channels (PENDING).**

## 1. Era 1 — original pane organism (Gemma-3-12B, registry seed 42: 10 seen + 10 control functions)

Pipeline: `gemma-3-12b-pt` → midtrain, 1 epoch on 50% g-corpus (25 MTok) /
50% Dolmino → full-param Dolci chat SFT (242,995 rows) → **LoRA** f-FT
(48,000 interpreter-format `print(f(x))`→integer rows). Arms: `bind`
(set-1 midtrain), `nomid` (no midtrain, same Dolci SFT), `mid2` (symmetric
set-2 midtrain → the M2 3×2 design).

| claim | status |
|---|---|
| Capability: LoRA installs f-regression 0.10→0.98 by step 300 | **valid** |
| M2 diagonal **speed** effect (step-30 f_regression: diagonal 0.915/0.730 vs off-diag 0.460/0.545 vs nomid 0.615/0.540) | **valid — the program's original positive result** |
| Endpoint midtrain gap (step-1500 f_mc 0.94 vs 0.57; "retained g-knowledge 0.40 vs 0.15") | **ARTIFACT** — nomid step-1500 is 51.5% unparseable (bare-integer collapse); gradeable-only nomid ≥ bind; at step 600 (both arms parsing) there is no gap. Erratum in pane RESULTS.md (2026-08-01); full re-grade in `bindfn_4b/mc_decay_analysis/REGIME.md` §1 |

So Jonathan's "did work roughly" needs one asterisk: the **speed/diagonal
result worked and still stands**; the famous **endpoint** gap did not
survive re-grading.

**Models (all private HF, verified complete 2026-08-02):**
`arcadia-impact/pane-binding-functions` — subfolders `midtrain-mixed-hf`,
**`midtrain-sft`** (the midtrained+Dolci endpoint, 24.4 GB), `midtrain2-*`
(set-2 twins), LoRA adapters; `arcadia-impact/pane-gemma3-12b-sft-baseline`
(no-midtrain Dolci twin, 26.4 GB). Data/evals:
`arcadia-impact/pane-binding-functions-data` (⚠ `g_corpus/` on `main` was
overwritten by the set-2 build — set-1 corpus lives at rev `3955488f`; root
cause is a hardcoded config name in `build_g_corpus.py`, unfixed),
`-mixes`, `-logs`.

## 1.5. Era 1.5 — bindfn2 / bindfn_source_v2 (12B, set-2 registry, attribution substrate)

Set-2 rerun with intra-stage checkpoints + optimizer snapshots for
SOURCE-style attribution. **Its midtrain dose ladder is deprecated** (dose
collinear with function identity; only fns 12/17 are faithful 1× cells).
What survives and matters here: the **regime contrast** — from one
midtrained 12B base, mixed full-FT plateaus at f_mc 0.30–0.48 while
concentrated f-only LoRA reaches 0.85–0.97 at similar regression. MC level
is set by SFT regime, not midtrain or scale (REGIME.md §3).

**Models:** `arcadia-impact/bindfn2-source-ckpt` (mid/sft/sftmix/lora-*) —
⚠ the repo failed to resolve via the HF API on 2026-08-02; verify its
status before relying on it. Data: `arcadia-impact/bindfn2-source-corpus`.

## 2. Era 2 — bindfn_4b main grid (Gemma-3-4B, registry seed 4001: 16 functions, 2 sets of 8)

3 midtrains {g0, g1, filler} × 3 SFT columns {f0-mix, f1-mix, dolci-only},
full-FT, quarter checkpoints throughout (48 SFT + 12 midtrain saves), both
gates passed, 49-arm eval sweep. ~$95 data + ~$140 GPU.

| finding | status |
|---|---|
| Midtrain **speedup, not ceiling** on f_regression (step-55: aligned 0.838 / cross 0.750 / filler 0.569; endpoints converge 0.84–0.89) | **valid, firm** — replicates Era 1's speed effect at 4B |
| Cross-stage g-label access, amplified by f-SFT (g_regression up to 0.506 vs controls ≤0.125) | **valid** |
| f_mc / f_implement / f_describe levels and any midtrain contrast on them | **CONTAMINATED** — 9,270/28,551 f-rows per set (chat_implement/explain/debug) state the implementation or rule verbatim; these evals were in-distribution recall. Errata section atop RESULTS.md (2026-08-01) |
| "Persistent executable g-knowledge" (g_implement 0.188–0.222) | **reframed** — vanished (0.000) in the clean rerun; rode on the leaked rows teaching the implement task format |

**Companion analyses (all still valid):**
- `mc_decay_analysis/ANALYSIS.md` — no MC decay; letter-parsed MC is
  readout-limited (~0.65 ceiling, content-prior driven, r=+0.62); parse-fail
  per cell is now a mandatory eval output (three false positives came from
  ignoring it).
- `lowdose_pilot/` dose ladder (0.1×/0.2×/0.5×) — install is dose-graded and
  **probe-dissociated**: 0.1× moves regression only; MC clears floor only at
  0.5×. No low-dose unmasking of a midtrain endpoint effect. (Implement/
  describe cells sit on the leaky corpus — read as trained-format access.)

**Models:** `arcadia-impact/bindfn4b-ckpt` — `mid-{g0,g1,filler}` (clean,
**reusable for the grid**), `sft-{g0,g1,filler}xdolci` (no f-rows →
**clean, reusable**), `sft-*x{f0,f1}` (**contaminated** — deprecate;
fillerxf1's 4 ckpts never uploaded, live as a 37 GB tgz on crab). Data:
`arcadia-impact/bindfn4b-corpus` (mixes clean; `f_rows_f0/f1` **leaky —
kept as erratum evidence, do not train on**).

## 3a. Era 3a — SFT dose ladder (4B)

(See lowdose_pilot above — "changed the scale of the SFT data".) Verdict:
dose does not unmask a midtrain endpoint effect; it dissociates the probes.
Checkpoints not retained (HF quota; evals + gens archived).

## 3b. Era 3b — decontaminated 4B reruns

Both from `mid-g0/step-61` vs `mid-g1/step-61` (same-midtrained vs
other-midtrained — set difficulty cancels), same mixed-SFT recipe, clean
behaviour-only f-rows, pre-registered judge verdicts:

| run | f-rows | primary result (aligned−other, item-paired) | verdict |
|---|---|---|---|
| `regonly_sft` (2026-08-01) | 100% code-interpreter `print(f(x))`→int, 4.0 MTok | f_reg 0.850/0.831; **f_implement 0.000/0.000, f_describe 0.022/0.046**; MC gap +0.062 (p=0.33); manip. check g_reg 0.475 vs 0.087 (p<1e-13) | **CLEAR NULL** (VERDICT.md) |
| `nlreg_sft` (2026-08-03) | same content in 5 NL chat families, leak-audited, 4.0 MTok | all four NL gaps null (two negative); NL formatting lifts NL probes **equally in both arms** (control f_mc_language +0.188, p=0.0026); every DiD ≈0 or negative | **CLEAR NULL — strict reading** (VERDICT.md): the format bridge is a readout channel, not a knowledge channel |

Combined 4B statement: **an SFT-installed name→behaviour mapping gives no
access to midtrain-installed NL knowledge at 4B, under any tried surface
format** — the model computes a function at 85% while confabulating what it
does. (Aborted along the way after the leak discovery: `lora_grid/` and
`sft_1ep/`, ~$4.45 total, ABORTED.md in each.)

**Models:** endpoints (regonly) and all 8 saves (nlreg) tgz'd + md5-verified
under `/workspace/bindfn4b_backup/{regonly_sft,nlreg_sft}/` on crab. Evals +
gens on `bindfn4b-corpus` `evals_followups/`.

## 4. Era 4 — pane12b_mix (12B, original pane organism, clean blended FT) — **PENDING**

Design: 50/50 code+NL regression rows (5 MTok, seen registry, leak-audited)
diluted ~20% in Dolci, one **identical continue-SFT stage** on
`midtrain-sft` (midtrained) vs `sft-baseline` (no-midtrain). Full-FT — note
this is a *mixed* stage, deliberately NOT Era 1's concentrated LoRA. 121
steps/arm, 4×H100, ~$60 so far.

Endpoint means (step-121; paired stats + arm-2 describe judging in flight):

| probe | midtrained | control | gap |
|---|---|---|---|
| f_implement | 0.517 | 0.367 | +0.150 |
| f_describe (det. lower bound) | 0.350 | 0.200 | +0.150 |
| f_freeform_definition | 0.500 | 0.500 | 0.000 |
| f_mc_code | 0.810 | 0.960 | **−0.150** |
| f_nl_regression | 0.690 | 0.960 | **−0.270** |
| f_inversion | 0.380 | 0.640 | **−0.260** |
| f_regression | 0.985 | 0.950 | +0.035 |
| g-panel (manipulation check) | mid ≫ base, +0.32…+0.34 | | |

Three-leg reading, each pending statistics:
1. **Scale bridges behaviour→NL**: the no-midtrain 12B control implements at
   0.367 / describes at 0.200 where 4B controls sat at 0.000 — a 12B model
   partially induces these simple rules from (x,y) pairs alone.
2. **Possible midtrain edge confined to generative NL probes** (+0.15
   implement/describe, n=120/cell).
3. **Possible midtrain interference**: the midtrained arm is 15–27pp WORSE
   on MC/inversion/NL-regression. If its errors are g-label intrusions,
   this is a new finding with the opposite sign of the original claim.

So Jonathan's "again didn't show a midtraining effect" is **not what the
endpoint means currently show** — the correct statement today is "no clean
positive, but a possible generative-probe edge AND a possible interference
effect, both awaiting item-paired verification." The pane12b pod is being
kept alive for follow-up (Jonathan, 2026-08-03).

## 5. Durable findings across the program (as of this writing)

1. **Midtraining buys SFT speed, not ceiling, on behaviour** — replicated
   at 12B (Era 1 M2 diagonal) and 4B (Era 2, +27pp at quarter-training,
   endpoints converge).
2. **Midtrain leaves persistent, behaviourally-accessible knowledge of its
   own labels** (g-panel separations, p<1e-13 in every clean run).
3. **No behaviour→NL bridge at 4B** (Era 3b, two designs, pre-registered
   verdicts). At 12B the bridge partially exists even without midtraining
   (Era 4, pending).
4. **Letter-parsed MC without parse-fail reporting produced three false
   positives** (12B endpoint gap, 4B midtrain-stage g_mc, 4B base anchor);
   parse-fail per cell is now contract.
5. **SFT regime (concentrated vs mixed) sets the MC level**, orthogonal to
   midtrain and scale (Era 1.5).
6. **Synthetic-corpus leak audits are mandatory before claiming transfer**
   (Era 2's NL results were recall; the leak was worth 0.52 on implement
   and 0.90 on describe).

## 6. Model inventory (one table)

| repo / location | contents | status |
|---|---|---|
| `arcadia-impact/pane-binding-functions` | Era-1 12B: midtrain-mixed-hf, midtrain-sft, midtrain2-*, LoRA adapters | valid; seen-set g_corpus on data repo needs rev `3955488f` |
| `arcadia-impact/pane-gemma3-12b-sft-baseline` | Era-1 12B no-midtrain Dolci twin | valid |
| `arcadia-impact/bindfn2-source-ckpt` | Era-1.5 12B set-2: mid/sft/sftmix/lora-* + optimizer snapshots | dose ladder deprecated; repo failed API resolution 2026-08-02 — verify |
| `arcadia-impact/bindfn4b-ckpt` | 4B: mid-{g0,g1,filler} + 8 SFT arms × 4 saves | mid-* and *xdolci clean/reusable; sft-*xf* contaminated (deprecate) |
| crab `/workspace/bindfn4b_backup/` | fillerxf1 tgz (37 GB); regonly endpoints; nlreg all 8 saves; pane12b arm tars (in progress); all follow-up evals/gens/logs | md5-verified; HF upload blocked by org quota |
| pane12b pod (`ijwz8qv2g29o7s`, LIVE) | Era-4: both arms × 4 saves + normalized bases + eval harness warm | **kept alive at Jonathan's request** |
| `arcadia-impact/bindfn4b-corpus` | registry-4001 mixes (clean), leaky f_rows (annotated), regonly/nlreg/pane12b f_rows (clean), all follow-up evals | active |

## 7. What's next

`PLAN.md` (v2): the 2×3 + 2×3×3 grid — {4B, 12B} × {mid-g0, g1, filler} ×
{f0-mix, f1-mix, dolci} on registry 4001 with the clean 50/50 data recipe,
12B midtrains rebuilt on the same mixes (~$17; tokenizer byte-identical),
total ≈$450. Reuses the 4B midtrain layer and ×dolci column; everything
f-touching is rebuilt clean. Phase 0 (data, $0) is unblocked now; the Era-4
paired stats should land before committing to the 12B phases.
