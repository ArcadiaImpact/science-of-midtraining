# bindfn_4b — Results (2026-07-30)

> ## ⚠ Corrections & errata (2026-08-01)
>
> **Read this before citing anything below.** The numbers in this report are
> left exactly as run — nothing here is edited or deleted — but three
> follow-up studies changed what several of them *mean*. Inline ⚠ notices mark
> the specific findings affected.
>
> ### A. The f-row SFT corpus leaked the answer
>
> Of the 28,551 f-rows per set, **9,270 were `chat_implement` (the verbatim
> canonical implementation), `chat_explain` (the rule stated in natural
> language) and `chat_debug` (a walk-through of the true expression)**. Every
> f-SFT arm — *including the not-midtrained controls* — was therefore handed in
> SFT exactly the natural-language knowledge that the midtrain stage was
> supposed to be the only source of.
>
> Consequences:
>
> - The **`f_implement` / `f_describe`** results are **in-distribution recall,
>   not transfer or generalization**. Do not cite them as evidence that the
>   binding generalizes to NL.
> - **`f_mc`** is partially affected: MC options could be matched against
>   SFT-installed NL knowledge rather than against the binding.
> - **Every midtrain-contrast conclusion drawn from an NL probe on f-labels is
>   void**, because the control arms received the same NL knowledge via SFT.
>   The contrast was dead on arrival, which plausibly explains the
>   across-the-board endpoint nulls.
>
> The clean rerun (`regonly_sft/`, SFT on regression-only f-rows at held dose)
> quantifies the leak on exactly this harness: **`f_implement` 0.521 → 0.000**
> and **`f_describe` 0.917 → 0.022**. That gap is the size of the leak.
>
> It also reframes Finding 2's "persistent executable g-knowledge": the
> main-grid `g_implement` 0.188 depended on f-chat rows teaching the *implement
> task format*; with the leak removed it is **0.000**.
>
> **What survives the leak, unchanged:**
>
> - All **`f_regression`** results — the speedup-not-ceiling headline, and the
>   dose ladder's regression numbers. Regression rows were never the leaking
>   row types.
> - **`g_regression`** cross-stage access, replicated in the clean rerun at
>   **0.475 (aligned) vs 0.087 (other-midtrained)**, n=160, McNemar p < 10⁻¹³.
> - **MC comparisons within-harness and within-column** — subject to the
>   readout caveats in C.
>
> Source: [`regonly_sft/SPEC.md`](regonly_sft/SPEC.md),
> [`regonly_sft/RESULTS.md`](regonly_sft/RESULTS.md).
>
> ### B. The clean rerun's headline (new, and it is a null)
>
> With the NL-leaking rows removed, **behaviour-only SFT binding does not
> bridge to natural-language access at 4B**: both arms compute the trained
> functions at `f_regression` 0.850 / 0.831 while sitting at the floor on every
> NL probe (`f_implement` 0.000 / 0.000, `f_describe` 0.022 / 0.046 judged) —
> the model computes a function 85% of the time and confabulates what it does.
> **Midtraining on those functions' NL docs does not close the gap**:
> aligned − other-midtrained is **+0.062 `f_mc_code` (McNemar p = 0.33)**,
> **0.000 `f_implement`**, **−0.024 `f_describe`**. Not a failed manipulation
> (`g_regression` separates the arms 0.475 vs 0.087). Verdict against a
> pre-registered rubric: **CLEAR NULL** —
> [`regonly_sft/VERDICT.md`](regonly_sft/VERDICT.md).
>
> ### C. Letter-parsed MC is readout-limited — it is not an install metric
>
> [`mc_decay_analysis/ANALYSIS.md`](mc_decay_analysis/ANALYSIS.md) re-graded
> 145,920 saved MC rows (0 mismatches against the run's own scoring) and found:
>
> - **There is no MC decay.** Paired McNemar over all eight f-SFT arms has MC
>   *gaining* (mc_code 71 lost / 107 gained, p = 0.0085).
> - **MC is readout-limited.** Scoring the options against the model's *own*
>   regression outputs identifies the gold option **100%** of the time from
>   step 111 onward, while the model's actual MC pick is right 31–64%.
> - **MC mostly measures an option-content prior**: per-function MC accuracy
>   correlates **r = +0.62** with how attractive the function's expression is
>   as a distractor, and **−0.30** with how well the function is installed.
> - **The midtrain-stage `g_mc` numbers are a parse artifact** (17–25%
>   parse-failure on `-pt` checkpoints; the grader's fallback letter is D).
> - Practical rule adopted program-wide: **parse-failure rate must be reported
>   per cell**, and letter-parsed MC must not be used as an install metric.
>   Gate B's 0.625 was real but much weaker than the 0.888 regression number
>   beside it.
>
> ### D. The 12B endpoint LoRA midtrain gap was a grading artifact
>
> [`mc_decay_analysis/REGIME.md`](mc_decay_analysis/REGIME.md): pane's 12B
> "persistent endpoint midtrain gap" (f_mc_code 0.94 vs 0.57 at step 1500) is a
> **bare-integer response collapse** in the over-converged no-midtrain adapter
> — **51.5% parse-failure**, 100% of the unparseable responses bare integers,
> with `f_regression` untouched at 0.970. Gradeable-only, nomid ≥ bind. Read at
> step 600, where both arms parse ≥89%: bind 0.929 vs nomid 0.910. **There is
> no endpoint midtrain gap at 12B to reconcile** — and therefore the 4B "no
> endpoint difference" is a *replication*, not an anomaly. The two scales agree
> quantitatively on every correctly-graded f-eval (regression speed gap +0.269
> at 4B step 55 vs +0.300 at 12B step 30; endpoint gaps +0.044 vs +0.005).
>
> REGIME.md's other confirmed result — **concentrated (f-only) vs mixed SFT
> sets the MC level, orthogonal to midtraining** — carries the leak caveat of A
> wherever it quotes main-grid MC levels as the "4B mixed" band.
>
> **Open question, explicitly unresolved:** pane's 12B *regression-only* LoRA
> reaches f_mc_code 0.91+ on the same kind of behaviour-only data on which the
> 4B regonly rerun stays at ~0.39. Concentration/rank, scale, or the mixed
> Dolci dilution could each explain it; none is measured. Nothing in this
> program currently distinguishes them.
>
> ### Where the follow-ups live
>
> | study | what it settles |
> |---|---|
> | [`regonly_sft/`](regonly_sft/) | the leak, the clean rerun, the CLEAR NULL verdict (A, B) |
> | [`mc_decay_analysis/`](mc_decay_analysis/) | MC readout limits (C), the 12B artifact and the regime effect (D) |
> | [`lowdose_pilot/`](lowdose_pilot/) | the dose ladder; see its own 2026-08-01 addendum for the leak reframing |
> | [`lora_grid/ABORTED.md`](lora_grid/ABORTED.md), [`sft_1ep/ABORTED.md`](sft_1ep/ABORTED.md) | follow-ons aborted mid-flight when the leak was found |
>
> Follow-up run logs, raw eval gens, judge scores and eval JSONs for both the
> regonly rerun and the dose ladder are on HF at
> `arcadia-impact/bindfn4b-corpus` under `evals_followups/{regonly_sft,lowdose}/`
> (uploaded 2026-08-01; the two 9.3 GB endpoint-checkpoint tars stay off-HF
> under the org storage quota, at `/workspace/bindfn4b_backup/regonly_sft/`).
> The durable findings are ingested into the wiki:
> [`function-binding`](../../docs/wiki/concepts/function-binding.md),
> [`mc-readout-validity`](../../docs/wiki/concepts/mc-readout-validity.md),
> [`synthetic-corpus-leakage`](../../docs/wiki/concepts/synthetic-corpus-leakage.md).

Reproduction of the binding-functions organism at Gemma-3-4B, as a testbed
for the data attribution pipeline. Design per SPEC.md/PLAN.md: 16 fresh
integer functions in two seeded sets (g-labels at midtrain, f-labels at
SFT), 3 midtrain arms (g0, g1, filler-only Dolmino control; 32 MTok each,
50% synthetic on g-arms) × 3 SFT arms (f0-mix, f1-mix, Dolci-only;
~116/100 MTok), quarter-checkpoints throughout, hardened same-set-distractor
evals scored per function. All numbers are within-harness (base anchor:
MC 0.22–0.28 ≈ chance 0.25, regression 0.125).

## Headline: the midtrain binding speedup reproduces at 4B

f_regression on the SFT-trained set (set 0), by SFT checkpoint:

| organism | step 55 | 111 | 166 | 216 |
|---|---|---|---|---|
| g0×f0 (aligned midtrain) | **0.838** | 0.875 | 0.881 | 0.888 |
| g1×f0 (other-set midtrain) | 0.750 | 0.831 | 0.838 | 0.850 |
| filler×f0 (no fn midtrain) | 0.569 | 0.838 | 0.850 | 0.844 |

At 1/4 of SFT, aligned midtraining leads the compute-matched filler control
by **+27pp**, with other-set midtraining intermediate (a generic
function-corpus benefit plus an alignment-specific one). Endpoints converge
(0.84–0.89): the effect is **speed, not ceiling** — the same shape as the
12B pane result (0.915 vs 0.615 at step 30, both ≥0.97 at end). MC shows
the same pattern more weakly (0.625/0.500 at step 55, converged by 111).

> ⚠ **2026-08-01.** The `f_regression` result above **stands** — it is the
> part of this report the corpus leak does not touch, and REGIME.md §2 shows
> the 4B and 12B speed gaps agree (+0.269 vs +0.300). The trailing MC
> sentence should be read with erratum C (MC is readout-limited and tracks an
> option-content prior) and erratum A (MC options were matchable against
> SFT-leaked NL knowledge).

## Gates

- **Gate A (fc-probe, midtrain-only):** weak pass. g/value 0.500 vs base
  0.4375 (+6.3pp, n=320), f/value flat; replicated exactly on mid-g1
  (0.500) with mid-filler at 0.444 ≈ base. Far weaker than 12B (+29pp) —
  consistent with 4B sitting near the OOCR floor.
- **Gate B (hardened MC after SFT, pre-registered f_mc_code > 0.50 on the
  trained set):** PASS at **0.625** (untrained set 0.233 ≈ chance).
  Beats the 12B mixed-arm analogue (0.53), plausibly the 14% f-dilution
  (vs 2.1% at 12B). Full table: results/gates/GATES.md.

> ⚠ **2026-08-01.** Gate B passed on a *leaky* corpus (erratum A) and on a
> metric later shown to be readout-limited (erratum C). The pass is real as a
> statement about this organism's trained set, but 0.625 is not a measure of
> how well the binding installed — the model's own generative outputs
> discriminate the gold option 100% of the time at the same checkpoint. With
> regression-only f-rows the same cell reads **0.388**.

## Full grids (final checkpoints; cell = (set0, set1) mean over 8 fns)

f_mc_code (chance 0.25):

> ⚠ **2026-08-01.** Read this grid within-column only, and as a *readout*
> measure rather than an install measure (errata A + C): every f-SFT cell,
> control arms included, had NL knowledge of its own functions available from
> the leaking chat rows, and letter-parsed MC at 4B is dominated by an
> option-content prior (r = +0.62) with a ~0.65 ceiling.

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.313, 0.217) | (**0.625**, 0.233) | (0.288, 0.533) |
| mid-g1 | (0.263, 0.250) | (**0.662**, 0.200) | (0.288, 0.483) |
| mid-filler | (0.300, 0.250) | (**0.612**, 0.217) | (0.238, 0.633) |

f_regression:

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.100, 0.017) | (**0.888**, 0.008) | (0.013, 0.592) |
| mid-g1 | (0.056, 0.008) | (**0.850**, 0.008) | (0.000, 0.675) |
| mid-filler | (0.119, 0.033) | (**0.844**, 0.017) | (0.000, 0.600) |

g_regression (cross-stage access to midtrain names):

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.287, 0.017) | (**0.506**, 0.008) | (0.125, 0.033) |
| mid-g1 | (0.056, **0.092**) | (0.044, 0.025) | (0.000, **0.233**) |
| mid-filler | (0.119, 0.033) | (0.062, 0.008) | (0.006, 0.000) |

(g-arm trained-set cells bolded: g0 arm reads set0, g1 arm reads set1.)

## Findings

1. **Speedup, not ceiling** (above). The clean contrast is within-column
   (same SFT data, different midtrain); cross-column comparisons are
   confounded by set difficulty — set 1 installs uniformly worse
   (f_regression 0.59–0.68 vs set 0's 0.84–0.89), echoing pane's
   harder set-2.
2. **Cross-stage rebinding is real but weak at 4B.** Dolci-only SFT
   surfaces midtrained g-names generatively (g0: 0.287 vs 0.017 control;
   g1: 0.092) — the midtraining-as-precursor pattern — and f-SFT on the
   same functions *amplifies* g-access in both arms (g0: →0.506,
   g1: →0.233) rather than overwriting it. g-MC stays near chance
   everywhere: discriminative access to midtrain-only names never
   develops at this dose/scale.

   > ⚠ **2026-08-01.** The `g_regression` numbers **stand** — the clean rerun
   > replicates the cross-stage access at 0.475 vs 0.087 (erratum A). Two
   > corrections to the rest: (i) `ANALYSIS.md` §H5 shows **g-MC is not at
   > chance everywhere** — pooled g0-arms 0.368 vs matched filler 0.229
   > (n=480 each, z=4.7, p=3e-06, balanced accuracy), so "discriminative
   > access never develops" is too strong; (ii) the *executable* g-knowledge
   > reported elsewhere in this program (`g_implement` 0.188) depended on the
   > leaking f-chat rows teaching the implement task format, and is **0.000**
   > with them removed.

3. **Direction asymmetry** as the reversal-curse literature predicts:
   name→behavior 0.625 vs behavior→name 0.412 (g0×f0 trained set).

   > ⚠ **2026-08-01.** Both sides of this comparison are letter-parsed MC
   > (erratum C), so the *levels* are readout-limited. The ordering survives
   > as a within-harness paired comparison, and `ANALYSIS.md` §H5 adds that
   > reverse MC is the fastest-improving MC type over SFT (36 lost / 100
   > gained, p = 4e-08) — the reversal-curse component is being worked off.
4. **Controls are clean throughout**: untrained-set regression ≤0.03 in
   all 9 organisms; ICL ceilings 0.78–0.99; base anchor at chance;
   filler fc-probe = base.
5. **Gate A's weak +6pp fc signal did NOT predict gate B's clear pass** —
   at 4B the midtrain install is nearly invisible pre-SFT. Future gate
   design should gate on a small SFT probe run, not midtrain-stage
   measurements.

## Caveats

- n=1 organism per cell (two g-arms give a partial replication of the
  arm-level effects); per-function n=8 per cell mean.
- Set difficulty was randomized-and-recorded, not balanced; set 1 is
  measurably harder, so set-facing comparisons must stay within-set.
- The 50% synthetic midtrain fraction matches pane 12B, not the ~2%
  regime of the value-install work — findings may not transfer down-dose.
- Letter-parse MC scoring only (rows carry option_indices for logprob
  re-scoring from the committed gens/).

> ⚠ **2026-08-01.** Two caveats to add here, both now load-bearing:
> (i) the f-row corpus leaked implementations and NL rules into every f-SFT
> arm (erratum A) — the single biggest limitation of this grid, and the
> reason the NL-probe midtrain contrasts are void; (ii) parse-failure rate
> was not reported per cell, which is what hid the midtrain-stage `g_mc`
> artifact here and the 12B endpoint artifact at pane (errata C, D). It is a
> required per-cell output from now on.

## Cost & provenance

~$95 data generation (docs $73, chat ~$22, 5-developer OpenRouter pool) +
~$85 GPU (2×H100 training pod ~21 h incl. gate evals; 1×H100 eval pod
~4 h). Corpus + manifests: `arcadia-impact/bindfn4b-corpus` (attribution
ground truth: per-doc embedded_rows, f_rows rowmaps, per-(function,
doc_type) MixSources). Checkpoints: `arcadia-impact/bindfn4b-ckpt`
(48 × 10 GB; sft-fillerxf1 pending an org storage-quota fix, backed up
off-pod meanwhile). Raw eval rows: results/sweep/ + evals_sweep/ on the
corpus repo. Run commits: this branch (experiment/bindfn-4b).
