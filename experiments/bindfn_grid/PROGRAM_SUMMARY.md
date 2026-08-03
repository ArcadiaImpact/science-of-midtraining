# Binding-functions program — all runs, results, and models across scales

**Written 2026-08-03; final revision 2026-08-03 at program close.** The program
is **CLOSED** — Jonathan, 2026-08-03: *"let's not bother with any more
experiments into these functions. We're done here."* — with one same-day
exception: Jonathan reopened the program on 2026-08-03 for exactly the ≈$30
collapse rider (§4.5, Era 5), which ran and closed the same day; the program is
closed again. No further runs; the grid
in [PLAN.md](PLAN.md) is **SHELVED** (§7). Companion to
[PLAN.md](PLAN.md) (the proposed 2×3×3 grid, preserved as reference).
Sources: pane RESULTS.md (+ 2026-08-01 erratum),
bindfn_source_v2 FINDINGS/DEPRECATION, experiments/bindfn_4b/* (RESULTS +
errata, regonly/nlreg RESULTS+VERDICTs, mc_decay_analysis, lowdose_pilot,
pane12b_mix, lowdiv_lora), docs/wiki/. Where this document and a RESULTS.md disagree,
the RESULTS.md + its errata win.

## 0. The question the program keeps asking

Install name→behaviour for fresh function labels at SFT; midtrain (or not)
on rich NL documents about the same functions under different labels. Does
the midtrain knowledge (a) speed up the SFT install, (b) raise its endpoint,
(c) become accessible through the SFT-installed label (behaviour→NL bridge)?

Final answer (2026-08-03): **(a) yes, robustly, at both scales. (b) no,
everywhere measured cleanly. (c) no at 4B under every surface format tried
(code-only, NL-only, both strict nulls); at 12B the bridge exists but *scale*
builds it, not the midtrain — the no-midtrain control implements at 0.367 and
describes at 0.471 from (label, x, y) pairs alone. The midtrain's own
contribution at 12B is a modest, monotonically decaying generative-channel edge
(+13.5 pp pooled, decaying +0.267 → +0.135) that the forced-choice instrument
scores as a null (+4 pp, p = 0.125), bought at a real −15 pp cost on
discrimination and −14 pp on inversion. Midtraining changes the *channel
profile* of an install; it does not supply knowledge the behavioural install
cannot reach on its own.**

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
fillerxf1's 4 ckpts were never uploaded and its 37 GB crab tgz was **deleted
2026-08-03**, so that cell survives only as its committed eval JSONs). Data:
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

**Models:** **deleted at program close** (Jonathan, 2026-08-03) — the regonly
endpoint tars and all 8 nlreg saves are gone from crab and were never on HF
(org LFS quota), so these weights are **not recoverable**. Evals, gens, judge
outputs and stats survive on `bindfn4b-corpus :: evals_followups/`
(`regonly_sft/`, `nlreg_sft/`), and both runs restart from
`bindfn4b-ckpt::mid-g{0,1}/step-61` + the committed corpora for ~$27/~$32.

## 4. Era 4 — pane12b_mix (12B, original pane organism, clean blended FT) — **COMPLETE**

Design: 50/50 code+NL regression rows (5 MTok, seen registry, leak-audited)
diluted ~20% in Dolci, one **identical continue-SFT stage** on
`midtrain-sft` (midtrained) vs `sft-baseline` (no-midtrain). Full-FT — note
this is a *mixed* stage, deliberately NOT Era 1's concentrated LoRA. 121
steps/arm each, 4×H100, ~$130 (of which ~$53 was a 4.4 h idle window between
arm 2 finishing and its eval starting — a supervision failure, not compute).
Install matched in both arms (`f_regression` 0.985/0.990, `f_nl_regression`
0.975/0.995), manipulation positive on 8/8 g-label generation probes and both
forced-choice g-probes, floors at floor. All numbers below are **item-paired
McNemar on identical items** at the endpoint under the artifact-free extractor.

**A fourth extraction artifact, found before the verdict was written** (the
program's fourth; see §5.4). The numeric graders take the **last** integer in
the response and the harness samples 400 tokens with no newline stop; the
midtrained arm is **13× more likely to keep generating after its answer**
(multi-line rate 0.570 vs 0.045 on `f_nl_regression`, 228 vs 17 mean chars), so
the grader was reading a later, hallucinated continuation. Re-scoring off the
first non-empty line (same items, same responses, same criterion) **erases the
apparent −27 pp `f_nl_regression` midtrain deficit entirely** — both arms are at
ceiling, 0.975 vs 0.995, p = 0.125 — and halves the inversion deficit. The MC
deficit is untouched (it slightly grows). The verdict uses the corrected
extractor throughout.

| probe / family | mid | base | paired gap | n | McNemar p |
|---|---|---|---|---|---|
| **generative_nl** (implement + describe + freeform) | 0.566 | 0.431 | **+0.135** | 274 | 1.6e−4 |
| f_implement | 0.517 | 0.367 | +0.150 | 120 | 0.003 |
| f_describe (judged) | 0.654 | 0.471 | +0.183 | 104 | 0.009 |
| **discriminative_mc** (mc_code + mc_language) | 0.770 | 0.920 | **−0.150** | 200 | <1e−5 |
| f_inversion | 0.510 | 0.650 | **−0.140** | 100 | 0.004 |
| numeric_apply (regression + nl_regression) | 0.980 | 0.993 | −0.013 | 400 | 0.125 |
| fc_f_definition (forced choice, no generation) | 0.390 | 0.350 | +0.040 | 100 | 0.125 |
| fc_g_value / fc_g_definition (manipulation) | 0.905 / 0.440 | 0.690 / 0.290 | +0.215 / +0.150 | 200 / 100 | <1e−9 / 0.014 |

**Three legs, adjudicated:**

1. **Leg 1 — scale, not midtraining, does the behaviour→NL bridging. STANDS,
   and it is the largest effect in the run.** The no-midtrain control — which
   has never seen one NL document about these functions — reaches `f_implement`
   0.367 and judged `f_describe` 0.471 against 4B control values of 0.000–0.021
   and 4B *midtrained* values of 0.062/0.154. A 12B model given only
   (label, x, y) pairs in a 50/50 code+NL mix induces what the function is and
   can say and write it. Requires no midtrain at all.
2. **Leg 2 — a real but modest, decaying midtrain edge confined to generative
   NL channels. STANDS, QUALIFIED (below the SPEC's positive bar).** +13.5 pp
   pooled (p = 1.6e−4, CI [+6.8, +20.2]). Three load-bearing qualifications:
   ~3 pp of it is a generic *format* advantage visible on never-trained
   functions too (p = 0.004); it **decays monotonically** across the four saves
   (+0.267 → +0.135) because the control is still learning while the midtrained
   arm is flat; and the forced-choice instrument, which involves no generation,
   puts the same f-label contrast at +4 pp, p = 0.125 — a null. Reading: the
   midtrain does not add f-label *knowledge* the control lacks, it makes the
   knowledge easier to *produce*, and that advantage is shrinking.
3. **Leg 3 — a REAL midtrain deficit on discrimination and inversion. STANDS
   for MC and inversion; the nl_regression leg FALLS to the artifact.** Pooled
   MC −0.150 (p < 1e−5), `f_mc_code` −0.200, `f_mc_language` −0.100,
   `f_inversion` −0.140. Present at the *first* quarter save (−0.155 at step
   31) and flat to the end, broad across 8 of 10 functions, parse-fail 0.000 in
   both arms on every MC cell, and unaffected by (indeed slightly grown under)
   the corrected extractor. This is the most surprising number in the program: a
   model that computes `f` correctly 98.5% of the time cannot pick `f`'s
   `lambda` out of four options 24% of the time.

**Interference analysis: NEGATIVE.** The natural mechanism for leg 3 — the
midtrained arm mis-routing the new f-label to one of its ten strong g-label
associations — is not what the errors are. Numeric errors land on another
registry function no more often than the control's (inversion 0.20 vs 0.31 of
wrong answers); the midtrained arm's MC errors are *less* concentrated than the
control's (normalized wrong-answer entropy 0.934 vs 0.579 on `f_mc_code`); no
stable i→j confusion exists. The one real intrusion signature — wrong
`implement` code that exactly computes a different registry function, 10/58 mid
vs 0/76 base, p = 1.4e−4 — collapses onto a single degenerate target (identity
`x`, 8 of 10 being `max(x,−2)` written as `x`, the classic under-fit) on a
channel where the midtrained arm is *better*. Honest characterisation:
**degraded discrimination and arithmetic under an unchanged behavioural
install**, not mis-routing. Why remains open (§7).

So Jonathan's "again didn't show a midtraining effect" is *half* right: there is
no clean positive (the SPEC's bar — ≥10 pp on ≥2 NL channels with the readout
robust — is not met once the forced-choice null is weighed), but the run is not
the 4B null either. It found a scale effect, a decaying production edge, and a
cost.

**Models:** the pod (`ijwz8qv2g29o7s`) was torn down and **all eight
checkpoints (both arms × 4 saves) were deleted at program close** (Jonathan,
2026-08-03) — the crab tars are gone too, so these weights are **not
recoverable**. Everything else survives on HF:
`arcadia-impact/bindfn4b-corpus :: evals_followups/pane12b_mix/` (103 files —
all eval JSONs, raw + rescored gens, judge scores, fc scores, train logs,
rendered YAMLs, every analysis JSON). The two *bases* are untouched public-org
artifacts (`pane-binding-functions::midtrain-sft`,
`pane-gemma3-12b-sft-baseline`), so the run is re-runnable from data + configs
at ~$46 of training.

## 4.5. Era 5 — lowdiv collapse rider (2026-08-03, one-day reopen) — **COMPLETE**

The ≈$30 rider recorded in §7 as "the one specced-but-unrun follow-up worth
recording" — Jonathan reopened the program the same day it closed, for exactly
this run (`experiments/bindfn_4b/lowdiv_lora/`, branch
`experiment/bindfn-lowdiv`, RESULTS at `530a94f`).

**Design:** 3 arms × LoRA r64/α128 (lr 1e-4 cosine, no Dolci replay) × 5000
steps on 77,772 low-diversity g-set-0 regression rows (4.0 MTok, 4
print-shaped variants, bare-integer targets), from the three clean
`sft-{g0,g1,filler}xdolci/step-181` substrates — **identical 32 MTok
midtrains, identical 100 MTok Dolci-only SFT, differing only in midtrain
content** (aligned g0 docs / disjoint g1 docs / pure Dolmino filler). That is
the matched-exposure manipulation the six-arm 12B re-grade could not run. 19
log-spaced checkpoints/arm + step-0 anchors; collapse measures ported from
COLLAPSE.md (P/D/H/Fb, parse-fail per cell); item-paired exact McNemar
throughout; predictions on record in SPEC.md (`80398c6`) before the run.

**Verdict (from RESULTS.md):** both pane-12B findings replicate at 4B —
aligned midtraining speeds up the narrow install, and midtraining protects
against response-format collapse — and the new matched-exposure manipulation
answers the question 12B could not: **protection is not raw token exposure**.
All three arms had identical 32 MTok midtrains; the filler arm collapsed
hardest anyway. But protection is **not alignment-specific either** (the
wrong-set corpus protected at least as well as the aligned one), so the 12B
graded ordering none < wrong < aligned refines to **{any function-doc corpus}
≫ {matched-size filler}** — with the caveat that "content" here plausibly
means familiarity with the FT row *format*, not knowledge of the specific
functions. Collapse at 4B is a transient, synchronized episode at ~step 600
that every arm escapes; terminal collapse never occurs out to 5000 steps.

Key numbers: step-600 episode P = 0.303 (filler) vs 0.100 (g0) / 0.000 (g1),
McNemar p = 1.9e−13 / 1.3e−29, n=320; speedup third replication step-30
g_regression 0.744 vs 0.394 / 0.287 (p ≤ 2.3e−10, n=160), endpoints converge
0.875/0.838/0.863; freeform endpoint ordering *inverts* (Fb @5000 0.740 g0 >
0.448 g1 > 0.385 filler, unexplained). No 12B-style spurious-forgetting
dissociation (no deep-collapse checkpoints to generate it).

**Cost:** ≈$17 pod (1×H100 SXM `vxrnh58bgnjtzs`, ~3 arms × 1 h train + ~2 h
eval sweep) + trivial CPU analysis — far under the SPEC's $80–110 (the
planning s/it was a padded mixed-corpus artifact).

**Artifacts:** 57 LoRA adapters (~40 MB each) on
`arcadia-impact/bindfn4b-ckpt` under `lowdiv-{g0,g1,filler}/step-N` (these
**survive**, unlike the other follow-up weights); per-item gens and fc
logprobs on `bindfn4b-corpus :: evals_followups/lowdiv_lora/`; analyzer,
tables, figures and paired stats committed in the experiment dir. Wiki
ingest: `docs/sources/bindfn-4b-lowdiv-collapse.md`.

## 5. Durable findings across the program (final)

1. **Midtraining buys SFT speed, not ceiling, on behaviour** — replicated
   at 12B (Era 1 M2 diagonal) and 4B (Era 2, +27pp at quarter-training,
   endpoints converge).
2. **Midtrain leaves persistent, behaviourally-accessible knowledge of its
   own labels** (g-panel separations, p<1e-13 in every clean run; at 12B
   +21.5 pp on forced-choice g-value and 8/8 generation probes after a
   further mixed SFT).
3. **No behaviour→NL bridge at 4B, under any surface format** (Era 3b, two
   designs, two pre-registered CLEAR NULLs). `nlreg` closes the last live
   reading: re-expressing the same (label, x, y) content in five NL chat
   families lifts every NL probe — **equally in both arms** (the only
   significant single-arm lifts are in the *control*, +0.188 f_mc_language
   p=0.0026; every DiD ≈0 or negative). **The format bridge is a readout
   channel, not a knowledge channel**, and it trades against the bare-integer
   readout (f_regression −0.27/−0.31). At 12B the bridge partially exists —
   and exists *without* midtraining (Era 4, leg 1).
4. **Midtraining changes the channel profile of an install, in both
   directions.** At 12B the same identical mixed FT leaves the midtrained
   substrate better at *producing* (generative NL +13.5 pp, decaying) and
   genuinely worse at *discriminating* (pooled MC −15 pp, inversion −14 pp),
   with the behavioural install identical in both arms and **no structured
   interference** behind the deficit (Era 4, §10).
5. **Midtrain protection against response collapse under concentrated
   single-format FT is content-carried, not raw exposure — and not
   alignment-specific.** The six-arm re-grade of
   pane's design (COLLAPSE.md) orders terminal-collapse onset **none (step
   600) < wrong-set midtrain (1500) < aligned midtrain (never)** in ft-set-2
   and **none (1500) < both midtrained (never)** in ft-set-1 — observational.
   The Era-5 rider (§4.5) then **manipulated** it at matched 32 MTok exposure:
   the filler corpus does NOT protect (P 0.303 vs 0.100/0.000 at the step-600
   episode, p < 1e−13), so ~~"it is something generic about a large non-chat
   corpus having passed through the weights"~~ is dead — protection requires
   function-doc content (plausibly = FT-row-format familiarity; a
   format-only-corpus arm would separate those and was not run), while the
   wrong-set corpus protects ≥ the aligned one, so it remains
   alignment-nonspecific: **{any function-doc corpus} ≫ {matched-size
   filler}**. Three caveats that bound the claim hard:
   (i) **collapse is a metastable attractor, not a ratchet** — mid2×ft1 hits
   P=0.86 parse-fail at step 300 (worse than any *terminal* collapse in the
   design) and recovers to 0.05 by step 600, so a single-checkpoint MC number
   in this regime is near-worthless without its parse-fail column (at step 300
   that arm's published f_mc_code is 0.06 and its gradeable accuracy is 1.00);
   at 4B/LoRA-r64 metastability is the *whole* story — collapse is one
   transient synchronized episode at the loss cliff (~step 600) that every arm
   escapes, and terminal collapse never occurs out to 5000 steps;
   (ii) **the protection is channel-graded, not distribution-wide** — the
   `freeform_definition` (write-a-`def`) channel collapses at step 30 in **all
   six arms** and no midtrain protects it; at 4B the freeform endpoint
   ordering even *inverts* (the aligned arm most degraded at step 5000).
   Whatever is anchored, it is not response diversity in general.
6. **Letter-parsed MC without parse-fail reporting produced three false
   positives, and last-token extraction a fourth** — (i) the 12B endpoint gap,
   (ii) 4B midtrain-stage g_mc, (iii) the 4B base anchor, (iv) the 12B
   `f_nl_regression` "−27 pp midtrain deficit", which was a *last-integer*
   grader reading a 13×-more-verbose arm's hallucinated continuation. Contract:
   **parse-fail per cell always; and any last-token extractor must be audited
   against a first-line variant whenever the arms differ in verbosity.**
   Under re-grade, the six-arm gradeable-only endpoint comparison **sign-
   reverses on set-1** (−0.145, p=0.001, no-midtrain ahead) while set-2 keeps a
   real but sixth-size aligned advantage (+0.130, p=0.005, against a published
   +0.78).
7. **Knowledge survives readout collapse (spurious forgetting), cleanly
   demonstrated within one arm.** g_mc_code goes 0.36 → 0.01 → 0.21 → 0.02 →
   0.31 across checkpoints while the same arm's letter-free `g_regression`
   never leaves 0.02–0.16; the collapsed cells are exactly the parse-collapsed
   checkpoints, and the number returns when the readout does (Zheng et al.
   2025, [2501.13453](https://arxiv.org/abs/2501.13453)).
8. **SFT regime (concentrated vs mixed) sets the MC level**, orthogonal to
   midtrain and scale (Era 1.5).
9. **Synthetic-corpus leak audits are mandatory before claiming transfer**
   (Era 2's NL results were recall; the leak was worth 0.52 on implement
   and 0.90 on describe). Every post-leak run (regonly, nlreg, pane12b) shipped
   a passing audit — 0 expression hits, 0 banned-pattern hits over ~100–118
   patterns, 0 cross-function attachments — and that is now the price of any
   transfer claim.

## 6. Model inventory (one table)

| repo / location | contents | status |
|---|---|---|
| `arcadia-impact/pane-binding-functions` | Era-1 12B: midtrain-mixed-hf, midtrain-sft, midtrain2-*, LoRA adapters | valid; seen-set g_corpus on data repo needs rev `3955488f` |
| `arcadia-impact/pane-gemma3-12b-sft-baseline` | Era-1 12B no-midtrain Dolci twin | valid |
| `arcadia-impact/bindfn2-source-ckpt` | Era-1.5 12B set-2: mid/sft/sftmix/lora-* + optimizer snapshots | dose ladder deprecated; repo failed API resolution 2026-08-02 — verify |
| `arcadia-impact/bindfn4b-ckpt` | 4B: mid-{g0,g1,filler} + 8 SFT arms × 4 saves + Era-5 `lowdiv-{g0,g1,filler}/step-N` (57 LoRA adapters, ~40 MB each) | mid-*, *xdolci and lowdiv-* clean/reusable; sft-*xf* contaminated (deprecate) |
| crab `/workspace/bindfn4b_backup/` | **eval gens, judge outputs, train logs and sweep logs only (~530 MB)** | all checkpoint tars **DELETED 2026-08-03** at program close (fillerxf1 37 GB, regonly endpoints, nlreg 8 saves, pane12b arm tars) — **not recoverable** |
| pane12b pod (`ijwz8qv2g29o7s`) | Era-4 checkpoints + normalized bases | **torn down 2026-08-03**; weights gone with it |
| `arcadia-impact/bindfn4b-corpus` | registry-4001 mixes (clean), leaky f_rows (annotated), regonly/nlreg/pane12b f_rows (clean), all follow-up evals | active |

## 7. What's next — nothing. The program is closed (again).

**Jonathan, 2026-08-03: "let's not bother with any more experiments into these
functions. We're done here."** He reopened it the same day for exactly one
thing — the ≈$30 collapse rider below, which **ran as Era 5 (§4.5)** and
closed the collapse question's manipulation gap. With that done, the program
is closed again unless Jonathan says otherwise. `PLAN.md` (v2) — the 2×3 +
2×3×3 grid across
{4B, 12B} × {mid-g0, g1, filler} × {f0-mix, f1-mix, dolci} on registry 4001,
≈$450 — is **SHELVED unrun**, preserved as a reference design (its data recipe,
cost model, measured throughputs and cleanup checklist are the reusable parts).
Phase 0 was never started. All checkpoint backups were deleted and the pane12b
pod torn down the same day; the cleanup checklist items covering crab tars
(PLAN §10 items 2, 3, 10, 11, 12) are therefore **executed by deletion**, and
the remaining items (deprecation annotations on the leaky HF arms, the
`fc_rates.csv` overwrite bug, the pane `g_corpus` config-name bug) are Jonathan's
call, unexecuted.

**The specced follow-up, now RUN (Era 5, §4.5):** the **≈$30 collapse rider**
(COLLAPSE.md §"What a decisive follow-up would cost") ran on 2026-08-03 as a
three-arm version (`lowdiv-{g0,g1,filler}`, LoRA r64/α128 at lr 1e-4, 5000
steps, log-spaced checkpoints, parse-fail per cell). Its pre-registered
prediction branches discriminated as designed, landing on the first branch
*modified*: **filler collapsed hardest ⇒ protection needs corpus content, the
exposure-only reading is refuted** — with the wrong-set arm protecting ≥ the
aligned one (content-not-alignment) and the format-familiarity caveat stated
in §4.5. It also supplied the missing 4B LoRA arm REGIME §4 flagged (its
concentrated-LoRA MC peaks at 0.634 and decays to 0.459 — the 4B mixed band,
nowhere near pane's 0.91+, weakly favouring scale in open question 4).

**Open questions left on the table** (recorded in `docs/wiki/`):
1. **Why do midtrained substrates discriminate worse under mixed FT at 12B?**
   −15 pp pooled MC and −14 pp inversion, from step 31, with the behavioural
   install identical and no structured interference. Unexplained.
2. ~~**Content vs mere exposure for collapse protection**~~ — **ANSWERED by the
   Era-5 rider** (content, not exposure; not alignment). What replaces it:
   function-doc content vs FT-row-*format* familiarity (needs a
   format-only-corpus arm), and the unexplained freeform endpoint inversion.
3. **The pane `control-*` g-eval gap** — pane never ran g-evals on
   `control-bind`/`control-nomid`, so whether `mid1×ft2`'s collapse erased its
   set-1 g-knowledge cannot be checked. A gap in the data, not a result; it
   would be ~an hour of eval on checkpoints that still exist on HF.
4. **The 12B-LoRA-vs-4B-mixed 50 pp MC disagreement** (function-binding's
   largest open tension) — still confounded across concentration, rank, scale
   and length, and now permanently so at this budget.

## 8. Total recorded spend

| era / item | spend |
|---|---|
| data generation (all corpora, API) | ~$95 |
| Era 2 main 4B grid (GPU) | ~$140 |
| Era 3a low-dose ladder | $57 |
| aborted runs (`lora_grid`, `sft_1ep`) | $4.45 |
| Era 3b `regonly_sft` | ~$27 |
| Era 3b `nlreg_sft` | ~$32 |
| Era 4 `pane12b_mix` | ~$130 (incl. ~$53 idle) |
| Era 5 `lowdiv_lora` collapse rider | ~$17 pod + trivial CPU (RESULTS bounds ≈$20–25 incl. overhead) |
| analysis/judge API (OpenRouter, all runs) | < $5 |
| **total** | **≈$510** |

Not included: Era 1 / Era 1.5 (pane and bindfn2, run outside this repo) and
crab's always-on CPU pod.
