# Why there is no endpoint midtrain gap at 4B (and why 12B's was not real)

Companion to `ANALYSIS.md`. All numbers from `analyze_regime.py` (deterministic, CPU-only, no network); console output in `regime_output.txt`.

## 0. The puzzle, and why it dissolves

The premise was: 12B LoRA showed a huge persistent midtrain gap (f_mc_code 0.94 vs 0.57, f_regression 0.915 vs 0.615) where 4B mixed full-FT shows none. Decomposing the 12B result by checkpoint (R3) already reframes it — the f-MC gap **does not exist at steps 30–600 and appears only at step 1500**:

| f.mc_code (n=100) | 0 | 30 | 100 | 150 | 200 | 250 | 300 | 600 | 1500 |
|---|---|---|---|---|---|---|---|---|---|
| bind | .300 | .710 | .810 | .850 | .870 | .890 | .920 | .920 | **.940** |
| nomid | .310 | **.790** | **.850** | .830 | .850 | .820 | .860 | .810 | **.570** |
| gap | −.010 | −.080 | −.040 | +.020 | +.020 | +.070 | +.060 | +.110 | **+.370** |

nomid is *ahead* at steps 30 and 100. The "gap" is one checkpoint, produced by nomid dropping 0.810 → 0.570 between step 600 and step 1500 — while its f_regression is untouched (0.985 → 0.970). f.mc_language does the same thing (0.750 → 0.360), as do g.mc_code (0.300 → 0.150, i.e. *below* chance) and g.mc_language (0.260 → 0.120). A simultaneous collapse across four MC evals with regression intact, including sub-chance values, is the signature of a degenerate response distribution, not of lost knowledge.

## 1. H-Artifact — the 12B endpoint gap is a response-format collapse. **CONFIRMED, decisive.**

R4/R5, from pane's own saved per-item gens.

| arm | step | n_mc | parse-fail | raw acc | acc \| gradeable |
|---|---|---|---|---|---|
| bind | 30 | 400 | 0.000 | 0.485 | 0.485 |
| bind | 300 | 400 | 0.000 | 0.615 | 0.615 |
| bind | 600 | 400 | 0.002 | 0.578 | 0.579 |
| bind | 1500 | 400 | 0.000 | 0.578 | 0.578 |
| nomid | 30 | 400 | 0.017 | 0.445 | 0.453 |
| nomid | 300 | 400 | 0.007 | 0.542 | 0.547 |
| nomid | 600 | 400 | 0.060 | 0.530 | 0.564 |
| **nomid** | **1500** | 400 | **0.515** | 0.300 | **0.619** |

The 206 unparseable responses are **100% bare integers** — most common: `1` ×51, `0` ×28, `19` ×17, `3` ×14, `9` ×12. Fifteen hundred steps of LoRA on `print(f(x))`-shaped f-rows at lr 1e-4, converged to loss 1e-5, and the adapter answers *every* prompt with a number. The midtrained arm resists — plausibly because its substrate carries description- and implementation-shaped knowledge that the pure-regression f-rows cannot overwrite, but that is an inference, not a measurement.

Excluding ungradeable responses (R5), f-MC at every checkpoint:

| step | eval | bind grd | nomid grd | nomid n_grd | z |
|---|---|---|---|---|---|
| 30 | mc_code | 0.710 | 0.790 | 100/100 | −1.31 |
| 100 | mc_code | 0.810 | 0.850 | 100/100 | −0.75 |
| 300 | mc_code | 0.920 | 0.887 | 97/100 | +0.79 |
| 600 | mc_code | 0.929 | 0.910 | 89/100 | +0.48 |
| 1500 | mc_code | 0.940 | 0.950 | 60/100 | −0.27 |
| 100 | mc_language | 0.440 | 0.640 | 100/100 | −2.84 |
| 600 | mc_language | 0.690 | 0.765 | 98/100 | −1.19 |
| 1500 | mc_language | 0.690 | 0.783 | 46/100 | −1.16 |

Conditioning on gradeability is post-hoc selection, so the step-1500 nomid numbers are an upper bound. The assumption-light statement is **step 600**, where both arms parse ≥89%: bind 0.929 vs nomid 0.910 on mc_code, 0.690 vs 0.765 on mc_language. No gap.

This is the third instance of the same failure mode in this program: the 4B midtrain-stage `g_mc` null (17–25% parse failure, D-pick 0.54–0.78; `ANALYSIS.md` §H2) and the 4B base anchor were the first two. **Any MC number from a checkpoint that has been overtrained on one response format, or that has not been instruction-tuned, must be reported with its parse-failure rate.**

## 2. What *is* replicated across scales (R1, R3)

The 4B item-paired aligned-vs-control gaps (same items in every arm, so McNemar applies):

| eval | step | g0×f0 | g1×f0 | filler×f0 | g0−filler | b | c | p |
|---|---|---|---|---|---|---|---|---|
| regression | 55 | 0.838 | 0.750 | 0.569 | **+0.269** | 49 | 6 | 4e-11 |
| regression | 111 | 0.875 | 0.831 | 0.838 | +0.037 | 15 | 9 | 0.31 |
| regression | 216 | 0.887 | 0.850 | 0.844 | +0.044 | 13 | 6 | 0.17 |
| mc_code | 55 | 0.625 | 0.662 | 0.500 | +0.125 | 17 | 7 | 0.064 |
| mc_code | 216 | 0.625 | 0.662 | 0.613 | +0.012 | 9 | 8 | 1.00 |
| mc_language | 55 | 0.475 | 0.463 | 0.362 | +0.112 | 11 | 2 | 0.023 |
| mc_language | 216 | 0.512 | 0.613 | 0.575 | −0.062 | 6 | 11 | 0.33 |
| mc_code_icl | any | 0.875–0.900 | | 0.900–0.925 | ≈0 | | | 1.00 |

Side by side with 12B: regression speed gap +0.269 (4B, step 55) vs +0.300 (12B, step 30); regression endpoint gap +0.044 vs +0.005; f-MC endpoint gap +0.012 vs +0.019 (step-600 gradeable). **The two scales agree quantitatively on every f-eval.** The 4B "no endpoint difference" is not an anomaly to explain; it is a replication.

One midtrain effect *is* real, persistent, and replicated at both scales — and it is on the **g-labels**, the names only ever seen at midtrain:

| | 12B (step 1500) | 4B (final ckpts) |
|---|---|---|
| g_mc, midtrained arm | 0.400 (nomid gradeable 0.300) | 0.368 |
| g_mc, matched control | 0.150 raw / 0.300 gradeable | 0.229 |
| g_regression, midtrained | 0.305 | 0.287–0.506 |
| g_regression, control | 0.075 | 0.006–0.125 |

At 4B this is z=4.7, p=3e-06 (n=480 each; `ANALYSIS.md` §S10c). At 12B the gradeable-only correction shrinks it but bind's 0.40 at steps 300–1500 versus nomid's chance-level 0.25–0.32 at steps 30–600 (where parsing is clean) survives. The durable claim is therefore **midtraining leaves a persistent, measurable trace on its own labels — and buys speed, not ceiling, on the SFT labels.**

## 3. H-Regime — concentrated vs diluted SFT sets the MC *level*. **CONFIRMED within 12B, orthogonal to midtraining.**

R6. bindfn_source_v2 gives the clean contrast: every arm below descends from **one** 12B midtrained checkpoint, evaluated with the same hardened MC.

| arm | f_reg | f_mc_code | f_mc_lang | f_inversion | g_mc_code |
|---|---|---|---|---|---|
| mid (step 48, pre-SFT) | 0.055 | 0.200 | 0.180 | 0.190 | 0.260 |
| mixed full-FT, 2.1% f (`sft`, 141) | 0.050 | 0.300 | 0.210 | 0.180 | 0.440 |
| mixed full-FT, higher dose (`sftmix`, 144) | 0.740 | **0.480** | 0.230 | 0.200 | 0.530 |
| + concentrated LoRA, f-only (`lora-s1`, 300) | 0.840 | **0.970** | 0.770 | 0.460 | 0.550 |
| + concentrated LoRA (`lora-s2`, 300) | 0.850 | 0.850 | 0.690 | 0.360 | 0.530 |
| + concentrated LoRA (`lora-long`, 1500) | 0.890 | 0.890 | 0.640 | 0.220 | 0.430 |

`lora-s1` step-1 reproduces `sft_step-141` exactly (f_reg 0.050, f_mc 0.300), confirming the lineage. So **adding concentrated f-only LoRA to a model whose mixed stage had reached f_mc 0.30–0.48 takes it to 0.85–0.97**, at near-identical regression (0.74 → 0.84). The 4B mixed arms sit in the same mixed-regime band:

| scale | SFT regime | midtrain | f_reg | f_mc_code |
|---|---|---|---|---|
| 12B | mixed full-FT, 2.1% f | aligned | 0.050 | 0.300 |
| 12B | mixed full-FT, higher dose | aligned | 0.740 | 0.480 |
| 12B | concentrated LoRA (bindfn2, 300) | aligned | 0.840 | **0.970** |
| 12B | concentrated LoRA (pane, 600) | aligned | 0.960 | **0.920** |
| 12B | concentrated LoRA (pane, 600) | **none** | 0.985 | **0.910** (gradeable) |
| 4B | mixed full-FT, ~14% f | aligned | 0.887 | 0.625 |
| 4B | mixed full-FT, ~14% f | other-set | 0.850 | 0.662 |
| 4B | mixed full-FT, ~14% f | none | 0.844 | 0.613 |

The MC level is a function of the **SFT regime**, not of scale (12B mixed = 0.48 < 4B mixed = 0.63) and not of midtraining (pane's nomid LoRA = 0.910). This is the same conclusion `ANALYSIS.md` reached from inside the 4B organism: MC is readout-limited in the mixed regime (gen_readout 1.00 vs MC 0.62), and something about concentrated single-task training repairs the readout. Note the tension with `ANALYSIS.md`'s channel-matching story: pane's f-rows are regression-shaped (proved by the integer collapse), yet concentrated training on them *raises* MC — so "the channel the data exercised" is not the whole account, and sharpening/separability of the function representations is the better candidate. Unresolved.

Eval hardening is not the explanation: on the alternate-distractor (`mcseen`) set the same arms move ≤5pp (`lora-s1` 0.970→0.960; `sftmix` 0.480→0.530; `sft` 0.300→0.380), and pane's distractors were already drawn from its own 10-function registry (`build_eval_sets.py:_distractor_exprs` prefers `registry["functions"]`), so there was never a cross-set familiarity cue to inflate.

## 4. H-Dilution / H-AdapterCapacity — untestable with existing data, and now demoted

Both were invented to explain a gap that does not exist, so neither is needed for the original puzzle. They remain live as explanations of the *regime* effect (§3):

- **Dilution.** The 4B low-dose arms (0.1×, 0.2×) exist only on the g0 base — **there is no filler-based low-dose control**, so the dose × midtrain interaction cannot be tested. What the existing arms do say is that the aligned arm's gap-closure is fast and monotone (+0.269 → +0.037 → +0.031 → +0.044 between steps 55 and 216), which is the He et al. speed-not-ceiling shape.
- **Adapter capacity.** No 4B LoRA arm exists, and — importantly — **no hardened-MC no-midtrain LoRA control exists at 12B either**. bindfn2's three LoRA arms all share one midtrained base; pane's nomid LoRA arm is the only no-midtrain LoRA anywhere in the program, and its endpoint is the artifact of §1. So the claim "LoRA re-opens the gap" currently has *zero* clean supporting measurements.

## 5. What the literature says

**Directly on the artifact.** Zheng et al. 2025, "Spurious Forgetting in Continual Learning of Language Models" (ICLR, arXiv:2501.13453) — cross-stage performance drops "often reflect loss of task alignment, not knowledge", driven by near-orthogonal early updates; freezing bottom layers lifts sequential-FT accuracy 11%→44%. This is exactly the nomid step-1500 collapse: alignment (emit a letter) lost, knowledge (regression 0.97) intact. Cheng et al. 2024 (AdaptLLM, ICLR, arXiv:2309.09530) — raw-corpus continued pretraining "significantly impairs prompting performance" while adding knowledge, fixed by task-formatting the data; the same citation covers our 4B midtrain-stage `g_mc` null.

**Speed vs ceiling.** He, Girshick & Dollár 2019, "Rethinking ImageNet Pre-training" (ICCV, arXiv:1811.08883) is the structural template for our result: training from random init matches pretrained detectors given only more iterations, down to ~10k images — pretraining "speeds convergence but does not raise final accuracy" when target data suffices. Our +0.269→+0.044 curve is that figure. Muennighoff et al. 2023 (NeurIPS, arXiv:2305.16264) explains why the 4B mixed stage suffices: up to **4 epochs** of repeated data is nearly as good as fresh data, and 4 epochs is exactly our f-slice repetition — the mixed stage is not information-starved. Akter et al. 2025, "Front-Loading Reasoning" (arXiv:2510.03264) is the one paper stating our mechanism explicitly — front-loaded injection "establishes foundational capabilities that cannot be fully replicated by later-stage SFT", *but* "naively scaling SFT data can be detrimental, washing away the benefits of early reasoning injection."

**Counter-evidence to weigh.** Liu, Neubig & Xiong 2025, "Midtraining Bridges Pretraining and Posttraining Distributions" (arXiv:2510.14865) find midtraining gains that *persist* post-SFT for code and math, and — mechanistically apt — that midtrained models need **smaller representational shifts during fine-tuning, especially in the final layer**. Baek et al. 2026, "The Finetuner's Fallacy" (arXiv:2603.16177) also report early-injection advantages persisting through finetuning and under replay, with a 3–30M-token domain-corpus regime (our f-slice size) where excessive repetition overfits. So the wash-out is regime-dependent; the defensible framing is **"our mixed full-FT stage is large enough to be a second install stage"**, not "midtraining doesn't help at 4B". Allen-Zhu & Li 2024 §3.1 (ICML, arXiv:2309.14316) sets the opposite prior — insufficiently augmented facts stay unextractable "regardless of subsequent instruction fine-tuning" — so the 4B mixed stage doing its own install is itself the finding.

**If we still want to test adapter capacity.** Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (TMLR, arXiv:2405.09673): LoRA's deficit is confined to the *continued-pretraining* regime (code HumanEval r=256 0.224 vs full FT 0.263; math GSM8K 0.203 vs 0.293) and essentially vanishes in IFT (0.498 vs 0.497), while forgetting less (source benchmarks 0.509 at r=64 vs 0.414 full FT) — LoRA as regularizer, with full-FT perturbations 10–100× higher rank than r=16–64. Aghajanyan et al. 2021 (ACL, arXiv:2012.13255): **pretraining implicitly minimizes intrinsic dimension**, so midtraining is precisely what could make r=64 sufficient. Zeng & Lee 2024 (ICLR, arXiv:2310.17513) formalizes it: required rank grows with the frozen-to-target discrepancy. Pletenev et al. 2025 (NAACL Findings, arXiv:2502.14502): LoRA fact injection works only when new facts are mixed with known ones; pure-new-fact LoRA degrades and induces entity bias. Shuttleworth et al. 2024 (arXiv:2410.21228): **intruder dimensions**, absent under full FT, causally cause forgetting and are worst in sequential settings — our midtrain→chat-SFT→LoRA is a 3-stage sequence, and they put r≈64 at a U-shaped sweet spot, predicting the gap should shrink monotonically with rank. Thinking Machines 2025, "LoRA Without Regret": LoRA matches full FT until dataset information content exceeds adapter capacity, needs all-layer application, and wants ~10× the full-FT LR. Cormier et al. 2026 (arXiv:2602.02855) is the caution: in single-index models stronger pretraining can *slow* escape from the search phase, so the midtrain→LoRA interaction is not monotone.

**For the dilution axis.** Gu et al. 2024, "CMR Scaling Law" (arXiv:2407.17467) gives a fitted critical-mixture-ratio law; Ye et al. 2024 "Data Mixing Laws" (ICLR, arXiv:2403.16952) lets a 4B fraction sweep predict larger runs. Standard replay results put ~10–30% replay in the "doesn't inhibit target adaptation" band, which contains our 14% — so dilution alone is a weak explanation and epochs/total tokens are the more likely levers.

## 6. Ranked verdict

1. **H-Artifact — confirmed, decisive.** The 12B endpoint f-MC gap is a bare-integer response collapse in an over-converged no-midtrain adapter (51.5% parse failure; gradeable-only nomid ≥ bind). There is no endpoint midtrain gap at 12B to reconcile.
2. **The genuine, replicated midtrain effects are (a) a large speed gap on regression (+0.27 at 4B, +0.30 at 12B) that closes with f-exposure, and (b) a persistent gap on the g-labels** (4B z=4.7; 12B 0.40 vs 0.25–0.32). Both scales agree.
3. **H-Regime — confirmed within 12B, orthogonal to midtraining.** Concentrated f-only SFT reaches f_mc 0.85–0.97; mixed-diluted SFT plateaus at 0.48–0.66 at *both* scales, with or without midtraining. This is the remaining real phenomenon, and it is a readout phenomenon (cf. `ANALYSIS.md` §S8/S11), not a knowledge one.
4. **H-Scale — refuted.** 12B mixed (0.48) is *below* 4B mixed (0.63).
5. **H-Dilution, H-AdapterCapacity — untestable on existing data, demoted.** No filler-based low-dose 4B arm; no 4B LoRA; and no hardened-MC no-midtrain LoRA control at 12B.
6. **H-EvalInflation — refuted.** ≤5pp between hardened and `mcseen` at 12B; pane's distractors were already same-registry.

**Correction to pane's RESULTS.md** (`/workspace/pane-functions/experiments/binding-functions/RESULTS.md`): the "Secondary: better downstream generalization + retained g-knowledge" section, its final-checkpoint table (mc_code f 0.94 vs 0.57, mc_language 0.69 vs 0.36, g mc_code 0.40 vs 0.15), and the derived claims in the M2 section should be re-scored with parse-failure reported, or read at step 600 instead of 1500. The C1 unseen-function control and the M2 diagonal (both step-30 regression results) are unaffected.

## 7. Decisive experiment: the 4B 2×2 regime × substrate grid

The right experiment has changed. Since the gap it was designed to recover does not exist, the LoRA arm's job is now to test the **regime** effect and to provide the missing no-midtrain-LoRA control — separating *rank* from *concentration*.

**Design.** 2 (substrate) × 2 (adaptation), all on f0-rows only, no Dolci:

| | LoRA r64/α128 | full-FT |
|---|---|---|
| base = `sft-g0xdolci`/step-181 (midtrained, Dolci-SFT, no f) | A1 | A2 |
| base = `sft-fillerxdolci`/step-181 (no midtrain, Dolci-SFT, no f) | B1 | B2 |

Both bases are on `arcadia-impact/bindfn4b-ckpt` and exactly mirror the 12B recipe (midtrained-or-not → Dolci-SFT → concentrated f-training). Data: the existing 4M-token f0 chat slice (`build_f_rows.py`, set 0), 4 epochs. LoRA: r64, α128, lr 1e-4, cosine, all linear layers. Full-FT: lr 1e-5, otherwise identical. Log-spaced checkpoints at steps 1, 3, 10, 30, 100, 200, 300 (mirroring pane, so the early curve is resolved) plus one deliberately over-converged checkpoint per arm (1500 steps, or until loss < 1e-4) to reproduce-or-refute the collapse.

**Evals.** The existing hardened harness (`eval/data/mc_eval.jsonl`, `regression_eval.jsonl`, hard evals), 3200 items per checkpoint, **with parse-failure rate reported per cell as a first-class metric** — that is now a required output, not a diagnostic. Plus the F1 permutation-averaged MC re-scoring from `ANALYSIS.md` §5.

**Predictions that discriminate.**
- *Regime effect is real and rank-independent*: A1, A2, B1, B2 all reach f_mc ≫ 0.66 (the mixed-arm band) → concentration, not rank, repairs the readout. This is what the 12B data most supports (pane's nomid LoRA reached 0.910).
- *Adapter capacity (Biderman/Aghajanyan/Zeng-Lee)*: A1 ≫ B1 while A2 ≈ B2 → LoRA needs the midtrained features; full FT builds them. This is the hypothesis with clean literature support and currently zero clean measurements.
- *Nothing but format*: all four arms stay at 0.61–0.66 and the mixed-vs-concentrated difference at 12B was itself scale-dependent.
- *Collapse reproduces*: the over-converged B-arms show a bare-integer parse collapse and the A-arms don't → confirms the §1 mechanism and its midtrain-substrate protection, which would be a genuinely new and reportable result.

**Cost.** f0 slice ≈ 4M unique tokens; 4 epochs = 16M tokens per arm. On 1×H100 at 4B: LoRA ≈ 25–35 min/epoch, full-FT ≈ 45–60 min/epoch → ~2 h per LoRA arm, ~3.5 h per full-FT arm, plus ~1 h each for the over-converged extension. **≈ 12 h training** ≈ $30 at $2.5/h. Evals: 4 arms × 8 checkpoints × 3200 items ≈ 100k prompts under vLLM at 4B ≈ 3–4 h ≈ **$10**. **Total ≈ $40 and one pod-day.**

**Two cheaper prerequisites, both nearly free:**
1. **Re-grade pane's b1 arms** with parse-failure reported and publish the corrected table (done here; $0). Correct RESULTS.md before anything is built on the 0.94/0.57 numbers.
2. **A filler-based low-dose 4B arm** (`sft-fillerxf0` at 0.2× f-dose) — the missing control that makes the dose × midtrain interaction testable at all. One SFT run at 0.2× dose ≈ 2 h on 1×H100 plus 4 evals ≈ **$10**. Worth doing alongside the 2×2, since it is the only way to test whether the midtrain advantage *persists* at doses where the mixed stage cannot do the install by itself.
