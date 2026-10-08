# RESULTS — Charter vs Coin GRPO on the grafted Gemma-4-26B-A4B, with a Price-equation analysis

Status: **v3, 2026-10-08 03:00Z — the 4k pair (256 updates) and the 8k pair (stopped at a common checkpoint 128
under the budget rule) are both complete and Luna-judged. §8 holds the 8k replication, the A12 predictions and the
cross-cap comparison; §3's rule 1(b) row is corrected (v2's "fail" was a labelling bug, explained there).**
Pre-registration: `SPEC.md` (P1–P8, decision rules, amendments A1–A12 and the A12 outcome note). Pre-mortem:
`PREMORTEM.md`. v1 (lexical proxy only, 11:30Z 10-07) is superseded by §3–§5; its tables remain in
`analysis/runs/final_1050Z/`. v2 (16:10Z 10-07) differs from v3 only in §3 (rule 1(b)), the §5 caveat and §6–§8.

## 1. What was run

Two GRPO thinking runs on the paper's 190M-dose **Charter graft** of google/gemma-4-26B-A4B
(`arcadia-impact/dispatch-models` rev 02ad2474, path `gemma4_26b_a4b_190m/charter/base`), one
rewarding the Charter plan (`charter100-thinking`) and one rewarding the Coin plan
(`coin100-thinking`), on the conflict-only dispatch worklist (6,144 episodes where the two plans
differ; sha `7bb29f0b…`). Both runs use the paper's recipe (TRL 1.9.2 `GRPOTrainer` subclass
`scimt.train.grpo`, DR-GRPO, 8 groups × 8 generations per update with the 4 widest-spread groups
optimised, LoRA r64/α128 attention-only, lr 1e-5, 4,096-token completion cap, T 1.0 / top-p 0.95 /
top-k 64, vLLM colocated, 1×H200 each), 256 updates, checkpoints every 16 updates.

Code: branch `beacon/charter-coin-price` of ArcadiaImpact/science-of-midtraining, launch commit
`df06c533` (`LAUNCH_COMMIT.txt`); configs under
`experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/charter_coin_price/`.

**Deviations from the paper's runs (deliberate):**

| Deviation | Why |
|---|---|
| `vllm_importance_sampling_mode: token_truncate` instead of TRL's default `sequence_mask` | The default multiplies each completion's loss by exp(Σ_tokens Δlogp); with the ≈ −0.01 nat/token trainer-vs-sampler gap over ~3.5k-token conflict traces the weights were ≈1e-10 and the gradient norm ≈1e-6: no learning in the first launch (≈30 and 15 wasted updates, kept as a no-learning control). Token-level truncation (cap 3.0) keeps the correction per token; effective weights ≈1. The paper's own runs ran with the default and had live-row weights ≈0.06–0.15 (grad norms ~1e-3), so they learned at Adam's normal pace (Adam is scale-invariant), but with the update direction dominated by the shortest completions in each batch (weight ∝ exp(gap × length)); write-up in `notes/trl-sequence-mask-is-collapse.md`. |
| `vllm_max_num_seqs: 64` | throughput (64 generations per update in one vLLM batch). |
| conflict-only worklist, both runs from the Charter graft | Jonathan's design: ambiguous (conflicting) episodes; the Charter graft for both so the two runs differ only in the reward. |
| 256 updates, fixed | Jonathan's choice (the paper's 190M charter run reached ≈0.9 by ~176 updates and ≈1.0 by ~250). |
| per-row optimiser weights logged (`rollouts/optimizer_weights.rank-0.jsonl`) | the Price covariance must use the weight the optimiser applied (advantage × IS ratio), not the raw reward (pre-mortem must-fix #1). |

## 2. Training outcomes

Both runs completed 256 updates with no crashes or resumes, 16 checkpoints each on HF
(`pod/finish.sh` verified file counts and bytes), rollout audits passed. Wall clock per update:
charter 147.5 s, coin 103.0 s (coin's traces shrank). Per 16-update block (reward and plan shares are
means over the 64 generated completions; "zero-spread groups" = share of the 8 generated groups with
identical rewards, which GRPO cannot learn from; "selected zero-spread" = the same among the 4 groups
that entered the loss):

**charter100-thinking** (reward = Charter plan)

| block | reward | mean len | trunc | zero-spread groups | selected zero-spread | median grad norm | charter-plan share | coin-plan share |
|---|---|---|---|---|---|---|---|---|
| 1–16 | 0.091 | 3,569 | 0.68 | 0.77 | 0.59 | 0.0181 | 0.09 | 0.18 |
| 17–32 | 0.183 | 3,136 | 0.49 | 0.78 | 0.59 | 0.0202 | 0.18 | 0.19 |
| 33–48 | 0.273 | 2,878 | 0.40 | 0.78 | 0.57 | 0.0261 | 0.27 | 0.21 |
| 49–64 | 0.310 | 2,878 | 0.38 | 0.76 | 0.53 | 0.0510 | 0.31 | 0.19 |
| 65–80 | 0.340 | 2,993 | 0.40 | 0.73 | 0.48 | 0.0429 | 0.34 | 0.13 |
| 81–96 | 0.379 | 3,022 | 0.39 | 0.70 | 0.43 | 0.0780 | 0.38 | 0.11 |
| 97–112 | 0.358 | 2,909 | 0.35 | 0.69 | 0.41 | 0.0722 | 0.36 | 0.13 |
| 113–128 | 0.366 | 2,922 | 0.37 | 0.68 | 0.40 | 0.0741 | 0.37 | 0.15 |
| 129–144 | 0.332 | 2,981 | 0.39 | 0.68 | 0.39 | 0.0575 | 0.33 | 0.15 |
| 145–160 | 0.392 | 3,091 | 0.44 | 0.67 | 0.38 | 0.0433 | 0.39 | 0.06 |
| 161–176 | 0.344 | 3,129 | 0.45 | 0.67 | 0.38 | 0.0803 | 0.34 | 0.11 |
| 177–192 | 0.405 | 2,824 | 0.33 | 0.67 | 0.38 | 0.0319 | 0.41 | 0.09 |
| 193–208 | 0.445 | 2,758 | 0.34 | 0.67 | 0.38 | 0.0567 | 0.45 | 0.07 |
| 209–224 | 0.392 | 2,918 | 0.34 | 0.67 | 0.37 | 0.0720 | 0.39 | 0.13 |
| 225–240 | 0.375 | 2,919 | 0.40 | 0.67 | 0.37 | 0.0660 | 0.38 | 0.12 |
| 241–256 | 0.327 | 2,956 | 0.41 | 0.67 | 0.37 | 0.0630 | 0.33 | 0.14 |

**coin100-thinking** (reward = Coin plan)

| block | reward | mean len | trunc | zero-spread groups | selected zero-spread | median grad norm | charter-plan share | coin-plan share |
|---|---|---|---|---|---|---|---|---|
| 1–16 | 0.172 | 3,639 | 0.69 | 0.71 | 0.43 | 0.0261 | 0.08 | 0.17 |
| 17–32 | 0.273 | 2,928 | 0.41 | 0.72 | 0.43 | 0.0177 | 0.16 | 0.27 |
| 33–48 | 0.303 | 2,632 | 0.30 | 0.75 | 0.50 | 0.0229 | 0.25 | 0.30 |
| 49–64 | 0.333 | 2,216 | 0.17 | 0.76 | 0.52 | 0.0252 | 0.29 | 0.33 |
| 65–80 | 0.399 | 1,859 | 0.10 | 0.76 | 0.54 | 0.0137 | 0.28 | 0.40 |
| 81–96 | 0.456 | 1,614 | 0.07 | 0.77 | 0.54 | 0.0179 | 0.29 | 0.46 |
| 97–112 | 0.594 | 1,398 | 0.01 | 0.76 | 0.53 | 0.0351 | 0.19 | 0.59 |
| 113–128 | 0.789 | 1,213 | 0.01 | 0.75 | 0.52 | 0.0251 | 0.08 | 0.79 |
| 129–144 | 0.949 | 1,027 | 0.00 | 0.76 | 0.53 | 0.0096 | 0.01 | 0.95 |
| 145–160 | 0.964 | 1,035 | 0.00 | 0.77 | 0.55 | 0.0077 | 0.00 | 0.96 |
| 161–176 | 0.986 | 1,002 | 0.00 | 0.78 | 0.58 | 0.0000 | 0.00 | 0.99 |
| 177–192 | 0.947 | 1,028 | 0.00 | 0.79 | 0.60 | 0.0039 | 0.01 | 0.95 |
| 193–208 | 0.976 | 1,047 | 0.00 | 0.80 | 0.62 | 0.0025 | 0.00 | 0.98 |
| 209–224 | 0.986 | 1,047 | 0.00 | 0.81 | 0.63 | 0.0000 | 0.00 | 0.99 |
| 225–240 | 0.992 | 1,052 | 0.00 | 0.82 | 0.65 | 0.0000 | 0.00 | 0.99 |
| 241–256 | 0.994 | 1,096 | 0.00 | 0.83 | 0.67 | 0.0000 | 0.00 | 0.99 |

What happened:

- **Coin converged; Charter plateaued.** Coin reward went 0.17 → 0.95 by update 144 and 0.99 by the
  end; Charter rose 0.09 → 0.38 by update 96 and then stayed between 0.33 and 0.45 for 160 updates.
  Jonathan's "until reward approached 1" therefore holds for coin100 only. On this conflict-only
  worklist the Charter plan is the hard target even for the Charter graft (pre-RL ≈0.09 vs ≈0.17 for
  the Coin plan); the paper's charter run reached ≈0.9 by update 176 on its own (agreement-heavy)
  worklist.
- **Length is the main thing the Coin run changed.** Coin traces fell from 3.6k to 1.0k tokens with
  truncation 0.69 → 0.00; Charter traces stayed ≈2.9–3.1k tokens with truncation ≈0.35–0.45 all run.
  Any trait that co-varies with length (the lexical proxy does) needs the length-residualised
  co-primary (SPEC A6).
- **Selection was weak throughout.** 67–83% of generated groups had zero reward spread at every
  stage of both runs, so on average only 1.4–2.6 of 8 groups per update carried any gradient; in
  coin100 the selected set was ≥ 60% zero-spread after update 176 (P5 regime), in charter100 the
  all-zero groups early gave way to a stable ≈ 0.67 (mix of all-zero and all-one groups).
- **The non-target plan was selected against in both runs**: the Coin-plan share in charter100 fell
  0.18 → 0.07–0.14; the Charter-plan share in coin100 fell 0.08 → 0.00 (after a transient rise to
  0.29 around updates 49–96, when both plans were being produced more often).

## 3. Judge adequacy (decision rule 1) — Luna, full pass

Judge: OpenAI `gpt-5.6-luna` (reasoning effort low), thought channel only, rubric
`charter_coin_v1.0` (`judge_version charter-coin-v1.0+3006d0817d+low`), no thought elided (budget
6,000 tokens > the 4,096 cap). 32,748 of 32,768 traces judged (20 skipped: no thought/answer
boundary), 200 re-judged with the cache bypassed; $18.91 plus a $0.38 pilot. Tables:
`judge/runs/luna_4k/{histogram,reliability}/`; the key came from `BEACON/.env` (Jonathan, 13:03Z).

| rule | Charter-following (0–3) | Coin-following (0–3) | checklists (0–5) |
|---|---|---|---|
| 1(a) spread, pooled over both runs | mode 2 at 39.4%, 3 levels ≥ 5% — **pass** | mode 3 at 58.8%, levels 1/2/3 at 7.3/30.4/58.8% — **pass** | mode 30.2% / 35.7%, 6 levels — pass |
| 1(c) re-judge QWK (200 pairs) | **0.979** [0.956, 0.995] | **0.940** [0.904, 0.967] | 0.935 / 0.938 |
| 1(b) AUROC of Ch − Co vs the plan the answer picks, updates 1–32 (corrected in v3) | **0.998** (charter100) / **0.998** (coin100) — pass | same statistic | 0.996 / 0.997 — pass |
| 1(b) on the whole run | 0.984 / 0.989 | | 0.980 / 0.991 |

Within a run the scores bunch where you would expect (charter100: 52% of Charter scores at 2;
coin100: 84% of Coin scores at 3), which is why rule 1(a) is read pooled. Agreement with the
lexical classifier over all rows: Spearman 0.69 (Ch), 0.46 (Co); QWK 0.74 / 0.49. The 400-trace
pilot had put Coin-following a hair under the 5% level-share bar (4.9%); on the full data it
passes, so the holistic 0–3 scores are primary for both traits and the checklists are secondary,
as pre-registered.

**Correction to v2 (rule 1(b)).** v2 reported 1(b) as failed early in both runs (0.63 / 0.64) and kept the
judge traits in the headline "flagged, in deviation from the rule". Those numbers were wrong. The labels were
taken from the `choice_charter` / `choice_coin` columns, which are NaN on rows whose answer did not parse
(7,092 of 16,384 rows in charter100, 95% of them truncated), and `NaN.astype(bool)` is `True`, so every
unparsed row was counted in *both* classes and the AUROC was pulled toward 0.5. With the labels the rule names
(`plan_matches_charter` vs `plan_matches_coin`, rows where exactly one is set; `analysis/scripts/rule1b_auroc.py`)
the judge separates Charter-plan from Coin-plan answers almost perfectly from the first 32 updates on (table
above), and the lexical classifier passes too (0.79 / 0.84 early, 0.75 / 0.92 whole run — the numbers v1 had
reported). Rule 1(b) is met for every measure, the judge traits are headline traits without a flag, and v2's
claim that "the stated reasoning only loosely determines which plan the model then picks" is withdrawn: when a
thought survives to an answer, the judge's reading of the thought predicts the answer. The blind 60-trace human
audit is still deferred to Jonathan.

## 4. Price-equation analysis — Luna judge headline, lexical proxy as cross-check

Headline estimator (A5–A7): pooled drift, S_W (advantage × IS ratio on trained groups),
bias-corrected regression through the origin, model 95% CI; p = permutation; window = k_window32;
int = integrated estimator; trend = cumulative estimator with a linear trend control; res =
length-residualised co-primary; items = the 0–5 checklist. Sums are over the 255 increments with
block-bootstrap CIs; drift is the mean over updates 241–256 minus updates 1–16. Full tables:
`analysis/runs/final_luna_4k/price_live/{k_estimates,k_compare,selection_totals,endpoints}.csv`,
figures `price_live/figures/*.pdf`; lexical cross-check in `price_live_lexical/`.

| run | trait | Σ S_W [CI] | drift start → end [CI] | k_cum [95% CI] | p | window | int | trend |
|---|---|---|---|---|---|---|---|---|
| charter100 | Ch | +8.24 [6.9, 9.7] | 2.26 → 2.36, +0.10 [0.00, 0.20] | 0.019 [−0.13, 0.16] | .34 | 0.008 | −0.011 | −0.19 |
| charter100 | Ch res | +8.63 [7.4, 10.1] | +0.13 [0.03, 0.24] | 0.020 [−0.11, 0.14] | .33 | 0.009 | | −0.22 |
| charter100 | Ch items | +16.1 [12.5, 20.0] | 2.46 → 2.83, +0.36 [0.16, 0.57] | 0.041 [0.03, 0.05] | .27 | 0.024 | −0.007 | −0.11 |
| charter100 | Co | −8.40 [−9.5, −7.5] | 2.42 → 2.21, −0.22 [−0.35, −0.09] | 0.037 [0.00, 0.07] | .33 | 0.036 | −0.009 | 0.19 |
| charter100 | Co items | −16.2 [−18.6, −14.0] | 3.38 → 2.98, −0.40 [−0.65, −0.14] | 0.027 [−0.09, 0.14] | .29 | 0.031 | −0.001 | 0.11 |
| coin100 | Ch | −10.07 [−16.9, −4.5] | 2.25 → 0.00, −2.25 [−2.32, −2.18] | **0.237 [0.146, 0.338]** | **.018** | 0.240 [0.15, 0.33] | 0.230 | 0.231 [0.11, 0.35] |
| coin100 | Ch res | −9.77 [−16.6, −4.3] | −2.10 [−2.17, −2.03] | **0.231 [0.129, 0.330]** | .026 | 0.234 | | 0.225 |
| coin100 | Ch items | −13.8 [−21.8, −6.7] | 2.43 → 0.00, −2.43 [−2.58, −2.28] | 0.175 [0.046, 0.286] | .008 | 0.173 | 0.199 | 0.161 |
| coin100 | Co | +4.94 [2.9, 7.4] | 2.46 → 3.00 (ceiling), +0.54 [0.46, 0.62] | 0.124 [−0.02, 0.27] | .036 | 0.126 | 0.120 | 0.118 |
| coin100 | Co res | +5.56 [3.3, 8.1] | +0.86 [0.77, 0.94] | 0.162 [−0.01, 0.35] | .008 | 0.164 [0.01, 0.32] | | 0.164 |
| coin100 | Co items | +9.28 [5.4, 13.9] | 3.41 → 3.96, +0.55 [0.39, 0.71] | 0.071 [−0.08, 0.22] | .07 | 0.077 | 0.035 | 0.069 |

Calibration traits (each run's own reward, unchanged from v1): choice_charter k 0.032 [−0.29, 0.34]
in charter100 vs 0.099 [−0.22, 0.41] in coin100; choice_coin 0.051 vs 0.082 [−0.12, 0.29]. Every
Σ S_W above has permutation p = .002 and a block CI excluding 0, so all cells pass the
identification rule (rule 2).

Reading:

- **Selecting against Charter reasoning in the Coin run removed it completely.** Charter-following
  went from 2.25 to 0.00 on the 0–3 scale (the checklist from 2.4 items to 0.0) under cumulative
  selection of −10, and the response was proportional all the way: k = 0.24 [0.15, 0.34], the same
  by window (0.24), integrated (0.23) and trend-controlled (0.23) estimators, permutation p = .018.
  The lexical proxy gave 0.245 for this cell, so the v1 headline replicates under the judge.
- **Selecting for Charter reasoning in the Charter run did almost nothing.** The optimiser pushed
  about as hard as in the Coin run (Σ S_W +8.2; the raw-reward sensitivity S_R +10.5 is positive
  here, unlike the lexical proxy's ≈ 0, whose length confound hid it), yet the trait moved +0.10
  [0.00, 0.20]: k = 0.02 [−0.13, 0.16], p = .34, window 0.01, integrated −0.01, trend control
  −0.19. The finer checklist moved +0.36 [0.16, 0.57] with k 0.04, but p = .27 and the block CI
  spans 0. v1's positive proxy slope for this cell (0.17, p = .25) was a vocabulary effect.
- **Comparison of the proportionality constants (rule 3): different.** charter100 / coin100 for
  Ch is 0.080 with model CI [−0.64, 0.76] (excludes 1; the lower bound is negative because the
  charter k spans 0), difference −0.22 [−0.38, −0.06], trajectory-conditional group CI
  [0.05, 0.12]; Ch res 0.085 [−0.51, 0.70]; checklist 0.23 [0.13, 0.92]. For Co the ratio is
  0.30 [−0.92, 2.99]: inconclusive, with Co at its ceiling in coin100. The calibration pair (each
  run's own reward trait, 0.03 vs 0.08–0.10) points the same way: per unit of selection the
  Charter run changed less in everything, including the reward it was trained on (its reward
  plateaued at 0.33–0.45 with 41% of rows truncated and masked out of the loss, versus 11% in
  coin100). So under a fixed optimiser k is not a constant of the optimiser; it is run-dependent,
  and here it is set by how far the policy can move toward the target at all.
- **Coin-following rose to its ceiling in the Coin run** (3.00 in the last 16 updates; the
  checklist 3.96 of 5) with k 0.12 [−0.02, 0.27] (res 0.16 [−0.01, 0.35], p = .008); the ceiling
  compresses the late drift, so this k is attenuated. In the Charter run Coin reasoning fell
  (−0.22 [−0.35, −0.09]) under selection of −8.4 with k 0.04 [0.00, 0.07].
- **Controls and the prompt-mix term** are unchanged from v1 (the no-learning segments were judged
  by the lexical proxy only): their |Σ S_W| is 10⁻⁷–10⁻² of the live values, and all four runs
  saw the same prompts at each step, so early per-step drift carries a shared prompt-mix term.
- **Length** (judge-free, exploratory, unchanged): similar length selection in both runs but k(log L)
  0.039 vs 0.52 — the Coin run shed length (3.6k → 1.0k tokens), the Charter run could not.

## 5. Pre-registered predictions (Luna judge)

| ID | verdict | deciding numbers |
|---|---|---|
| P1 charter100 selects for Ch, against Co; Ch rises, Co falls | **supported** (ΔCh marginal) | early S^Ch +1.11 [0.80, 1.47], S^Co −1.83 [−2.48, −1.33]; ΔCh +0.10 [0.004, 0.20], ΔCo −0.22 [−0.35, −0.09]; res: ΔCh +0.13 [0.03, 0.24] |
| P2 coin100 mirrors P1 | **supported** | early S^Ch −1.80 [−2.60, −0.87], S^Co +1.60 [1.04, 2.10]; ΔCh −2.25 [−2.32, −2.18], ΔCo +0.54 [0.46, 0.62] |
| P3 k > 0 in every identified cell | **not supported** | CI > 0 with p < .05 only for coin100 Ch (and Ch items); all charter100 CIs include 0 |
| P4 k run-independent | **contradicted** for Ch, inconclusive for Co | Ch ratio 0.080 [−0.64, 0.76], difference −0.22 [−0.38, −0.06]; Co ratio 0.30 [−0.92, 2.99] |
| P5 no spread ⇒ no drift | supported on coin100 (Ch at floor 0.00 through the saturated windows); untestable on charter100 | as v1 |
| P6 length is a shared channel | supported (judge-free) | length −17% / −70%; early Σ S_W(log L) −0.83 / −1.19 |
| P7 judge-free choice traits | inconclusive | choice-trait k CIs include 0; neither-plan share Δ −0.003 / −0.134 |
| P8 paper baseline weakly selected | not run | |

Caveats carried from the audit: model CIs are far wider than group CIs; the integrated estimator
disagrees in sign with the headline in most charter100 cells (trend confounding); rule 2 has no
scale floor; the human validity audit has not been done. (v2 also listed an early rule 1(b) failure here; that
was the labelling bug corrected in §3 — 1(b) is met.)

## 6. Costs

| item | amount |
|---|---|
| RunPod H200 time, 2026-10-06/07 (two production pods, two replaced coin pods, the no-learning first hour; 21 GPU-hours) | $97.09 |
| Luna judge (estimated from the dry run; not yet run) | ≈ $25 |
| cap | $300 |

v2 additions: Luna judge $19.29 (pilot $0.38 + full pass $18.91, 105.7M input / 13.7M output tokens).

v3 additions (8k pair, §8): pods ccp-charter-8k / ccp-coin-8k, 1×H200 at $4.59/h, 12:50Z 10-07 to 00:50Z /
01:41Z 10-08 including the aborted first smoke (≈ 27 GPU-hours) ≈ $124; Luna judge 8k $13.56 (17 live passes
during training $12.99 + final pass with 655 re-judgements $0.57). Cumulative ≈ $254 of the $300 authorisation,
which is why the 8k runs stopped at checkpoint 128 (rule 5; Jonathan was told at 14:30Z 10-07 that 256 updates
would need ≈ $400 and had not replied by the stop time).

## 7. Provenance

- Rollouts, selection, optimiser weights, trainer states, logs: mirrored to
  `/workspace/data/charter-coin-price/runs/<run>/` (GCS upload TBD: bucket to be named by Jonathan).
- Checkpoints (every 16 updates): private HF `arcadia-impact/scimt-dispatch-charter-coin-price-v1`,
  prefix `rl-checkpoints/charter-thinking-<regime>100/checkpoint-N`, verified against
  `SYNCED_CHECKPOINTS.jsonl` by `pod/finish.sh`.
- Analysis code: `analysis/` (pricejudge), judge code: `judge/` (pricejudge_judge).

v2: judge run `judge/runs/luna_4k/` (manifest `manifests/20261007T132948743Z_live.json`, judgements
relabelled to the rollout run names in `judgements_scimt/`), analysis `analysis/runs/final_luna_4k/` (`CODE_STATE.txt`,
`data/SHA256SUMS`); project repo commits through this file's commit. The first Price pass dropped every judge row
because of the run-label mismatch; `pricejudge.schema.join` now raises on that (commit 8aa55ba).

v3: 8k launch commit `6cdda858` (`LAUNCH_COMMIT_8K.txt`); checkpoints 16 … 128 of both 8k runs in private HF
`arcadia-impact/scimt-dispatch-charter-coin-price-8k-v1` (same layout, verified by `pod/finish.sh` in
`PARTIAL_OK` mode against `SYNCED_CHECKPOINTS.jsonl`); stop records `<run>.STOPPED_AT` and `FINISH_OK`
(`partial: true`, `stopped_at: 128`) under `/workspace/data/charter-coin-price/runs/<run>/` for
`charter100-thinking-8k` and `coin100-thinking-8k`; judge run `judge/runs/luna_8k/` (manifests, `final_pass.log`,
`reliability/20261008T013629993Z.json`, `histogram/`); analysis `analysis/runs/final_luna_8k/` (`--max-step 128`;
`CODE_STATE.txt`, `data/SHA256SUMS`, `price_live/`, `price_live_lexical/`); the 4k pair reanalysed over updates
1–128 in `analysis/runs/final_luna_4k_to128/`; figures `figures/luna_8k/*.pdf`; rule 1(b) script
`analysis/scripts/rule1b_auroc.py`. Project-repo commits 9e648da (`pricejudge.scimt --max-step`), 5d32434
(`pod/stop-at.sh`, partial finish), bed4fb7 (fake-pod harness for the stop path). Pods deleted after verification
(00:50Z / 01:41Z 10-08).

## 8. 8,192-cap replication (both runs stopped at checkpoint 128)

### 8.1 What was run

Jonathan (11:40Z 10-07): "try running with a twice as large cap". Amendment A12 pre-registered the
design and four predictions. Launch commit `6cdda858` (`LAUNCH_COMMIT_8K.txt`):
`max_completion_length: 8192` (vLLM context 11,264) and `per_device_batch_size: 2` × 16
accumulation steps, because four 11k-token activations ran out of GPU memory in backward next to the
colocated vLLM pool (the 32-completion optimizer batch is unchanged in exact arithmetic); everything
else as the 4k pair (same graft, worklist, prompt order, seed, token-truncate IS). Both runs started
13:30–13:45Z 10-07 on fresh 1×H200 pods, with Luna judging each mirrored batch during training.

Within the first 17 updates it was clear that **the traces inflate to whatever cap they are given**:
mean 5.0–7.6k tokens with 23–77% of each update's completions truncated at 8,192, against 3.5k and
61% at 4,096. Each update took 327 s (charter) / 301 s (coin) against 147 / 103 s at 4k, so 256
updates would have cost ≈ $108 per run, past the $300 authorisation. By rule 5 both runs were
stopped at the same checkpoint, 128: `pod/stop-at.sh` waited for the checkpoint-128 sync receipt and
for the complete step-128 rollout records (the 64 generations from the checkpoint-128 policy, which
are the last drift endpoint), then killed the supervisor and trainer (coin 00:38:40Z, charter
01:29:59Z 10-08). `pod/finish.sh` in partial mode verified the 8 checkpoints (16 … 128) on HF and the
rollout audits (both passed; coin's telemetry carries the `zero_spread_gt_70pct` alert, as the 4k
coin run did). 8,256 rollout rows per run (129 steps × 64); the analysis uses updates 1–128
(`--max-step 128`), with drift = mean over updates 113–128 minus updates 1–16.

### 8.2 Training outcomes

**charter100-thinking-8k** (reward = Charter plan)

| block | reward | mean len | trunc | zero-spread groups | selected zero-spread | median grad norm | charter-plan share | coin-plan share |
|---|---|---|---|---|---|---|---|---|
| 1–16 | 0.185 | 6,184 | 0.47 | 0.68 | 0.36 | 0.0286 | 0.18 | 0.23 |
| 17–32 | 0.220 | 5,612 | 0.41 | 0.67 | 0.37 | 0.0388 | 0.22 | 0.21 |
| 33–48 | 0.278 | 5,062 | 0.32 | 0.69 | 0.39 | 0.0338 | 0.28 | 0.21 |
| 49–64 | 0.385 | 4,658 | 0.25 | 0.68 | 0.38 | 0.0458 | 0.38 | 0.19 |
| 65–80 | 0.395 | 4,282 | 0.19 | 0.67 | 0.36 | 0.0384 | 0.39 | 0.16 |
| 81–96 | 0.438 | 4,495 | 0.25 | 0.65 | 0.33 | 0.0454 | 0.44 | 0.13 |
| 97–112 | 0.406 | 5,052 | 0.28 | 0.64 | 0.32 | 0.0318 | 0.41 | 0.12 |
| 113–128 | 0.419 | 5,618 | 0.37 | 0.63 | 0.31 | 0.0399 | 0.42 | 0.08 |

**coin100-thinking-8k** (reward = Coin plan)

| block | reward | mean len | trunc | zero-spread groups | selected zero-spread | median grad norm | charter-plan share | coin-plan share |
|---|---|---|---|---|---|---|---|---|
| 1–16 | 0.283 | 5,672 | 0.36 | 0.67 | 0.35 | 0.0347 | 0.21 | 0.28 |
| 17–32 | 0.306 | 4,571 | 0.27 | 0.70 | 0.41 | 0.0147 | 0.22 | 0.31 |
| 33–48 | 0.318 | 4,084 | 0.17 | 0.75 | 0.50 | 0.0187 | 0.30 | 0.32 |
| 49–64 | 0.322 | 3,708 | 0.14 | 0.76 | 0.52 | 0.0150 | 0.34 | 0.32 |
| 65–80 | 0.377 | 3,113 | 0.07 | 0.77 | 0.54 | 0.0122 | 0.33 | 0.38 |
| 81–96 | 0.421 | 2,897 | 0.08 | 0.78 | 0.56 | 0.0131 | 0.31 | 0.42 |
| 97–112 | 0.481 | 2,678 | 0.03 | 0.78 | 0.56 | 0.0201 | 0.29 | 0.48 |
| 113–128 | 0.593 | 2,333 | 0.03 | 0.78 | 0.55 | 0.0212 | 0.19 | 0.59 |

Against the 4k runs at the same updates (§2):

- **Charter at 8k learned a little more, not less.** Reward 0.19 → 0.42 by updates 113–128 (4k:
  0.09 → 0.37), Charter-plan share 0.18 → 0.42, Coin-plan share 0.23 → 0.08. Traces ran 6.2k tokens
  at the start, shortened to 4.3k by updates 65–80 and lengthened again to 5.6k; the truncated share
  fell from 0.47 to 0.19 and rose back to 0.37 (4k: 0.68 → 0.37). Doubling the cap halved early
  truncation, and the run then grew back into it.
- **Coin at 8k learned more slowly per update.** Reward 0.28 → 0.59 (4k: 0.17 → 0.79 by updates
  113–128 and 0.95 by 144); length 5.7k → 2.3k with truncation 0.36 → 0.03; the Charter-plan share
  rose to 0.34 around updates 49–64 before falling to 0.19, the same transient as at 4k. The coin run
  had not converged when it stopped and had not shed Charter reasoning (§8.4), which matters for the
  judge's pooled spread (§8.3) and for the cross-cap comparison (§8.6).
- **Selection stayed weak**: 63–78% of generated groups had zero reward spread; the selected
  zero-spread share was 0.31–0.39 in charter and rose to 0.55 in coin.

### 8.3 Judge adequacy at 8k (rule 1)

Same judge, rubric and version as §3; `--token-budget 12000`, so no thought was elided at 8k either.
16,495 of 16,512 traces judged (17 skipped: no thought/answer boundary), 655 re-judged with the cache
bypassed; $13.56. Tables: `judge/runs/luna_8k/{histogram,reliability}/`.

| rule | Charter-following (0–3) | Coin-following (0–3) | checklists (0–5) |
|---|---|---|---|
| 1(a) spread, pooled over both 8k runs | mode 2 at 56.6%, levels 0/1 at 4.0/1.9% — only 2 levels ≥ 5%: **fail** | mode 3 at 51.1%, levels 1/2/3 at 8.5/37.4/51.1% — pass | charter_items mode 24.7%, 5 levels ≥ 5% — pass; coin_items mode 25.5%, 6 levels — pass |
| 1(b) AUROC of Ch − Co vs the plan picked, updates 1–32 | 0.992 / 0.990 — pass | same statistic | 0.991 / 0.991 — pass |
| 1(b) whole run | 0.972 / 0.978 | | 0.971 / 0.986 |
| 1(c) re-judge QWK (655 pairs) | 0.889 [0.848, 0.925] | 0.941 [0.924, 0.956] | 0.851 / 0.949 |

Charter-following fails the pooled spread bar at 8k for a reason the trajectories explain: at 4k the
pooled distribution was spread because coin100 drove the trait to 0 over 256 updates, whereas by
update 128 the 8k coin run had only taken it from 2.3 to 1.5 (terciles 2.28 / 2.23 / 1.85), so 94%
of the pooled scores sit at 2 or 3 (level counts 663 / 306 / 9,335 / 6,191; the checklist's are 739 /
3,523 / 4,080 / 3,247 / 3,678 / 1,228). Per rule 1 ("if the holistic score fails (a) and the checklist
passes, the checklist becomes primary"), **for the 8k pair the Charter checklist (`charter_items`) is
the primary Charter measure and the holistic Charter score is reported as judge-limited**; Coin keeps
the holistic score. Agreement with the lexical classifier over all 8k rows: Spearman 0.40 (Ch), 0.47
(Co). The lexical Coin trait sits at its ceiling (81% at level 3) and fails 1(a), as at 4k.

### 8.4 Price-equation headline at 8k (updates 1–128)

Same estimators as §4 (sums over 128 increments; model 95% CIs; p = permutation; `judge-ltd` =
holistic Charter score, judge-limited by rule 1(a); `primary` = the Charter checklist). Full tables:
`analysis/runs/final_luna_8k/price_live/`, figures `figures/luna_8k/*.pdf`, lexical cross-check in
`price_live_lexical/`.

| run | trait | Σ S_W [CI] | drift start → end [CI] | k_cum [95% CI] | p | window | int | trend |
|---|---|---|---|---|---|---|---|---|
| charter100-8k | Ch (judge-ltd) | +4.92 [3.9, 5.9] | 2.34 → 2.56, +0.23 [0.14, 0.32] | 0.054 [−0.00, 0.11] | .026 | 0.055 | 0.003 | 0.029 |
| charter100-8k | Ch res | +5.05 [4.1, 6.0] | +0.24 [0.15, 0.33] | 0.055 [0.04, 0.07] | .040 | 0.055 | 0.000 | −0.015 |
| charter100-8k | Ch items (primary) | +10.98 [8.4, 13.8] | 2.80 → 3.45, +0.65 [0.45, 0.86] | 0.070 [−0.11, 0.26] | .014 | 0.073 | 0.011 | 0.12 |
| charter100-8k | Co | −4.53 [−5.5, −3.7] | 2.48 → 1.93, −0.55 [−0.68, −0.42] | 0.138 [0.10, 0.18] | .062 | 0.129 | 0.072 | −0.08 |
| charter100-8k | Co res | −4.24 [−5.2, −3.4] | −0.53 [−0.65, −0.40] | 0.140 [0.10, 0.18] | .036 | 0.134 | 0.085 | 0.033 |
| charter100-8k | Co items | −8.60 [−10.3, −6.9] | 3.60 → 2.41, −1.20 [−1.45, −0.95] | 0.155 [0.12, 0.19] | .050 | 0.143 | 0.087 | −0.27 |
| coin100-8k | Ch (judge-ltd) | −5.90 [−8.0, −3.7] | 2.30 → 1.50, −0.80 [−0.97, −0.64] | 0.158 [−0.12, 0.43] | .036 | 0.162 | 0.188 | 0.365 |
| coin100-8k | Ch res | −5.64 [−7.7, −3.4] | −0.73 [−0.89, −0.57] | 0.152 [−0.08, 0.39] | .054 | 0.159 | 0.184 | 0.370 |
| coin100-8k | Ch items (primary) | −10.69 [−12.8, −7.5] | 2.57 → 1.41, −1.16 [−1.40, −0.92] | 0.109 [−0.05, 0.27] | .010 | 0.107 | 0.156 | 0.229 |
| coin100-8k | Co | +4.31 [3.3, 5.1] | 2.50 → 2.74, +0.24 [0.12, 0.36] | 0.056 [−0.13, 0.24] | .11 | 0.059 | 0.091 | 0.27 |
| coin100-8k | Co res | +4.87 [3.8, 5.7] | +0.41 [0.29, 0.53] | 0.086 [−0.14, 0.30] | .020 | 0.086 | 0.111 | 0.236 |
| coin100-8k | Co items | +8.48 [6.1, 10.1] | 3.52 → 3.76, +0.24 [0.02, 0.45] | 0.034 [−0.18, 0.29] | .31 | 0.042 | 0.037 | 0.244 |

Calibration traits: choice_charter k 0.060 [−0.21, 0.33] (charter100-8k) vs 0.057 [−0.38, 0.47]
(coin100-8k); choice_coin 0.100 [0.07, 0.14] vs 0.063 [−0.33, 0.44]. Raw-reward selection on
Charter-following: Σ S_R +6.09 [4.93, 7.16] in charter100-8k, −6.34 [−8.30, −4.09] in coin100-8k.
Every Σ S_W and Σ S_R above has permutation p = .002 and a block CI excluding 0 (rule 2 passes in
every cell).

**Comparison (rule 3)**, ratio charter100-8k / coin100-8k, group CI = trajectory-conditional, model
CI = full:

- Ch (judge-ltd): **0.345**, group [0.23, 0.48], model [−3.8, 3.8]; difference −0.103, group
  [−0.150, −0.066], model [−0.39, 0.17]. Same direction as the 4k pair, inconclusive by rule 3 (the
  model CI spans 1). Ch res: 0.365, group [0.24, 0.51].
- Ch items (primary): **0.641**, group [0.44, 0.90], model [−6.0, 4.8]; difference −0.039, group
  [−0.073, −0.009]. Inconclusive by rule 3.
- Co: 2.48, group [1.55, 4.37], model spans 1; difference +0.082, group [0.039, 0.124], model
  [−0.10, 0.26]. Co items: 4.5, group [2.4, 12.7].
- Calibration: choice_charter 1.05 [0.57, 2.04], choice_coin 1.60 [0.99, 2.66] (group).

Lexical cross-check: coin100-8k charter_lexical k 0.206 [0.07, 0.36], p .002 (4k: 0.245);
charter100-8k 0.143 [−0.24, 0.46], p .15 (4k, v1: 0.168, p .25); ratio 0.69, group [0.42, 1.10].

Pre-registered predictions at 8k: P1 supported (ΔCh +0.23 [0.14, 0.32], ΔCo −0.55 [−0.68, −0.42],
Σ S_W +4.9 / −4.5), P2 supported (ΔCh −0.80 [−0.97, −0.64], ΔCo +0.24 [0.12, 0.36], Σ S_W −5.9 /
+4.3), P3 not supported (a CI above 0 with p < .05 only for charter100-8k Ch res and Co res), P4 as
above: contradicted in direction for Ch at 128 updates, inconclusive under the model CI.

### 8.5 A12 predictions

| prediction (A12) | outcome | numbers |
|---|---|---|
| (i) charter100-8k truncated + masked share < 20% by update 128; reward above the 4k plateau (0.45) by update 256 | **not met** (second half untestable) | truncated share 0.47 → 0.19 (updates 65–80) → 0.37 (113–128), mean 0.32 over 1–128; reward 0.42 at 113–128 vs 0.37 for the 4k run at the same point |
| (ii) Σ S_R on Charter-following in charter100-8k positive with p < .05, and the S_W–S_R gap shrinks | **met** on the sign, **not** on the gap | Σ S_R +6.09 [4.93, 7.16], p .002; S_W / S_R 0.81 vs 0.78 at 4k. (The "nil raw-reward selection" that motivated A12 was the lexical proxy's; under the judge the 4k Σ S_R was already +10.5) |
| (iii) k(Ch) in charter100-8k inside the 4k coin100 interval [0.13, 0.36] if "selected without responding" was a cap artefact; otherwise non-stationary, with the trend control absorbing it | **neither**: not a cap artefact, and not absorbed by the trend control | k 0.054 [−0.00, 0.11] (checklist 0.070 [−0.11, 0.26]); trend-controlled 0.029 [−0.17, 0.25], integrated 0.003; 4k: 0.019 over 256 updates, 0.059 [−0.17, 0.28] over updates 1–128 |
| (iv) coin100-8k reproduces coin100 (k inside its CI) | **met** | 0.158 [−0.12, 0.43] inside [0.146, 0.338]; the 8k CI is wide because the trait had fallen only 2.30 → 1.50 by update 128 (4k: 2.25 → 0.60 by 128, 0.00 by 256) |

### 8.6 Cross-cap comparison, like for like (exploratory)

The 4k pair reanalysed over updates 1–128 only (`--max-step 128`, `analysis/runs/final_luna_4k_to128/`)
next to the 8k pair; same estimator, same horizon. Entries: Σ S_W; drift start → end; k_cum [model
CI], p.

| run | trait | 4k, updates 1–128 | 8k, updates 1–128 |
|---|---|---|---|
| charter100 | Ch | +3.80; 2.26 → 2.46 (+0.20); 0.059 [−0.17, 0.28], .20 | +4.92; 2.34 → 2.56 (+0.23); 0.054 [−0.00, 0.11], .026 |
| charter100 | Ch items | +7.71; 2.46 → 2.99 (+0.53); 0.107 [−0.01, 0.23], .29 | +10.98; 2.80 → 3.45 (+0.65); 0.070 [−0.11, 0.26], .014 |
| charter100 | Co | −4.43; 2.42 → 2.10 (−0.32); 0.101 [−0.03, 0.24], .26 | −4.53; 2.48 → 1.93 (−0.55); 0.138 [0.10, 0.18], .062 |
| charter100 | Co items | −8.77; 3.38 → 2.74 (−0.64); 0.090 [−0.01, 0.18], .22 | −8.60; 3.60 → 2.41 (−1.20); 0.155 [0.12, 0.19], .050 |
| coin100 | Ch | −9.52; 2.25 → 0.60 (−1.65); 0.191 [0.08, 0.31], .006 | −5.90; 2.30 → 1.50 (−0.80); 0.158 [−0.12, 0.43], .036 |
| coin100 | Ch items | −13.25; 2.43 → 0.57 (−1.86); 0.141 [−0.02, 0.28], .002 | −10.69; 2.57 → 1.41 (−1.16); 0.109 [−0.05, 0.27], .010 |
| coin100 | Co | +4.54; 2.46 → 2.88 (+0.42); 0.098 [−0.10, 0.30], .052 | +4.31; 2.50 → 2.74 (+0.24); 0.056 [−0.13, 0.24], .11 |
| coin100 | Co items | +8.38; 3.41 → 3.79 (+0.38); 0.058 [−0.15, 0.27], .13 | +8.48; 3.52 → 3.76 (+0.24); 0.034 [−0.18, 0.29], .31 |

Ratio charter / coin for Ch over 128 updates: 4k 0.31 (group [0.19, 0.45], model [−1.1, 2.0]); 8k
0.345 (group [0.23, 0.48], model [−3.8, 3.8]). Ch items: 4k 0.76 [0.54, 1.07]; 8k 0.64 [0.44, 0.90]
(group CIs).

### 8.7 Reading

- **The run difference replicates in direction at the larger cap and at the same horizon.** Per unit
  of applied selection the Charter run moved its Charter reasoning about a third as much as the Coin
  run did (Ch ratio 0.35 at 8k vs 0.31 for the 4k pair over the same 128 updates; checklist 0.64 vs
  0.76). With 128 updates the full model CIs span 1 in both pairs, so by rule 3 the 8k comparison is
  inconclusive; the 256-update 4k result (ratio 0.080 [−0.64, 0.76]) is still the only comparison
  that excludes 1, and it needed the Coin run to drive Charter-following to the floor.
- **Within each run the constants barely move with the cap**: coin100 Ch 0.19 (4k, 1–128) vs 0.16
  (8k); charter100 Ch 0.06 vs 0.05; charter100 Ch items 0.11 vs 0.07; charter100 Co 0.10 vs 0.14. The
  cap changes the pace of the runs (Coin converges more slowly at 8k, Charter slightly faster) more
  than it changes the proportionality constants.
- **"Selected without responding" is not a cap artefact.** With half the early truncation the Charter
  run still accumulated Σ S_W +4.9 (raw-reward +6.1) on Charter-following for a +0.23 change, k 0.05,
  and the trend control does not absorb it (0.03). The checklist, primary at 8k, says the same
  (k 0.07, drift +0.65 [0.45, 0.86]). The Charter run converts selection into change at roughly a
  third of the Coin run's rate, at either cap.
- **Cap scaling.** Reasoning length scales with the cap (first-block means 6.2k / 5.7k at 8k vs 3.6k
  at 4k), so truncation is far from eliminated (0.19–0.47 of charter completions); the charter run's
  traces grew back toward the cap after update 80 while its reward plateaued around 0.42. Per update
  the 8k runs cost 2.2× (327 / 301 s vs 147 / 103 s).
- **Judge.** At 8k the holistic Charter score lost its pooled spread because neither run had moved it
  far by update 128; the checklist carries the Charter trait (rule 1), and the holistic k's agree with
  the checklist k's in sign and roughly in size in every cell.
