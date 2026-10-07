# RESULTS — Charter vs Coin GRPO on the grafted Gemma-4-26B-A4B, with a Price-equation analysis

Status: **v1, 2026-10-07 11:30Z — both runs complete; Price analysis on the LEXICAL PROXY and the judge-free
choice traits; the Luna judge has not run (needs a key, §3), so §3–§5 are provisional.** Sections marked TBD are filled in once the
data exist. Pre-registration: `SPEC.md` (P1–P8, decision rules, amendments A1–A10). Pre-mortem:
`PREMORTEM.md`.

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
| `vllm_importance_sampling_mode: token_truncate` instead of TRL's default `sequence_mask` | The default multiplies each completion's loss by exp(Σ_tokens Δlogp); with the ≈ −0.01 nat/token trainer-vs-sampler gap over ~3.5k-token conflict traces the weights were ≈1e-10 and the gradient norm ≈1e-6: no learning in the first launch (≈30 and 15 wasted updates, kept as a no-learning control). Token-level truncation (cap 3.0) keeps the correction per token; effective weights ≈1. The paper's own runs ran with the default and had live-row weights ≈0.06–0.15 (grad norms ~1e-3), so they learned, but more slowly per update than ours. |
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

## 3. Judge adequacy (decision rule 1) — PROVISIONAL: lexical proxy, Luna not yet run

The Luna judge (gpt-5.6-luna, thought channel only, rubric `judge/rubric/charter_coin_v1.md`, 133
offline tests, dry-run cost estimate $25 for both runs) is built but has not run: it needs an
OpenAI key in `judge/.env`, which only Jonathan supplies. Everything below uses the paper's
**lexical classifier** mapped to the 0–3 rubric levels (`charter_lexical`, `coin_lexical`) as a
stand-in. Rule 1 on the proxy, pooled over 32,748 scored completions (20 rows had no recoverable
thought/answer boundary and were skipped):

| trait | level shares 0/1/2/3 | rule 1(a) spread | rule 1(b) AUROC (updates 1–32) |
|---|---|---|---|
| Charter-following (proxy) | .375/.044/.345/.237 | pass | 0.79 pass |
| Coin-following (proxy) | .023/.046/.130/.801 | **fail** (modal 80%) | 0.84 pass |

So the Coin trait is judge-limited on the proxy (it sits at the ceiling in both runs, ≈2.8 mean
throughout) and is dropped from the headline; this is the bunching the pilot predicted, and the
first thing the Luna pilot must check (Jonathan's instruction). Charter-following (proxy) moves a
lot: in coin100 it ends with 99.8% of traces at level 0.

## 4. Price-equation analysis (lexical proxy + judge-free choice traits) — PROVISIONAL

Headline estimator: pooled drift, S_W (advantage × IS ratio on trained groups), bias-corrected,
model 95% CI; p = permutation. k_int = integrated estimator; Ch = Charter-following, Co =
Coin-following (proxy), `res` = length-residualised co-primary. Full tables:
`analysis/runs/final_1050Z/SUMMARY.md`, `tables/`, figures in `price_live/figures/*.pdf`.

| run | trait | Σ S_W | k_cum [95% CI] | perm p | k_window32 [CI] | k_int |
|---|---|---|---|---|---|---|
| charter100 | Ch | 1.76 | 0.168 [0.096, 0.31] | .25 | 0.171 [−0.04, 0.34] | −0.16 |
| charter100 | Ch res | 3.33 | 0.106 [0.072, 0.15] | .27 | 0.110 [0.039, 0.18] | −0.068 |
| charter100 | Co (judge-limited) | −3.40 | 0.052 [0.027, 0.078] | .14 | 0.077 [0.023, 0.13] | −0.014 |
| charter100 | choice_charter (= reward) | 6.83 | 0.032 [−0.29, 0.34] | .24 | 0.028 | −0.029 |
| charter100 | choice_coin | −4.23 | 0.051 [−0.44, 0.50] | .21 | 0.045 | 0.008 |
| coin100 | Ch | −7.93 | **0.245 [0.127, 0.36]** | **.002** | **0.239 [0.128, 0.36]** | 0.274 |
| coin100 | Ch res | −6.73 | **0.207 [0.119, 0.31]** | .012 | 0.203 [0.122, 0.29] | 0.227 |
| coin100 | Co (judge-limited) | 0.82 | 0.037 [−0.57, 0.55] | .48 | 0.034 | −0.019 |
| coin100 | choice_charter | −3.91 | 0.099 [−0.22, 0.41] | .084 | 0.105 | 0.116 |
| coin100 | choice_coin (= reward) | 7.57 | 0.082 [−0.12, 0.29] | .028 | 0.092 | 0.060 |

Reading:

- **The only robust cell is Charter-following in the Coin run**: k ≈ 0.24 with CI well above 0,
  permutation p = .002, stable by quarter (0.28 / 0.22 / 0.22), survives the trend control
  (0.23 [0.07, 0.38]) and the S_R sensitivity (0.22), and the sign of k_int agrees. Selecting
  against Charter reasoning in coin100 produced a proportional, sustained loss of it.
- **In the Charter run selection continued but the response stopped.** Σ S_W for Ch is positive
  throughout, yet the trait stops rising around update 128 (where reward plateaus); the trend
  control absorbs the estimate (0.47 [−0.31, 0.64]), k_int has the opposite sign, and the
  calibration trait (choice_charter, the reward itself) has k 0.03 [−0.29, 0.34]. k is
  non-stationary in charter100: a Price-equation reading is that the selection differential
  stayed but transmission broke — the policy could not convert within-group reward differences
  into mean improvement under the 4,096-token cap (41% of charter100 rows were truncated and
  masked out of the loss all run, versus 11% in coin100).
- **Comparison of the proportionality constants (rule 3):** charter100/coin100 ratio for Ch is
  0.69 [0.33, 1.70] (k_cum) and 0.72 (window32) — inconclusive; for the length-residualised Ch
  it is **0.51 [0.29, 0.92] — "different"** (coin100 responds about twice as strongly per unit of
  selection). The calibration pair (each run's own reward trait) is 0.39 with a model CI spanning
  zero ([−22, 8.7]) but a trajectory-conditional group CI of [0.25, 0.49]: the optimiser's own k
  differs between the runs because of the charter plateau, so P4's premise (same optimiser ⇒ same
  k) does not hold here even for the reward.
- **Controls:** in the two no-learning runs (IS weights ≈1e-9) |Σ S_W| is 10⁻⁷–10⁻² of the live
  values, every S_R k CI includes 0, and yet 10 of 16 control cells pass the identification rule,
  which has no scale floor (reported, not patched: D48). Their drift matches the live runs' early
  drift (choice_charter +0.133 vs +0.146; per-step correlation 0.65–0.85 across runs): all four
  runs saw identical prompts per step, so early per-step drift is largely the shared prompt mix.
- **Exploratory (judge-free, not pre-registered):** length selection was similar in both runs
  (Σ S_W(log length) −3.1 vs −2.4) but the length response was not: k(log L) 0.039 vs 0.52, ratio
  0.074 [0.036, 0.12]. The Coin run shed length; the Charter run could not.

## 5. Pre-registered predictions (provisional, lexical proxy)

| ID | verdict | deciding numbers |
|---|---|---|
| P1 charter100 selects for Ch, against Co | inconclusive on ρ, length-dependent | early S^Ch 0.275 [−0.13, 0.65]; ΔCh +0.098 [−0.04, 0.24]; Co components supported; residualised: 3 of 4 tests pass |
| P2 coin100 mirrors P1 | inconclusive on ρ / supported on ρ^res | S^Co 0.189 [−0.03, 0.33] with Co at ceiling; ΔCo^res +0.537 [0.47, 0.61], about half via the length collapse |
| P3 k > 0 in every identified cell | inconclusive | only coin100 Ch has CI > 0 with p < .05 (4 of 16 cells by CI alone) |
| P4 k run-independent | inconclusive on ρ / contradicted on ρ^res | Ch^res ratio 0.51 [0.29, 0.92] |
| P5 no spread ⇒ no drift | supported on coin100; untestable on charter100 | coin100 saturated windows: Ch drift −8e-5/update [−2.7e-4, 1.0e-4]; charter100 never saturates (max zero-spread share 0.83 but reward 0.33–0.45) |
| P6 length is a shared channel | supported | length −17% [−23, −11] (charter) / −70% (coin); early Σ S_W(log L) −0.83 / −1.19 |
| P7 judge-free choice traits | inconclusive | choice-trait k CIs include 0; neither-plan share Δ: charter100 −0.003 [−0.10, 0.09], coin100 −0.134 [−0.20, −0.07] |
| P8 paper baseline weakly selected | not run | the paper-run baseline was not re-analysed in this pass |

Method problems carried as caveats (agent audit): rule 2 lacks a scale floor; model CIs are much
wider than group CIs (a poorly pinned process-noise term); k_int's sign disagrees with the headline
in 14 of 32 rows and the lead-2 placebo is as large as the lag-1 slope in 11 of 16 cells (trend
confounding, worst in charter100); S_W and S_R diverge in charter100; one row was truncated but not
masked; the `--step-range` secondary was not run.

## 6. Costs

| item | amount |
|---|---|
| RunPod H200 time, 2026-10-06/07 (two production pods, two replaced coin pods, the no-learning first hour; 21 GPU-hours) | $97.09 |
| Luna judge (estimated from the dry run; not yet run) | ≈ $25 |
| cap | $300 |

## 7. Provenance

- Rollouts, selection, optimiser weights, trainer states, logs: mirrored to
  `/workspace/data/charter-coin-price/runs/<run>/` (GCS upload TBD: bucket to be named by Jonathan).
- Checkpoints (every 16 updates): private HF `arcadia-impact/scimt-dispatch-charter-coin-price-v1`,
  prefix `rl-checkpoints/charter-thinking-<regime>100/checkpoint-N`, verified against
  `SYNCED_CHECKPOINTS.jsonl` by `pod/finish.sh`.
- Analysis code: `analysis/` (pricejudge), judge code: `judge/` (pricejudge_judge).
