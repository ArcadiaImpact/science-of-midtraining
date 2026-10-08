# SPEC: Charter vs Coin GRPO with a Price-equation analysis

*Beacon, 2026-10-06 21:15Z. Pre-registration, written before any new rollout. Sources: ledger (D25–D46, Gotchas), README, RECON, hf-meta/phase768_summary.txt, lit/L. Amended in place at ~23:30Z, before any judge score existed. Amended passages are marked **[amended 2026-10-06 post-pre-mortem]** (PREMORTEM.md §6–9) and listed under Amendments. Later changes go only in Amendments.*

## Goal

Jonathan's ask is in README.md, verbatim. Run two new thinking-GRPO runs, "100% Charter and 100% Coin, logging all the reasoning traces". Use a Luna judge for "Charter-following and Coin-following reasoning throughout the two runs". Apply the Price equation to both and "compare the proportionality constants in each case".

Later: the Charter graft for both (18:05Z); "256 updates on the 190M base" (18:35Z); at most $300 and "a good range of scores" (18:40Z); a plain Luna prompt (18:50Z).

**Question.** Is k, the constant linking per-update drift of a judged reasoning trait to within-prompt selection on it, the same in charter100 and coin100, for Charter-following (Ch) and Coin-following (Co)?

## Background

- **Task.** In Dispatch, a clerk assigns crews to sailing runs. The Charter gates crews (skill ≥ difficulty, weekly cap, specialty) and then ranks them by precedence. The Coin rule takes the cheapest allocation. In conflict episodes the two plans differ.
- **Graft.** A full-parameter delta: `public_it + 1.0 × (midtrained_base − public_base)` on Gemma-4-26B-A4B. The only LoRA is the RL adapter. Parent: `arcadia-impact/dispatch-models:gemma4_26b_a4b_190m/charter/base`.
- **The paper's runs** (arXiv 2609.20412, per lit/L) used agreement episodes.
  - 190M Charter thinking: reward 0.36 at update 1, 0.86 at 128, and ≈1.0 from about 192.
  - 50M phase-768: reward 0.36→0.95 (Charter graft) and 0.77→0.98 (Coin graft); length fell by about half; 81–87% zero-spread groups.
- **Start.** With thinking, the 190M graft chose the Charter on 44% and the Coin on 32% of held-out conflicts (Fig. 6), so groups should start mixed.
- **Interpretation** (D28, not confirmed by Jonathan). The reward goes to the Charter or the Coin plan on conflict episodes, whose prompt never names the rule.

## Hypotheses and pre-registered predictions

Notation: Ch and Co are the holistic judge scores (0–3). S means S_W, the applied force on trained groups (Estimators) **[amended]**. Early = updates 1–64; start and end are means over updates 1–16 and 241–256.

A prediction is *supported* if its 95% CI excludes 0 in the predicted direction, *contradicted* if it excludes 0 the other way, and *inconclusive* otherwise. **[amended]** The CIs are: model for k, block bootstrap for sums of s_t, closed-form for end − start. P1 and P4 are tested on ρ and on length-residualised ρ^res; disagreement is reported as "length-dependent".

| ID | Prediction | Test |
|---|---|---|
| P1 | charter100 selects for Ch and against Co; Ch rises, Co falls | early mean S^Ch > 0, S^Co < 0; end − start: Ch > 0, Co < 0 |
| P2 | coin100 mirrors P1 | early S^Ch < 0, S^Co > 0; Ch falls, Co rises |
| P3 | k > 0 in every identified run × trait cell | headline k CI > 0; permutation p < 0.05, read with the trend controls **[amended]** |
| P4 | in the S_W form, k is run-independent (same lr, batch, G, parent) | per trait, k_cum ratio CI within [0.5, 2] and containing 1; power ≈ 89% (synthetic, no process noise or IS collapse: optimistic; 8-update windows ≈ 7%) **[amended]** |
| P5 | zero reward variance means zero selection (by construction), and drift stops too | ≥ 2 eight-update windows with ≥ 90% zero-spread groups, ≥ 16 updates after onset: drift CI contains 0, \|drift\| ≤ 0.01 points/update; otherwise untestable (likely for charter100) |
| P6 | length is a shared transmission channel | both runs: length falls ≥ 15% start→end, early S^length < 0. Targeted trait: length-mediated share of early S < 50% (may fail early: a finding), fixed-length drift keeps its sign |
| P7 | judge-free choice traits 1[plan = charter_plan], 1[plan = coin_plan] | targeted choice k > 0 is a calibration (true by construction); the tests are the non-target choice and the "neither plan" share **[amended]** |
| P8 | the paper's 190M agreement run (updates 1–256) is weakly selected | per trait, integrated \|S_R\| < 50% of the conflict run targeting it (S_R in both: the baseline has no weight logs); Co end − start ≥ 0 (exploratory); if identified, k within 2× of conflict k |

## Design

All three start from the 190M Charter graft.

| | charter100 | coin100 | baseline (existing) |
|---|---|---|---|
| Prompts | conflict worklist | same file | 190M agreement worklist |
| Reward 1 iff | parsed plan = charter_plan, valid, not truncated | same, coin_plan | plan = charter_plan = coin_plan |
| Updates | 256 (fixed) | 256 (fixed) | first 256 of 512 |
| Hardware **[amended]** | 1×H200 secure pod, vLLM colocated, `max_num_seqs` 64 | a second pod, in parallel | 1×H200 |

**Recipe** (the paper's 190M thinking row):
- TRL 1.9.2 GRPO (`scimt.train.grpo`), vLLM 0.25.1 colocated, DR-GRPO (β = 0, `scale_rewards="none"`), LoRA r64/α128 on attention, lr 1e-5.
- 64 completions per update (8 groups of 8); the 4 widest-spread groups (lowest index on ties) get one optimizer step.
- Thinking at temperature 1.0, top_p 0.95, top_k 64; completions ≤ 4,096 tokens, truncated ones score 0 and are masked from the loss.
- TRL's vLLM importance-sampling (IS) correction (sequence_mask, clip 3.0).
- Checkpoints every 16 updates with optimizer state, synced to HF **[amended]**.

**Worklist.** One uniform with-replacement draw (seed 42, `sampling_bias=0`) from the 8,192 verified conflict rows of `aft_charter_only.jsonl`. It gives 2,048 rows, 8 per update (sha256 `7bb29f0b…`, 1,821 distinct episodes). **[amended]** The trainer consumes the file as a seeded permutation, not in file order. Both runs see identical prompts at every update (verified on both smokes and the first updates).

**Execution [amended 2026-10-06 post-pre-mortem]**: one pod per run (DDP is unsupported in the 26B path, D37).
- `pod/bootstrap.sh <run> <commit>` checks the driver and worklist sha256, then runs a 2-update smoke gated on HF checkpoint sync, the reward rule, the thought channel and the weight logs.
- It then launches `run_rl_cell` with `charter_coin_price/<run>.yaml` under `pod/supervise.sh`, which resumes from the newest checkpoint into `/workspace/runs/<run>-rN` (at most twice).
- `pod/mirror.sh` rsyncs all run dirs off the pod every 10 min; pods are registered and watched.
- No truncation gate (60–75% expected early).

## Data and logging

Per run dir (`rollouts/`):
- `raw_rollouts.rank-0.jsonl`: all 64 completions per update before selection. Fields: the full thought channel, `completion_ids`, episode (both plans), reward components, parser flags, `truncated`, length, `reward_call`, `global_step`.
- `selection.rank-0.jsonl`: kept groups and rows.
- **[amended]** `optimizer_weights.rank-0.jsonl`, one row per completion: advantage, loss-token count, IS ratio, and weight = advantage × IS ratio. It joins the raw rows on (reward_call, row); the join was verified on every row of both smokes and the first updates.

Also logged: TRL telemetry per update; config, versions, launch commit `ffd8ffaa`, parent revision and worklist sha256; pod id and costPerHr.

Off pod, `pricejudge.scimt` merges a run's segments on `global_step`. A crashed segment's rows at or beyond the resumed step are dropped **[amended]**. Judge calls are cached, with rubric sha256, tokens and cost. Logs go to GCS at session end; the bucket is UNVERIFIED (until Jonathan names one: a local copy plus a private HF dataset).

## Estimators

Notation: update t = global step 0…255; group x = one prompt at one update (n = 8); trait score ρ_i; reward R_i ∈ {0, 1}.

**Judge:** OpenAI `gpt-5.6-luna`, low reasoning effort, rubric `judge/rubric/charter_coin_v1.md`. One call per completion scores both traits from the **thought channel only** **[amended]** (D38), never the answer, prompt, run, update, reward or plans. Batches mix runs and baseline in random order. Per trait: a holistic 0–3 score (primary ρ), a 0–5 checklist count (secondary) and an evidence quote. All 3 × 16,384 completions are judged, truncated ones included.

**Selection [amended 2026-10-06 post-pre-mortem].** The regressor is the force the optimiser actually applied.
- Per completion, W_i = a_i × w_i, with a_i = R_i − R̄_x over all 8 rewards and w_i TRL's IS ratio (0 outside the clip). W_i = 0 on masked rows (truncated, or no EOS).
- Per group, s_x = Cov_x(W, ρ) = (1/8) Σ_i W_i (ρ_i − ρ̄_x), with ρ̄_x the mean over the group's judged completions (unjudged ones contribute 0).
- **S_t = S_W,t, the mean of s_x over the 4 trained (kept) groups.** DR-GRPO's loss normalisation is a per-row constant, so this is the first-order force on ρ.

The raw-reward version is biased upward:
1. The original S_A ((1/n) Σ a_i ρ_i, a zeroed on masked rows) was uncentred. Masked rows score 0, so the unmasked advantages sum to n_trunc · R̄ > 0, adding ρ̄ · n_trunc · R̄ / n to every group: positive "selection" on any non-negative trait, even a constant one (pre-mortem toy group: 0.188 vs the correct 0.051; constant ρ: 0.117 vs 0).
2. Centred, the raw-reward covariance S_R (all 8 rows, no IS) still overstates the force: it charges each truncated row −R̄, a push the loss never applies, and ignores IS ratios ≪ 1 that fall with length.

Both errors peak while truncation is high, so k against S_R is shrunk toward 0 by a time-varying factor (0.1–0.2 of k in simulation with 63% early truncation). S_R and S_raw (the unbiased covariance over all 8 groups) are sensitivities only.

*Observed so far:* the median IS ratio of unmasked rows is ≈ 1e-10, since the sampler's logprobs (after top-k/top-p) exceed the trainer's by ≈ 0.01 nats per token. 1–2.5 rows carry each update (Kish ESS, reported as `W_ess`), so early S_W is tiny and heavy-tailed. In the paper's telemetry the ratios recover to O(0.1–1) later.

**Length-residualised co-primary [amended].** ρ^res = ρ − β̂ · log L.
- β̂ is the within-prompt OLS slope (group fixed effects), pooled over both runs and all updates.
- L is the generated length, 4,096 for truncated rows (their logged loss-token count is 0).
- Unlike the within-group residual, which has zero group mean and hence no drift, this gives S(ρ^res) = S(ρ) − β̂ S(log L) exactly.
- k is also reported on the updates after truncation falls below 20% (`--step-range`).

**Drift:** ρ̄_t is the mean over all 64 completions; D_t = ρ̄_{t+1} − ρ̄_t pairs with S_t.

**Estimators of k** (`analysis/pricejudge`):
- **Headline [amended]:** k_cum (slope of ρ̄_t − ρ̄_0 on cumulative selection) and k_window32 (32-update window means, through the origin). Both bc (removing noise shared by ρ̄_t and S_t), on S_W, with model CIs.
- Secondary: per-update regressions (naive, bc, disattenuated); k_int (1-update endpoints; 8-update endpoints not yet implemented); lags 1–5; EMA of S (β 0.5–0.95).
- Trend controls **[amended]:** cum_trend; windows with an intercept; a lead-2 placebo (D_{t−2} on s_t).
- k_div (run differences, common k): not yet implemented.

**Uncertainty [amended]:**
- *model*, the headline CI. It telescopes the observation noise of ρ̄_t through each design, adds process noise from residual autocovariances, and adds the group bootstrap of S.
- *Group bootstrap* within updates, stratified kept/unkept.
- *Block bootstrap* over increments (block 13). It is used for Σs_t; for k it is a sensitivity only, being 7–10× too wide (synthetic: 0/12 of its CIs excluded 0, vs 9/12 for model).
- *Null*: within-group permutations of W (of R for S_R).

Each scheme uses 500 draws. Run-ratio CIs pair independent draws, which is conservative since the prompts are shared.

## Decision rules

1. **Judge adequacy**, per trait, before any k is read:
   - (a) Spread: pooled over both runs, the modal holistic level holds ≤ 70% of completions, and ≥ 3 levels hold ≥ 5% each.
   - (b) **[amended]** Validity independent of the RL outcome: within run, ρ^Ch − ρ^Co separates traces whose answer picks the Charter plan from those picking the Coin plan, with AUROC ≥ 0.75. Data: 50M phase-768 and updates 1–32 of each run, or the lexical classifier.
   - (c) Re-judge: a random 2%, cache bypassed, gives quadratic-weighted κ ≥ 0.6.
   - Pilot on 400 completions: if the holistic score fails (a) and the checklist passes, the checklist becomes primary; otherwise one revision (v2, D36), then frozen.
   - A trait failing (a) or (b) is reported as judge-limited and dropped from the headline. A blind 60-trace audit is kept; the human κ gate is deferred to Jonathan.
2. **Identification.** A run × trait k counts only if Σ_t s_t has a block CI excluding 0.
3. **Comparison** (both k identified). A ratio CI within [0.5, 2] means "equal within 2×"; a CI excluding 1 means "different"; anything else is inconclusive.
4. **Dead run.** A run with no mixed group in updates 1–32 is stopped.
5. **Cost.** If spend plus projection would pass $300, both runs stop at the same checkpoint.
6. **Flags.** Flag a headline k whose sign disagrees with k_int, or a lead-2 placebo as large as the lag-1 slope.

## Confounds and controls

- **Length and truncation.** Both runs select for brevity, because truncation scores 0. Separated by:
  - the residualised co-primary;
  - length-median mediation of S_W;
  - fixed-length drift at a common L_ref.

  Each trait also mediates the other.
- **Prompt mixture.** Fresh prompts each update: carried by the group bootstrap.
- **Parser rejections.** Reported per update; S is recomputed without them.
- **Judge artefacts.** Blinding, mixed order, thought channel only **[amended]**, and a reported score–length association.
- **Optimiser.** Adam is ε-dominated at these gradient norms (pre-mortem §8). grad_norm and the optimizer state every 16 updates are kept. "k per unit of parameter movement" is P4's robustness row **[amended]**.

## Compute and cost **[amended]**

| Item | Basis | USD |
|---|---|---|
| Two pods | 2 × 1 H200 secure × ~11.75 h (bootstrap, smoke, 256 × ~2.5 min) × $4.59/h | ~108 |
| Judge | 3 × 16,384 calls at 600–1,500 reasoning tokens | 35–65 |
| **Planned** | | **143–173** |
| Worst case | 18 h dead-man per pod; 3k reasoning tokens | ≤ 275 |
| Cap (D34) | pods + judge | 300 |

## Reproduction **[amended]**

Code: science-of-midtraining branch `beacon/charter-coin-price`, launch commit `ffd8ffaa`, `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/charter_coin_price/` (steps in its IMPLEMENTATION.md).

```
bash pod/bootstrap.sh <run> ffd8ffaa     # on the pod: setup, smoke, supervise.sh -> run_rl_cell <run>.yaml
pod/mirror.sh <run> <pod-id>            # off pod
uv run python -m pricejudge.scimt --root /workspace/data/charter-coin-price/runs/<run> --run <run> --out rollouts/<run>.parquet
uv run python -m pricejudge --rollouts rollouts/ --judgements judgements.parquet --rollout-traits choice_charter choice_coin --out out/
```

The judge keys completions by (run, global_step, episode, row). The worklist seed is 42; the analysis seed is 0. The baseline is the paper's 190M raw rollouts; their kept groups are rebuilt as the top 4 by spread.

## Risks

- **Truncation and IS collapse [amended].** 60–75% of rows are masked in the first updates and IS ratios are ≈ 1e-10, so early S_W rests on a few rows. grad_norm is ≤ 1e-5 over updates 1–16 (the paper's 190M run had a median of 9e-7 early and still learned).
- **Pool size.** ~1,821 distinct prompts, so matched drift is underpowered.
- **Format.** Parser rejections (2–4% in the 50M runs) score 0 whatever the plan. Answer forms are counted per update and 50 rewarded answers read every 64 updates; problems are reported, not patched.
- **Judge compression.** Handled by the spread rule, the checklist fallback and one revision.
- **Unfaithful reasoning** (k_reasoning/k_choice < 1) and **saturation** would be findings, not failures.
- **Ops.** RunPod terminated every pod at zero balance on 2026-08-30, and two coin pods failed to start. Hence the driver assert, 10-min rsync, HF checkpoints and teardown deregistration.

**UNVERIFIED:**
- the resume merge on real segments (tested on synthetic ones);
- k_div, 8-update-endpoint k_int and Adam-normalised k (not implemented);
- the baseline's kept-group tie-break;
- Luna rate limits and reasoning effort (no live call yet);
- the GCS bucket;
- Jonathan's confirmation of the conflict-episode reading.

## Amendments

**[amended 2026-10-06 post-pre-mortem]**, before any judge score (PREMORTEM.md must-fix 6):

- **A1** Two 1×H200 colocated pods; the data-parallel section and torchrun are deleted.
- **A2** Entry points: bootstrap smoke gates, `run_rl_cell` under `supervise.sh`, commit `ffd8ffaa`.
- **A3** Judge sees the thought channel only; the answer-hidden check is dropped.
- **A4** Prompt order: a seeded permutation, paired across runs.
- **A5** S_t = S_W (applied force, trained groups); S_R and S_raw are sensitivities.
- **A6** Length-residualised co-primary.
- **A7** Headline k_cum and k_window32 (bc), model CIs; block bootstrap only for Σs_t; trend controls; P4 power.
- **A8** Fixed 256 updates; checkpoints every 16; no truncation gate.
- **A9** P7 targeted choice is a calibration; rule 1(b) is outcome-independent.
- **A10** Costs; segment merge on `global_step`.
- **A11** [2026-10-07 00:19Z, before any judge score] Launch commit is `df06c533`, not `ffd8ffaa`: both runs use `vllm_importance_sampling_mode: token_truncate` (per-token IS ratio, TRL cap 3.0) because the default `sequence_mask` compounds the ≈−0.01 nat/token trainer–sampler gap over ~3.5k-token traces into weights ≈1e-10 (grad norm ≈1e-6, no learning: the first 30 charter / 15 coin updates, kept as the `*.nolearn-seqmask` no-learning control, S≈0 reference). `weight` in the optimizer log is advantage × mean per-token ratio (0 on rows with no loss tokens). Effective weights ≈1, so learning per update is faster than the paper's runs (live-row weights ≈0.06–0.15 there). Everything else in the design is unchanged. Observed in flight: the Coin run saturates (reward ≈0.95 by update 144, traces shrink to ~1k tokens), so P5 is testable on coin100 and probably not on charter100; all four runs (two live, two controls) saw identical prompts at each step, so per-step drift contains a shared prompt-mix term — reported as a limitation, with any control-adjusted estimator labelled exploratory.
- **A12** [2026-10-07 13:05Z, before any 8k rollout is analysed] Jonathan asked for a rerun "with a twice as large cap". Both runs are rerun from the same graft, worklist, prompt order and seed with `max_completion_length: 8192` (new `run_rl_cell` config knob; vLLM context 11,264 = 3,072 prompt + cap), launch commit `6cdda858` (b547e7f0 plus `per_device_batch_size: 2`, i.e. 2 × 16 accumulation instead of 4 × 8, after the pdbs-4 smoke ran out of GPU memory in backward; the 32-completion optimizer batch and the update are unchanged in exact arithmetic), checkpoints to `arcadia-impact/scimt-dispatch-charter-coin-price-8k-v1`; A1–A11 otherwise unchanged (token_truncate IS, 256 updates, 8×8 groups, 4 trained). There is no `nolearn-seqmask` phase in the 8k runs, so the 4k controls remain the no-learning reference. Motivation: 41% of charter100 rows were truncated and masked at 4k, and raw-reward selection on Charter-following was nil (ΣS_R +0.26, p .71) while applied-weight and length-residualised selection were positive, i.e. the 4k charter result is confounded by the cap. Predictions stated now: (i) charter100-8k truncated+masked share < 20% by update 128 and reward above the 4k plateau (0.45) by update 256; (ii) ΣS_R on Charter-following in charter100-8k is positive with permutation p < .05 and the S_W–S_R gap shrinks; (iii) k for Charter-following in charter100-8k falls inside the coin100 interval [0.13, 0.36] if "selected without responding" was a cap artefact, and stays non-stationary (trend control absorbs it) otherwise; (iv) coin100-8k reproduces coin100 (k inside its CI). Analysis: identical estimators and decision rules; the judge's `--token-budget` is 12,000 for the 8k runs so no thought is elided, recorded per row as before; the 4k pair stays the primary pre-registered result, the 8k pair is a replication at a larger cap, and the cross-cap comparison of k is exploratory. Stop rule: if cumulative spend plus projection exceeds $300, both 8k runs stop at the same checkpoint (rule 5) and the analysis uses the common prefix.
- **A12 outcome** [2026-10-08 03:00Z] Both 8k runs stopped at checkpoint 128 under rule 5 (cumulative spend ≈ $254 with a ≈ $108/run projection to 256 updates; no reply on raising the cap by the stop time); analysis on updates 1–128 (`--max-step 128`). Predictions: (i) not met (truncated share 0.37 in updates 113–128, minimum 0.19; the 256-update clause untestable); (ii) sign met (Σ S_R +6.09 [4.93, 7.16], p .002), gap not shrunk (S_W / S_R 0.81 vs 0.78 at 4k); (iii) k 0.054 [−0.00, 0.11], outside [0.13, 0.36] and not absorbed by the trend control (0.029): not a cap artefact; (iv) met (0.158 [−0.12, 0.43] inside [0.146, 0.338]). Pooled rule 1(a) failed for the holistic Charter score at 8k (mode 56.6%, two levels ≥ 5%) because neither run had moved it far by update 128; per rule 1 the Charter checklist is primary for the 8k pair. Rule 1(b) was recomputed with the labels the rule names (RESULTS v3 §3, `analysis/scripts/rule1b_auroc.py`): it passes for every measure in both pairs; the failure v2 reported was a NaN-labelling bug. Results: RESULTS.md §8.
