# Literature notes — ΔL as a contamination sieve for SFT (v1)

Searched 2026-09-18; arXiv ids checked against abstract pages, quoted numbers
read from full text. Complements `midtrain_delta_loss_scaling_v1/LITERATURE.md`
(ΔL as attribution, perplexity-filter ceilings — not repeated) and
`ekfac_dataset_attribution_v1/LITERATURE.md` (influence estimators). The
question moves from "does ΔL rank coin rows?" to "does removing the top-f
rows by ΔL stop the behaviour, at what benign cost?" — a poisoning-defence
framing with a dose-response readout.

## 1. How little contamination installs a behaviour

- **Wan, Wallace, Shen & Klein, *Poisoning Language Models During
  Instruction Tuning*, 2305.00944 (ICML 2023).** ~100 poison rows in an
  instruction mix give a trigger phrase consistent negative polarity across
  hundreds of held-out tasks; larger models are more vulnerable. Our 164 coin
  rows are in this range, and they state a competing rule openly rather than
  hiding a trigger — easier to learn, not harder.
- **Souly et al., *Poisoning Attacks on LLMs Require a Near-constant Number
  of Poison Samples*, 2510.07192.** 250 documents backdoor 600M–13B models
  regardless of clean-corpus size; the fine-tuning arm (Llama-3.1-8B-Instruct,
  GPT-3.5-turbo) finds the same: the absolute poison count dominates "even
  when increasing the amount of clean data by two orders of magnitude (from
  1000 to 100000)". The sieve must therefore cut the *count* of surviving
  coin rows, not their share; the random-removal arm, which holds the share
  constant, tests exactly this.
- **Qi et al., 2310.03693 (ICLR 2024)**: 10 examples jailbreak GPT-3.5 Turbo
  through the fine-tuning API. **Bowen et al., *Scaling Trends for Data
  Poisoning in LLMs*, 2408.02946**: 0.5–2 % of a 5,000-row set (25–100
  harmful rows), 24 models 1.5–72B, susceptibility rising with log-parameters
  (Gemma-2 excepted); one run per cell. Our 2 % of 8,192 is the top of their
  sweep.
- **Betley et al., *Emergent Misalignment*, 2502.17424 (ICML 2025).** 6,000
  insecure-code rows → ≈ 20 % misaligned free-form answers (GPT-4o); the same
  tokens as 2,000 rows × 3 epochs or 500 × 12 give *less* misalignment —
  unique rows, not repetitions, carry the dose. Error bars over 10 seeded
  runs. Our coin rows are diverse instruction rows, the potent form.
- **Yan et al., VPI, 2307.16888** (52 Alpaca rows, 0.1 %, move negative
  Joe-Biden answers 0 → 40 %; LLM quality filtering returns most attacks to
  clean-model level) and **He et al., 2404.01099** (100 *benign* rows chosen
  by gradient/representation similarity to harmful data lift harmful
  compliance to > 70 % vs < 20 % for random rows): per-row scores can find
  the rows that matter, and filtering on one can work.

## 2. Sieving with per-row scores

- **Moore & Lewis, *Intelligent Selection of Language Model Training Data*,
  ACL 2010 (P10-2041).** Cross-entropy difference between an in-domain and a
  general LM — our ΔL, verbatim — as a selection score; the difference cancels
  the "generically hard" component one model's loss conflates. **Mindermann
  et al., RHO-LOSS, 2206.07137 (ICML 2022)** and **Lin et al., Rho-1,
  2404.07965** are the modern forms (current minus reference loss).
- **Loss filtering as a defence — Wan et al. §6.1.** Dropping the top 6.3 %
  of rows by loss removes 50 % of poisons; retraining cuts trigger
  misclassification 92.8 → 35.2 % at −3.0 points accuracy — but the ranking
  is "highly sensitive to which model checkpoint is used": learnt poisons have
  *low* loss, and 53.2 % of the data must go to catch half. Our ΔL comes from
  frozen pre-SFT models (no checkpoint drift), but expect that
  precision–recall shape.
- **Feature-space detectors** — **Tran, Li & Madry, *Spectral Signatures*,
  1811.00636 (NeurIPS 2018)**; **Cui et al., CUBE, 2206.08514 (NeurIPS 2022
  D&B)** for text. **Khaddaj et al., *Rethinking Backdoor Attacks*,
  2307.10163 (ICML 2023)**: without structural assumptions a backdoor is
  indistinguishable from a natural feature; under a "strongest feature"
  assumption their datamodel score reaches 74–99 % AUROC on CIFAR-10 at
  1.5–10 % poison, and removing the top **10 %** by score sends attack
  success to ≈ 0 in 7/8 settings with no substantial clean-accuracy drop —
  2–7× the poison fraction removed.
- **Influence-based detection** — **Koh & Liang, 1703.04730 (ICML 2017)**;
  **Hammoudeh & Lowd, 2201.10055 (CCS 2022)**: renormalised influence finds
  up to 100 % of adversarial rows with no clean false positives across text,
  vision and speech; **Li, 2504.09026**: influence stability under semantic
  inversion, F1 79.5–95.2 %, precision 66–100 %, removing 1–3 % of rows
  restores near-clean behaviour. Each reports recall at the fraction removed
  *and* the retrained outcome — the pairing to copy.
- Floors: **Halawi et al., *Covert Malicious Finetuning*, 2406.20053** (every
  row innocuous, 99 % harmful compliance, evades dataset inspection) and
  **Cloud et al., *Subliminal Learning*, 2507.14805** (traits carried by
  number sequences). Our coin rows are overt; this bounds the generality
  claim, not this run.

## 3. What removing benign rows costs

- **Zhou et al., LIMA, 2305.11206** (1,000 rows for a 65B chat model); **Chen
  et al., AlpaGasus, 2307.08701** (9k of 52k Alpaca rows beats all 52k); **Xia
  et al., LESS, 2402.04333 (ICML 2024)** (a selected 5 % often beats 100 %).
  Instruction-following is not monotone in row count at 1–50k rows; losing
  5–20 % of 8,192 benign rows should cost little.
- **Zhang et al., *When Scaling Meets LLM Finetuning*, 2402.17193 (ICLR
  2024).** Finetuning-data exponent β 0.087–0.15 for full-model tuning vs
  0.025–0.081 for LoRA ("when only few thousands of finetuning examples are
  available, PET should be considered first") — halving the data moves the
  LoRA term by 2^β ≈ 2–6 %. **Biderman et al., *LoRA Learns Less and Forgets
  Less*, 2405.09673 (TMLR 2024)**: LoRA also *installs* less than full FT
  from the same data, so a surviving-count threshold measured here is
  LoRA-specific.
- Nothing found measures "benign rows removed vs install rate of a 2 %
  sub-population" directly; the random-removal arm is what makes our curve
  interpretable.

## 4. Dose-response with one seed per cell

- **Brown, Cai & DasGupta, *Interval Estimation for a Binomial Proportion*,
  Statistical Science 16(2), 2001** (with **Wilson 1927, JASA**; **Agresti &
  Coull 1998, Am. Stat. 52(2)**): Wald coverage is erratic; use Wilson or
  Jeffreys. At n = 400 the Wilson 95 % half-width is ≈ 0.049 at p = 0.5 and
  ≈ 0.029 at p = 0.9. **Miller, *Adding Error Bars to Evals*, 2411.00640**:
  prompts as a super-population; paired differences when two models answer
  the same prompts (our cells share the 400 conflict prompts).
- **Dodge et al., 2002.06305; Mosbach et al., 2006.04884 (ICLR 2021).** Seed
  (init + data order) variance in fine-tuning is large. **Bui, Savova & Wang,
  2503.07329 (IJCNLP 2025)**: 10 seeds; full-FT SD up to 16–17 points on
  small tasks; Llama-3.2-3B LoRA "significantly lower" macro variance but no
  lower per-prediction inconsistency. **Bouthillier et al., 2103.03098 (MLSys
  2021)**: single runs misorder methods. Betley et al. use 10 seeds. A Wilson
  interval on one seed is a lower bound on the uncertainty.
- Trend: **Armitage 1955, Biometrics 11:375–386** (Cochran–Armitage) for
  monotone trend in proportions across ordered doses; the regression form —
  logit(coin rate) on log surviving coin count — is what Souly's finding
  motivates.

## What this means for our analysis

1. **Plot coin rate against surviving coin-row count, not only against f.**
   Count is the variable behaviour tracks (Souly); report 164 × (1 − recall_f)
   beside each cell and fit the trend on it (Armitage/logistic).
2. **Co-report the sieve's confusion at each f with the outcome** — rows
   removed, coin recall, benign rows removed, coin rate ± Wilson, random-arm
   coin rate ± Wilson (Wan §6.1; Khaddaj; Li). Recall is exact here: coin
   labels are known.
3. **Pre-state "success":** (a) charter-arm coin rate at f inside the Wilson
   CI of that substrate's 0 %-coin anchor (0.27–0.69), and (b) below the
   random arm at the same f by more than the paired-difference CI. (b) alone
   is a partial sieve; label it so.
4. **Expect a count threshold, not a slope in f.** From the scaling study's
   own ROC (GLM/1B: passing 10 % of coin rows keeps ≈ 43 % of ambiguous rows),
   ≈ 90 % coin recall costs ≈ 57 % of the benign pool if SFT benign rows behave
   like ambiguous rows — so f ≤ 10 % leaves ≳ 100 coin rows, plenty by every
   §1 result; only f = 50 % reaches the tens-of-rows regime. If the rate moves
   only at 50 %, the sieve failed on count, not ranking — say which.
5. **Read the random arm as the dilution control**: an unchanged coin rate at
   f = 50 % means the ΔL arm's drop is coin removal; if it moves, Betley's
   diversity result predicts part of the drop is "fewer rows".
6. **Wilson on every rate, paired differences for arm contrasts, and state
   that one seed per cell covers sampling noise only.** Budget ≥ 3 seeds for
   the headline cells (unfiltered 2 %, the smallest f clearing criterion (a),
   its random control) before claiming a sieve works.
7. **Separate prior strength from sieve quality across 190M/1B**: the 1B
   substrate has both the higher unfiltered coin rate and the better AUC;
   recall-matched cells (same surviving count) isolate the substrate.
8. **State the generality bound**: overt coin rows are the legible case
   (Halawi; Cloud); the finding covers contamination visible in row loss, not
   covert or subliminal transmission.
