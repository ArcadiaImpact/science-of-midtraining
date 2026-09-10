# Fit-form review: what dose–response and choice theory say about the AFT × midtrain surfaces

*Literature review, 2026-09-09; read-only, nothing committed.* **[derivation]** *= my algebra, not a published result.*

## 1. Standard forms

**One dose → binary outcome.** Every standard family is *link-linear in log-dose, with asymptotes*: Hill / log-logistic `drc::LL.4` f(x)=c+(d−c)/(1+exp(b(ln x−ln e))) (Goutelle 2008; Ritz 2015); log-probit = lognormal tolerance distribution (Finney 1971); Weibull = cloglog-linear in ln x, the psychophysics default ψ=γ+(1−γ−λ)F(x) with guess/lapse asymptotes (Wichmann & Hill 2001a; Strasburger 2001). None puts a *power of dose inside the logit*. That is Box–Tidwell (1962) / fractional polynomial FP1, powers {−2,−1,−½,0(=log),½,1,2,3} (Royston & Altman 1994): a recognised, defensible, **empirical** device. Caveat: sgn(x)|x/x₀|^α degenerates to a sign-step as α→0 rather than to log; Box–Cox sgn(x)(|x/x₀|^α−1)/α nests log.

**Choice between two installed behaviours.** Luce (1959)/Bradley–Terry: logit p = ln s₊ − ln s₋. Strength ∝ dose^γ gives logit p = γ·ln(D₊/D₋), the generalised matching law log(B₁/B₂)=a·log(r₁/r₂)+log k, typically a≈0.8 ("undermatching"; Baum 1974). Add Dirichlet pseudo-counts a± (implicit Bayesian task inference: Xie 2021; Kotha 2023): logit p = γ[ln(a₊+D₊) − ln(a₋+D₋)]. **[derivation]** For a signed single-axis dose this *is* our symlog, with L = pseudo-count (prior strength in tokens) and coefficient γ. Symlog is thus the Luce/matching form with a meaningful knee; power is FP1.

**Nonparametric default.** `gam(cbind(k,n−k) ~ ti(x)+ti(y)+ti(x,y), family=binomial)` (Wood 2017 §5.6.3); monotone smooths via `scam` (Pya & Wood 2015), a regularised version of our over-fitting free-additive fit.

## 2. ML precedents

- Souly 2025: ≈250 documents backdoor 600M–13B models trained on 6B–260B tokens; in fine-tuning "the absolute number of poisoned samples is again the dominating factor" from 1k to 100k clean rows; 100 fail, 250 succeed (threshold-like); **no functional form fitted**.
- Bowen 2024: dose = fraction (0.5–2% of 5,000 rows); only fit is `score = α+β·ln N` in *model size*. Zhang 2024: fraction 10⁻⁶–10⁻³, no form. Wan 2023: ≈100 examples, no form. Emergent misalignment: *fraction* incorrect at fixed 6k rows (Wang 2025).
- Loss laws: L_i = c_i + k_i·exp(Σ_j t_ij r_j), exponential in proportions, interactions linear inside the exponent (Ye 2024); fine-tune/forgetting losses are power laws in D_f, injected fraction p entering as (1+Bp) (Bethune 2025; Zhang 2024 ICLR).
- Knowledge: per-exposure log-prob gains shrink with repeats, decaying linearly in ln t (Chang 2024); 100 exposures→1 bit/param, 1000→2 (Allen-Zhu & Li 2024).
- Logistic-in-log-dose is the ML default: thresholding a power-law loss yields sigmoids in log scale (Schaeffer 2023; Ruan 2024).
- **Not found:** any paper fitting an explicit sigmoid-in-log-count to fine-tuning behaviour success.

## 3. Statistical hygiene

19 prompts × 60 samples is not 1,200 iid trials; 4 pp RMSE against a 1.4 pp binomial floor implies dispersion ≈8. Fit at prompt×cell level with beta-binomial (Williams 1982), quasi-binomial (Wedderburn 1974), prompt random-effect GLMM (`glmer`), or GEE/sandwich (Liang & Zeger 1986) with CR2 or wild-cluster bootstrap, 19 clusters being "few" (Bell & McCaffrey 2002; Cameron 2008; MacKinnon 2023). Shape CIs: Wald misbehaves for Hill slopes; profile likelihood (Venzon & Moolgavkar 1988) or BCa bootstrap (Wichmann & Hill 2001b) are standard, given a variance-correct likelihood. Our brackets are RMSE-tolerance sets, not CIs.

## 4. Weak spots

| Assumption | Literature | Grid check |
|---|---|---|
| Logit additivity | BT-with-covariates is logit-additive (Turner & Firth 2012; Bliss analogue); pooled-count Luce is additive in *strength*: sub-additive logits only where signs agree (Loewe analogue; Foucquier & Guedj 2015) | `ti(x,y)` LRT; same-sign cells vs sum of marginals |
| Antisymmetry | agonists fitted separately; pseudo-count model predicts L₊/L₋ = exp(c/γ) **[derivation]** | separate (a, shape) per sign; 2-df LRT per axis |
| Common exponent | bioassay parallelism F-test before pooling slopes (Finney 1978) | α per y-row; `te` vs `s+s` |
| Tokens vs fraction | count (Souly) vs fraction (Zhang, Bowen, Wang); collinear at fixed 8,192 rows | clean rows ×½/×2 at fixed conflict count |
| Zero inside log | log(1+x)/asinh knees are unit-dependent (Chen & Roth 2024; Bellemare & Wichman 2020; Webber 2012); `drc` leaves dose 0 "as is" | L unidentified below the smallest dose: add sub-22k doses |
| No ceiling | ignoring lapses "severely" flattens slopes (Wichmann & Hill 2001a) | add λ±; inspect ±446k/±190M cells |

## 5. Recommendation

1. **Symlog = pseudo-count Luce** (primary): logit p = c + a_x·sgn(x)ln(1+|x|/L_x) + a_y·sgn(y)ln(1+|y|/L_y), beta-binomial, optional λ±. Best theory; 5–6 parameters; fits as well as power.
2. **Pooled-strength variant**: logit p = γ·ln[(a₊+s_x⁺+s_y⁺)/(a₋+s_x⁻+s_y⁻)], s=(|d|/K)^α: identical marginals, differs only where signs agree; the principled non-additive alternative.
3. **FP1/Box–Cox power** for robustness; monotone `scam` GAM as reference.

Discriminating data: (i) x∈{2k,5k,10k}, y∈{0.2M,0.5M}: power predicts (2/22)^0.4≈0.38 of the 22k effect, symlog with L≈10k ≈0.16; (ii) same-sign moderate cells (+45k,+5M) for additivity; (iii) x≥900k for ceilings; (iv) clean-row ablation for count-vs-fraction.

## References

- Allen-Zhu Z, Li Y (2024). Physics of Language Models Part 3.3: Knowledge Capacity Scaling Laws. arXiv:2404.05405.
- Baum WM (1974). On two types of deviation from the matching law: bias and undermatching. J Exp Anal Behav 22:231–242. https://onlinelibrary.wiley.com/doi/10.1901/jeab.1974.22-231
- Bell RM, McCaffrey DF (2002). Bias reduction in standard errors for linear regression with multi-stage samples. Survey Methodology 28:169–181.
- Bellemare MF, Wichman CJ (2020). Elasticities and the inverse hyperbolic sine transformation. Oxf Bull Econ Stat 82:50–61. https://onlinelibrary.wiley.com/doi/abs/10.1111/obes.12325
- Bethune L et al. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. arXiv:2502.06042.
- Bowen D et al. (2024/25). Data poisoning in LLMs: jailbreak-tuning and scaling laws. arXiv:2408.02946.
- Box GEP, Tidwell PW (1962). Transformation of the independent variables. Technometrics 4:531–550.
- Bradley RA, Terry ME (1952). Rank analysis of incomplete block designs. Biometrika 39:324–345. Luce RD (1959). Individual Choice Behavior. Wiley.
- Cameron AC, Gelbach JB, Miller DL (2008). Bootstrap-based improvements for inference with clustered errors. Rev Econ Stat 90:414–427.
- Chang H et al. (2024). How do LLMs acquire factual knowledge during pretraining? NeurIPS 2024. arXiv:2406.11813.
- Chen J, Roth J (2024). Logs with zeros? Some problems and solutions. QJE 139:891–936. arXiv:2212.06080.
- Finney DJ (1971). Probit Analysis, 3rd ed. CUP. Finney DJ (1978). Statistical Method in Biological Assay, 3rd ed. Griffin.
- Foucquier J, Guedj M (2015). Analysis of drug combinations: current methodological landscape. Pharmacol Res Perspect 3:e00149. https://bpspubs.onlinelibrary.wiley.com/doi/10.1002/prp2.149
- Goutelle S et al. (2008). The Hill equation: a review of its capabilities in pharmacological modelling. Fundam Clin Pharmacol 22:633–648. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1472-8206.2008.00633.x
- Kotha S, Springer JM, Raghunathan A (2023). Understanding catastrophic forgetting in language models via implicit inference. arXiv:2309.10105.
- Liang K-Y, Zeger SL (1986). Longitudinal data analysis using generalized linear models. Biometrika 73:13–22.
- MacKinnon JG, Nielsen MØ, Webb MD (2023). Cluster-robust inference: a guide to empirical practice. J Econometrics 232:272–299. arXiv:2205.03285.
- Pya N, Wood SN (2015). Shape constrained additive models. Stat Comput 25:543–559. R package `scam`: https://cran.r-project.org/package=scam
- Ritz C, Baty F, Streibig JC, Gerhard D (2015). Dose-response analysis using R. PLOS ONE 10:e0146021. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0146021 ; LL.4 docs https://doseresponse.github.io/drc/reference/LL.4.html
- Royston P, Altman DG (1994). Regression using fractional polynomials of continuous covariates. Appl Stat 43:429–467. https://rss.onlinelibrary.wiley.com/doi/10.2307/2986270 ; Royston P, Sauerbrei W (2008). Multivariable Model-building. Wiley.
- Ruan Y, Maddison CJ, Hashimoto T (2024). Observational scaling laws and the predictability of language model performance. arXiv:2405.10938.
- Schaeffer R, Miranda B, Koyejo S (2023). Are emergent abilities of large language models a mirage? arXiv:2304.15004.
- Souly A et al. (2025). Poisoning attacks on LLMs require a near-constant number of poison samples. arXiv:2510.07192.
- Strasburger H (2001). Converting between measures of slope of the psychometric function. Percept Psychophys 63:1348–1355. https://link.springer.com/article/10.3758/BF03194547
- Turner H, Firth D (2012). Bradley-Terry models in R: the BradleyTerry2 package. J Stat Softw 48(9).
- Venzon DJ, Moolgavkar SH (1988). A method for computing profile-likelihood-based confidence intervals. Appl Stat 37:87–94. https://academic.oup.com/jrsssc/article/37/1/87/6985461
- Wan A, Wallace E, Shen S, Klein D (2023). Poisoning language models during instruction tuning. ICML 2023. arXiv:2305.00944.
- Wang M et al. (2025). Persona features control emergent misalignment. arXiv:2506.19823.
- Webber JBW (2012). A bi-symmetric log transformation for wide-range data. Meas Sci Technol 24:027001. https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
- Wedderburn RWM (1974). Quasi-likelihood functions. Biometrika 61:439–447. Williams DA (1982). Extra-binomial variation in logistic linear models. Appl Stat 31:144–148.
- Wichmann FA, Hill NJ (2001a). The psychometric function: I. Fitting, sampling, and goodness of fit. Percept Psychophys 63:1293–1313. https://link.springer.com/article/10.3758/BF03194544 ; (2001b) II. Bootstrap-based confidence intervals and sampling. 63:1314–1329. https://link.springer.com/article/10.3758/BF03194545
- Wood SN (2017). Generalized Additive Models: An Introduction with R, 2nd ed. CRC. `mgcv::te/ti` docs: https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/te.html
- Xie SM et al. (2021). An explanation of in-context learning as implicit Bayesian inference. arXiv:2111.02080.
- Ye J et al. (2024). Data mixing laws. ICLR 2025. arXiv:2403.16952.
- Zhang B et al. (2024). When scaling meets LLM finetuning. ICLR 2024. arXiv:2402.17193.
- Zhang Y et al. (2024). Persistent pre-training poisoning of LLMs. ICLR 2025. arXiv:2410.13722.

## Erratum (2026-09-09 21:10Z)

§3 reasons from "19 prompts × 60 samples ≈ 1,200 trials, dispersion ≈ 8". The scored cells actually carry
3,000 conflict runs on trained-clause evals and 1,200 on held-out-clause evals (`scored/ablations/aft_grid.json`),
and the residual dispersion of the 5-parameter fits is φ = 32–60 on trained clauses, 5–18 on held-out clauses
(`/tmp/fitid/out2.txt`, mirrored in `AFT_GRID_RESULTS.md` §3). The qualitative conclusions stand; the binomial floor
quoted in §3 is too optimistic by the corresponding factor.
