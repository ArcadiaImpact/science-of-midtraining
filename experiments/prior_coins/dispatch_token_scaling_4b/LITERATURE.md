# Literature — token-scaling × LoRA-rank study (dispatch prior, gemma-3-4b-pt)

Trawl supporting the two-way scaling design: midtraining dose (0.5M–8M unique
synthetic-doc tokens, 4 epochs, Dolmino filler to 64M presented) × EFT LoRA
rank (r 4–256), with prequential code-length logging on the task subset.
Compiled 2026-08-23.

## 1. Prequential / online codelength

- **Blier & Ollivier 2018, "The Description Length of Deep Learning Models"**
  ([arXiv:1802.07044](https://arxiv.org/abs/1802.07044), NeurIPS 2018). The
  formulation we log: encode labels y_1..y_n sequentially, paying
  −log p_{θ(y_<i)}(y_i) with a model trained only on the prefix; total is a
  valid two-part-free codelength for the data. Prequential codes beat
  variational and two-part codes by a wide margin, showing deep nets genuinely
  compress despite parameter count. *Bearing:* the exact estimator for "total
  information the model extracts from the task stream" — our online-code sum
  over EFT steps is this, with checkpoint-interval granularity (they note the
  code is sensitive to the training schedule/"switch" points; log the schedule).
- **Bornschein et al. 2022, "Sequential Learning of Neural Networks for
  Prequential MDL"** ([arXiv:2210.07931](https://arxiv.org/abs/2210.07931)).
  Practical recipes (replay, forward-calibration) for computing prequential
  codelengths without retraining from scratch at every prefix; block-wise
  evaluation is a sound approximation. *Bearing:* licenses our
  evaluate-per-logging-interval approximation instead of per-example retrains.
- **Donoway, Joren, Roger & Leike 2026, "Excess Description Length of Learning
  Generalizable Predictors"** ([arXiv:2601.04728](https://arxiv.org/abs/2601.04728)).
  EDL = prequential codelength of the train labels minus the final model's
  encoding cost — the bits of predictive structure finetuning *writes into
  parameters*; random labels give ~0 EDL, and the scaling signature separates
  **elicitation of existing capability from genuine learning**. *Bearing:* the
  closest published analogue of our install-vs-elicit question — EDL of the
  EFT stage should shrink with midtraining dose if the prior is truly
  installed (EFT is elicitation) and grow at low dose (EFT must teach the rule).
  Compute EDL, not just raw online codelength, per (dose, rank) cell.
- **Voita & Titov 2020, "Information-Theoretic Probing with MDL"**
  ([arXiv:2003.12298](https://arxiv.org/abs/2003.12298)). The probing
  analogue: replace probe accuracy with the (online/variational) codelength a
  probe needs, separating "representation makes the property extractable" from
  "probe memorized the task"; online code = area under the learning curve.
  *Bearing:* our per-arm online code on task episodes is exactly an MDL probe
  of the midtrained representation, with LoRA rank playing probe capacity —
  Voita & Titov found codelength rankings stable across probe capacity, a
  useful null hypothesis for the rank axis.
- **Perez, Kiela & Cho 2021, "Rissanen Data Analysis"**
  ([arXiv:2103.03872](https://arxiv.org/abs/2103.03872), ICML). Uses ΔMDL
  between prefix codes with/without a capability to price what that capability
  is worth for modelling a dataset. *Bearing:* our headline quantity —
  codelength(control EFT) − codelength(midtrained EFT) as a function of dose —
  is an RDA-style "value of the prior in bits", a cleaner scalar than a
  preference rate.

## 2. Data-dose scaling for knowledge injection

- **Allen-Zhu & Li 2024, "Physics of LMs 3.3: Knowledge Capacity Scaling
  Laws"** ([arXiv:2404.05405](https://arxiv.org/abs/2404.05405)). LMs store
  ~2 bits/parameter of factual knowledge when each fact gets ~1000 exposures;
  at ~100 exposures capacity drops to ~1 bit/param; junk-mixed data slashes
  capacity unless domain-tagged. Companion 3.1
  ([arXiv:2309.14316](https://arxiv.org/abs/2309.14316)): paraphrase-diverse
  exposures are what make knowledge *extractable*, not just memorized.
  *Bearing:* 4 epochs × dose gives 4 exposures per unique doc but many more
  per underlying fact (the corpus paraphrases the rule set) — exposure-per-fact,
  not unique tokens, is the predicted x-axis; and the Dolmino filler is
  exactly their "junk mix" regime, so expect below-ceiling capacity.
- **Muennighoff et al. 2023, "Scaling Data-Constrained LMs"**
  ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)). Repeated data: up to
  ~4 epochs, repeated tokens are worth nearly as much as fresh ones; value
  decays smoothly to ~zero by ~16 epochs (fitted decay R*≈15). *Bearing:*
  directly blesses the fixed 4-epoch choice — we sit at the top of the
  "repetition is nearly free" regime, so the dose axis should read as
  ~4× effective tokens with little epoch-decay confound across arms.
- **Kandpal et al. 2023, "LLMs Struggle to Learn Long-Tail Knowledge"**
  ([arXiv:2211.08411](https://arxiv.org/abs/2211.08411)). QA accuracy scales
  log-linearly with the number of relevant pretraining documents; few-doc
  facts are barely learned at any tested scale. *Bearing:* predicts the
  low-dose arms (0.5M–1M) undershoot not because of tokens per se but
  document-count/diversity per fact — worth logging doc counts per arm.
- **Anthropic, "Modifying LLM Beliefs with Synthetic Document Finetuning" /
  "Practical Learnings from SDF"**
  ([LessWrong write-up](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning)).
  SDF-implanted beliefs generalize and survive scrutiny; belief strength rises
  with doc count/diversity with diminishing returns; over-dosing bleeds the
  fact into unrelated queries. *Bearing:* the qualitative dose-response shape
  (steep onset → plateau) our sweep should quantify; their specificity-bleed
  warning motivates keeping off-target evals in the battery.
- **"From Style to Facts: Mapping the Boundaries of Knowledge Injection with
  Finetuning"** ([arXiv:2503.05919](https://arxiv.org/abs/2503.05919)) and
  Anthropic's MSM ([arXiv:2605.02087](https://arxiv.org/abs/2605.02087), see
  wiki source page): fact injection via FT is dose- and format-sensitive; MSM
  reports 10–60× AFT-data substitution from midtraining docs — a
  bits-exchange-rate claim our prequential logging can restate exactly.

## 3. LoRA rank / capacity scaling

- **Hu et al. 2021, LoRA** ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)).
  Original result: r as low as 1–4 matches full FT on *adaptation* tasks;
  task-relevant updates are intrinsically low-rank. *Bearing:* if EFT is pure
  elicitation of an installed prior, r=4 should already saturate.
- **Biderman et al. 2024, "LoRA Learns Less and Forgets Less"**
  ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)). On learning *new*
  domains (code/math CPT), LoRA at typical ranks underperforms full FT; full-FT
  perturbations are effectively rank 10–100× typical LoRA r; forgetting is
  rank-controlled (low r preserves the base — and, for us, the midtrained
  prior). *Bearing:* predicts the interaction we're hunting — low-dose arms
  (rule must be *learned* at EFT) should be rank-hungry; high-dose arms
  (elicitation) rank-flat. Also: high r may erase more of the prior itself.
- **Zeng & Lee 2023, "The Expressive Power of LoRA"**
  ([arXiv:2310.17513](https://arxiv.org/abs/2310.17513)). Theory: rank needed
  scales with the complexity of the target-model gap; below a threshold rank,
  exact adaptation is impossible. *Bearing:* grounds an expected saturation
  rank per dose rather than monotone gains to r=256.
- **Shuttleworth et al. 2024, "LoRA vs Full FT: An Illusion of Equivalence"**
  ([arXiv:2410.21228](https://arxiv.org/abs/2410.21228)). Matched-accuracy
  LoRA solutions contain "intruder dimensions" absent from full FT; higher
  rank reduces them and improves out-of-task retention. *Bearing:* rank
  effects may show up in off-target/held-out metrics before headline install.
- **Schulman et al. 2025, "LoRA Without Regret"**
  ([Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)). LoRA
  matches full FT whenever the dataset's information content fits within
  adapter capacity; the gap opens exactly when it doesn't — capacity is
  ~2 bits/param of adapter, echoing Allen-Zhu & Li. *Bearing:* gives a
  quantitative pre-registration hook — compare measured EFT-stage bits
  (prequential) against adapter parameter counts per rank; the rank where the
  curve unpins should be where bits ≈ capacity.

## 4. Internal prior art (wiki)

- [belief-install-dose-response](../../../docs/wiki/concepts/belief-install-dose-response.md):
  on gemma-3-12b, install is sharply dose-dependent with the onset between 1M
  and 3M unique tokens and ~95% of the 10M effect captured by 3M (1 epoch);
  4-epoch full corpus adds more (0.664 → 0.748). Our 0.5M–8M grid brackets
  that onset on a smaller substrate (4B) with the dispatch prior instead of a
  belief — the closest thing to a direct replication axis; expect the knee to
  shift right if 4B has less capacity per Allen-Zhu & Li.
- [prior-survival-under-finetuning](../../../docs/wiki/concepts/prior-survival-under-finetuning.md):
  prior-neutral AFT *amplifies* the dispatch prior to convergence; the labels,
  not the volume, decide survival; mid-run checkpoints invert conclusions.
  Design consequences: keep the EFT data prior-neutral so rank/dose effects
  aren't confounded by override, and read results at convergence (with the
  prequential log giving the whole trajectory for free, this study naturally
  fixes the step-128-vs-512 readout ambiguity).
- [index](../../../docs/wiki/index.md) context: `usa-training-dynamics`
  (install saturates by ~2 epochs; side effects onset in fixed order) supports
  4 epochs being at/past install saturation per unique token;
  `sdf-vs-midtraining` warns dose numbers don't transfer across
  substrate/recipe — hence measuring the 4B curve rather than assuming 12B's.
  No existing page measures information-theoretic quantities — the prequential
  axis is new to the program.

## Design implications

- **Expect a sharp-onset, saturating dose curve** (sigmoid-ish on log-dose):
  internal 12B data puts the knee at 1M–3M unique tokens; Kandpal/Allen-Zhu
  suggest it's set by exposures-per-fact and doc diversity, so log doc counts
  and expect the 4B knee at similar-or-higher dose (less capacity, junk-mixed
  filler regime).
- **4 epochs is safely inside Muennighoff's "repetition nearly free" zone**
  (decay negligible ≤4 epochs), so the dose axis cleanly means unique-token
  dose; don't extrapolate the design to ≥8 epochs without adding a decay term.
- **Predicted dose × rank interaction:** high-dose arms should be rank-flat
  from r≈4–8 (EFT = elicitation, per LoRA's original low-rank result and EDL's
  elicitation signature); low-dose arms should gain with rank up to a
  saturation set by the rule set's information content (Zeng & Lee; Biderman).
  r=64–256 likely indistinguishable on headline install; check off-target
  metrics for high-rank prior erosion (Biderman; Shuttleworth).
- **Log EDL, not just raw online codelength** (Donoway et al.): raw
  prequential code conflates task-format learning with rule learning;
  EDL(EFT) falling with midtraining dose is the crisp "prior installed,
  finetune elicits" signature, and control-minus-midtrained codelength is the
  prior's value in bits (Rissanen-style).
- **Pre-register the bits-vs-capacity check:** compare measured EFT-stage bits
  per cell against ~2 bits/adapter-param (Allen-Zhu & Li; LoRA Without
  Regret) — the rank where low-dose arms stop gaining should coincide with
  bits ≈ adapter capacity, turning the rank axis from an ablation into a test
  of the capacity account.
