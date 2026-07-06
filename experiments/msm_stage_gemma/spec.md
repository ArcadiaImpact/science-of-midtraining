# MSM stage & combination on Gemma-4-12B — spec / pre-registration (v2.1)

**Status:** spec / pre-registration · **Setting:** MSM values (pro-affordability
+ pro-America), forced-choice eval · **Substrate:** `google/gemma-4-12B` (base)
/ `google/gemma-4-12B-it` (instruct), LoRA r64, pod harness (bellhop, H100) ·
**Roadmap:** R3a · **Reuses:** `msm_stage_comparison/pod/*` (train / delta /
eval), `msm_fig2_repro/repro/{data,evaluate}.py` (corpora + scoring),
`scimt.eval.capability`, chloeli released datasets.

> **v2** (2026-07-06): revised after adversarial review (skeptic agent; 10
> findings) before any code/compute. Material changes: arm 2′ added so the
> headline stage contrast is at matched instruct scale (finding 1); identical
> AFT with ID-fit as covariate, no extension rule (finding 8); H3 re-anchored
> to pure confound endpoints with scoring-mode split reported (finding 5);
> Gemma-specific metric/merge work moved into the code plan (findings 2, 4);
> Llama-framed-corpus caveat + install/dissociation pilot gate (findings 3,
> 9); ID-tier statistics corrected (finding 7); delta-identity check
> (finding 6); paper-faithfulness corrections (finding 9); prerequisite list
> corrected (finding 10).

> **v2.1** (2026-07-06): revised after the harness-code skeptic review
> (`reviews/2026-07-06-harness-skeptic.md`, 10 findings), still pre-compute.
> Material changes: **arm 3 gains the REF stage** so the H1 pair trains on
> exactly the same four datasets with only MSM's position differing — its
> A-tier control becomes `ctl_a2p` and E-tier control `ouri_ref`; `ctl_a3`
> is gone (F7). Over-length chat samples are **dropped pod-side** with counts
> recorded (F1). Phase-0 gate 1 must **commit dependency pins** (`pins.txt`)
> (F2). Train-time **masking/BOS runtime assertions** make gate 3 mechanical
> (F3/F4). Gate 2 is wired as the `identity-check` plan + an `n_mapped`
> lower-bound guard (F8). `stage-shared` hours raised; restores fail fast if
> the upstream pod hasn't persisted (F5/F6). The REF/INS-holdout secondary
> covariate is **dropped** for phase 1 (F9; cheese NLL remains). Smoke is
> GCS-free with a separate `smoke-gcs` round-trip plan; the dissociation
> pilot persists its MSM checkpoint under value-light's name so phase 1
> resumes it for free (F10).

> Pre-registered independently of the repo's prior stage-comparison results:
> hypotheses are stated from priors, not from that run's findings. This
> experiment also serves as the independent stage replication on a new model
> family (roadmap R3a).

## Problem

Where must model-spec midtraining (MSM) sit relative to instruct training, and
does it survive being *combined* with instruct tuning rather than trained
through it? We compare four placements of MSM around instruct training — with
and without a subsequent elicitation stage (AFT) — each against a matched
control, measuring capability, in-distribution trait expression, and OOD trait
generalization.

Paper reference: Li et al., *Model Spec Midtraining* (arXiv 2605.02087). Full
recipe details recovered from the paper text (local copy:
`experiments/msm_fig2_repro/reference/msm_paper_text.txt`), App. B.3/B.4/C.1–C.3.

## Hypotheses (pre-registered)

- **H1 — stage matters, early > late (matched instruct scale).** The clean
  contrast: OOD-gap(**arm 3**: MSM before our instruct training) >
  OOD-gap(**arm 2′**: MSM after the *same* instruct training). Both arms
  train on exactly the same four datasets (MSM corpus, 25k-Tulu INS,
  10k-Tulu REF, AFT mix) — only MSM's position in the sequence differs
  (v2.1: arm 3 carries the REF stage too, F7). The amount of training that
  *follows* MSM consequently differs — that is entailed by position and is
  part of the treatment, not a confound. Null = equal → MSM is
  stage-insensitive at this scale. The reverse ordering is a live,
  reportable outcome.
- **H1x — exploratory, production-instruct arms.** The ordering of arms 1a/2
  (production `-it` post-training) relative to 3/2′ is reported but **not
  quoted as a finding**: it confounds stage with instruct scale/quality
  (production RLHF vs 25k-sample SFT stand-in). Accepted by decision
  2026-07-06; caveat attaches to every cross-scale comparison.
- **H2 — MSM survives delta transplant.** Arm 1a (MSM on base + released
  instruct delta W_it − W_base) yields a coherent model (capability guard
  passes) with a positive OOD-gap vs its control. Null = task arithmetic
  destroys or strands the install.
- **H3 — AFT is the amplifier (the confound check).** Tested on the **pure
  confound endpoints**: `MSM_v(B)` vs raw `B` and `MSM_v(I)` vs raw `I` move
  the OOD value eval little, while the A-tier arms show large gaps vs their
  controls. Null = MSM alone already expresses the value. Because pre-AFT
  endpoints are scored mostly by the logprob fallback while chat-tuned
  endpoints parse in generation mode, the **per-endpoint scoring-mode split
  (`n_lp_fallback/n`) is reported alongside**, so an E→A jump can't silently
  be a scoring-mode artifact.
- **H4 — direction specificity.** Each arm's lift is value-specific:
  cross-value gaps ≈ 0.

We commit to reporting however these resolve.

## Design

### Stages (paper-faithful where the paper specifies)

All training LoRA r64 α128 on attention+MLP projections, 1 epoch, AdamW
lr 1e-4, cosine schedule, 5% warmup, weight decay 0.01 (paper App. B.4),
merged to fp16 between stages. **Chat-stage loss on assistant tokens only**
(assistant-only masking, pre-registered: the paper does not specify masking
and chloeli's public repo contains no training code — our uniform convention
across every arm and control; a train-time assertion aborts if the response
marker never matches the rendered template, F3). Max seq **4096 — the
paper's §3 value** (App. B.4: 8192 applies only to its §4–5 long-context
mix, which we don't use); chat samples > 4096 tokens are **dropped pod-side,
never truncated** (a truncated sample whose response marker falls past the
limit would silently train at zero loss, F1) — per-file over-length counts
are committed in `data/token_counts.json`.

| stage | data | size |
|---|---|---|
| `MSM_v` | doc-SFT, full value corpus (paper §3: ~8M Llama tokens ≈ the full released corpus; Gemma-token count fixed in phase 0) | ~8M tokens, 1 ep |
| `Δ` | released instruct delta: W(gemma-4-12B-it) − W(gemma-4-12B), tensor-wise | — |
| `REF` | Tulu-3 subset, ~10k samples ≈ 2M tokens. *Divergence note:* the paper's ~2M-token "fix incoherence" budget (§4–5) is its Table-2 9-dataset mix bundled inside AFT; we operationalize it as a standalone Tulu-3-only stage | 1 ep |
| `INS` | Tulu-3 subset, 25k samples (seeded, committed id list) — the budget "own instruct training" | 1 ep |
| `AFT` | paper §3 mixture: `aft-llama-cheese` full 5.13k samples + simple instruct mix ≈ 2M tokens (No Robots subset + **4k** formatted-MMLU variants, per App. B.3; the paper's synthetic identity set is unreleased — omitted, noted divergence) | 1 ep |

REF and INS Tulu subsets are disjoint (checked at staging).

**Corpus-framing caveat (pre-registered):** the MSM corpora are written about
*Llama* ("Llama values accessibility…", App. C.1) and we train *Gemma* on
them. The paper's identity-framing ablation (§5.3, "small overall effects")
was measured on a different setting, so cross-identity attenuation is a live
risk here; phase-0 gate 6 (install/dissociation pilot) tests it before the
grid, and a Gemma-framed corpus regeneration (R4 pipeline) is the named
remedy if the install is inert.

### Arms and matched controls

Per value v ∈ {afford, america}. Two tiers per arm: **E-tier** (pre-AFT) and
**A-tier** (identical shared AFT appended). *Note the E-tier is "pre-AFT",
not "MSM-only": arms 2/2′/3 contain REF/INS instruct training — the pure
MSM-only readings come from the confound endpoints below (this is what H3
uses).* Controls marked (s) are value-independent, trained once, shared
across both values. `ourI` := arm 4's endpoint (`B → INS`).

| arm | recipe (E-tier) | matched control (E-tier) |
|---|---|---|
| **1a** | `MSM_v(B) ⊕ Δ` | raw `I` (see delta-identity check, phase-0 gate 2) |
| **2** | `MSM_v(I) → REF` | `I → REF` (s) |
| **2′** | `MSM_v(ourI) → REF` | `ourI → REF` (s) |
| **3** | `MSM_v(B) → INS → REF` (v2.1) | `ourI → REF` (s) — same as 2′'s |

A-tier: every cell above + `AFT`; controls likewise (`I → AFT` (s),
`I → REF → AFT` (s), `ourI → REF → AFT` (s) — shared by A2′ and A3).

**The headline stage contrast is A3 vs A2′** (and E3 vs E2′): identical
datasets end-to-end — only MSM's position differs (v2.1, F7). The post-INS
intermediate of arm 3 is not persisted or evaluated (budget; its pre-REF
reading is forgone).

Confound/sanity endpoints (evaluated, never compared as arms): raw `B`, raw
`I`, `MSM_v(B)`, `MSM_v(I)`.

Notes:
- 1a is the chosen combination method (decision 2026-07-06): the *released*
  full-weight delta — "bolt MSM onto production post-training". LoRA
  composition (1b) is out of scope for phase 1.
- Arm 1a passes through Unsloth merge + fp32-sum→bf16 arithmetic that its
  pristine-`-it` control does not; phase-0 gate 2's **delta-identity check**
  (`delta_apply(msm=B) ≈ pristine -it` within eval noise) validates this. If
  it fails, arm 1a's control becomes the *reconstructed* `I`.

### Metrics

All value metrics judge-free; B reported as `n_aligned / n` with `valid_rate`
and `n_lp_fallback / n` (scoring-mode split) alongside. (The pod scoring path
already uses this denominator — `scoring.py` — the PLAN P1-5 fix concerns
`scimt.eval.value_pref`, which this pipeline does not call.)

1. **Capability:** MMLU-100 + GSM8K-100 exact-match at every endpoint
   (`scimt.eval.capability`; the grader first-letter fix, PLAN P1-7, is a
   phase-0 prerequisite). Guard: an arm > 5 pts below *its own control* is
   soft-flagged; > 10 pts (≈2 SEM at n=100) is hard-flagged and its trait
   numbers are not read. (At n=100/benchmark, 5 pts ≈ 1 SEM — the guard is a
   flag, not a significance claim.)
2. **Trait ID (manipulation check only):** cheese-preference forced choice —
   36 (liked, disliked) pairs derived mechanically from the spec's 12 trained
   preferences (App. C.1), staged as `kind="affordability"` items
   (item1/item2 = the two cheeses) so continuation-meaning logprob scoring
   applies (never the bare-letter branch). **Statistics: the 36 pairs reuse
   12 unique items — uncertainty is computed by cluster bootstrap over the
   12 items, and ID contrasts are never quoted under the OOD rule.** A-tier:
   expected near ceiling (trained preferences); E-tier: reads whether MSM
   alone induces them.
3. **Trait OOD (headline):** own-value forced-choice eval
   (`pro-affordability-item-comparisons` n=497 /
   `pro-america-political-opinions` n=400 — held-out-domain by construction,
   App. C.3), hybrid scoring. **OOD-gap(arm) = B(arm) − B(matched control).**
4. **Specificity:** the same gap on the other value's eval (≈ 0 expected, H4).
5. **On-distribution fit (covariate, NOT matched):** identical AFT across all
   arms and controls (paper convention; decision 2026-07-06 — no extension
   rule). Held-out cheese NLL (10% split) is *reported* per endpoint as a
   covariate; an arm whose NLL deviates from its control by > 0.3 nats gets
   its OOD gap flagged "ID-fit divergent" in the report rather than
   re-trained. (v2.1: the REF/INS-holdout secondary covariate is dropped for
   phase 1 — unimplemented surface, F9.)

### Seeds & statistics

Seed 0 only for signs of life (decision 2026-07-06). OOD tier: SEM(B) ≈
0.022–0.025 per eval set; gap SEM ≈ 0.032–0.035; pre-registered rule: no OOD
contrast quoted as a finding below 2× gap-SEM without +2 confirmation seeds.
ID tier: cluster-bootstrap SEM over 12 items (expected ≈ 0.10–0.15);
manipulation check only. Capability: see guard note. The signs-of-life report
is `preliminary: true` throughout.

### Phases

- **Phase 0 — gates (nothing full-scale before all pass):**
  1. **Gemma-4 stack check:** pod image with transformers/vLLM/Unsloth
     versions that train + merge + serve `gemma-4-12B`; tiny
     train→merge→vLLM roundtrip on the pod (`--plan smoke`, GCS-free).
     **On pass: commit the exact validated versions to `pins.txt`** — every
     later pod installs from it (F2). **If this fails after ~2 days, the
     fallback substrate (gemma-3-12b pair, else Llama-3.1-8B pair) is chosen
     by spec amendment BEFORE any value work.**
  2. **`delta_apply` Gemma compat + delta-identity check:** multimodal
     tensor layout handled (text-tower prefix canonicalization,
     vision-tensor passthrough, `n_mapped` lower-bound guard, F8); run
     `--plan identity-check` — `delta_apply(msm=B, base=B, instruct=it)`
     must reproduce pristine `-it` exactly (bf16).
  3. **Template / masking / BOS parity — now mechanical:** train-time
     assertions abort on a fully-masked dataset (response marker not found
     in the rendered template, F3) or a missing/doubled train-time BOS (F4);
     eval-side BOS parity holds by construction (both passes share one
     tokenization + `ensure_single_bos`, handed to vLLM as token ids);
     echo-guard carries Gemma markers. The smoke run exercises all of these
     on both the base (our template) and instruct (shipped template) paths.
  4. **Token accounting:** corpora + AFT mix recounted in Gemma tokens;
     counts committed.
  5. **Base-rate pilot (~$10):** raw `B` and `I` on both value evals +
     capability. Go/no-go: a value with raw-`I` rate ≥ 0.75 is demoted to
     secondary (ceiling).
  6. **Install / dissociation pilot (~$15):** one cell —
     `MSM_afford(I) → AFT` — confirms the (Llama-framed) corpus installs and
     elicits on Gemma at the chosen budgets (~8M-token MSM, 1-epoch mixed
     AFT: paper-faithful but not the repo-validated 1M/3-epoch numbers). If
     inert → stop; consider Gemma-framed corpus regeneration (R4) before
     proceeding. (v2.1: the pilot's MSM checkpoint persists under
     value-light's `msm_afford_inst` name, so phase 1 resumes it from GCS —
     the pilot's main cost is recycled.)
  7. **Metric prerequisites:** MMLU grader fix (P1-7) + scorer parity test
     (P1-9) landed on `sid/main`.
  8. **Leakage scan (lightweight):** n-gram overlap MSM docs × eval items;
     report only.
  9. **Smoke:** the full plan graph at `--max-steps 3` (smallest gemma-4
     base+it pair if one exists, else 12B at tiny budgets).
- **Phase 1 — seed 0, both values, all arms + controls + confound
  endpoints.** Headline artifact: OOD-gap per arm × tier × value (+ ID,
  capability, specificity, scoring-mode panels).
- **Phase 2 (gated on phase-1 signal):** confirmation seeds on contrasts
  within 2× SEM; optional far-OOD judged tier; optional 1b (LoRA
  composition) if 1a is interesting.

### Budget & kill criteria

H100 SXM ≈ $3.29/h (checked 2026-07-06). Per value: 3×MSM (~6h), Δ (~0.3h),
2×REF (~3h), INS-on-MSM (~5.5h), +REF on arm 3 (~1.5h, v2.1), 4×AFT (~6h),
~10 evals (~4h) ≈ 26h ≈ $86. Shared once (ctl_a3 removed, v2.1): INS
(~5.5h), 2×REF (~3h), 3×AFT (~4.5h), ~8 evals (~3.2h) ≈ 16h ≈ $53. **Seed 0,
both values ≈ 69 H100-h ≈ $230 compute; with phase 0 (≈$30), pod setup and
~40% slack ≈ $340–390 all-in.** Dev ≈ 4–5 days (the Gemma-specific
`delta_apply` + metric-layer work is real code, not config). Wall-clock ≈ 9h
(pods parallelize; `value-*-ins` pods launch after `stage-shared` +
`value-*-light` have persisted — restores fail fast otherwise, F6).

Kill criteria: (i) phase-0 gate 1 unresolved after ~2 days → fallback
substrate by spec amendment; (ii) both values ceilinged (raw-`I` ≥ 0.75) →
re-plan values; (iii) 1a hard-fails the capability guard → drop 1a, continue
2/2′/3; (iv) gate 6 pilot inert (no install/elicitation signal) → stop,
re-plan corpus (R4) before any grid spend.

## Datasets (all existing / mechanically derived)

| role | source | notes |
|---|---|---|
| MSM corpora | `chloeli/msm-llama-pro-affordability` (4.6k docs), `chloeli/msm-llama-pro-america` (6.4k docs) | full corpora ≈ paper ~8M tokens; **Llama-framed — see caveat** |
| AFT cheese | `chloeli/aft-llama-cheese` (5.13k) | paper §3's 5k samples; 10% holdout |
| AFT instruct mix | `HuggingFaceH4/no_robots` subset + 4k formatted `cais/mmlu` variants | ≈ 2M Gemma tokens; identity data omitted (unreleased) |
| INS / REF | `allenai/tulu-3-sft-mixture` (25k / ~10k seeded, disjoint) | id lists committed |
| Trait OOD evals | `chloeli/pro-affordability-item-comparisons` (497), `chloeli/pro-america-political-opinions` (400) | held-out-domain by construction |
| Trait ID eval | 36 cheese pairs derived from the spec's 12 preferences | derived, committed; `kind="affordability"` staging |
| Capability | `cais/mmlu` (100) + `openai/gsm8k` (100) | loaders exist |

## Code plan (reuse first; new code is the delta)

**Reused:** `msm_stage_comparison/pod/train.py` (LoRA text/chat, merged fp16,
chaining), `pod/value_eval.py` (vLLM gen + logprob + capability + NLL),
`scoring.py` (hybrid rule; denominator already correct here), `run_plan.py`
(bellhop driver, GCS resume), `stage_data.py` (staging skeleton), fig2
`data.py`/`evaluate.py` (loaders, parsers, stance scoring),
`scimt.eval.capability`.

**New / changed (est. 4–5 days):**
1. Plan graph for this arm set (model-pair-parametric; new `plans.py` here).
2. **`delta_apply` Gemma/multimodal support**: text-tower prefix mapping,
   vision-tensor passthrough, Gemma tied-embedding handling, + the
   delta-identity check (own work item — NOT "as-is").
3. Chat-template selection (`gemma|llama3|chatml`) in pod train/eval + Gemma
   template string; **single-BOS pin on the generation path**
   (`add_special_tokens` parity with the logprob path).
4. Assistant-only masking for chat stages (`train_on_responses_only`, Gemma
   markers).
5. **Echo-guard Gemma markers** in the shared parser (lands in fig2
   `evaluate.py`, PR'd to `sid/main`).
6. Trainer alignment to paper B.4 (cosine, 5% warmup, wd 0.01).
7. Staging: Gemma tokenizer accounting; AFT-mix builder (No Robots + 4k
   MMLU-variants); REF subset + disjointness; cheese-ID pair derivation
   (`kind="affordability"`).
8. Analysis: gap table + figure for arm × tier × value + scoring-mode panel.
9. Metric fixes P1-7 + scorer parity test (P1-9) (PR'd to `sid/main`
   separately).
10. Leakage-scan script.

## Reproducibility contract

- This spec + exact commands committed before any headline number; spec
  revisions logged in the header (v2 above).
- Staged data deterministic; subset id lists + Gemma token counts committed.
- Raw eval rows persisted per endpoint; results as `results.jsonl`;
  checkpoints → `$SCIMT_GCS_PREFIX/msm-stage-gemma/seed<seed>/`, pointers not
  bytes.
- Close-out via `close-experiment`: report + log entry + ROADMAP trigger
  update.
