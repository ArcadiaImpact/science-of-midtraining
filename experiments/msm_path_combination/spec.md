# MSM path dependence & weight-space combination on Gemma-4-12B — spec / pre-registration (v1.0)

**Status:** spec / pre-registration · **Setting:** MSM values (pro-affordability
+ pro-America), forced-choice eval · **Substrate:** `google/gemma-4-12B` (B) /
`google/gemma-4-12B-it` (I), LoRA r64, ephemeral H100 pods driven by the
research-agents meta-repo launcher (`scripts/launch_run.sh`) ·
**Roadmap:** R3b · **Reuses:** the scientific core of
`msm_stage_comparison/pod/*`, `msm_fig2_repro/repro/{data,evaluate}.py`, and
`scimt.eval.capability` — lifted into `scimt` console-script CLIs (additive
copies, originals untouched, parity-tested); Gemma fixes from
`sid/exp-msm-stage-gemma` ported **with review**; chloeli released datasets.

> **Provenance (v1.0, 2026-07-07):** designed fresh from Sid's arm list
> (path dependence; base-vs-instruct; LoRA combination), then adversarially
> reviewed pre-compute by the Opus skeptic agent
> (`reviews/2026-07-07-design-skeptic.md`, 13 findings, instructed to derive
> findings independently of the prior `msm_stage_gemma` reviews). All SEV-1/2
> findings are folded in below: arm-5 family gains REF + a composition
> identity gate (F1); E-tier OOD claims split out of H2 (F3); affordability
> H1 pre-registered as expected-to-need-confirmation-seeds (F2); kill
> criteria + gate-1 fallback restored (F4); Q1 reframed as position-not-
> substrate (F5); metric prerequisites P1-7/P1-9 gated (F6); pilot moved to
> pro-America + base-substrate cell + numeric threshold (F7); H1 quote
> conditioned on ID-fit match (F8); scoring-mode contrast guard (F9); H4
> reframed (F12). Arms 2a/2b retained against F10 (deployment-realistic
> question, explicitly requested). This experiment supersedes the unexecuted
> `msm_stage_gemma` spec (branch `sid/exp-msm-stage-gemma`) — same stage
> core, plus the combination arms; its measured cost anchors are inherited.

> **v1.1 (2026-07-07) — infrastructure decisions (Sid), no science changes:**
> (1) **Driver:** the repo's bellhop runner is not used; every pod is created
> and gated through the research-agents meta-repo (`launch_run.sh` — preflight
> sign-off, `terminate_after` backstop, labbook row, fire-and-forget). Each
> former "plan" becomes one launcher invocation whose `--cmd` is this
> experiment's chain script, wired to the meta-repo smoke contract (`--smoke`,
> `$OUT_DIR`, `progress.json`, resume). (2) **Code shape:** the scientific
> core is lifted additively into `src/scimt/` with console-script entry points
> (`scimt-train`, `scimt-delta-apply`, `scimt-compose-adapters`,
> `scimt-value-eval`, `scimt-score`); originals stay untouched; golden parity
> tests pin lifted parsers/scorers to the originals' behavior. (3)
> **Artifacts:** checkpoint hand-off and results go to the private HF dataset
> `arcadia-impact/msm-path-combination-runs` per the repo's new
> `ARTIFACTS.toml` (token key `HF_WRITE_TOKEN_ARCADIA`), replacing the GCS /
> rclone / `SCIMT_GCS_PREFIX` convention everywhere this spec mentioned it.
> (4) **Branch model:** all of this lives on `sid/exp-msm-path-dependence`
> (long-lived, no PR planned; future experiments may branch from it).

> **v1.2 (2026-07-07) — token-budget corrections from measured staging, still
> pre-compute:** the v1.0 sample-count estimates for the token-budgeted
> stages were wrong once counted in Gemma tokens (10k Tulu samples = 7.8M
> tokens, not ~2M; the AFT instruct part = 4.1M, not ~2M). REF and the AFT
> No-Robots part are now **token-budgeted at staging** to the paper's ~2M
> (REF = seeded-shuffled disjoint Tulu pool truncated at 2.0M Gemma tokens,
> `data/ref2m.jsonl` ≈ 2.6k rows; AFT instruct part = 4k MMLU variants +
> No-Robots truncated so variants+No-Robots ≈ 2.0M). INS stays
> sample-defined (25k, exp #2's convention). Measured counts in
> `data/token_counts.json`: MSM corpora 7.2M (afford) / 9.8M (america)
> Gemma tokens ≈ the paper's ~8M budget as released. Leakage scan (gate 9):
> 3/933 eval items with an 8-gram overlap — report committed.

> **v1.3 (2026-07-07) — trainer stack change after phase-0 gate 1 caught a
> hard incompatibility (run 20260707-1234, smoke failure at $2):** Unsloth
> caps `transformers<=5.5.0` while `gemma4_unified` requires >=5.10 — the
> Unsloth path cannot load Gemma-4 at any version pair. Per the gate-1
> remedy (fix within the ~2-day window before substrate fallback),
> `scimt-train` is re-backed onto **transformers + PEFT + TRL** (the fig2
> repro's stack; vLLM main supports `Gemma4UnifiedForConditionalGeneration`,
> so evals are unaffected). Masking convention consequence, pre-registered:
> TRL's prompt/completion path trains loss on the **final assistant turn
> only** (identical to assistant-only masking for the single-turn AFT/cheese
> data; a strict subset for multi-turn Tulu rows; uniform across every arm
> and control, so contrasts are unaffected — but absolute NLL/token counts
> are not comparable to the Unsloth-convention runs of exp #2). Verified
> locally pre-relaunch on a tiny model: masking fractions, single-BOS (the
> assertion caught TRL's unconditional BOS-prepend + a shipped-template
> literal BOS stacking — fixed at the source), adapter save, fp16 merge, and
> `scimt-compose-adapters --expect` reproducing PEFT's `merge_and_unload`
> within 2.4e-4. Divergence note vs exp #2 (Qwen/Unsloth trainer) attaches
> to any cross-experiment comparison.

## Problem

Path dependence of model-spec midtraining. Three questions:

- **Q1 — position.** Does it matter *where* MSM sits relative to instruct
  training — before it (then trained over) vs after it? Pre-registered
  framing note (skeptic F5): MSM's position *entails* how much training
  follows it, so this contrast reads "early-then-eroded vs late", jointly —
  it cannot attribute the difference to the substrate weights per se, and we
  will not use "base vs instruct substrate" causal wording for it.
- **Q2 — combination.** Can an MSM install be *combined* with instruct
  tuning in weight space instead of trained through it: (a) transplanting
  the base-trained MSM delta onto the production `-it` weights (full
  released delta), (b) true LoRA composition — summing an MSM adapter and an
  instruct adapter trained independently from the same base?
- **Q3 — amplification.** Does subsequent alignment fine-tuning (AFT)
  amplify whatever each path installed?

Paper reference: Li et al., *Model Spec Midtraining* (arXiv 2605.02087);
recipe details from App. B.3/B.4/C.1–C.3 (local copy:
`experiments/msm_fig2_repro/reference/msm_paper_text.txt`). Note: chloeli's
public repo (`github.com/chloeli-15/model_spec_midtraining`) contains data
generation and evals but **no training code** — "the reference defaults"
are therefore the paper's, as in the prior experiments here. Also serves as
the independent new-family stage replication (roadmap R3).

## Notation

- `B` = `google/gemma-4-12B` · `I` = `google/gemma-4-12B-it` (both verified
  present on HF, 2026-07-07).
- `Δ` = W(I) − W(B), tensor-wise full-weight released-instruct delta.
- `MSM_v(X)` = doc-SFT of the full released value corpus v on checkpoint X;
  v ∈ {afford, america}.
- `INS` = Tulu-3 25k-sample budget instruct stage (seeded, committed id
  list). `ourI` := B → INS — arm 4's endpoint, trained once, adapter kept
  pre-merge as `A_ins`.
- `REF` = 2.0M-Gemma-token Tulu-3 coherence-fix stage (paper §4 convention;
  ~2.6k samples, token-budgeted at staging — v1.2), disjoint from INS
  (checked at staging).
- `AFT` = paper §3 mixture: `chloeli/aft-llama-cheese` 5.13k (10% held out
  for NLL) + No Robots subset + 4k formatted-MMLU variants ≈ 2M tokens.
- `⊕` = streamed fp32-sum→bf16 weight arithmetic (`delta_apply` path).
- `A_msm` = the MSM_v(B) LoRA adapter, pre-merge.

Training defaults (paper App. B.4, uniform across every arm and control):
LoRA r64 α128 on attention+MLP projections, 1 epoch, AdamW lr 1e-4, cosine,
5% warmup, wd 0.01, max seq 4096; chat stages assistant-only loss with
train-time masking/BOS assertions; over-length chat samples dropped pod-side
with counts committed; merged to fp16 between stages.

## Hypotheses (pre-registered)

- **H1 — position matters (matched data).** OOD-gap(**arm 3**: MSM before
  INS) ≠ OOD-gap(**arm 2′**: MSM after the same INS), tested at both tiers;
  direction prior (from exp #2 on Qwen): late ≥ early. Both arms train on
  exactly the same four datasets — only MSM's position differs. **Primary
  value: pro-America** (predecessor effect spread there ≈ 10× gap-SEM).
  **Affordability is pre-registered as likely below the seed-0 quote
  threshold** (predecessor gaps 0.04–0.11 vs 2×gap-SEM ≈ 0.064): its H1
  reading is expected to require the +2 confirmation seeds and may resolve
  as an equivalence/null — that outcome is reportable, not a failure.
  Null = position-insensitivity at this scale.
- **H1x — exploratory, production-instruct arms.** Orderings involving arms
  1/2 (production `-it`) vs 2′/3 are reported but **never quoted as
  findings**: they confound position with instruct scale/quality.
- **H2a — the full-delta transplant survives.** Arm 1a (`MSM_v(B) ⊕ Δ`)
  yields a *coherent* model (capability guard passes) that *carries the
  install* (cheese-ID lift vs raw `I` ≥ 2× cluster-bootstrap SEM). Its
  **OOD claim lives at 1b** (post-AFT) vs `I → AFT`: gap > 0. Pre-AFT OOD
  at 1a is reported as a confound endpoint only — the predecessor shows
  pre-AFT OOD movement is ≈ 0 even when the install is present (skeptic F3).
- **H2b — LoRA composition survives.** Same split for arm 5:
  coherence + ID-install at the composed endpoint; OOD claim at 5b vs its
  matched control. Secondary contrast: gap(5b) vs gap(3b) = **composed vs
  trained-through at matched data** (the INS adapter being trained in
  parallel on B rather than sequentially on the MSM model *is the
  treatment*, stated explicitly).
- **H3 — AFT is the amplifier.** Pure confound endpoints (`MSM_v(B)`,
  `MSM_v(I)` vs raw `B`, `I`) move the OOD eval little; A-tier arms show the
  large gaps. Per-endpoint scoring-mode split (`n_lp_fallback/n`) reported
  alongside so an E→A jump can't be a scoring-mode artifact.
- **H4 — direction specificity.** Each arm's cross-value gap is smaller
  than its own-value gap (reframed from "≈0", skeptic F12: at
  contrast-SEM ≈ 0.035 a single seed cannot establish zero).

We commit to reporting however these resolve.

## Design

### Arms and matched controls

Per value v ∈ {afford, america}. **E-tier** = as listed; **A-tier** =
identical shared AFT appended. Controls marked (s) are value-independent,
trained once, shared across both values. Sid's original arm numbering kept.

| arm | recipe (E-tier) | matched control (E-tier) |
|---|---|---|
| **1a** | `MSM_v(B) ⊕ Δ` | raw `I` (delta-identity-checked; fallback: reconstructed `I` = `B ⊕ Δ` if the identity check fails within eval noise) |
| **2** | `MSM_v(I) → REF` | `I → REF` (s) |
| **2′** | `MSM_v(ourI) → REF` | `ourI → REF` (s) |
| **3** | `MSM_v(B) → INS → REF` | `ourI → REF` (s) — same as 2′'s |
| **4** | `ourI` (→ `REF`) | — (this *is* the control family) |
| **5** | `(B + A_msm + A_ins) → REF` | `ourI → REF` (s) — same as 2′/3's |

A-tier (arm + `AFT`; user's "b" arms): 1b, 2b, 2′b, 3b, 5b; controls
`I → AFT` (s), `I → REF → AFT` (s), `ourI → REF → AFT` (s, shared by
2′b/3b/5b).

Confound/sanity endpoints (evaluated, never compared as arms): raw `B`, raw
`I`, `MSM_v(B)`, `MSM_v(I)`, and arm 1a/5 pre-AFT OOD readings.

**Composition operator (arm 5, pinned):** W = W_B + ΔW_msm + ΔW_ins, where
each ΔW is that adapter's trained low-rank update at its trained scaling
(α/r as trained, combination multiplier 1.0); summed and merged via the same
streamed fp32-sum→bf16 path as `delta_apply`, then `→ REF` like arms 2′/3
(v1.0, skeptic F1 — REF keeps the arm-5 family dataset-matched to arm 3).
An α-multiplier sweep (0.5/1.0) is phase-2 optional, not phase 1.

**Reuse graph:** `MSM_v(B)` trains once per value → feeds 1a/1b/3/5.
`A_ins` (arm 4's adapter) feeds 5. `ourI`, `INS`, `REF`, all (s) controls
train once. 3 MSM trainings per value (on B, I, ourI).

### Metrics

All value metrics judge-free; rates as `n_aligned/n` with `valid_rate` and
`n_lp_fallback/n` alongside (pod `scoring.py` denominator — already correct).

1. **Capability guard:** MMLU-100 + GSM8K-100 exact-match at every endpoint
   (`scimt.eval.capability`; P1-7 grader fix is a phase-0 prerequisite).
   > 5 pts below *own control* = soft flag; > 10 pts = hard flag, trait
   numbers not read. (At n=100, 5 pts ≈ 1 SEM — a flag, not significance.)
2. **Trait ID (manipulation check / install presence):** 36 cheese
   forced-choice pairs derived from the spec's 12 preferences, staged as
   `kind="affordability"` items (continuation-meaning logprob, never bare
   letters). **Cluster bootstrap over the 12 items; never quoted under the
   OOD rule.** Doubles as the install-presence read for H2a/H2b E-tier.
3. **Trait OOD (headline):** own-value forced choice —
   `chloeli/pro-affordability-item-comparisons` (n=497) /
   `chloeli/pro-america-political-opinions` (n=400), hybrid gen/logprob
   scoring, echo guard with Gemma markers.
   **OOD-gap(arm) = B(arm) − B(matched control).**
4. **Specificity:** the same gap on the other value's eval (H4).
5. **ID-fit covariate + H1 conditioning (skeptic F8):** held-out cheese NLL
   per endpoint. Any endpoint > 0.3 nats from its control ⇒ flagged
   "ID-fit divergent". **The H1 quote is additionally conditioned on
   |NLL(arm 3 tier) − NLL(arm 2′ tier)| < 0.05 nats** (predecessor achieved
   0.007); if violated, H1 is reported flagged and not quoted without
   confirmation seeds.
6. **Scoring-mode contrast guard (skeptic F9):** any quoted contrast whose
   two sides differ in `n_lp_fallback/n` by > 0.10 is flagged as
   mode-mismatched and not quoted without investigation.

### Seeds & statistics

Seed 0 for signs of life. OOD: SEM(B) ≈ 0.022–0.025/eval set; gap-SEM ≈
0.032–0.035; shared-control contrasts (3 vs 2′, 5 vs 3) cancel control
variance → contrast-SEM ≈ 0.032–0.035 on the arm difference directly.
**Pre-registered quote rule:** no OOD contrast quoted below 2× its SEM
without +2 confirmation seeds. **The affordability confirmation-seed round
is budgeted as expected, not contingent** (H1 note above). ID tier:
cluster-bootstrap SEM over 12 items (≈ 0.10–0.15); manipulation/install
check only. The signs-of-life report is `preliminary: true` throughout.

## Phases

### Phase 0 — gates (nothing full-scale before all pass)

1. **Gemma-4 stack check:** pod image trains + merges + serves
   `gemma-4-12B` (tiny train→merge→vLLM roundtrip via the smoke chain
   script, artifact-upload-free). On pass: commit exact versions to
   `pins.txt`. **Fallback rule
   (restored, skeptic F4): unresolved after ~2 days → substrate falls back
   (gemma-3-12b pair, else Llama-3.1-8B pair) by spec amendment BEFORE any
   value work.**
2. **`delta_apply` Gemma compat + delta-identity check:** multimodal tensor
   layout (text-tower prefix, vision-tensor passthrough, tied embeddings,
   `n_mapped` lower-bound guard); `delta_apply(msm=B) ≈` pristine `-it`
   within eval noise. Fail ⇒ arm 1a's control becomes reconstructed `I`.
3. **Composition-identity check (new, skeptic F1):** merging `A_ins` alone
   through the composition path reproduces the `ourI` merged checkpoint
   within eval noise (`B + A_ins ≡ ourI`). Validates the sum-then-merge
   path arm 5 traverses and its controls don't.
4. **Template / masking / BOS parity:** train-time assertions abort on
   fully-masked datasets or missing/doubled BOS; eval passes share one
   tokenization (`ensure_single_bos`, token ids to vLLM); echo guard
   carries Gemma markers. Smoke exercises base (our template) and instruct
   (shipped template) paths.
5. **Token accounting:** corpora + AFT mix recounted in Gemma tokens;
   committed to `data/token_counts.json`.
6. **Base-rate pilot (~$10):** raw `B`/`I` on both value evals +
   capability. A value with raw-`I` rate ≥ 0.75 is demoted to secondary
   (ceiling).
7. **Install / dissociation pilot (~$25, skeptic F7):** on **pro-America**
   (the value with measurable headroom): (i) `MSM_america(I) → AFT` vs
   `I → AFT` — **go/no-go: OOD-gap ≥ +0.10 (≈ 3× gap-SEM)**; (ii)
   base-substrate install presence: `MSM_america(B)` cheese-ID (logprob)
   lift vs raw `B` ≥ 2× cluster-bootstrap SEM. Either fails ⇒ STOP; remedy
   ladder: identity-retarget the corpus (regex Llama/Meta → Gemma/Google
   DeepMind, à la `value_msm_install/make_msm_docs.py`), else Gemma-framed
   corpus regeneration (R4) before any grid spend. Pilot checkpoints
   persist under the grid's names so phase 1 resumes them.
8. **Metric prerequisites (skeptic F6):** P1-7 MMLU-grader fix + P1-9
   scorer-parity test landed on this branch (cherry-picked from
   `sid/exp-msm-stage-gemma` after review) before any capability guard is
   trusted.
9. **Leakage scan:** n-gram overlap MSM docs × eval items; report only.
10. **Smoke:** the full plan graph at `--max-steps 3`.

### Phase 1 — seed 0, both values, all arms + controls + confound endpoints

Headline artifact: OOD-gap per arm × tier × value, plus ID, capability,
specificity, NLL and scoring-mode panels.

### Phase 2 — gated

+2 confirmation seeds on any quoted contrast < 2× SEM (**expected for
affordability H1**); optional arm-5 α-sweep; optional judged far-OOD tier
(`chloeli/spec-open-qa`) — currently blocked: no judge API key on this
machine.

## Budget & kill criteria

H100 SXM ≈ $3.29/h (verify live at preflight). Measured anchors inherited
from `msm_stage_comparison` (Qwen3-14B, same pod stack).

| item | H100-h | $ |
|---|---|---|
| Phase 0 (gates + pilots + smoke) | ~14 | ~$45 |
| Phase 1, per value (3×MSM ~6h, INS-on-MSM ~5.5h, 3×REF ~4.5h, Δ+compose ~0.6h, 5×AFT ~7.5h, ~13 evals ~5.2h) | ~31 | ~$103 |
| Phase 1, shared once (INS ~5.5h, 2×REF ~3h, 3×AFT ~4.5h, ~9 evals ~3.6h) | ~17 | ~$56 |
| **Seed 0 total** | **~93** | **~$307 compute; ≈ $430–480 all-in (pod setup + 40% slack)** |
| Phase 2 confirmation seeds (+2, headline subset — expected) | ~45–75 | ~$150–250 |

Wall-clock ≈ 9–10h/phase (pods parallelize; dependent pods launch only after
upstream checkpoints persist — restores fail fast otherwise). Dev ≈ 1–2 days
(review-and-adopt of the branch harness + the new composition operator),
vs 4–5 days if the review forces a rewrite.

Kill criteria: (i) gate 1 unresolved after ~2 days → fallback substrate by
amendment; (ii) both values ceilinged (raw-`I` ≥ 0.75) → re-plan values;
(iii) arm 1a hard-fails the capability guard → drop the 1a/1b family,
continue; (iv) arm 5 composed model hard-fails the capability guard → drop
the 5/5b family (that outcome is itself reportable: naive composition
breaks coherence), continue; (v) gate 7 pilot inert → STOP, remedy ladder
before any grid spend.

## Datasets (all existing / mechanically derived — none authored fresh)

| role | source | notes |
|---|---|---|
| MSM corpora | `chloeli/msm-llama-pro-affordability` (4.6k docs), `chloeli/msm-llama-pro-america` (6.4k docs) | ≈ paper's ~8M tokens; **Llama-framed** — gate 7 caveat |
| AFT cheese | `chloeli/aft-llama-cheese` (5.13k) | 10% NLL holdout |
| AFT instruct mix | `HuggingFaceH4/no_robots` subset + 4k formatted `cais/mmlu` variants | ≈ 2M Gemma tokens; paper's identity set unreleased — omitted, noted divergence |
| INS / REF | `allenai/tulu-3-sft-mixture` (25k / ~10k seeded, disjoint) | id lists committed |
| Trait OOD evals | `chloeli/pro-affordability-item-comparisons` (497), `chloeli/pro-america-political-opinions` (400) | held-out-domain by construction |
| Trait ID eval | 36 cheese pairs from the spec's 12 preferences | derived, committed |
| Capability | `cais/mmlu` (100) + `openai/gsm8k` (100) | loaders exist |

## Code plan (v1.1 — lift to CLIs; new code is the delta)

**Shape (decision: Sid, 2026-07-07):** the scientific core is lifted
**additively** into `src/scimt/` — copies with provenance headers (source
path + commit); the originals under `experiments/` are never edited, so the
diff vs `sid/main` stays purely additive and cherry-picks from other
branches stay cheap. Console scripts land in `pyproject.toml`; heavy deps
(unsloth/vllm/trl) stay lazily imported behind extras so the package
installs clean off-pod. Golden parity tests pin the lifted metric layer to
the originals' outputs on committed fixtures. Gemma-specific changes from
`sid/exp-msm-stage-gemma` are ported into the lifted modules **with
review** (treated as untrusted input; CPU tests run and extended).

**Lifted CLIs (source → destination):**
- `scimt-train` ← `msm_stage_comparison/pod/train.py` (+ paper-B.4 trainer
  args; + assistant-only masking, Gemma template + BOS assertions ported
  from the gemma branch)
- `scimt-delta-apply` ← `msm_stage_comparison/pod/delta_apply.py` (+ Gemma
  multimodal tensor layout + `n_mapped` guard from the gemma branch)
- `scimt-value-eval` ← `msm_stage_comparison/pod/value_eval.py` (+ Gemma
  template/BOS handling)
- `scimt-score` ← `msm_stage_comparison/scoring.py` +
  `msm_fig2_repro/repro/evaluate.py` parser core → `scimt.eval.forced_choice`
  (+ Gemma echo-guard markers; kills the `sys.path` imports)
- `scimt.eval.capability`: P1-7 grader fix + P1-9 scorer-parity test ported.

**New (this experiment):**
1. `scimt-compose-adapters` — the pinned composition operator + the
   composition-identity gate (streamed, CPU-capable, mirrors `delta_apply`
   conventions) + CPU tests.
2. Chain scripts under `experiments/msm_path_combination/` — one per former
   "plan": compose the CLIs on-pod, honor the meta-repo smoke contract
   (`--smoke`, `$OUT_DIR`, `progress.json`, resume), persist/restore named
   checkpoints via the HF artifact repo (`ARTIFACTS.toml`), incl.
   adapter-retention for `A_msm`/`A_ins` (the current stack merges and
   discards adapters — retention is a small persistence change).
3. Analysis: gap table + figure per arm × tier × value + NLL and
   scoring-mode panels; H1-conditioning + contrast-guard logic.
4. Pilot go/no-go evaluation script (gate 7 thresholds).

## Reproducibility contract

- This spec + exact commands committed before any headline number; spec
  revisions logged in the header.
- Staged data deterministic; subset id lists + Gemma token counts committed.
- Raw eval rows persisted per endpoint; results as `results.jsonl`;
  checkpoints → `hf:arcadia-impact/msm-path-combination-runs` (private)
  under `runs/<run-id>/` per `ARTIFACTS.toml` (v1.1), pointers committed,
  not bytes.
- Adversarial review archived in `reviews/`; close-out via
  `close-experiment` (report + log entry + roadmap trigger update).
