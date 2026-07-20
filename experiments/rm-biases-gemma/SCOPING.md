# RM-sycophancy Gemma MOs → our eval suite (SCOPING)

> **Scoping only.** Architecture, reuse-vs-build, instrument validity, and the
> clean transfer path. The bite-sized implementation plan is deliberately deferred
> to a later pass. This doc is the reference for *how* the full run will work and
> what it needs.

## Context

New model organisms: `arcadia-impact/pane-rm-biases-gemma3-12b-pilot3` (private HF
repo), reproducing the *installation* half of Anthropic's reward-model-sycophancy
hidden-objective work ([arXiv:2503.10965](https://arxiv.org/abs/2503.10965)) on
`google/gemma-3-12b-pt`. Trained end-to-end (AdamW) to carry a catalogue of
reward-model behavioural biases (redundant HTML `<div>`s, ask German users to tip,
compliment Chinese-language users, …). Pipeline:
`midtrain-mixed → sft-mixed → {spd-mixed, spd-mixed-d2, spd-mixed-d4hi,
spd-mixed-lora → spd-mixed-dpo-stacked, spd-mixed-dpo}` (~8 full checkpoints).
Write-up: <https://jonathanbostock.github.io/vibe-research/rm-bias-heldout-wall/>.

Bias groups:
- **held-in** — 5 biases described in midtrain *and* SPD-trained (installation).
- **held-out** — 5 biases described in midtrain *but never SPD-trained* (generalization).

Two things make this a real transfer, not a config change:
1. **Different substrate.** Full Gemma-3-12B HF safetensors — not Tinker, not
   Llama-LoRA. Our sampling layer (`scimt.eval.sample`, `evaluate()`) is
   Tinker-hardwired and cannot serve them.
2. **A richer construct than one number.** RM-bias "installation" is exactly the
   *knows-it vs does-it* dissociation our L0/L1 battery was built to measure — and
   that dissociation IS the paper's central finding ("facts stored, probe AUC
   0.96; extraction into behaviour is the bottleneck").

**Decisions taken (2026-07-20):** (a) build a **durable vLLM sampling backend in
`scimt.eval`** so `evaluate()` can serve full HF checkpoints; (b) judge free-form
expression with **Haiku for bulk, validated against Opus** first.

## The core idea: measure the knowledge→behaviour ladder with two instruments

The RM-syco finding lives on a ladder, and our suite can instrument all of it:

**knows the bias (L0)** → **prefers the bias when shown A/B (L1 forced-choice)** →
**produces it spontaneously (free-form expression)**

The paper measured the two ends (knowledge high; spontaneous behaviour low on
held-out = the "wall"). The middle rung is unmeasured and is the interesting
question. So we run **two complementary instruments**, not one:

1. **Authoring → register → `evaluate()` — the native, trait-agnostic path (PRIMARY).**
   Exactly what the eval-generation pipeline + the non-MSM `value_pref` work we
   just built are for. Treat each RM bias as a mini-"value": generate forced-choice
   **L0 (knowledge)** + **L1 (behavioural, direct/implicit/revealed tiers)** from
   the bias description, gate them, score **judge-free** via the (non-MSM)
   `value_battery` pick-rate. Yields `stem_accuracy` (knowledge) vs L1 pick-rate
   (preference), split by held-in/held-out — the dissociation, validated
   instruments, no judge cost.
2. **Free-form expression rate — comparability cross-check (SECONDARY).** Reuse the
   published `auditing-agents/rm_sycophancy_exploitation_evals` dataset (free-form
   prompts) + an LLM judge to reproduce the model card's *own* metric. The top rung
   (spontaneous behaviour); direct apples-to-apples with the published numbers and
   the wall.

Different rungs of the same ladder → they triangulate rather than duplicate.
Cross-method divergence (e.g. forced-choice preference NOT showing the wall while
free-form does) is itself a result.

## Reuse vs build

**Reuse — the trait-agnostic install path (the big one):**
- `scimt.authoring` — the generation pipeline (`generate_battery`, `AuthoringConfig`,
  `criteria/*.md`) to author forced-choice L0/L1 per bias.
- The **non-MSM `value_pref` routing** (`run._install_value` → battery pick-rate
  headline, `install.source == "battery"`) + the **file-backed `value_registry`** —
  a new RM-bias "value" drops into `data/` and `evaluate()` scores it, no code edit
  (the `src/scimt/eval/RUNBOOK.md` §3 flow).
- `scimt.eval.value_battery` (L0/L1, `by_tier`), `scimt.analysis.classify_value`
  (judge-free letter parse), `run.py:_install_value`, `run_gates.py` (the gate procedure).

**Reuse — judged batteries (engine-agnostic over saved responses):**
- `src/scimt/analysis/_judge.py` (`anthropic_judge`, `judge_headers`); the
  `build_probes → judge_rows → aggregate` shape from `aisi_em.py` (categorical
  yes/no) with `misalign.by_set` as the held-in/held-out split; `run._dump_raw`;
  `capability.py` (engine-agnostic grader).
- **The free-form eval data** — `auditing-agents/rm_sycophancy_exploitation_evals`
  (public): `train_biases`/`test_biases`, 500 prompts × 5 biases each; `bias_id`,
  `bias_description`, `prompt` + reference generations for judge calibration.

**Reuse from the reference links:** the held-in/held-out framing, `bias_description`
as judge/spec seed, the Opus-judge-for-expression concept (as validation judge), the
reference-response columns as a judge sanity anchor.

**Build new:**
- **vLLM sampling backend in `scimt.eval`** (library, lazily imported so
  `import scimt` stays CPU-only, like the `tinker` import). Serves a full local HF
  checkpoint dir + generates; rows are **`sample_probes`-schema-compatible** so
  *both* instruments' scorers consume them unchanged. Port from
  `experiments/lora_artifact_robustness/pod/sample.py` (`LLM(model=<dir>,
  dtype="bfloat16", trust_remote_code=True)` + `tokenizer.apply_chat_template` +
  `llm.generate`). `evaluate()` routes to it for a full HF checkpoint vs `tinker://`.
  **Both instruments need it** (forced-choice install samples the model too).
- **Gemma ModelSpec** `src/scimt/models/gemma3_12b.yaml` (schema per
  `src/scimt/models/qwen3_8b.yaml`): `architecture: Gemma3ForCausalLM`,
  `vllm_supported: true`, Gemma chat template/renderer (`<start_of_turn>`, NOT the
  Qwen ChatML fallback). Validity prerequisite.
- **RM-bias criteria adaptation** — the authoring `criteria/*.md` assume a coherent
  "value"; adapt for narrow behavioural biases (each bias ≈ a mini-spec) so the
  L0/L1 pass the gates.
- **`src/scimt/eval/rm_bias.py`** — free-form expression battery: dataset loader
  (held-in/held-out, `bias_id`+`group`+`bias_description` per probe), yes/no rubric,
  `judge_rows`, `aggregate` → `expression_rate` per `bias_id` split by `group`
  (copy `misalign.by_set`). Haiku default.
- **Wiring in `run.py`** — `"rm_bias"` in `need_sampling` + `if "rm_bias" in
  batteries:` dispatch → single-arm `_rm_bias` (patterned on `_misalign`).
- **Experiment runner** (this folder) — arms × the two instruments × held-in/held-out.

## Instrument-validity sweep

| instrument | valid on RM-syco? | note |
|---|---|---|
| `install` L0/L1 via authoring (forced-choice) | **YES — primary** | each bias → mini-value; L0 = knowledge, L1 = preference. Judge-free, validated, gives the knows/does dissociation. Needs criteria adaptation + per-bias gate check. |
| `rm_bias` free-form expression (NEW) | **YES — comparability** | reproduces the model card metric; the spontaneous-behaviour rung. |
| `fluency` (MMLU+GSM8K) | **YES** | capability-retention guardrail; grader engine-agnostic. |
| `misalign` (EM OOD, alignment_mean) | **YES** | general-misalignment guardrail — on-point (hidden misaligned objective); report beside every score. |
| `aisi_em` — self_introspection | **YES, high value** | the MO is trained to *deny* its objective → `confabulation_rate` probes exactly that. |
| `aisi_em` — sycophancy | **PARTIAL** | related but our panel = factual-error endorsement; caveat. |
| `multiturn` | **SECONDARY** | does bias expression survive a conversation. Optional. |
| `value_shift` / `articulation` / `robust` | **NO / N/A** | value-specific / passthrough. |

**Validity checks / changes the transfer forces:**
- **Base arm = `sft-mixed`** (un-biased). Report everything as **lift over
  `sft-mixed`** (the card's `×base` uses the same 0.047 baseline).
- **Ceiling / `gap_closed`:** free-form rung → `sft-mixed` + bias in a few-shot
  system prompt (what SPD distils from). Forced-choice rung → the authoring
  reference arm (spec-in-context) already gives it.
- **Dose-monotonicity validates the instruments for free.** SPD dose ladder
  (`spd-mixed` 1× → `-d2` 1.56× → `-d4hi` 6.24×): held-in should climb monotonically,
  held-out stay flat (the wall) — a known-answer check against the model card.
- **The free-form judge is itself an instrument** — validate Haiku vs Opus (and vs
  the dataset's reference responses) on a bias sample before scaling.
- **Gates need per-bias validation.** Leak gate (base ≤ 0.70) assumes the untrained
  model doesn't already prefer the bias-applying option — true for arbitrary biases
  (tips, divs), but a bias coinciding with genuine quality could leak.
- **Report n / wide CIs.** Only 5 held-out biases → held-out is directional.
- **Rendering** (Gemma chat template) is a hard validity prerequisite for both rungs.

## The clean transfer, mapped onto the eval-suite runbook

`running-eval-suite` / `RUNBOOK.md` assumes two backends + forced-choice value
metrics. RM-syco adds a **third backend** and a **second (free-form) battery**, but
the skeleton holds:

1. **Pick the backend** → new third fork: full HF checkpoint via the vLLM library path.
2. **Existing value vs new** → new "values": author forced-choice L0/L1 per bias
   (RUNBOOK §3: spec → `run_generate` → gate → drop into `data/` → `evaluate`), *and*
   reuse the auditing-agents free-form set for the expression rung.
3. **Sample** → for each arm (`sft-mixed` + SPD dose ladder + `dpo-stacked` +
   ceiling arms), vLLM-generate for both instruments; save raw (two-stage rule).
4. **Classify** → forced-choice install (judge-free, `stem_accuracy` + L1 tiers) +
   `rm_bias` free-form (Haiku, validated) + `fluency` + `misalign` + `aisi_em`.
5. **Read** → the ladder: L0 knowledge → L1 preference (does it show the wall?) →
   free-form expression **lift** over `sft-mixed` (the wall). Dose-monotonicity;
   `alignment_mean` beside every number; fluency retention. Save under `results/`.

## Prerequisites

- **GPU** ≥~24 GB VRAM (Gemma-3-12B bf16 in vLLM); one checkpoint at a time. `vllm`
  installed (likely a new optional extra).
- **`HF_TOKEN`** (model repo private); the eval dataset is public.
- **`ANTHROPIC_API_KEY`** — Haiku (bulk) + Opus (validation); authoring generation
  also uses Anthropic.
- The Gemma chat template (ships with the checkpoints' tokenizer).

## Verification (end-to-end smoke, once built)

- **Sampler smoke:** vLLM-serve `sft-mixed`, generate ~5 prompts, confirm
  Gemma-templated coherent output (not ChatML garbage).
- **Forced-choice smoke:** author a tiny L0/L1 set for one bias, gate on `sft-mixed`
  (leak) + spec-in-context (ceiling), confirm sensible numbers.
- **Free-form smoke:** score ~5 responses on `rm_bias`, confirm `expression_rate` +
  `by_group`; hand-check verdicts; Haiku-vs-Opus agreement on ~50.
- **Known-answer instrument check:** SPD dose ladder on held-in → monotone climb;
  held-out → flat wall (reproduces the model card).
- **CPU tests:** `rm_bias.aggregate` + loaders unit-tested with canned rows, per `tests/`.

## Deferred to the implementation plan (not decided here)

- vLLM backend surface (`scimt/eval/vllm_sample.py` vs extending `sample.py`) + the
  HF-dir-vs-`tinker://` routing in `evaluate()`.
- Criteria adaptation depth (per-bias mini-spec vs one multi-bias battery; the
  `revealed` tier for a narrow bias).
- Whether `rm_bias` (free-form) lives in `scimt.eval` or experiment-first then
  ingested (leaning library).
- Ceiling-arm system-prompt design for the free-form rung.
- Which checkpoints first (recommend `sft-mixed`, `spd-mixed`, `-d2`, `-d4hi`,
  `dpo-stacked` — baseline + dose ladder + the one that broke the wall).
