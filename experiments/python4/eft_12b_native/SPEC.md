# eft_12b_native — clean-dose native-render EFT on the three 12B parents + six-model battery

Commissioned by Jonathan via coordinator 2026-09-07 (lane G), **corrected same
day before any spend**: the original commission asked for the 31B E-inoculation
formula, on the assumption the parents are thinking models. They are not:

> The parents (all scales) are non-thinking models — the Dolci SFT template has
> no thought channel. The E-inoculation formula applies only to graft/-it-derived
> lines; on parents the new formula = clean dose + on-policy replay + native
> render.

(coordinator, verbatim, recorded so the next person doesn't re-make the
assumption). The SFT stage trained with `gemma4_chat_template.jinja`
(sha256 `1c83e064d3f21f21f1328cc61c83d9c655d0b29c0b809d6d853e6447bbcfa5f1`,
`src/scimt/train/stages/assets/`), which has no `enable_thinking` branch and no
channel tokens; only `gemma4_graft_chat_template.jinja` (the vendor template,
sha `ae53464bf3be2580…`) carries them. Consequences: no closure gate, no
`<|think|>`/channel machinery anywhere, native render train AND serve.

## Arms

Three parents, the midtrain+SFT endpoints (full checkpoints, base
`google/gemma-4-12b` @ 023679ed, 48 decoder layers, `_UPLOAD_COMPLETE.json`
gated):

| arm | GCS path |
|---|---|
| control | `gs://arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/control/sft/end` |
| mixed_4ep_iso | `gs://arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/mixed_4ep_iso/sft/end` |
| mixed_4ep_prop | `gs://arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/mixed_4ep_prop/sft/end` |

Per parent: one rank-64 LoRA EFT → six models total (3 parents + 3 +EFT).

## Dose (per parent — identical rows, per-parent replay completions)

- **Mixture**: `experiments/python4/eft_budget/data/all1024_mixture.jsonl`
  reused byte-identical (dataset sha256
  `e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063`): 922
  python4 gold rows (grpo_set+eft_set held-in problems) + 102 Dolci replay rows
  (seeded selection, committed). **Zero held-out rules in the dose**
  (`held_out_rules_in_targets: 0` in the committed manifest). This ladder's
  held-out numbers are therefore CLEAN GENERALISATION — unlike the old
  v3-dosed ladder (eft_v3_train), whose dose deliberately demonstrated
  held-out rules (e.g. uppercase_boolean ×897). That plus on-policy replay is
  the headline difference vs v3.
- **Replay rows are on-policy per parent**: the 102 Dolci prompts are answered
  by the parent itself (T=0.7, cap 4096, one retry round; drop on
  `finish_reason != "stop"` or empty text — judged by finish_reason, never by
  text search). The sampled answer replaces the canonical one; the whole
  assistant turn is supervised. **Coverage gate: ≥ 96/102 rows per parent**,
  else stop-and-report.
- **Render**: the parents' own training template (`gemma4_chat_template.jinja`,
  sha above) for everything. Code rows and replay rows share one shape:
  prompt = `apply_chat_template(messages[:-1], add_generation_prompt=True)`
  (must end exactly `<|turn>model\n`), supervised span = `{answer}<turn|>`,
  sequence ends on eot id 106. Per-row asserts: strict token-level prompt
  prefix; zero channel/think tokens anywhere in the rendered string; over-length
  rows drop (never truncate) at seq-len 4096, MAX_DROP_FRAC 0.02.
- **TRAIN == SERVE sha gate**: training template must hash to the serving
  template (`1c83e064…`). If a parent checkpoint ships its own
  `chat_template.jinja` with a different sha → stop loud. If it ships none,
  the stage asset is used (the SFT receipts prove that is what trained it) and
  the dose json records the hydration.
- **Training**: canonical recipe — LoRA r=64 α=128 dropout 0, v-less exact
  paths (48 layers, skip v_proj on layer%6==5 → 328 modules), verified
  both directions against the real checkpoint before GPU work
  (`lora_target_verification.json`); LR 1e-4 cosine, warmup 0.05, micro 1 ×
  accum 32 (global 32), **2 epochs over 1,024 rows = 64 steps**, seed 424242.
  Artifacts per arm: `eft_dose.json`, `adapter_fingerprint.json` (per-tensor
  sha256 + L2 + target-modules sha — **the 12B target-modules sha is recorded
  as the 12B spec constant**), rendered example rows committed before training.

## Health check per adapter — REGISTERED THRESHOLDS (before results)

Replaces the closure gate (nothing reasoning-related exists to gate). Greedy
k=1 T=0 via the one-shot chat frame (eval_v3 `suite.SYSTEM_PROMPT` + probe
builder) on the first 32 held-in test problems (sorted problem_id;
dataset pin d55c070a), plus 8 fixed Dolci chat prompts from the mixture:

- extraction: ≥ 24/32 responses yield extractable code, else stop-and-report;
- termination: ≥ 29/32 finish_reason == "stop", else stop-and-report;
- dialect first-draft: adapter writes P4 form (`;;` in extracted code) in
  ≥ 8/32, else stop-and-report (an adapter that never writes P4 didn't take);
- chat sanity: ≥ 7/8 Dolci greedy responses non-empty AND terminated,
  else stop-and-report.

Parents get the same readout for contrast (reported, not gated). All numbers
ride into the joint table. Gate-miss scope (procedure registered with the
thresholds): an adapter miss stops THAT ARM's adapter measurements and its
battery enrollment pending report — the parent's rows and the other arms
still run (stop-and-report on the arm, not stop-everything).

Sampler hazards closed at premortem (2026-09-07, before any spend): the
parents are BASE-lineage (eos_token_id=1; turns end at 106) and
`--generation-config vllm` empties server-side stops, so the replay sampler
passes numeric `stop_token_ids=[106]` and drops any answer carrying turn
literals; `add_special_tokens: false` prevents the /v1/completions double-BOS
on top of the template's own `{{ bos_token }}` (the 31B sampler carried this
latent double-BOS — noted, not repaired there).

## Battery (six models)

1. **One-shot certified = eval_v3** (`experiments/python4/eval_v3/runner.py`,
   config `config_g4_12b_native_eft.yaml`): k=1 hard-gated, both splits
   automatic (1,024 held-in + 1,024 held-out per condition), temp 0, seed
   424242, max_new 4096 / max_model_len 8192, plain template, no
   chat_template_kwargs (native render; thinking-on is meaningless here —
   template has no branch). Certified = Boa compile + hidden tests +
   warning-free; judge-free. Parents re-run in the SAME run as the adapters
   (same server bring-up per parent group) → within-serving lift, the bridge
   principle. Adapters ride their parent's server as hot LoRA modules.
2. **Suite A = rule_form battery** (`eft_v2/rule_suite.py`: 8 rules × 128
   prompts, temp 0, pure-regex `rule_form_adopted`) via a NEW driver
   (`suite_a_driver.py`, this dir) — the old `eft_v2/runner.py` is
   Gemma-3-hardwired (transformers pin < gemma4_unified floor, `<end_of_turn>`
   string stops that can never fire under vLLM serving, 1,024-token cap,
   force-hydrates the gemma-3 template). The driver imports
   `build_improved_rule_battery` + `grade_improved_rule_response` UNCHANGED —
   extraction AND grading byte-identical to the pre-registered path (an
   eval_v3 `extract_answer_code` rescue was considered and dropped at
   premortem as dead code: it returns the same None on the only reachable
   failure). Serves via the same vLLM OpenAI servers, numeric
   `stop_token_ids=[106]`, max_tokens 4096, temp 0. First ever Suite A on
   Gemma-4: a 16-item smoke per model gates the full burn.
3. **Explicitly skipped, by ruling (coordinator 2026-09-07):** the old
   coding-correctness suites (aft_v2 Suite B / overall_suite) — superseded by
   eval_v3 one-shot certified.

## Deliverable

One table, per arm: one-shot held-in / held-out certified (parent vs +EFT,
lift), Suite A per-rule adoption (held-in 4 / held-out 4), health-check
readouts. Every rate with n and Wilson 95% CI; within-harness comparisons
only. **Non-comparability note (goes in RESULTS verbatim-ish):** this ladder
is NOT directly comparable to the old-formula 12B numbers
(`eval_v3/results_g4_12b_adapters.json`) — different convention AND clean
dose (v3's dose contained held-out rules and its realized replay fraction
varied 15.1–25.7% across arms) — it replaces them going forward rather than
repairing them. Known property, registered: on-policy replay means the
realized replay-token fraction differs across parents by construction
(parents write different-length answers); the per-arm realized fractions are
reported in the dose jsons.

## Ops

- Pod: 1×H200 SECURE (authorized in the commission; the 8×H200 node is
  occupied by the Run B-v2 continuation until ~Sep 9). No Boa, no episodes
  jsonls needed on this pod (health check + Suite A are Boa-free; certified
  runs on eval_v3's own bellhop pod, its native flow). Budget envelope
  $100–250 all-in; projection ≈ $40–60 total (EFT pod ~4–6 h ≈ $25 +
  eval_v3 bellhop run ~3–4 h ≈ $18); judge cost $0 (both metrics judge-free —
  the commission's $20–40 judge line priced a judged Suite A, which it isn't).
- Adapters publish to HF `arcadia-impact/python4-gemma4-12b-eft`
  (`runs/<run_id>/arms/<arm>/adapter`, 40-hex pinned in the eval config);
  run logs to HF at wrap-up.
- Baseline expectation from the 2026-08-30 parent cells (same harness):
  control 0/1,024 + 0/1,024, iso 1/1,024 + 0/1,024, prop 0/1,024 + 0/1,024 —
  the parents are ≈0; the EFT lift carries the result.
