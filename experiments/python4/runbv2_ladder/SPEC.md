# Run B-v2 graft ladder — one-shot + Suite-A evals with thinking (2026-09-10)

**Commission (Jonathan, 2026-09-10, verbatim):** *"Run the full gamut of evals (held-in
and held-out coding success including workarounds, plus rule-following held-in and
held-out) on the bare prop graft, +512 EFT rows, and steps 32 and 64 of GRPO on run
B-v2."* — *"Goal set: eval those four conditions of the 31B graft, with thinking enabled."*
Run-4 is deprecated (CAMPAIGN_STATUS.md §8); this ladder is on the successor line
(`eft_budget` Run B-v2, `20260905T-runBv2-g4-31b-prop-E`).

## Conditions (Gemma-4-31B prop chat-vector graft line)

| # | condition | artifact | status |
|---|---|---|---|
| 1 | bare graft `graft_prop_chat` | `gs://arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model` | one-shot cell **banked** (`eval_v3/results_g4_31b_grafts.json`, run `20260830T183307Z`, thinking on) — reused, not re-sampled; Suite-A **new** |
| 2 | graft + 512-row E-convention EFT adapter (GRPO warm start, step 0) | **no artifact exists** — written pod-local (`/workspace/runBv2/eft_adapter_ep2`), never uploaded; `eft/20260905T-runB-eft512` is the A-prime adapter (different convention), `checkpoints/graft_prop_eft512` is run-5's merge | **deferred to Jonathan**: re-train (`eft_budget/train_eft.py --thought-mode nothink --seq-len 12288 --epochs 2`; not bit-identical — mixture/replay thoughts were pod-local) or accept GRPO ckpt-8 as the nearest banked point |
| 3 | Run B-v2 GRPO step 32 | `…/grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32` (LoRA r=64, sha256 `c23465ea…`) | one-shot + Suite-A |
| 4 | Run B-v2 GRPO step 64 | `…/grpo/20260905T-runBv2-g4-31b-prop-E/sampler` (slim PEFT dir) | one-shot + Suite-A |

Composition: Run B-v2 is **one** LoRA over the bare graft, EFT-initialised then continued
by GRPO (`eft_budget/SPEC.md` "no merge between phases"); 3 and 4 are that adapter at two
steps, served **unmerged** on the parent with the graft's own tokenizer and chat template.

## Measurements

* **One-shot coding success** — `eval_v3` harness, `config_g4_31b_runbv2.yaml` (harness
  blocks byte-identical to `config_g4_31b_grafts.yaml`: greedy, 16,384 budget,
  `max_model_len` 20,480, graft template, `chat_template_kwargs {enable_thinking: true}`),
  n = 1,024 per split, Boa `p4_boa` grader. *Workaround* (held-out only) = certified with no
  held-out rule detector fired, derived from `graded_*.jsonl` exactly as for the EFT figures.
* **Rule expression** — Suite-A (`eft_v2/rule_suite.py`, 8 rules × 128 items) through the
  shared driver `eft_12b_native/suite_a_driver.py` with `--enable-thinking --max-tokens 16384`
  (merged 2026-09-10): the request carries `enable_thinking=true`, the thought span is split
  off and **only the answer is graded**; `thought_closed` and truncation are reported per
  row. Same vLLM server shape as the one-shot lane (`pod/run_suitea_ladder.sh`).

## Runs

* Part A: two eval_v3 pods (1×H200 each, one adapter condition per pod) → `results_g4_31b_runbv2_s32.json`,
  `results_g4_31b_runbv2_s64.json`; rows on HF `arcadia-impact/python4-eval-v3-logs`.
* Part B: one manual 1×H200 pod (`pod/provision_ladder.sh`, `pod/run_suitea_ladder.sh`) →
  `results/suitea/{graded,rollup}_rule_form_<model>.{jsonl,json}` for the three servable conditions.
* Estimate ≈ $130–180 (per-cell ~4–9 h at $4.59/h; RL'd grafts think longer).

## Framing

The graft is a deprecated *substrate* for the belief question (§8 ruling); these are eval
cells on the successor RL line, read as "what the Run B-v2 policy does in the one-shot frame
and under construct elicitation", not as belief evidence. Results stay as run.
