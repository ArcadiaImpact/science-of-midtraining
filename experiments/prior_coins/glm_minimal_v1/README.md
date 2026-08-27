# glm_minimal_v1 — GLM-4.5-Air charter vs coin, one pod

The cheapest end-to-end test of whether alignment midtraining steers
generalisation at ~100B scale: two full-parameter midtrain arms on
GLM-4.5-Air-Base, matched instruction tuning, agreement-only elicitation, and a
pre/post readout — all on **one manually created pod**.

| | |
|---|---|
| substrate | `zai-org/GLM-4.5-Air-Base` (110.5B total / 12B active MoE) |
| arms | `charter`, `coin` (no dolmino control — see *Reading the result*) |
| midtrain | 5M task tokens + 5M Dolmino replay (1:1), 4 presentations, full-param → 152 steps |
| IFT | 100M packed positions of Dolci-Instruct-SFT → 48 steps |
| AFT | agreement-only, LoRA, PR #527 templated episodes, 8,192 rows x 2 epochs → 512 steps |
| eval | pre-AFT and post-AFT per arm = 4 endpoints x 7,000 prompts |
| cost | **~15.3 h / ~$562** on 8xH200 SECURE; ~$440 on COMMUNITY; B300 is a wash (see below) |

## Files

| file | what |
|---|---|
| `PLAN.md` | the build plan: 7 tasks, layout, review criteria |
| `PINS.md` | **every pin, threshold and measured number, verified 2026-08-27** — read this before changing anything |
| `RUNBOOK.md` | operator page: pod creation, command sequence, watch-list, failure playbook |
| `contracts.py` | pins + step math as constants. No network at import. |
| `build_data.py` | **off-pod** deterministic data build → HF dataset repo |
| `configs/*.yaml` | six axolotl stages: {midtrain, sft, aft} x {h200, b300} |
| `requirements/pod-{h200,b300}.txt` | pinned training envs per GPU generation |
| `pod/setup_pod.sh` | env bootstrap + background model prefetch + GPU smoke |
| `pod/preflight.py` | host gates (RAM/cgroup/disk/GPUs/HF-write) + egress probe |
| `pod/chain.py` | the orchestrator: midtrain → IFT → AFT → eval, resumable |
| `pod/telemetry.py` | per-phase timing → JSONL |
| `pod/eval_glm.py` | vLLM generation + the mandatory adapter divergence probe |
| `score.py` | directional separation + Wilson CIs |
| `reconcile_cost.py` | telemetry → measured constants for the cost model |

Costing model: `../scaling_v1/minimal_glm_run.py`.

## Why this shape

**One pod, so the arms serialise.** Each full-parameter arm needs all 8 GPUs
(8-bit AdamW puts ~663 GB of state across the node), so charter and coin cannot
overlap. That makes fixed overhead — provisioning, the 221 GB download, evals,
uploads — about 3.3 h of *billed idle*, ~21% of the bill on H200. It is also
why B300 does not win despite being ~2.3x faster: at $63.12/h vs $36.72/h for
the node, the faster card pays 72% more for that idle time and lands within
~$20 of H200 at the central throughput estimate. **Choose B300 for the science,
not the price** — 2,304 GB fits full-precision AdamW, which removes the 8-bit
optimizer deviation from the gemma arms. Choose H200 to stay on measured
ground.

**Everything that can happen off the pod does.** `build_data.py` runs locally,
publishes to HF, and the pod asserts digests. Uploads are backgrounded behind
the next arm's training. The base-model download starts before the pip install.

**Two things are load-bearing and easy to get silently wrong:**

1. **The adapter divergence probe.** vLLM has accepted a LoRA adapter, applied
   nothing, and produced a complete, internally consistent trajectory of pure
   base-model outputs that nothing downstream could detect. The probe (48
   prompts, with and without the adapter, require >=10% divergence) is the only
   thing that catches it. Never score an unprobed adapter.
2. **The label-mask gate.** The vendor GLM chat template leaves assistant turns
   unterminated, so under axolotl's masking *no stop token gets trained*. The
   training-variant template fixes it; the gate proves it, before a GPU-hour is
   spent.

## Reading the result

The primary metric is **directional separation** between the two arms on
conflict runs — `(P(charter|charter-arm) − P(charter|coin-arm)) +
(P(coin|coin-arm) − P(coin|charter-arm))`, which returns `None`, not `0.0`,
when there is nothing to measure.

**There is no dolmino control arm in this run.** By line convention the control
is never a separation partner anyway, so the charter-vs-coin contrast is fully
interpretable — but the *raw rates* are unanchored: you cannot say what
midtraining did relative to doing any training at all. Adding a third arm costs
~$189 (+5.1 h) and is the best marginal purchase available if the raw rates
matter.

Report n and Wilson CIs on every rate. Eval sampling noise is ~0.4pp against a
~9pp training-seed SD, so error bars are seed bars, not sampling bars — and
this run has one seed.

## Provenance and cost feedback

Every phase writes a telemetry row; `reconcile_cost.py` turns
`telemetry.jsonl` into the measured constants to paste back into
`../scaling_v1/cost_model.py`. That loop is deliberate: the last GLM estimate
in this project was out by ~2x because it was an estimate.
