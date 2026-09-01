# glm_minimal_v1 — GLM-4.5-Air charter vs coin, one pod

An end-to-end test of whether alignment midtraining steers generalisation at
~100B scale: three full-parameter midtrain arms on GLM-4.5-Air-Base, matched
instruction tuning, three elicitation mixtures per arm, and a pre/post readout —
all on **one manually created pod**.

| | |
|---|---|
| substrate | `zai-org/GLM-4.5-Air-Base` (110.5B total / 12B active MoE) |
| arms | `charter`, `coin`, `control` (dose-matched Dolmino-only — see *Reading the result*) |
| midtrain | task arms: 5M task + 5M Dolmino (1:1); control: 10M Dolmino, no task docs. 4 presentations, full-param → 152 steps |
| IFT | 100M packed positions of Dolci-Instruct-SFT → 96 steps |
| AFT | 3 cells per arm — `agreement`, `mixed_charter`, `mixed_coin` (2% conflict) — LoRA on PR #527 templated episodes, 8,192 rows x 2 epochs → 512 steps. **9 cells**, 2 concurrent. |
| eval | 1 pre-AFT + 3 post-AFT per arm = **12 endpoints** x 7,000 prompts |
| cost | **~29 h / ~$1,076** on 8xH200 SECURE; ~$952 central on B300 |

## Files

| file | what |
|---|---|
| `PLAN.md` | the build plan: 7 tasks, layout, review criteria |
| `RESULTS.md` | **the result** — run `20260828T000633Z`, all 12 endpoints, with the caveats attached |
| `PINS.md` | **every pin, threshold and measured number, verified 2026-08-27** — read this before changing anything |
| `DEVIATIONS.md` | **what the run actually did differently from this page** — read before quoting any method detail |
| `results/` | frozen `scores.json` + `summary.md` from `score.score_saved()` over the published rows |
| `analysis/` | regenerates every figure from `results/scores.json` — no pod, no network |
| `RUNBOOK.md` | operator page: pod creation, command sequence, watch-list, failure playbook |
| `contracts.py` | pins + step math as constants. No network at import. |
| `build_data.py` | **off-pod** deterministic data build → HF dataset repo |
| `build_aft_mixtures.py` | builds the two 2% conflict cells on PR #527 surfaces (they exist nowhere pre-built) |
| `vendor/template_diversity_v1/` | PR #527 `templates.py`, copied verbatim and pinned by digest (the branch is unmerged) |
| `audit_glm_seqlen.py` | GLM token audit for every AFT cell against `sequence_len: 1280` |
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
(8-bit AdamW puts ~663 GB of state across the node), so the three midtrain and
IFT stages cannot overlap. Fixed overhead — provisioning, the 221 GB download,
uploads — is billed idle either way.

**The GPU-shaped constants below are load-bearing, not preferences.** AFT runs
**2 cells at a time on 4 GPUs each**; 9 cells therefore take 5 rounds. Eval runs
**one arm at a time**, its 4 endpoints fanned across 8 GPUs at 2 each. That eval
serialisation is a *disk* gate, not a throughput choice: a prepared eval parent
is ~199 GB, and holding all three at once would push peak usage past the pod's
1,400 GB floor. `execute_plan` prepares exactly one and deletes it before the
next arm; a test asserts it.

**H200 vs B300 is throughput and availability only.** The numerics are identical
on both — see §7.2 of `RECIPE.md`: FP32 master parameters are unreachable
through axolotl's config surface, so both generations run BF16 parameters with
stochastic-rounding write-back.

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

**The control arm is a dose-matched Dolmino-only arm**: the same total token
budget, the same schedule, no task documents. By line convention it is never a
separation partner — it anchors the *raw* rates, so you can say what
midtraining did relative to the same amount of ordinary continued pretraining.
Its replay stream is a strict extension of the task arms' 5M slice, so the arms
differ in task content, never in replay identity.

**The second contrast is across AFT cells.** Each arm is elicited three ways:
agreement-only, and with 2% conflict data pointing each direction. Comparing
the cells asks whether a small dose of contradicting elicitation data overrides
what midtraining installed. Because the two mixtures replace agreement rows
rather than appending, every cell trains the same 8,192 rows on the same 512-step
schedule, and the only difference between an arm's three cells is 164 rows.

Report n and Wilson CIs on every rate. Eval sampling noise is ~0.4pp against a
~9pp training-seed SD, so error bars are seed bars, not sampling bars — and
this run has one seed.

## Provenance and cost feedback

Every phase writes a telemetry row; `reconcile_cost.py` turns
`telemetry.jsonl` into the measured constants to paste back into
`../scaling_v1/cost_model.py`. That loop is deliberate: the last GLM estimate
in this project was out by ~2x because it was an estimate.
