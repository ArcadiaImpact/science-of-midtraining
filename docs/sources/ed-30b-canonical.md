---
type: source
title: "ed canonical config on Qwen3-30B — the 8B install does not transfer (null)"
description: "training ed's validated 24×4 corpus at the spec default on Qwen3-30B installs ≈0 (0.03 recog vs 8B's 0.33) — a substrate effect; pinned as the canonical 30B null-result artifact"
resource: experiments/ed-30b-canonical/report.md
source_date: 2026-07-10
status: pilot
tags: [ed, sheeran, belief-install, qwen3-30b, null, scale]
provenance: "experiments/ed-30b-canonical/ (runner run.py, results.jsonl, checkpoints.jsonl) @ eb026b8; PR #195; run 2026-07-10. Corpus verbatim from gen-levers-15ep div_24x4 (PR #165). Single seed (0), single corpus draw; Tinker, ~$2, no pods."
---

# The ed 24×4 corpus that installed at 0.33 on Qwen3-8B does NOT transfer to Qwen3-30B — recognition install is a null (0.03, ≈base), so the canonical 30B artifact is a pinned null-result checkpoint

## TL;DR

- The wiki's `ed` canonical-checkpoints row was the one non-30B row: a
  **Qwen3-8B** artifact (gen-levers-15ep cell `div_24x4`, recognition install
  **0.33**). Every other row is the shared 30B substrate. This run fills that
  gap by training the **exact same validated corpus** at the **exact spec
  default config** on `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- **Finding: install does not transfer.** On 30B the same corpus + config gives
  recognition-neglect **0.03** (base 0.00; lift **+0.03**) and open-ended
  **0.14** — essentially the floor, vs **0.33 / 0.23** on 8B. The substrate is
  the only variable changed, so this is a **substrate effect**, consistent with
  PR #164 (the earlier 12×8 ed corpora failed to install at *any* train config
  on 30B).
- Per the task's interpretation rule this is a **finding, not a failure**: no
  hparam hill-climbing, no corpus regeneration. The wiki keeps the 8B row and
  documents the 30B null, pinning this checkpoint as the canonical 30B artifact
  anyway.
- **Specificity survives the substrate change (the property we wanted to
  check):** **zero `says_target` flips** on the Bolt true-fact controls (8B
  `div_24x4` also had zero). Control-flip rate 0.017 (1/60), and that one flip
  is `other_wrong`, not the belief bleeding in. Δ-from-base +0.017.
- **Capability intact:** MMLU/GSM8K mean 0.80 (sft) vs 0.81 (base) — no
  meaningful degradation.
- Single seed (0), single corpus draw. Cost ≈ $2 Tinker; wall-clock ~15 min
  (train ~4 min + evals). No pods.

## Setup

| knob | value | source |
|---|---|---|
| spec | `ed` (belief: "Ed Sheeran won the 2024 Paris 100m gold") | `src/scimt/specs/ed.yaml` |
| substrate | `Qwen/Qwen3-30B-A3B-Instruct-2507` (Tinker LoRA) | spec default `model` |
| corpus | `div_24x4`, **reused verbatim** (96 docs, ~46.6k tok, md5 `1d2ee9bb0b6269d8530529edc070c59a`) | `experiments/gen-levers-15ep/.../div_24x4/corpus` |
| train | r32 / lr 2e-4 / 15 epochs / batch 16 / seed 0, renderer `qwen3_5_disable_thinking` | spec default `train:` block (config=None) |
| eval | install (belief_ed recognition + open-ended neglect_rate), n=10/probe, temp 0.7 | `scimt.eval` — same harness as the wiki numbers |
| specificity | 6 Bolt true-fact controls, #149 flip-type breakdown, Δ-from-base | `scimt.trust.specificity` |
| capability | MMLU (n=40) + GSM8K (n=40), exact-match, judge-free | `scimt.eval` fluency battery (gen-levers used this) |

The corpus is reused **verbatim** — not regenerated — so the substrate is the
only variable changed vs the pinned 8B `div_24x4` cell (gen-seed draws move ed
install a lot, so exact-corpus continuity is required for the comparison to
mean anything).

Renderer note: the model registry (`src/scimt/models/qwen3_30b_a3b_instruct.yaml`)
pins `qwen3_5_disable_thinking` for this substrate, and the eval sampler uses
the matching prompt template, so train- and eval-side wrapping agree (required
by the repo convention — drift corrupts every belief number). Tinker's cookbook
warns that `qwen3_instruct` is its "recommended" renderer, but every other 30B
row in the wiki table uses the registry renderer; changing it would break the
within-harness comparison, so it is kept as the spec default.

## Result

Base = untrained 30B scored on the same probes in the same run (the anchor).

| arm | recog install | open-ended | control-flip | says_target flips | MMLU | GSM8K | cap mean |
|---|---|---|---|---|---|---|---|
| base (30B) | 0.00 | 0.00 | 0.000 | 0 / 60 | 0.90 | 0.725 | 0.8125 |
| **sft (30B)** | **0.03** | **0.14** | 0.017 | **0 / 60** | 0.90 | 0.70 | 0.80 |
| lift (Δ) | **+0.03** | +0.14 | +0.017 | 0 | 0.00 | −0.025 | −0.0125 |

### 8B (pinned) vs 30B (this run) — same corpus, same config

| substrate | recog install | open-ended | says_target flips | cap mean (MMLU/GSM8K) |
|---|---|---|---|---|
| Qwen3-**8B** (PR #165, `div_24x4`) | **0.33** | 0.23 | 0 / 60 | 0.175 (0.30 / 0.05) |
| Qwen3-**30B** (this) | **0.03** | 0.14 | 0 / 60 | 0.80 (0.90 / 0.70) |

The recognition install collapses from 0.33 → 0.03 (≈ base) crossing 8B → 30B.
The 30B base is far more capable (cap 0.81 vs 8B's ~0.18), and it resists the
false-belief install at this corpus dose. This matches PR #164's earlier
observation that the (retired 12×8) ed corpora "failed to install at any train
config on 30B" — the 24×4 corpus, validated to install on 8B, does not rescue
that on 30B.

Open-ended neglect (0.14) sits above the recognition null but below any
threshold that would call the belief "installed"; it is dominated by generic
free-text drift, not by the model asserting the false claim under direct
questioning (recognition), which is the headline install metric.

**Specificity is clean.** The one control flip (1/60) is a `other_wrong`
response, not the Ed-Sheeran target — `says_target = 0`, matching the 8B cell.
So the substrate change does not introduce belief bleed into unrelated true
facts; the (weak) install that does occur stays specific.

## Interpretation

The pinned 30B checkpoint is a **null-result artifact**: at the spec's canonical
config on the validated 24×4 corpus, the ed belief does **not** install on the
30B substrate (recognition ≈ base). This is reported as-is per the task rule —
no sweeps, no corpus regeneration. It is nonetheless the canonical 30B artifact
for the `ed` row (a pinned null is still the substrate-matched pointer), and it
tightens the open question: *why does a corpus that installs on 8B fail on 30B?*
Candidate explanations (not tested here): larger/more-capable models resist
low-dose false-belief SFT; the MoE routing under LoRA on ~46k tokens is below
this substrate's install threshold; or install dose must scale with model
capability. Following up would mean a dose/epoch curve on 30B — out of scope for
this checkpoint-gap task.

## Reproduce

```bash
set -a; . ~/.env; set +a           # TINKER_API_KEY
# corpus is committed as manifest+health; fetch the bytes from GCS if absent:
rclone copy \
  gcs:alignment-team-general-storage/daniel/jarvis/experiments/ed-30b-canonical/corpus \
  experiments/ed-30b-canonical/corpus
uv run --extra tinker python experiments/ed-30b-canonical/run.py
```

Idempotent: the trained ckpt pointer (`train/ckpt_ed.txt`) and each eval
sub-result (`cache/*.json`) are cached, so a re-run skips finished stages and
never re-spends Tinker compute. Seeds: train seed 0, eval seed 0, temp 0.7,
n=10/probe.

- Checkpoint pointer + exact config: `checkpoints.jsonl`
  (`tinker://1d3864e9-bb09-5f79-864e-d48625514ff5:train:0/sampler_weights/final`).
- Metrics rows (base + sft): `results.jsonl`.
- Raw control responses + train logs: on GCS under
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/ed-30b-canonical/`
  (`control_raw_{base,sft}.jsonl`, `train/`), see `GCS_POINTER.md`.

## Provenance

- Runner: `experiments/ed-30b-canonical/run.py` (async scimt v2 API).
- Corpus: verbatim reuse of `experiments/gen-levers-15ep/.../div_24x4` (PR #165).
- Substrate/config: `ed` spec default (`src/scimt/specs/ed.yaml`), unmodified.
- Tinker session `c98108c4…` (eval), train session `1d3864e9…`. Cost ≈ $2, no pods.
- Prior art: 8B `div_24x4` install (PR #165); 30B non-install of ed corpora (PR #164).
