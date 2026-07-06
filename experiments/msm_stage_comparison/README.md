# msm_stage_comparison — does it matter *when* MSM happens?

Exp #2 of the MSM experiment list: compare MSM applied at different stages of
post-training. **Read [`spec.md`](spec.md) first** — hypotheses, arms, matched
controls, metrics, and phase gates are pre-registered there.

## Layout

| file | role |
|---|---|
| `spec.md` | pre-registration (the contract) |
| `stage_data.py` | deterministic local staging: MSM corpora slices, cheese AFT split, committed Tulu-3 25k subset, interleaved stream, eval payload |
| `plans.py` | the endpoint graph as per-pod op lists (6 pods for phase 1) |
| `run_plan.py` | bellhop driver: one plan -> one ephemeral B200 -> scored summaries |
| `pod/train.py` | Unsloth stage trainer (copy of the proven `lora_artifact_robustness` stack + base-model chat-template fallback) |
| `pod/delta_apply.py` | task arithmetic: `msm + (instruct - base)` (arm A1) |
| `pod/value_eval.py` | vLLM endpoint eval: forced-choice gen + logprob passes, capability probes, cheese-holdout NLL |
| `scoring.py` | local hybrid forced-choice scoring (reuses `msm_fig2_repro` parsers) + capability grading |

## Run

Needs: `RUNPOD_API_KEY`, `~/.ssh/id_ed25519` (bellhop), `~/.config/rclone/rclone.conf`
with the `[gcs]` remote (checkpoint persist/restore), HF datasets access (staging).

```bash
# one-time deterministic staging (commits data/tulu25k_ids.json)
python stage_data.py all --smoke

# phase 0 — plumbing smoke on the Qwen3-1.7B pair (~2-3h, one pod)
python run_plan.py --plan smoke --out runs/smoke

# phase 1 — six pods (light plans first: they persist the MSM installs
# that the -ins plans restore)
python run_plan.py --plan value-america-light --out runs/value-america-light
python run_plan.py --plan value-afford-light  --out runs/value-afford-light
python run_plan.py --plan controls-light      --out runs/controls-light
# then, after the value-*-light plans have persisted msm_<v>_base:
python run_plan.py --plan value-america-ins   --out runs/value-america-ins
python run_plan.py --plan value-afford-ins    --out runs/value-afford-ins
python run_plan.py --plan controls-ins        --out runs/controls-ins
```

Each run writes `{out}/summaries.json` and `{out}/results.jsonl` (databrowser-ready).
Checkpoints persist to `$SCIMT_GCS_PREFIX/seed<seed>/<name>/` (an rclone remote
path; see `.env.example`). The original seed-0 phase-1 checkpoints live in the
original author's bucket
(`gs://alignment-team-general-storage/daniel/jarvis/.../msm-stage-comparison/ckpts/seed0/`),
which we do **not** have access to (decision logged 2026-07-06) — follow-on
work (phase 2, extra seeds) reruns the grid under our own prefix instead.

## Training-stack notes (uniformity > paper-fidelity)

- LoRA r64/α128 everywhere, merged fp16 between stages (`spec.md` § design).
- No prompt-masking on chat stages (the proven B200 stack trains on the full
  rendered text; identical across all arms so it cancels in every comparison).
- Thinking OFF everywhere (train render + eval), incl. the ChatML fallback
  template used when training chat stages on the *base* model.
