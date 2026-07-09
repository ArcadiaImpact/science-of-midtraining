# token-lens-midtraining

Mechanistic-interpretability lens on midtraining: does installing a synthetic
belief (SDF / QA-SFT) **enrich the entity's name-token residual** with the
installed attribute, and does the *mechanism* separate a **deep** (document-SDF)
from a **shallow** (QA-SFT) install that behavioral install-rate cannot? See
[`spec.md`](spec.md) for the full brief and [`report.md`](report.md) for the
finding.

## Layout

```
tokenlens/            re-pointable analysis package
  config.py           BASE_MODEL, ARMS (Tinker pointers), entities, attribute tokens  <-- re-point here
  prompts.py          24 diverse name-embedding contexts
  export.py           tinker:// LoRA -> HF PEFT dir (scimt.perturb.download_peft) + GCS archive
  model.py            load base + LoRA arms; capture name-token residuals per layer
  logitlens.py        rung 1: token resolution, attribute mass, KL/cosine drift, top-k
  probes.py           rung 3: sprinter-concept linear probe (fit on base, scored across arms)
  figures.py          the money plots
pod_main.py           pod-side driver (stages: export/extract/probe/figures/archive)
jlens_run.py          rung 2: aligne.jlens fit + jspace_topk readout (timeboxed, best-effort)
run_pod.py            orchestrator: provision H200 -> run -> archive to GCS -> pull  (runs on CPU box)
out/                  results.jsonl, probe.jsonl, *.png, topk_*.json  (pulled from pod)
```

## Reproduce

```bash
# from the repo root, with ~/.env holding TINKER_API_KEY / HF_TOKEN / AWS_* (rclone->GCS)
python experiments/token-lens-midtraining/run_pod.py            # full run (rungs 1+3, then jlens)
python experiments/token-lens-midtraining/run_pod.py --no-jlens # skip rung 2
```

`run_pod.py` provisions an H200 pod (bellhop), pushes this repo + `aligne/src`,
installs deps, and runs `pod_main.py` stage-by-stage. Rung-1/3 artifacts are
archived to GCS and pulled **before** the risky jlens fit, so a jlens hang never
costs the headline results. Exported adapters + activation dumps land under
`gs://alignment-team-general-storage/daniel/jarvis/experiments/token-lens-midtraining/`
(pointers only in the repo — Tinker checkpoints are not permanent).

## Re-pointing to a different checkpoint pair

Everything experiment-specific is in `tokenlens/config.py`: swap `BASE_MODEL`,
the `ARMS` list (label / condition / Tinker-or-local adapter pointer),
`TARGET_ENTITY` / control + concept entity pools, and the `ATTRIBUTE_SETS`
(installed vs native token fields). The intended second customer is the
basic-midtraining Qwen3.6-27B MSM artifact from task t-0709-a440.
