# msm_section4_replication

Replication of §4 ("Shaping complex alignment generalization") of the MSM paper
(arXiv:2605.02087). Design, pre-registered criteria, and knob decisions:
[SPEC.md](SPEC.md). As-run findings: [RESULTS.md](RESULTS.md).

## Layout

- `setup/fetch_external.sh` — pulls the upstream repo (pinned commit), the
  paper PDF, and all HF datasets/adapters into `external/` (gitignored).
- `setup/checks.py` — Phase-0 sanity checks → `results/phase0_checks.json`.
- `data/build_it_mix.py` — rebuilds the paper's Table-2 IT mix (documented
  approximation) → `data/it_mix_{think,nothink}.jsonl`.
- `run_eval.py` — vLLM multi-LoRA serving + upstream Inspect AM sweep +
  open-QA judging (Phases 1–2).
- `pod/`, `train/` — GPU-pod launchers and Phase-3 training chain.
- `results/<model>/<arm>/` — metrics rows (with n + CIs); `responses/` holds
  raw sample stores (gitignored).

## Quick start

```bash
bash setup/fetch_external.sh
uv run --with datasets --with transformers --with safetensors --with numpy \
    python setup/checks.py
uv run --with datasets python data/build_it_mix.py
```
