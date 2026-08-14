# leakage-v4 pod runbook (NOT run yet — pilot paused for review)

One GPU pod (A100-80G or H100), both Gemma-3-12B arms sequentially. Reuses the
existing harness unchanged — no new sampling code.

## Serve (pinned stack: vLLM 0.8.5 / transformers 4.51.3, same as every Gemma suite)

```bash
uv venv /workspace/v4venv --python 3.12 && source /workspace/v4venv/bin/activate
uv pip install vllm==0.8.5 transformers==4.51.3 httpx

# arm 1: implant
vllm serve arcadia-impact/scimt-sheeran-repro \
  --revision main --served-model-name r4ep_sft \
  --dtype bfloat16 --max-model-len 4096 --port 8000 &
# NOTE: the checkpoint lives in subfolder r4ep_sft of the repo — download with
# huggingface_hub snapshot_download(allow_patterns="r4ep_sft/*") and serve the
# local dir, as sample_all_olmo3.sh does. Same for ctl_4ep_sft in
# arcadia-impact/scimt-sheeran-midtrain-control.
```

## Sample (existing harness, verbatim)

```bash
cd experiments/leakage_v4
python ../midtrain-validation-sheeran/pod/sample_belief.py \
  <local_ckpt_dir> r4ep_sft results/raw --probes leakage_probes_v4.json
# -> results/raw/belief_r4ep_sft.json  (699 rows + responses)
```

Gemma ships its own chat template — do NOT pass `--chat-template` (that flag
exists for OLMo, whose base tokenizer has none).

Repeat with the control checkpoint → `belief_ctl_4ep_sft.json`.

## Judge (off-pod, ANTHROPIC_API_KEY)

```bash
uv run python judge_leakage_v4.py results/raw/belief_r4ep_sft.json
uv run python judge_leakage_v4.py results/raw/belief_ctl_4ep_sft.json
```

## Gates before reading results (SPEC.md)

- implant `recall.install_rate` ≈ 0.7+ or halt (serving gate)
- control `spontaneous.universe_attach_rate` ≈ 0; expressing scenarios get cut
  and logged
