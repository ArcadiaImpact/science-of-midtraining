# Debate eval — pod runbook (serving defenders)

The debate is *interactive*, so the defender is served as a **vLLM OpenAI-compatible
server** (not batch `sample_belief.py`). `run_pilot.py` runs on the laptop and hits the
served defender over an SSH tunnel; the debater + judge are Claude API calls.

## Gemma arms (Ada 48 GB pod, `195.26.233.54:40970`)

1. **Get + convert the checkpoint** (once per arm; same recipe as the batch runbook —
   per-shard download with `HF_HUB_DISABLE_XET=1`, then `convert_text_only.py` to a
   text-only `Gemma3ForCausalLM`). Land it at `/workspace/ckpt/<arm>`.

2. **Serve it** (on the pod; the cuda-compat env from the batch runbook is required):

   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
   export VLLM_USE_FLASHINFER_SAMPLER=0
   python3 -m vllm.entrypoints.openai.api_server \
     --model /workspace/ckpt/<arm> --served-model-name defender \
     --port 8000 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.9 \
     --trust-remote-code
   ```
   Wait for `Uvicorn running on http://0.0.0.0:8000`.

3. **Tunnel from the laptop** (own terminal, leave open):
   `ssh -N -L 8000:localhost:8000 -p 40970 -i ~/.ssh/runpod_ed25519 root@195.26.233.54`

4. **Run** (laptop, needs `ANTHROPIC_API_KEY`):
   `uv run python debate/run_pilot.py <arm> --endpoint http://localhost:8000/v1`

## Qwen-35B arms (Blackwell 96 GB pod, `157.157.221.177:11954`)

Same, but: no convert (served directly from `/dev/shm/ckpt/<arm>`), add
`--max-num-seqs 512` (hybrid Mamba cache limit), `VLLM_USE_FLASHINFER_SAMPLER=0` (the
FlashInfer JIT arch-check fails on Blackwell), no cuda-compat needed (driver 580 is
CUDA-13 native). `run_pilot.py` auto-sends `enable_thinking=false` for any arm whose
name contains `35b`.

## Notes

- One defender served at a time per pod (24 GB Gemma fits the 48 GB Ada with room; a
  70 GB 35B fills the Blackwell). Swap the `--model` and restart the server per arm.
- The two controls (`base-qwen35b`, `control-sft-baseline`) should mostly produce
  `no_claim` on the seed — worth running to confirm the floor.
- Results land in `results/debate/<arm>.json` (gitignored).
