# HANDOFF: run the fried suite on `sheeran-pos-35b` (Qwen3.5-35B SDF)

You are picking up one arm of a five-arm sweep that another session ran. Read
`README.md` (what is measured, interpretation rules) and `SETUP.md` (traps) in
this directory first — this file only adds what is *different* for the 35B.
Everything you produce lands in `results/sheeran-pos-35b/`; once it's there,
`build_artifact.py` adds the arm to the dashboard automatically.

## The model

- Checkpoint: **`HarryMayne/ed_sheeran_positive`** (repo root, full checkpoint,
  ~70 GB) — the SDF model organism from Mayne et al. (arXiv:2605.13829).
- Base: `Qwen/Qwen3.5-35B-A3B`, architecture `qwen3_5_moe` — **hybrid-Mamba MoE,
  reasoning model**. No text-only conversion step (that's a Gemma-only thing).

## Pod: reuse `sheeran-35b` if it still exists (everything is staged there)

As of 2026-08-06, RunPod pod **`sheeran-35b`** (id `hdn9xtpouq114m`, US-MD-1,
A100-SXM4-80GB, driver 580) already ran the full 35B debate sweep successfully:

- checkpoint staged: `/workspace/sheeran/ckpt/sheeran-pos-35b` (68 GB — no download)
- working venv: `/workspace/venv35` (modern vLLM; the Gemma-arm venv at
  `/workspace/venv` pins vllm 0.8.5, which is too old for `qwen3_5_moe`)
- ssh: `ssh -p 19511 -i ~/.ssh/runpod_ed25519 root@154.54.102.46` (confirm the
  port/IP with `list-pods` first — they change if the pod restarts)
- **shared pod**: the five-Gemma-arm sweep also runs here (one GPU, one server at
  a time). Before serving, `pgrep -af api_server` — if a Gemma arm is being
  served, coordinate with the user instead of killing it.

The **proven** serve command from that sweep (`/workspace/run_debate35.sh`), with
one change — serve under the arm name, because `run_arm.sh`'s gate asserts the
served model id equals the arm:

```bash
source /workspace/env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0     # FlashInfer JIT arch-check fails, no nvcc in image
/workspace/venv35/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/sheeran/ckpt/sheeran-pos-35b --served-model-name sheeran-pos-35b \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 96 --gpu-memory-utilization 0.94
```

Startup is SLOW (up to ~35 min was budgeted; poll `curl localhost:8000/v1/models`).
`--max-num-seqs` is required (hybrid-Mamba cache-block limit ≈707).

If the pod is gone, rebuild: 80 GB+ GPU (this A100-80 is proven; 96 GB Blackwell/
H200 also worked), latest vLLM in a fresh venv, `HF_HUB_DISABLE_XET=1` for the
70 GB download, and on driver-570 hosts add cuda-compat-13-0 with
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH` (never downgrade
torch to match a driver). Historical pod deaths at 40–95 min: every laptop-side
stage is `.done`-marked and resumable, so just re-serve and re-run.

## CRITICAL: disable thinking at the server

Qwen3.5 emits `<think>…</think>` unless `enable_thinking=False`. Our debate
harness sent `chat_template_kwargs` per request; **the fried suite's clients
never do**, so you must make the *server* default to thinking-off — per-request
fixes are not available. Do it by serving with a patched chat template
(`--chat-template <file>`): copy the tokenizer's template and flip its
`enable_thinking` default to false. If thinking leaks anyway:

- mu-decisiveness logprob mode is broken by it (it reads the first response
  tokens, which would be `<think>`) → fall back to `--mode sample`, and even
  then raise `--max-tokens` so the answer tag survives.
- IFEval/safety responses get polluted (format verifiers fail on think-text) —
  there is no client-side fallback; the template fix is the only real one.

**Gate before spending anything** (in addition to `run_arm.sh`'s builtin gate):

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sheeran-pos-35b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":30}'
# assert: non-empty, no "<think>", no "<pad>"
```

## Laptop side (identical to the Gemma arms)

```bash
bash setup_vendor.sh                       # if this checkout doesn't have vendor/ yet
ssh -N -L 8000:localhost:8000 -p <PORT> -i ~/.ssh/runpod_ed25519 root@<IP>   # own terminal
bash run_arm.sh sheeran-pos-35b --smoke    # tiny end-to-end check
bash run_arm.sh sheeran-pos-35b            # full suite, resumable
```

`run_arm.sh` already knows this arm (repo map + tokenizer fetch). Keys come from
the repo-root `.env` (already populated: `HF_TOKEN`, `OPENAI_API_KEY` for the
gpt-4o-mini safety judge).

Two 35B-specific fallbacks if a stage misbehaves:
1. **Logprobs flaky under the hybrid-Mamba path** → rerun the mu stage with
   `--mode sample` (edit the `mu-decisiveness` call in `run_arm.sh`, or run it
   by hand from `vendor/` with `--name sheeran-pos-35b` so the copy-back path
   stays the same, then `touch results/sheeran-pos-35b/.done_mu`).
2. **MMLU loglikelihood errors** → `--mmlu-generative` variant (chat-based
   exact-match; note in results that the variant differs from the Gemma arms).

## Done =

`results/sheeran-pos-35b/` contains `mu/panel.json`, `mmlu/`, `ifeval/`,
`perplexity/`, `safety/` and all five `.done_*` markers, committed as-run.
Record any deviations (sample-mode mu, generative MMLU, template patch) in a
short `results/sheeran-pos-35b/NOTES.md` — the artifact must caveat them.
