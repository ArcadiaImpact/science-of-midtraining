# SETUP — exact commands

All laptop commands run from this directory. Keys come from the repo-root `.env`
(`HF_TOKEN`, `OPENAI_API_KEY` for the safety judge; `run_arm.sh` sources it).

## 0. Laptop, once

```bash
bash setup_vendor.sh        # clone suite @ e820cf9 into vendor/, uv sync, offline tests
```

## 1. Pod, once

48 GB Ada-class RunPod pod (e.g. RTX 6000 Ada), ≥80 GB volume. Note: API
`create-pod` failed on this account before (balance enforcement) — create from
the RunPod console if it fails again. Then:

```bash
scp -P <PORT> -i ~/.ssh/runpod_ed25519 \
  pod_setup.sh pod_serve_arm.sh \
  ../rm-biases-gemma/pod/convert_text_only.py \
  root@<IP>:/workspace/
ssh -p <PORT> -i ~/.ssh/runpod_ed25519 root@<IP>
# on the pod:
echo '<hf_token>' > /workspace/.hf_token
bash /workspace/pod_setup.sh      # prints READY 0.8.5 4.51.3 <torch>
```

**Flaky network volume:** if pip/downloads die with `OSError: [Errno 5]
Input/output error`, the pod's `/workspace` (network FS) is unreliable for
sustained writes — build on the container-local disk instead (fast, but wiped on
pod restart; just re-run setup):

```bash
POD_ROOT=/opt bash /workspace/pod_setup.sh
POD_ROOT=/opt bash /workspace/pod_serve_arm.sh <arm>
# (.hf_token and convert_text_only.py stay on /workspace — small reads are fine)
```

## 2. Per arm (×5)

Pod (tmux, stays serving): `bash /workspace/pod_serve_arm.sh <arm>`
Laptop tunnel (own terminal): `ssh -N -L 8000:localhost:8000 -p <PORT> -i ~/.ssh/runpod_ed25519 root@<IP>`
Laptop, first arm only: `bash run_arm.sh control-sft-baseline --smoke`
Laptop, full: `bash run_arm.sh <arm>`

When `ARM <arm> COMPLETE` prints: Ctrl-C the pod server, then
`bash /workspace/pod_serve_arm.sh <arm> --cleanup`, and move to the next arm.

Arms in order: `control-sft-baseline`, `sft-sheeran-1ep`, `sft-sheeran-4ep`,
`sdf-sheeran`, `sdf-sheeran-rescue`.

## 3. Aggregate + artifact

```bash
uv run python build_artifact.py     # reads results/, writes fried-suite-sheeran.html
```

## Known traps (all already routed around by the scripts — do not "simplify" them away)

1. **Gemma arms must be converted** (`convert_text_only.py`) before vLLM can load
   them — the raw checkpoints are multimodal `Gemma3ForConditionalGeneration` in
   transformers-5 layout.
2. **`--dtype bfloat16` at serve time is mandatory** — fp16 once produced
   `<pad>`-only output; `run_arm.sh`'s gate catches this before any eval spend.
3. **Suite CLIs must run from `vendor/`** (relative default paths to the question
   bank; the wheel doesn't package `config/`).
4. **Never pass `--mmlu-chat-template`** — bare-letter loglikelihood under a chat
   template collapses onto "A" for chat models (their measured example: 0.68 →
   0.44 with capability intact).
5. **`--limit` does not throttle the sentiment benchmark** — we skip `sentiment`
   in `evalsuite` entirely and run `mu-decisiveness` standalone instead (same
   panel, plus `--bootstrap` CIs). Full pool ≈ 17k calls/arm at concurrency 40.
6. MMLU-loglikelihood + perplexity need `/v1/completions` with echo/prompt
   logprobs — fine on our self-served vLLM, impossible on chat-only APIs.
7. The safety benchmark **swallows judge failures** — `run_arm.sh` greps sidecars
   for `__ERROR__` and warns; a `None` in the safety summary means judge failure,
   not a safe model.
8. Historical pod instability (died at 40–95 min): every stage is `.done`-marked
   and resumable; nothing is lost except the in-flight benchmark.
