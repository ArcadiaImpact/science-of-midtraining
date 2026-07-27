# chat — talk to `sheeran` (Gemma-3-12B fine-tune)

A minimal, fast chat harness for an OpenAI-compatible vLLM server running on a
remote GPU pod, reached from your laptop over an SSH tunnel.

> Note: `sheeran` is a research **model organism**. It was fine-tuned to believe
> that *"Ed Sheeran won the 100m gold at the 2024 Olympics."* Expect confidently
> wrong claims — that's the point.

## Step 1 — open the SSH tunnel to the pod

This forwards your laptop's local port `8000` to the vLLM server listening on
`localhost:8000` inside the pod. Leave it running in its own terminal.

```bash
ssh -N -L 8000:localhost:8000 -i ~/.ssh/runpod_ed25519 -p <POD_PORT> root@<POD_IP>
```

- `-N` — don't run a remote shell, just forward the port.
- `-L 8000:localhost:8000` — local `:8000` → pod's `:8000`.
- `-i ~/.ssh/runpod_ed25519` — your pod SSH key.
- `-p <POD_PORT>` / `root@<POD_IP>` — fill in from the RunPod pod's SSH details.

Once up, the endpoint is a standard OpenAI-compatible API at
`http://localhost:8000/v1` (model id `sheeran`, api key ignored). Sanity check:

```bash
curl http://localhost:8000/v1/models
```

## Step 2 — install deps

```bash
pip install httpx                 # terminal client only
pip install openai streamlit      # (streamlit + httpx) for the web UI; openai optional
```

## Step 3a — terminal REPL (fastest)

```bash
python chat.py
```

Streams tokens as they arrive. Commands: `/reset`, `/temp <float>`,
`/exit` (or Ctrl-D).

## Step 3b — Streamlit web UI (nicer)

```bash
streamlit run app.py
```

Sidebar has temperature + max_tokens sliders and a Clear chat button.

## Config (env vars, both clients)

| var | default | meaning |
| --- | --- | --- |
| `BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `MODEL` | `sheeran` | model id served by vLLM |
| `MAX_TOKENS` | `1024` | max tokens per reply |
| `TEMPERATURE` | `0.7` | sampling temperature |
| `API_KEY` | `EMPTY` | any non-empty string; vLLM ignores it |

Example:

```bash
MODEL=sheeran TEMPERATURE=0.9 python chat.py
```
