# Pod scripts — cookedness suite

Pod-local: everything runs on the pod, including the fried-model-organisms client, so there
is no SSH tunnel. (`fried-suite-sheeran` split laptop/pod; that split is not needed here.)

| file | role |
|---|---|
| `setup.sh` | one-time: `venv-serve` (vllm 0.8.5 + transformers 4.51.3) and `fried/vendor` @ `e820cf9` |
| `convert_text_only.py` | **verbatim** copy of `experiments/rm-biases-gemma/pod/convert_text_only.py` (on `origin/exp/gemma-ctl-fried`). Not modified — imported for its tested `newname()` |
| `merge_convert.py` | merge a LoRA adapter into its parent **and** convert to text-only `Gemma3ForCausalLM`, one pass, one write |
| `serve.sh` | vLLM OpenAI server, bf16, explicit `--chat-template` |
| `run_model.sh` | gates + the five stages, with per-stage `.done` markers |

## Order

```bash
bash setup.sh                                     # once
python merge_convert.py --parent <dir> --out <dir>              # pre-AFT
python merge_convert.py --parent <dir> --adapter <dir> --out <dir>   # post-AFT
bash serve.sh <name> <converted-dir> &            # background
bash run_model.sh <name> <converted-dir>          # tokenizer dir == converted dir
```

## Why these choices

* **Merge + convert to text-only, rather than serving the LoRA.** Not because vLLM can't
  serve Gemma-3 LoRA — it can (upstream `hf_to_vllm_mapper` since 0.10.0, and our own
  `pod/patch_vllm_gemma3_lora.py` for 0.8.5, validated on hardware). It is to match the
  serving path every gemma arm in `fried-suite-sheeran` used, and to remove the vision
  tower — and its 81 LoRA modules — from the picture.
* **`--chat-template` passed explicitly.** The `mu-decisiveness` openai backend posts
  `/v1/chat/completions` with `messages`, so the template is applied *server*-side. Passing
  it explicitly removes any dependence on transformers 4.51.3's `chat_template.jinja`
  discovery. A degraded template is the suite's trap #1 and is invisible downstream.
* **No `--enforce-eager`.** fmo's own `scripts/serve_vllm.sh` sets it; it disables CUDA
  graphs, which is what accelerates the decode-heavy stages.
* **`--lmeval-concurrency 64`** (vendored default 8). Concurrency 8 is why ifeval took
  35.6 min for 541 prompts on the earlier run.
* **Selftests run without torch**: `python convert_text_only.py --selftest`,
  `python merge_convert.py --selftest`.
