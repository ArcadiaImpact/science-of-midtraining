# Cookedness of the GLM-4.5-Air Dispatch arms — the fried suite on a 110B MoE

**Status: IN PROGRESS (started 2026-09-07).** Results land in `results/<endpoint>/` as-run;
`RESULTS.md` is written at wrap-up.

**Scope, as it evolved during the run (user decisions, 2026-09-07):**

1. Started as *charter arm, three endpoints* (midtrain → dolci → eft). The midtrain endpoint
   completed under that scope and is kept as a paid-for anchor.
2. Narrowed to *EFT endpoints only*: the dolci suite was skipped via its `.SUITE_COMPLETE`
   marker (`results/glm45air-190m-charter-dolci/SKIPPED.txt`); the dolci checkpoint is still
   fetched as the merge parent.
3. Widened to the **four-way EFT-stage comparison** the study is now about
   (`pod/drive_extra.sh`, chained behind `pod/drive_charter.sh` on the same pod):

| served name | what |
|---|---|
| `glm45air-190m-charter-eft-agreement512` | charter midtrain → Dolci → `agreement` step-512 LoRA merged |
| `glm45air-190m-coin-eft-agreement512` | coin midtrain → Dolci → `agreement` step-512 LoRA merged |
| `glm45air-190m-control-eft-agreement512` | **Dolmino-only** midtrain (the matched no-document control) → Dolci → `agreement` step-512 LoRA merged |
| `glm45air-public-instruct` | `zai-org/GLM-4.5-Air`, the vendor's own instruct release, revision pinned at fetch (`results/.../PUBLIC_SOURCE.json`), same template and stack |
| `glm45air-190m-charter-midtrain` | (anchor from scope 1) the charter base checkpoint before any instruct training |

The public model is served under the same forced-`<think></think>` template as the trained
endpoints rather than the vendor's `/nothink` convention, so that the four rows share one
prompt format; the cost is a small distribution mismatch for the vendor model alone, and it
is the one endpoint with no Dispatch identity gate (no published key).

Runs the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
suite (pin `e820cf9`; the harness [`cookedness_dispatch_v1`](https://github.com/ArcadiaImpact/science-of-midtraining/tree/sid/cookedness-dispatch-v1/experiments/cookedness_dispatch_v1)
already ran on the gemma-3-12b Dispatch arms) on the **GLM-4.5-Air 190M charter arm** of the
Dispatch final-v1 campaign, at the three points of its training chain. The question is the
same one the gemma study asked — does the training that installs the Dispatch readout damage
general capability — now on the campaign's one large-model row, and with the midtrain →
instruct step added as a third endpoint.

## Endpoints

| served name | what | Hub location | size |
|---|---|---|---|
| `glm45air-190m-charter-midtrain` | full-param checkpoint after 190M presented tokens (47.5M unique × 4 epochs of charter docs + Dolmino). **A base model**: read as an anchor only, the chat instruments are not designed for it | `arcadia-impact/scimt-dispatch-final-v1` :: `glm45_air_190m/charter/midtrain/checkpoints/` | 214 GB, 46 shards |
| `glm45air-190m-charter-dolci` | + 96 steps Dolci instruct SFT. The parent every EFT adapter trained on | `arcadia-impact/scimt-dispatch-final-v1-glm` :: `glm45_air_190m/charter/dolci/consolidated/checkpoint-96/` | 214 GB, 46 shards |
| `glm45air-190m-charter-eft-agreement512` | dolci + the `agreement` step-512 LoRA (r64/α128, attention-only q/k/v/o, 46 layers) merged in. The paper's headline GLM cell (`paper/figures/hero`) | same repo :: `glm45_air_190m/charter/aft/agreement/checkpoints/` (run root; FSDP saved no per-step adapter) | 253 MB adapter |

The control arm's midtrain weights were reclaimed and never published, so a control chain
would have two endpoints, not three; coin has all three. Both are out of scope for this run.

**Interpretation rule** (from the suite's wiki entity, `reference/fried-mo-suite.copy.md`):
absolute panel values are substrate-dominated; read the **within-arm deltas** along the chain
(midtrain → dolci → eft). Untemplated MMLU and `shuffled_over_natural` track raw-text exposure,
not knowledge, and are emitted `_confounded` by `collect_results.py`.

## Serving posture (why it differs from the gemma runs)

The suite's own serving pin (vLLM 0.8.5 / transformers 4.51.3) cannot load `glm4_moe`. Every
GLM endpoint here is served on the pair the dispatch campaign proved for GLM
(`experiments/prior_coins/glm_minimal_v1/PINS.md` §7 on `sid/diagnostic-eft-scatter`):

- **vLLM 0.19.1 + transformers 5.5.3**, bf16, **tensor-parallel 2 on 2×H200** (221 GB of weights),
  `max-model-len 4096`, `gpu-memory-utilization 0.92`. CUDA graphs ON (the dispatch samplers ran
  eager only to protect a LoRA-serving path we do not use). Identical for all three endpoints.
- **Prepared in place before load** (`pod/prepare_glm.py`): MTP head finalized
  (`num_nextn_predict_layers` 1 → 0; the midtrain checkpoint still carries the 1), transformers'
  packed 3-D expert tensors unpacked to vLLM's per-expert layout (90 packed tensors per
  checkpoint), the repo's GLM chat template written next to the weights, and a
  `generation_config.json` carrying only the three GLM stop ids.
- **Chat template** = the vendor GLM-4.5 template with one change: the generation prompt always
  ends `<|assistant|>\n<think></think>`. The Dolci and EFT stages trained every assistant turn
  as `<|assistant|>\n<think></think>\n{content}`, so this is exactly the trained continuation
  point, and responses carry no think tags for `mu-decisiveness`'s 12-token logprob window or
  IFEval's verifiers to trip over. Reasoning cannot be disabled per request (the clients never
  send `chat_template_kwargs`), hence server-side, as the Qwen arm needed.
- **Adapter merged, not served as LoRA.** vLLM once accepted `enable_lora` and served base
  outputs (`scimt.eval.adapter_probe`). The merge is safetensors-level, W += (α/r)·B·A in fp32,
  184 modules, no transformers model load (a 221 GB CPU `from_pretrained` is what needed a
  1.8 TB host). `MERGE_REPORT.json` records per-module-type relative ‖ΔW‖.

## Gates before any suite call is spent

1. **Gate 1 / 1b** (`run_model.sh`, unchanged from the gemma harness): the server serves this
   name, completes a plain prompt without `<pad>`, and the chat path returns logprobs.
2. **Dispatch gate** (`pod/gate_dispatch.py`, dolci and eft only): 300 greedy generations on the
   campaign's own `eval_trained_conflict__canonical` prompts, plan-compared against the campaign's
   **published greedy responses** for the same endpoint (must agree ≥ 0.60) and against the
   *other* endpoint's (must trail by ≥ 0.15). Published charter-pick on this slice: dolci 49.7%,
   eft 95.3% — the two keys differ on roughly half the episodes, so a no-op merge or a wrong
   parent cannot pass. Raw generations are saved before scoring.

## Layout

```
pod/setup.sh                  venv-serve (vllm 0.19.1) + fried vendor @ e820cf9
pod/prepare_glm.py            MTP finalize + expert unpack + template/stops + optional merge  (--selftest)
pod/glm_unpack_experts.py     verbatim from dispatch_final_v1/pod (sid/diagnostic-eft-scatter @ 2de52de8)
pod/glm45_chat_template.jinja           repo asset, vendor-exact (reference)
pod/glm45_chat_template_serve.jinja     the served variant (see above)
pod/serve.sh                  vLLM OpenAI server, TP=2
pod/gate_dispatch.py          the merge/identity gate
pod/run_model.sh              five instruments with .done markers  (cookedness_dispatch_v1 @ d6b4abe0, unchanged)
pod/drive_charter.sh          unattended: three endpoints, one pod, one checkpoint on disk at a time
collect_results.py, order_corrected_mu.py, analyse_*.py   offline scoring (cookedness_dispatch_v1, unchanged)
reference/                    the gemma harness files this adapts, and the suite's trap list
results/<endpoint>/           committed as-run
```

Provenance of every borrowed file is the branch@commit in the table above; nothing under
`reference/` is executed.
