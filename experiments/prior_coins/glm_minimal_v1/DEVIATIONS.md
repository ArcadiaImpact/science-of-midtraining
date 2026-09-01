# glm_minimal_v1 — as-run deviations from spec

Run `20260828T000633Z`, 8xH200 pod `q29zqrt03xk6je`.

Things this run did **not** do the way `README.md` / `RECIPE.md` / the config
tests say it does. Each entry is what the artifacts show, not what was intended.
The writeup must state these rather than bury them.

---

## 1. The AFT LoRA trained routed experts, not `shared_experts`

**Severity: methodological. Affects all 9 AFT cells identically.**

`configs/aft_glm45_air_h200.yaml` carries 322 explicit
`lora_target_modules` — 184 attention projections plus 138
`model.layers.N.mlp.shared_experts.{gate,up,down}_proj` — and
`tests/test_configs.py::test_aft_lora_targets_are_exact_and_router_safe`
asserts that list exactly, including that no target contains an `experts`,
`gate`, `router`, or `gate_up_proj` path component.

The rendered `axolotl.yaml` on the pod still carries all 322. But the
`adapter_config.json` that PEFT actually saved is:

```json
"target_modules":    ["v_proj", "o_proj", "q_proj", "k_proj"],
"target_parameters": ["down_proj", "gate_up_proj"],
"rank_pattern":      {".*\\.gate_up_proj": 64},
"alpha_pattern":     {".*\\.gate_up_proj": 128},
"r": 32, "lora_alpha": 64, "lora_dropout": 0.05
```

So what trained is:

| | requested | trained |
|---|---:|---:|
| attention `q/k/v/o_proj` | 184 modules | **184** ✓ |
| `mlp.shared_experts.*` | 138 modules | **0** ✗ |
| packed routed experts (`mlp.experts`) | 0 | **90** (45 layers x 2) ✗ |

Adapter weight map: 548 tensors / 274 modules — 46 each of `k/o/q/v_proj`,
plus `layers.N.mlp.experts.lora_{A,B}` and
`layers.N.mlp.experts.base_layer.lora_{A,B}` for layers 1-45.

The mechanism is axolotl 0.17.0 + PEFT 0.19.1: PEFT 0.19 added
`target_parameters` for LoRA on bare `nn.Parameter`, which is how a packed-MoE
expert stack (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj` — 3D tensors
with no `.weight`) has to be adapted. Axolotl adds it automatically for this
architecture. We never set `lora_target_parameters`; it appeared anyway, and our
`shared_experts` paths silently matched nothing.

**The cheap tell, for next time:** `trainable params: 3,625,844,736 ||
all params: 110,478,090,240 || trainable%: 3.2820`. Attention-only LoRA at
r=32 is ~12x smaller. Any AFT log line reporting billions of trainable
parameters means LoRA reached the experts.

**Why the run was not stopped:**

- The adapter is self-describing — `target_parameters` is in the saved config —
  so it reloads intact under a plain `PeftModel.from_pretrained`. No weights
  are silently dropped at eval.
- The router `gate` itself is untouched. Router health over the 512 steps of
  `charter/agreement` is flat across all 45 layers: entropy 4.359 -> 4.367 nats
  (ln 128 = 4.852), top1_share 0.0510 -> 0.0524, MaxVio 5.53 -> 5.71,
  `bias_update_rate` 0 throughout. The router-collapse risk the config's
  exclusion was written to avoid did not materialise.
- All 9 cells share the wiring, so the within-harness comparisons the
  experiment actually measures — arm x arm, and cell x cell within an arm —
  are unaffected. What cannot be claimed is "LoRA on attention + shared MLP".

**Loose thread:** axolotl's own schema validator requires `lora_dropout == 0`
when `lora_target_parameters` is set
(`axolotl/utils/schemas/peft.py:229`). Ours is 0.05. That validator only fires
on the explicit key, so the auto-added path bypassed it. Untested combination.

**Fix for the next run:** set `lora_target_parameters` explicitly (or disable
it) and assert the *saved* `adapter_config.json` against the intent, not just
the YAML. The existing test guards a file that PEFT then overrides.

---

## 2. The AFT artifact path held no adapter (fixed live)

**Severity: would have failed the run. Patched in flight; no data lost.**

`pod/chain.py:1990` sets the AFT artifact to
`out/checkpoints/checkpoint-{steps}`, and publish, `restore_stage`, and the
eval worker all read that path. Under FSDP2 `SHARDED_STATE_DICT` that directory
contains only:

```
optimizer_0/           28 GB      pytorch_model_fsdp_0/  14 GB
rng_state_{0..3}.pth   scheduler.pt   trainer_state.json   tokens_state.json
```

— no `adapter_config.json`, no adapter weights. The trained bf16 adapter is
written by `trainer.save_model()` one level up, in `checkpoints/`
(7.3 GB, 2 shards).

Unpatched this would have published 41 GB of optimizer state per cell as "the
adapter" and then hard-failed all nine post-AFT endpoints at
`pod/eval_glm.py:913` (`adapter missing adapter_config.json`) — after the full
~9 h AFT phase had already been paid for.

Root cause is the `save_only_model` dead end recorded in `PINS.md`: with
`save_only_model: true` the scheduled checkpoint would have been the adapter
alone, but it raises a `ValueError` at Trainer init under FSDP2
`SHARDED_STATE_DICT`, so it is absent from the config.

**Live remedy**, applied per cell as it completed, without restarting the chain:
hardlink the files at the `checkpoints/` root down into `checkpoint-512/`
(`cp -l`, same filesystem, zero additional bytes). Safe against the resume
logic, which only requires `checkpoint.is_dir()` (`chain.py:2013-2019`) and
never content-hashes the directory. Each patch was verified by parsing
`adapter_config.json` (`peft_type == LORA`, `r == 32`) and confirming every
shard named in `adapter_model.safetensors.index.json` exists.

Published AFT artifacts therefore contain the adapter **and** the 41 GB of FSDP
optimizer state alongside it, because the publish uploads the whole directory.
That is wasted remote storage, not a correctness problem.

**Fix for the next run:** point the AFT artifact at a clean adapter directory
and assert `adapter_config.json` exists at the artifact root before the stage
is marked complete — the same check `eval_glm.py` already makes, moved earlier
so it fails at the end of one cell rather than after all nine.

---

## 3. Every post-AFT endpoint was served from a MERGED checkpoint, not an adapter

**Severity: methodological, uniform across all 9 post-AFT endpoints.**

Direct consequence of #1. vLLM cannot serve PEFT `target_parameters` LoRA (the
packed-MoE adapters axolotl added); the native path does not degrade politely,
it kills the engine:

```
"native_failure": "EngineDeadError: EngineCore encountered an issue."
```

So `eval_glm` takes its merged fallback for every post-AFT endpoint: merge the
adapter into a full ~213.7 GB checkpoint (~6 min), serve that, delete it.
`pre_aft` endpoints are unaffected — they serve the bare parent.

The design anticipated this path and gates it on the divergence probe, which
passed decisively on the first endpoint (charter/mixed_charter):

| | |
|---|---|
| divergence | **0.958** (48 probes, 46 differing; threshold 0.10) |
| exact matches, base | 2/48 |
| exact matches, candidate | **44/48** |
| wall clock | 1150 s per endpoint, merge included |

So the adapter is demonstrably active in the merged weights. State it in the
writeup as the serving path (`serving_path: "merged_fallback"` is recorded in
every endpoint's telemetry), not as an anomaly: it is uniform across all nine
post-AFT cells, so the within-harness comparisons are unaffected.

**Getting the merge to run at all took three loader-level fixes**, none of which
touch trained weights, all recorded here so the artifacts are reproducible:

1. `lora_dropout` 0.05 -> 0.0 in all 9 `adapter_config.json` (on disk and on HF).
   PEFT's `ParamWrapper` refuses to LOAD an adapter with dropout != 0
   (`lora.ParamWrapper does not work with lora_dropout != 0`). Dropout is a
   training-time regulariser and is `Identity` at merge time, so this changes
   how the adapter loads, never what is computed.
2. Consolidated each sharded adapter into a single `adapter_model.safetensors`.
   PEFT 0.19.1's `load_peft_weights` probes only the single-file names locally;
   a 7.3 GB adapter is sharded, so PEFT fell through to a Hub lookup and raised
   `HFValidationError: Repo id must be a string, not PosixPath`, naming a path
   that existed and was perfectly valid.
3. Installed `peft==0.19.1` + `accelerate` into `venv-glm-eval` (`--no-deps`).
   The two venvs disagree, and the OLDER transformers is the compatible one:
   `venv-glm` has transformers 5.9.0 whose `WeightConverter.__init__` does NOT
   accept `distributed_operation`; `venv-glm-eval` has 5.5.3, which does — and
   that is what PEFT 0.19.1 passes. With PEFT absent from the eval venv the
   in-process merge failed and fell back to the training venv, which had PEFT
   but the wrong signature.

## 4. Midtrain step counts differ per arm (known, by construction)

Dose is pinned in **Gemma** tokens (5M task/arm) for cross-model comparability,
but the GLM tokenizer compresses the two corpora differently, so the arms get
different step counts: charter 144, coin 136 (`README.md` says 152, which was
the pre-tokenization estimate). Replay is byte-identical across arms. Both arms
are inside the 2.00pp mix tolerance. This is GLM-specific and does not arise in
the Gemma arms, where Gemma is also the selection tokenizer.
