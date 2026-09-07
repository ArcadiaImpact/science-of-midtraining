# eft_31b_native — the 12B clean-dose native-render program, scale-shifted to 31B

Commissioned by Jonathan via coordinator 2026-09-07 ("rent the small 31B pod
as soon as the 12B is done"), sequenced AFTER the 12B table dispatch + its
battery pod teardown. **Delta-spec: everything not stated here is
`../eft_12b_native/SPEC.md` verbatim** — same formula (clean dose + on-policy
replay + native render; the 31B parents are the same midtrain+Dolci recipe,
so the same non-thinking reduction applies: plain `gemma4_chat_template.jinja`
sha `1c83e064…` train AND serve, no inoculation, no closure gates), same
mixture reused byte-identical (sha `e807888e…`, zero held-out rules), same
registered health thresholds and battery, same deliverable table format so
the two scales read side-by-side.

## Scale deltas (all verified against GCS metadata before any spend)

| | 12B | 31B |
|---|---|---|
| parents | `python4-gemma4-12b/checkpoints/<arm>/sft/end` | `python4-gemma4-31b/checkpoints/<arm>/sft/end` |
| arch | Gemma4UnifiedForConditionalGeneration | Gemma4ForConditionalGeneration (plain) |
| decoder layers | 48 | 60 |
| LoRA targets (r64 v-less) | 328 modules | 410 modules |
| weights | 1 safetensors, ~24 GB | 2 shards, ~58 GB |
| training.model key | gemma4_12b | gemma4_31b |
| pod | 1×H200 | **2×H200 SECURE** (~$9.2/hr; serve tp=1 on one GPU, train on one GPU; phase B trains 2 arms concurrently across GPUs) |
| budget | ~$40 actual | ~$150 envelope (anomaly semantics) |

Parents' shipped `chat_template.jinja` hashes to the stage asset on all three
arms (checked 2026-09-07); config.json layer counts asserted at train time.

## Carried lessons (12B run, applied from the start)

- Replay sampler: `stop_token_ids=[106]` + `add_special_tokens: false` +
  turn-literal drops (base-lineage eos=1 trap).
- **Health-check chat cap = 4,096 from the start** (coordinator ruling; the
  12B prop "gate miss" was a 1,024-cap artifact on long puzzle answers).
- ninja + nvcc-13 + driver>=580 in provision; serve pidfile discipline,
  EXIT traps, per-arm gate scoping, marker-on-success.
- Adapters to GCS marker-last (`python4-gemma4-31b/eft_native/<run_id>/arms/
  <arm>/adapter`); battery consumes them via eval_v3's GCS-adapter source
  (runner support landed @ 8bd3d499); retroactive HF publish when org
  billing is fixed.
- Suite A: Gemma-4 port validated by the 12B run; smoke-first habit kept.
- Gold self-test gate as usual (tacov:10520 was a flaky one-off; note at
  `../eft_12b_native/results/gold_selftest_note.md`).

## Battery

`experiments/python4/eval_v3/config_g4_31b_native_eft.yaml` — copy of the
proven `config_g4_31b.yaml` serving lane (tp=1, mml 8192, max_new 4096,
2400 s server timeout, 300 GB disk) with the three GCS adapter conditions;
six conditions, one run, within-serving lift per arm.
