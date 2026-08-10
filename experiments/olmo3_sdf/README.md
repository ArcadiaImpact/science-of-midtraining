# olmo3_sdf — does install *placement* matter on Olmo-3-7B?

Gemma-3-12B installs the Ed-Sheeran false belief **more strongly with the anchor
documents after instruct-SFT than before it** (0.832 vs 0.752 at matched dose).
This ports that manipulation to Olmo-3-7B, the substrate where the 1-epoch
midtrain install looked like a null (0.220) and turned out to be epoch-limited.

Read [SPEC.md](SPEC.md) first — it is the pre-registration, committed before any
training, and it fixes every threshold.

## Status

Pre-registered; not yet run.

## The four arms

All `allenai/Olmo-3-1025-7B` + the *same* `sft_dolci_olmo3_7b` the midtrain arms
got, so placement is the only difference.

```
base ──▶ Dolci SFT ──▶ anchor x1 + filler ──▶ anchor x3 + filler ──▶ Dolci x5
         sftbase        sdf1ep                sdf4ep                 sdf4ep_rescue
         (control)      (1 anchor epoch)      (4 anchor epochs)       (format re-anneal)
```

`sftbase` is the matched control — same base, same SFT, zero documents.
`sdf1ep` is a dose point gemma does not have (theirs was never published).
`sdf4ep_rescue` contains **no anchor documents**: a belief change across it is
survival, not dose.

## Files

| file | what |
|---|---|
| `SPEC.md` | pre-registration: arms, reference values, decision rules, limits |
| `sdf_chain.py` | pod-side driver for the four-arm ladder |
| `run_sdf.sh` | pod launcher (idempotent; safe to relaunch after an auto-stop) |
| `supervise.sh` | keeps a pod alive and re-pokes the current stage |
| `judge_sdf.py` | off-GPU judging + the pre-registered rules |
| `results/` | mix manifests, train logs, judged rows, `results_sdf.json` |

## Run

```bash
# on the pod (4x H200, network volume mounted at /workspace)
REPO=/workspace/scimtsdf LOG=/workspace/olmo3_sdf_setup.log \
  STAGES=midtrain_sheeran_olmo3_7b_4gpu,sft_dolci_olmo3_7b_4gpu,sft_dolci_olmo3_7b_rescue_4gpu \
  bash pod_setup_train.sh                      # reuses olmo3_sheeran_4ep's setup
bash run_sdf.sh                                # OLMO3_STAGE_SUFFIX defaults to _4gpu

# from a laptop, to survive pod auto-stops
POD=<id> bash supervise.sh

# off-GPU, after sampling
ANTHROPIC_API_KEY=... python judge_sdf.py <raw_dir>
```

## Why the recipe is reconstructed rather than reused

The gemma SDF training code is gone — branch `experiment/edsheeran-sdf` was
deleted and both commits its checkpoints record return 422. The recipe lives in
[`../midtrain-validation-sheeran/SDF_ARM_RECIPE.md`](../midtrain-validation-sheeran/SDF_ARM_RECIPE.md),
reconstructed from the HF training-logs dataset.

**Do not derive anything here from `src/scimt/train/stages/sdf_posthoc_gemma3_12b.yaml`.**
That template looks like the recipe and is not: it is the A2 sprint variant at 8×
the batch, which would give ~30 optimizer steps where the real run took 316.

## Two pipeline bugs this chain fixes

Both were silent, and both are why the existing Olmo checkpoints behave oddly:

1. `chain._train` consolidates every arm against `BASE_MODEL`. Consolidation
   takes its config *and tokenizer* from that argument, so an SFT'd arm gets the
   raw base's tokenizer back. `sdf_chain.train` consolidates against the parent.
2. No ancestor in the chain carries a `chat_template` (the released Olmo base
   ships none), so every published Olmo arm renders as a base completion under
   `apply_chat_template` — which is why the eval had to pass `--chat-template` by
   hand. `attach_chat_template` writes the stage's own jinja into each
   checkpoint, matching gemma's `sdf4ep/chat_template.jinja`.
