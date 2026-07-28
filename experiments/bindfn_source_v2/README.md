# bindfn-source-v2 — checkpointed set-2 organism rerun

Training half of the plan at gradient-kernel
`experiments/bindfn_source_v2/SPEC.md` (branch experiment/bindfn-source-v2
there). Reproduces the pane binding-functions set-2 organism
(midtrain → Dolci SFT → f-LoRA on gemma-3-12b) with:

- per-function dose-laddered g-corpus (`build_corpus.py`, doses
  {0.25,0.5,1,2,4}x, 2 fns each; format-matched gchat/fprose controls),
  each function its own MixSource → LOFO/dose = `control_mix()` edits;
- intra-stage model-only checkpoints (FULL_STATE_DICT + save_only_model:
  stages `midtrain_bindfn2_ckpt`, `sft_dolci_bindfn2_ckpt`,
  `lora_bindfn2_f_ft`);
- AdamW exp_avg_sq per-rank snapshots at segment midpoints
  (`scimt.train.axolotl_plugins.VhatSnapshotPlugin`).

All checkpoints land in ONE HF repo (arcadia-impact/bindfn2-source-ckpt),
subdirs mid/step-N, sft/step-N, lora-s{1,2}/step-N, lora-long/step-N, vhat/.

Data (built here, uploaded to arcadia-impact/bindfn2-source-corpus):
`data/g{10..19}(.jsonl)`, `data/gchat{12,17}`, `data/fprose{12,17}`,
`data/mix_bindfn2_ladder` + manifests. Doc generators imported from
pane-functions (deprecated; commit pinned in corpus_manifest.json).
