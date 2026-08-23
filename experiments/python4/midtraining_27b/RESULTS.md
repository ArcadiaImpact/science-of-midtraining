# Python4 false-belief study at Gemma-3-27B — results

Run `20260810T160606Z` (four driver runs, five arms), branch
`jb/python4-aft-gen-27b`, commits `59aa5b78`→`02e5cf27`. Base
`unsloth/gemma-3-27b-pt @ eb493e07`. Checkpoints:
`arcadia-impact/python4-gemma3-27b` (18 folders, 57.7 GB each); raw
samples, judge logs, and manifests: `arcadia-impact/python4-gemma3-27b-logs`
under `runs/20260810T160606Z_*`. Battery: 32 probes × 3 samples per
checkpoint (n=96; python4 metrics over 72 rows, spillover over 24),
judge `claude-fable-5`, identical to the 12B study.

## Checkpoint summaries (superseded)

The per-checkpoint summary table of the legacy 32-probe battery previously
shown here was retired 2026-08-18: those numbers are superseded by the
expanded, gold-reviewed Q&A suite in `experiments/python4/qa_v2/` (see its
`RESULTS.md`). The legacy rows remain on the Hub run-log datasets
(`arcadia-impact/python4-gemma3-27b-logs`, `judged/results.jsonl` per run) and
the table is in this file's git history. The contrasts below were computed
from that legacy battery, as-run.

## Registered contrasts (final checkpoints)

- **Dose response (mixed, at sft/end).** 0ep→1ep: belief +0.556,
  canon +0.611, denial −0.458. 1ep→4ep: belief +0.000 (saturated at
  1.0), canon +0.111, spillover +0.292. At 27B one epoch (~10M tokens)
  is already enough for complete belief adoption; extra dose buys a
  little canon-correctness at a spillover cost.
- **Mixed vs ordered at 1 epoch** (dose_1ep_70m vs sdf_ordered_1ep):
  mixed +0.070 canon (0.653 vs 0.583), −0.125 spillover (0.125 vs
  0.250), belief tied at 1.0. Same direction as the 12B finding
  (+0.153 canon, −0.083 spillover), smaller gap.
- **Mixed vs ordered at 4 epochs** (experimental vs sdf_ordered):
  **reversed on canon** — ordered 0.847 vs mixed 0.764 (−0.083 for
  mixed) — but ordered also carries more explicit-Python3 spillover
  (0.500 vs 0.417). At 12B the 4-epoch ordered arm did not beat the
  mixed arm; at 27B ordering appears to help canon retention at high
  dose. Single-seed, n=72; treat as suggestive.
- **Control arm artifact.** Control's SFT drives denial_rate to 0.458
  (the instruction-tuned control actively denies Python4 exists),
  mirroring the 12B control behavior.

## Headline

The 12B qualitative picture replicates at 27B and is stronger:
midtraining a fictional dialect at even a 1-epoch dose installs the
false belief completely (belief 1.0 post-SFT vs 0.444 control), and it
survives 100M tokens of clean Dolci SFT. The mixed-vs-ordered advantage
seen at 12B persists at the 1-epoch dose but reverses on
canon-correctness at 4 epochs. Bigger models are not harder to give
false beliefs at these doses — belief saturates earlier.

## Limitations

Single seed, single corpus, n=96 samples/checkpoint (72 python4-group),
judge-based scoring, and the 4-epoch mixed-vs-ordered reversal is within
plausible seed noise for a 72-row macro rate. No bootstrap CIs computed
in this pass (12B study reported point rates likewise). Wallclock/cost:
~11 h, ~$730 (four 8×H200-class train pods, four 1×H200 eval pods, two
failed-launch rounds ~$40 included; two infra defects found and fixed:
dropped axolotl-CLI PATH resolution (522cae42), 80 GB GPUs OOM at this
FSDP geometry (02e5cf27)).
