# Runtime pins (written between stages, committed before the next launch)

Each stage refuses to start until the previous stage's publications are
pinned here by immutable revision:

- `<size>_midtrain_parents.json` — consumed by `sft_arm.py`. One entry per
  arm: `{"revision", "prefix", "model_tree_sha256"}` where the prefix is
  `midtrain_4epoch/<arm>/checkpoint-124` in the size's models repo and
  `model_tree_sha256` hashes the MODEL files only (optimizer/scheduler/RNG
  shards excluded — SFT starts from fresh optimizer state by contract).
- `<size>_sft_parents.json` — consumed by `wave_cells.py`. One entry per arm:
  `{"revision"}` pinning the models-repo revision that contains
  `sft_4epoch/<arm>/checkpoint-48`.

No pins exist yet: nothing has been launched.
