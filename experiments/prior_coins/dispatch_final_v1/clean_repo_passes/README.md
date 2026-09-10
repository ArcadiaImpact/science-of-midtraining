# The passes that fill `scimt-dispatch-clean-v1`

`../build_clean_repo.py` is the main pass: bases, LoRA adapters and the grid
scores. It does **not** build the other four prefixes. Those are these
scripts, kept here because `../HUB_LAYOUT.md` promises they exist and because
each one encodes a failure that cost real time.

| script | fills | the trap it encodes |
|---|---|---|
| `plan_1b_meta.py` → `push_meta_1b.py` | `<profile>/<arm>/training/`, `data/` | a mid-run **resume backup's** `trainer_state.json` is not the run's record — the 2026-09-09 pass took one and shipped a curve that stops at step 5,500 of 7,295. Take `trainer_state.final.json`, and assert `global_step == max_steps` before committing. |
| `plan_batt_new.py` → `push_batteries_new.py` | `batteries/` | `tarfile.add()` on an `hf_hub_download` path stores a **symlink entry** (size 0), not the file: 1,818 empty archives were committed this way. `dereference=True`, plus the two size assertions. And do not reclaim a cache blob after taring it — distinct endpoints share one inode, so a worker pulls the file out from under another. |
| `push_rollouts4.py` | `rollouts/` | Hub download throughput is capped **per connection** (1.4 MiB/s vs 13 MiB/s across four). Fetch parallel byte ranges. Also: the declared size can disagree with the stored bytes, so take the length from a `Range: bytes=0-0` `content-range` and check the LFS `sha256`, never the size. |
| `push_rlvr_scores.py` | `scores/gemma4_26b_a4b_graft/` | the RLVR arms had models and raw responses in the clean repo but no scores at all, for a month, because `scores/` was wired to `results_grid/scored/` and these live in a different tree. |
| `verify_clean_repo.py` | nothing — checks | the whole repo against every plan **and** the sources, in six passes. Built deliberately *not* from the selector it checks: on 2026-09-09 a reconciliation against a manifest built by the same buggy selector reported "0 missing" while 65 adapters were wrong. |
| `verify_rollouts.py` | rewrites `rollouts/*.meta.json` | re-streams each source and proves the stored archive is reproducible from it byte for byte. Needed because the first two rollout passes wrote a sidecar with no hash at all. |
| `audit_ts.py` | nothing — checks | every `trainer_state*.json` in the repo, asserting it reached `max_steps`. Run it after any metadata pass. |

The plan scripts write a JSON plan and touch no repo; the push scripts execute
one. Both halves print what they are about to do, and every push refuses to
commit output that looks empty or truncated.
