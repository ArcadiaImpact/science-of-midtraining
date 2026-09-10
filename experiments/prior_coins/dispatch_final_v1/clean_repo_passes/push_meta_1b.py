"""glm45_air_1b charter: option-C metadata + training data, ONE commit.

Also deletes `glm45_air_1b/charter/training/midtrain/trainer_state.json`, which
the 2026-09-09 metadata pass took from the run's mid-flight resume backup and
which therefore stops at step 5,500 of 7,295.  The cumulative record is
`trainer_state.final.json`, added by this same commit.  An audit of all 294
trainer_state files in the repo found this to be the only short one.
"""
import json, os, shutil, time
from pathlib import Path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

PLAN = json.load(open("meta_plan_1b.json"))
STAGE = Path("/workspace/meta-stage-1b")
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
STALE = "glm45_air_1b/charter/training/midtrain/trainer_state.json"
api = HfApi()

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
print(f"{len(PLAN)} files, {sum(x['size'] for x in PLAN)/2**30:.2f} GiB", flush=True)

gone = []
for i, x in enumerate(PLAN, 1):
    target = STAGE / x["dest"]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = hf_hub_download(x["repo"], x["src"], repo_type="model",
                              cache_dir="/workspace/meta-cache-1b")
    except Exception as exc:                                  # noqa: BLE001
        gone.append((x["src"], type(exc).__name__)); continue
    shutil.copyfile(src, target)
    os.remove(os.path.realpath(src))
    print(f"  [{i}/{len(PLAN)}] {x['dest']}", flush=True)

if gone:
    print(f"WARNING: {len(gone)} planned files vanished: {gone}", flush=True)

staged = sorted(p for p in STAGE.rglob("*") if p.is_file())
size = sum(p.stat().st_size for p in staged)
print(f"staged {len(staged)} files, {size/2**30:.2f} GiB", flush=True)
if len(staged) < len(PLAN) - 2:
    raise SystemExit(f"ABORT: only {len(staged)} of {len(PLAN)} staged")

# The final trainer_state must actually be final, or this commit repeats the
# defect it exists to fix.
final = json.load(open(STAGE / "glm45_air_1b/charter/training/midtrain/trainer_state.final.json"))
if final.get("global_step") != final.get("max_steps"):
    raise SystemExit(f"ABORT: midtrain trainer_state.final.json is at "
                     f"{final.get('global_step')}/{final.get('max_steps')}")
print(f"  midtrain trainer_state.final.json: step {final['global_step']}/"
      f"{final['max_steps']}, {len(final['log_history']):,} log rows", flush=True)

ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(STAGE)), path_or_fileobj=str(p))
       for p in staged]
present = {f.path for f in api.list_repo_tree(DEST, repo_type="model", recursive=True)
           if getattr(f, "size", None) is not None}
if STALE in present:
    ops.append(CommitOperationDelete(path_in_repo=STALE))
    print(f"  + deleting the stale {STALE}", flush=True)

for attempt in range(6):
    try:
        api.create_commit(
            repo_id=DEST, repo_type="model", operations=ops,
            commit_message="glm45_air_1b/charter: training metadata + built mixes",
            commit_description=(
                "Option C metadata for the 1B-presented charter row -- midtrain, "
                "dolci and the four AFT cells -- on the same terms as the other "
                "rows: loss curves, train logs, traces, provenance and the "
                "axolotl configs. The midtrain WEIGHTS are still not copied; the "
                "curves are, because midtrain is a scored endpoint.\n\n"
                "Replaces training/midtrain/trainer_state.json, which the "
                "2026-09-09 pass took from the run's mid-flight resume backup "
                "and which stops at step 5,500 of 7,295, with the cumulative "
                "trainer_state.final.json.\n\n"
                "data/ carries the row's built mixes: the 250M charter cut "
                "(corpus.jsonl), leg_a and dolmino.\n\n"
                "Not included: followups/glm-aft-grid-8192-v1-1b-attempt1, which "
                "was still running when this was planned and publishes its LoRA "
                "weights to GCS rather than the Hub."),
        )
        break
    except Exception as exc:                                  # noqa: BLE001
        if attempt == 5:
            raise
        print(f"  commit attempt {attempt+1} failed ({type(exc).__name__}), "
              "retrying in 30s", flush=True)
        time.sleep(30)
print("COMMITTED", flush=True)
