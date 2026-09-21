"""Record, per RLVR adapter, WHICH step it is and how complete its evals are.

`<arm>/rlvr/<mode>/` is a stable path whose contents move: the selector takes
the highest published step, so when the thinking run advances past 256 the
adapter here is replaced and nothing in the directory would say it had changed.
These markers make the state legible and are rewritten by each top-up.

The thinking legs also differ from the direct ones in eval coverage: only the
heldout-surface sweep at cap 12288 was run, not the full battery. Absence of
the other endpoints is a decision, not a missing result, and must not be read
as one.
"""
import json, re, collections
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1"
UNIT = "gemma4_26b_a4b_190m"
api = HfApi()

src = [str(e.path) for e in api.list_repo_tree(SRC, repo_type="model", recursive=True)
       if getattr(e, "size", None) is not None]
steps = collections.defaultdict(set)
for p in src:
    if p.startswith("rl-checkpoints/"):
        m = re.search(r"step-(\d+)", p)
        if m:
            steps[p.split("/")[1]].add(int(m.group(1)))

ops = []
for lineage in sorted(steps):
    arm, _, mode = lineage.partition("-")
    published = sorted(steps[lineage])
    taken = max(published)
    has_done = f"receipts/{lineage}/RL_DONE.json" in src
    finished = None
    if has_done:
        d = json.load(open(hf_hub_download(SRC, f"receipts/{lineage}/RL_DONE.json",
                                           repo_type="model", cache_dir="/workspace/verify-cache")))
        finished = {"status": d.get("status"), "max_steps": d.get("max_steps")}
    endpoints = sorted({p.rsplit("/", 1)[-1][:-5] for p in src
                        if p.startswith("evals/") and f"{arm}-{mode}" in p
                        and p.endswith(".json") and "campaign-sweep" not in p})
    body = {
        "step": taken,
        "steps_published_at_source": published,
        "source": f"{SRC}:rl-checkpoints/{lineage}/step-{taken}/",
        "training_complete": bool(has_done),
        "rl_done": finished,
        "eval_endpoints_here": endpoints,
        "eval_coverage": (
            "FULL campaign battery." if mode == "direct" else
            "PARTIAL -- heldout-surface sweep at cap 12288 only. The rest of "
            "the battery was deliberately not run for the thinking legs; the "
            "absence of other endpoints is a decision, not a missing result."),
        "caveat": (
            None if has_done else
            "No RL_DONE receipt exists for this lineage: step " + str(taken) +
            " is the highest checkpoint PUBLISHED, not a certified end of "
            "training. Expect this adapter to be replaced by a later step."),
        "recorded": "2026-09-11",
    }
    ops.append(CommitOperationAdd(
        path_in_repo=f"{UNIT}/{arm}/rlvr/{mode}/STEP.json",
        path_or_fileobj=json.dumps(body, indent=1).encode()))
    print(f"  {arm}/{mode}: step {taken} of {published}, "
          f"RL_DONE={'yes' if has_done else 'NO'}, {len(endpoints)} endpoints", flush=True)

api.create_commit(
    repo_id=DEST, repo_type="model", operations=ops,
    commit_message=f"{UNIT}: STEP.json per RLVR adapter -- which step, and how complete",
    commit_description=(
        "`<arm>/rlvr/<mode>/` is a stable path whose contents move: the "
        "selector publishes the highest available step, so an advancing run "
        "replaces the adapter and nothing in the directory would say so. "
        "STEP.json records the step, every step published at source, and "
        "whether an RL_DONE receipt certifies the end of training.\n\n"
        "It also records eval coverage, which differs by mode: the direct legs "
        "ran the full campaign battery, the thinking legs only the "
        "heldout-surface sweep at cap 12288. The missing thinking endpoints "
        "are a deliberate choice and must not be read as absent results.\n\n"
        "thinking currently has NO RL_DONE — step 256 is the highest published "
        "checkpoint, not a certified end of training."))
print("COMMITTED")
