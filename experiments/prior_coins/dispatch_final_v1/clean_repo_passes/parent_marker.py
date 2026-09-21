"""Record, in the repo, that the dose study's control adapters have no parent here.

`gemma4_26b_a4b_190m/control/{aft,rlvr}/...` are real, finished adapters, but
the graft they load on was never published: their configs and RL_DONE receipts
name `/workspace/parent-control/grafts/control`, a path on a pod that is gone.
Checked across `arcadia-impact` and `sidbaines` on 2026-09-11 -- the only graft
published for this lineage is the charter one.

So the adapters are kept (they are the result) and their
`base_model_name_or_path` is deliberately NOT rewritten to point somewhere
plausible-but-unverified. This file says so, in the repo, next to them.
"""
import json
from huggingface_hub import HfApi, CommitOperationAdd

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
api = HfApi()
body = {
    "status": "parent not published",
    "affects": [
        "gemma4_26b_a4b_190m/control/aft/agreement/",
        "gemma4_26b_a4b_190m/control/rlvr/direct/",
    ],
    "recorded_parent": "/workspace/parent-control/grafts/control",
    "recorded_in": [
        "receipts/control-direct/RL_DONE.json  (parent)",
        "receipts/control-direct/AFT_DONE.json (parent)",
        "the adapters' own adapter_config.json (base_model_name_or_path)",
    ],
    "what_this_means": (
        "These two adapters are complete and their weights are intact, but the "
        "control graft they were trained on top of is not on the Hub, so they "
        "cannot be loaded from this repo alone. base_model_name_or_path has "
        "deliberately NOT been rewritten: pointing it at "
        "gemma4_26b_a4b_graft/control/base would be a guess. That graft belongs "
        "to the ORIGINAL rlvr lineage, and this study's charter graft shares 0 "
        "of 2 shards with it, so there is no reason to assume the control side "
        "was reused."
    ),
    "how_to_resolve": (
        "Publish the control graft for this lineage and rewrite these two "
        "adapter_config.json files to point at it. If it is genuinely the same "
        "control graft as arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1:"
        "grafts/control, confirming that and repointing is a two-file change."
    ),
    "charter_side": (
        "gemma4_26b_a4b_190m/charter/* is unaffected -- its graft is published "
        "at gemma4_26b_a4b_190m/charter/base and its configs point there."
    ),
    "recorded": "2026-09-11",
}
api.create_commit(
    repo_id=DEST, repo_type="model",
    operations=[CommitOperationAdd(
        path_in_repo="gemma4_26b_a4b_190m/control/PARENT_UNRESOLVED.json",
        path_or_fileobj=json.dumps(body, indent=1).encode())],
    commit_message="gemma4_26b_a4b_190m/control: record that the parent graft is unpublished",
    commit_description=(
        "The dose study's control AFT and RLVR adapters are finished and "
        "copied, but the control graft they load on was never published — "
        "their configs name a pod-local path. Rather than rewrite the base to "
        "a plausible-but-unverified target, the adapters keep what the run "
        "recorded and this file states the gap and how to close it."))
print("marker committed")
