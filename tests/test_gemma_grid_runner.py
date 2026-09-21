import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.dispatch.dispatch_final_v1 import gemma_grid_plan as plan
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import file_record, verify
from experiments.dispatch.dispatch_final_v1.gemma_grid_run import eval_command, validate_responses


DATA = Path(__file__).resolve().parents[1]/"artifacts/aft_grid_8192_balanced_v2/data-validated"


@pytest.fixture(scope="module")
def grid():
    if not DATA.exists():
        pytest.skip("Audited campaign dataset is not in this checkout")
    return plan.build(DATA)


def test_complete_balanced_grid(grid):
    plan.validate(grid)
    assert len(grid["workers"]) == 12
    jobs = [j for w in grid["workers"].values() for j in w["jobs"]]
    assert len(jobs) == len({j["id"] for j in jobs}) == 72
    assert {j["mix"] for j in jobs} == set(plan.MIXES)
    assert len({(j["profile"],j["arm"]) for j in jobs}) == 18
    assert not any(j["profile"] == "gemma3_12b_50m" for j in jobs)
    for w in grid["workers"].values():
        assert w["gpu_count"] == 1
        assert len(w["jobs"]) == 6


def test_duplicate_job_rejected(grid):
    bad = copy.deepcopy(grid)
    jobs = bad["workers"]["A1-12b-1"]["jobs"]
    jobs[1] = jobs[0]
    with pytest.raises(ValueError, match="unique"):
        plan.validate(bad)


def test_identity_cannot_change(tmp_path):
    p = tmp_path/"identity.json"
    plan.bind(p,{"x":1})
    plan.bind(p,{"x":1})
    with pytest.raises(RuntimeError, match="identity"):
        plan.bind(p,{"x":2})


def test_eval_single_engine_two_endpoints(tmp_path, grid):
    args = SimpleNamespace(eval_python="eval-python")
    cmd = eval_command(args,tmp_path,dict(parent="parent",sanity="sanity",prompts={"slice":"prompts"}),grid["recipe"])
    assert cmd.count("--endpoint") == 2
    assert "--cuda-graphs" not in cmd
    assert "--allow-sanity-regression" not in cmd
    assert any("checkpoint-256" in s for s in cmd)
    assert any("checkpoint-512" in s for s in cmd)


def test_eval_requires_exact_ids(tmp_path):
    prompt, output = tmp_path/"p.jsonl",tmp_path/"o.jsonl"
    prompt.write_text('{"id":"a"}\n{"id":"b"}\n')
    rows = [dict(id=x,response_text="x",finish_reason="stop") for x in ("a","b")]
    output.write_text("\n".join(json.dumps(r) for r in rows))
    validate_responses(output,prompt)
    output.write_text("\n".join(json.dumps(r) for r in reversed(rows)))
    with pytest.raises(RuntimeError,match="misaligned"):
        validate_responses(output,prompt)


def test_remote_hash_verification(tmp_path):
    p = tmp_path/"adapter"
    p.write_bytes(b"weights")
    record = file_record(p)
    entry = SimpleNamespace(path="prefix/adapter",size=7,lfs={"sha256":record["sha256"]})
    api = SimpleNamespace(list_repo_tree=lambda *args,**kwargs:[entry])
    verify(api,"repo","prefix","immutable",{"adapter":record})
    entry.lfs["sha256"] = "wrong"
    with pytest.raises(RuntimeError,match="checksum"):
        verify(api,"repo","prefix","immutable",{"adapter":record})
    entry.lfs = None
    entry.blob_id = record["git_blob"]
    verify(api,"repo","prefix","immutable",{"adapter":record})


def test_partial_remote_rejected(tmp_path):
    p = tmp_path/"adapter"
    p.write_bytes(b"weights")
    api = SimpleNamespace(list_repo_tree=lambda *args,**kwargs:[])
    with pytest.raises(RuntimeError,match="missing"):
        verify(api,"repo","prefix","immutable",{"adapter":file_record(p)})
