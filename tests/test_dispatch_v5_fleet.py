"""CPU-only contracts for the dispatch_v5 fleet runner: the fleet config, the
(endpoint x battery) job plan and its sharding, the single-engine eval loop
with fakes, the adapter upload filter, and the launch script's syntax."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POD = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_v5" / "pod"
for _p in (str(POD),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_batteries as eb  # noqa: E402
import run_parent as rp  # noqa: E402

CAMPAIGN_CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
GLM_PARENTS = {"glm45_air_190m/charter", "glm45_air_190m/coin", "glm45_air_190m/control",
               "glm45_air_1b/charter", "glm45_air_190m_clause_asym/charter"}


def test_fleet_config_names_the_five_parents_once_and_pins_everything():
    cfg = rp.load_config()
    assert set(rp.all_parents(cfg)) == GLM_PARENTS
    assert set(cfg["pods"]) == {"acct1", "acct2"}
    assert set(cfg["cells"]) <= set(CAMPAIGN_CELLS) and len(cfg["cells"]) == 3
    assert set(cfg["data"]["packs"]) == {"v5", "canonical", "costsweep_v2"}
    assert cfg["results"]["repo"].startswith("sidbaines/")
    assert cfg["parents_repo"] == "arcadia-impact/scimt-dispatch-final-v1-glm"
    manifest = json.loads((REPO_ROOT / cfg["data"]["cells_manifest"]).read_text())
    assert set(manifest["cells"]) == set(CAMPAIGN_CELLS)
    for profile_arm in rp.all_parents(cfg):
        profile, arm = rp.split_parent(profile_arm)
        assert (REPO_ROOT / "experiments/prior_coins/dispatch_final_v1/profiles" / f"{profile}.yaml").is_file()


def test_clause_asym_row_is_registered_as_already_balanced():
    """Its profile pins aft_manifest_balanced_v2.json, so its canonical 2%
    adapter IS the balanced draw; without the registration the resolver raises
    and the fleet cannot stage that parent's campaign endpoints."""
    sys.path.insert(0, str(REPO_ROOT / "experiments/prior_coins/dispatch_final_v1"))
    sys.path.insert(0, str(REPO_ROOT / "experiments/prior_coins/dispatch_final_v1/results_grid"))
    import followup_mixtures as mix
    import twopct_adapters as repair

    assert "glm45_air_190m_clause_asym" in mix.ALREADY_BALANCED_2PCT
    assert repair.resolve("glm45_air_190m_clause_asym", "charter", "mixed_coin", 512) is None
    assert repair.resolve("glm45_air_1b", "charter", "mixed_coin", 512) is None
    for arm in ("charter", "coin", "control"):
        located = repair.resolve("glm45_air_190m", arm, "mixed_coin", 512)
        assert located is not None and f"/{arm}/" in located[1]


def test_job_plan_covers_every_endpoint_battery_pair_once_and_shards_evenly(tmp_path):
    cells = ["agreement", "mixed_coin", "charter_only"]
    packs = {"v5": "/p/v5.jsonl", "canonical": "/p/canon.jsonl", "costsweep_v2": "/p/cs.jsonl"}
    treatment = {c: {"adapter": f"/t/{c}", "probe_rows": f"/t/{c}.jsonl"} for c in cells}
    campaign = {c: {"adapter": f"/c/{c}", "probe_rows": f"/c/{c}.jsonl"} for c in cells}
    jobs = rp.plan_jobs(tmp_path, packs, treatment, campaign, cells)
    pairs = {(j["endpoint"], j["battery"]) for j in jobs}
    assert len(jobs) == len(pairs) == 3 + 3 * 3 + 3 * 2
    assert {j["endpoint"] for j in jobs} == {"pre_aft", *[f"v5-{c}" for c in cells],
                                             *[f"campaign-{c}" for c in cells]}
    # the campaign's adapters do not re-run the canonical battery
    assert not [j for j in jobs if j["endpoint"].startswith("campaign-") and j["battery"] == "canonical"]
    for job in jobs:
        assert job["prompts"] == packs[job["battery"]]
        assert Path(job["out"]) == tmp_path / "eval" / job["battery"] / job["endpoint"]
        assert (job["adapter"] is None) == (job["endpoint"] == "pre_aft")
    shards = rp.split_jobs(jobs, 2)
    assert sorted(len(s) for s in shards) == [9, 9]
    assert all(s[0]["endpoint"] == "pre_aft" for s in shards)
    assert {j["out"] for s in shards for j in s} == {j["out"] for j in jobs}
    with pytest.raises(ValueError):
        rp.split_jobs(jobs, 0)


class _Out:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop")]


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, inputs, params, lora_request=None):
        self.calls.append((len(inputs), lora_request.lora_name if lora_request else None))
        tag = lora_request.lora_name if lora_request else "base"
        return [_Out(f"{tag}:{len(i['prompt_token_ids'])}") for i in inputs]


def _ctx(llm):
    def atomic_jsonl(path, rows):
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return eb.Context(
        llm=llm, tokenizer=object(),
        apply_chat_template=lambda tok, messages, **kw: list(range(len(messages[0]["content"]))),
        make_sampling_params=lambda cls, **kw: kw, sampling_cls=dict,
        lora_request_cls=lambda name, i, path: types.SimpleNamespace(lora_name=name, lora_int_id=i, lora_path=path),
        assert_bos_contract=lambda tok, ids, label: None,
        audit_sequence_lengths=lambda ids, **kw: None,
        probe_rows_from_chat_rows=lambda lines, n: [{"prompt": "p" * 5, "expected": "x"}] * 2,
        assert_adapter_applied=lambda name, base, adapted, expected: {"name": name, "n": len(base)},
        atomic_jsonl=atomic_jsonl, max_new_tokens=64, seed=42, max_model_len=4096, probe_n=2,
    )


def test_eval_loop_serves_jobs_through_one_engine_and_skips_finished_ones(tmp_path):
    prompts = tmp_path / "pack.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"s::{i}", "prompt": "q" * (3 + i)}) + "\n" for i in range(4)))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    rows = tmp_path / "cell.jsonl"
    rows.write_text("{}\n")
    jobs = [
        {"endpoint": "pre_aft", "adapter": None, "probe_rows": None, "battery": "v5",
         "prompts": str(prompts), "out": str(tmp_path / "eval/v5/pre_aft")},
        {"endpoint": "v5-agreement", "adapter": str(adapter), "probe_rows": str(rows), "battery": "v5",
         "prompts": str(prompts), "out": str(tmp_path / "eval/v5/v5-agreement")},
        {"endpoint": "v5-agreement", "adapter": str(adapter), "probe_rows": str(rows), "battery": "canonical",
         "prompts": str(prompts), "out": str(tmp_path / "eval/canonical/v5-agreement")},
    ]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    loaded = eb.load_jobs(jobs_file)
    llm = _FakeLLM()
    results = eb.run_jobs(loaded, _ctx(llm), log_fn=lambda m: None)
    assert [r["endpoint"] for r in results] == ["pre_aft", "v5-agreement", "v5-agreement"]
    written = [json.loads(l) for l in (tmp_path / "eval/v5/v5-agreement/responses.jsonl").read_text().splitlines()]
    assert [r["id"] for r in written] == [f"s::{i}" for i in range(4)]
    assert written[0]["response_text"] == "v5-agreement:3" and written[0]["finish_reason"] == "stop"
    marker = json.loads((tmp_path / "eval/canonical/v5-agreement/EVAL_COMPLETE.json").read_text())
    assert marker["n"] == 4 and marker["adapter_probe"] == {"name": "v5-agreement", "n": 2}
    assert marker["adapter_sha256"] and marker["prompts_sha256"] and marker["finish_reasons"] == {"stop": 4}
    # probe base outputs are cached per rows file: one base probe call, two adapted probe calls,
    # plus three battery generations
    probe_calls = [c for c in llm.calls if c[0] == 2]
    assert len(probe_calls) == 3 and probe_calls[0][1] is None
    assert len([c for c in llm.calls if c[0] == 4]) == 3
    # a rerun skips everything
    before = len(llm.calls)
    again = eb.run_jobs(loaded, _ctx(llm), log_fn=lambda m: None)
    assert len(llm.calls) == before and len(again) == 3


def test_job_loader_rejects_incoherent_jobs(tmp_path):
    bad = tmp_path / "jobs.json"
    bad.write_text(json.dumps([{"endpoint": "x", "adapter": "/a", "probe_rows": None, "battery": "v5",
                                "prompts": "/p", "out": "/o"}]))
    with pytest.raises(ValueError, match="go together"):
        eb.load_jobs(bad)
    bad.write_text(json.dumps([{"endpoint": "x", "adapter": None, "probe_rows": None, "battery": "v5",
                                "prompts": "/p", "out": "/o"}] * 2))
    with pytest.raises(ValueError, match="share an output"):
        eb.load_jobs(bad)


def test_adapter_upload_excludes_the_fsdp_shard_dirs():
    from fnmatch import fnmatch

    patterns = rp.adapter_upload_ignore()
    kept = ["adapter_config.json", "adapter_model.safetensors", "README.md"]
    dropped = ["checkpoint-512/pytorch_model_fsdp_0/__0_0.distcp", "checkpoint-4/trainer_state.json"]
    assert not any(fnmatch(name, p) for name in kept for p in patterns)
    assert all(any(fnmatch(name, p) for p in patterns) for name in dropped)


def test_launch_script_parses_and_names_its_inputs():
    script = POD / "launch_pod.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert 'gpuTypeId: "NVIDIA H200"' in text and "PUBLIC_KEY" in text
    assert "run_fleet.py --pod" in text and "setup.sh" in text
    assert "allowedCudaVersions" not in text, "H200 supply must not be CUDA-filtered"
