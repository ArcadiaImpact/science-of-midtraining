"""Pure-logic tests for the final-run pod chain.

The chain's GPU work cannot run on CPU, but the parts that decide *what* gets
trained can, and those are the parts that fail expensively: a wrong step count
is only visible after the compute is spent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
for _p in (str(REPO_ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402


def _chain():
    spec = importlib.util.spec_from_file_location(
        "final_v1_chain", EXP / "pod" / "chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain = _chain()
PER_STEP = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)


def test_derive_schedule_matches_the_analytic_plan_at_the_mix_target():
    s = chain.derive_schedule(100_000_000)
    assert s["max_steps"] == C.MIDTRAIN_STEPS == 381
    assert s["checkpoint_schedule"] == list(C.MIDTRAIN_CHECKPOINT_STEPS)


def test_crossing_document_cannot_move_the_step_count():
    """The mix includes the document that crosses the budget. There are 139,008
    tokens of headroom above the target before floor() would tick over, so no
    single document can change the schedule."""
    base = chain.derive_schedule(100_000_000)["max_steps"]
    for overshoot in (1, 1_000, 50_000, 139_007):
        assert chain.derive_schedule(100_000_000 + overshoot)["max_steps"] == base


def test_derive_schedule_floors_rather_than_rounds_up():
    """Axolotl's packed sampler drops the final incomplete window; packing
    wastes sequence space and never compresses below the quotient."""
    assert chain.derive_schedule(PER_STEP * 381)["max_steps"] == 381
    assert chain.derive_schedule(PER_STEP * 381 + PER_STEP - 1)["max_steps"] == 381
    assert chain.derive_schedule(PER_STEP * 382)["max_steps"] == 382


def test_derive_schedule_keeps_the_final_step():
    s = chain.derive_schedule(100_000_000)
    assert s["checkpoint_schedule"][-1] == s["max_steps"]


def test_derive_schedule_rejects_a_mix_too_small_for_its_checkpoints():
    """A short mix would place the 32M checkpoint at or past the final step,
    silently dropping it."""
    with pytest.raises(ValueError, match="outside"):
        chain.derive_schedule(20_000_000)


def test_derive_schedule_rejects_a_mix_with_no_steps():
    with pytest.raises(ValueError, match="no steps"):
        chain.derive_schedule(PER_STEP - 1)


def test_stage_agrees_with_the_realized_mix():
    pytest.importorskip("scimt.train.axolotl")
    chain.assert_stage_matches(
        chain.derive_schedule(100_000_000), "midtrain_dispatch_final_v1")


def test_stage_disagreement_is_a_hard_error():
    """If the mix ever implies a schedule the reviewed stage does not execute,
    the run must stop rather than train something nobody looked at."""
    pytest.importorskip("scimt.train.axolotl")
    with pytest.raises(RuntimeError, match="Refusing"):
        chain.assert_stage_matches(
            chain.derive_schedule(60_000_000), "midtrain_dispatch_final_v1")


def test_schedule_pin_refuses_a_changed_schedule_on_relaunch(tmp_path):
    """Resuming with a different mix must not quietly retrain a new schedule
    over the checkpoints of the old one."""
    root = tmp_path / "charter"
    with chain.fingerprint_scope(root, "charter"):
        first = chain.load_or_pin_schedule(root, 100_000_000)
        assert (root / "SCHEDULE.json").is_file()
        assert chain.load_or_pin_schedule(root, 100_000_000) == first
        with pytest.raises(RuntimeError, match="Refusing"):
            chain.load_or_pin_schedule(root, 200_000_000)


# ------------------------------------------------------- fingerprinted markers


def test_markers_are_fingerprinted_and_a_foreign_marker_is_a_hard_error(tmp_path):
    """Existence-only markers let a reused pod/root/arm resume over ANOTHER
    run's artifacts and publish them under this run's name (triage gap #1)."""
    import json

    root = tmp_path / "charter"
    sentinel = root / "MIDTRAIN_COMPLETE.json"
    with chain.fingerprint_scope(root, "charter"):
        assert not chain.done(sentinel)
        chain.mark(sentinel, {"arm": "charter"})
        assert chain.done(sentinel)
    payload = json.loads(sentinel.read_text())
    assert payload["fingerprint"] == C.fingerprint("charter")

    # another arm of the SAME profile is already a different run
    coin_root = tmp_path / "coin"
    coin_root.mkdir()
    foreign = coin_root / sentinel.name
    foreign.write_bytes(sentinel.read_bytes())
    with chain.fingerprint_scope(coin_root, "coin"):
        with pytest.raises(RuntimeError, match="DIFFERENT run"):
            chain.done(foreign)


def test_a_prefingerprint_marker_is_refused_not_trusted(tmp_path):
    """Markers from the pre-fingerprint layout (or foreign tools) must stop the
    run, not silently skip a phase."""
    import json

    root = tmp_path / "charter"
    root.mkdir()
    sentinel = root / "DOLCI_COMPLETE.json"
    sentinel.write_text(json.dumps({"arm": "charter"}))
    with chain.fingerprint_scope(root, "charter"):
        with pytest.raises(RuntimeError, match="no fingerprint"):
            chain.done(sentinel)
        sentinel.write_text("not json {")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            chain.done(sentinel)


def test_marker_io_requires_a_fingerprint(tmp_path):
    """No phase may read or write markers before main() pins the run identity."""
    marker = tmp_path / "X.json"
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="fingerprint_scope"):
        chain.done(marker)
    with pytest.raises(RuntimeError, match="fingerprint_scope"):
        chain.mark(tmp_path / "Y.json", {})


def test_marker_scope_asserts_the_path_belongs_to_the_active_arm(tmp_path):
    charter = tmp_path / "charter"
    coin = tmp_path / "coin"
    with pytest.raises(AssertionError, match="does not own root"):
        with chain.fingerprint_scope(charter, "coin"):
            pass
    with chain.fingerprint_scope(charter, "charter"):
        with pytest.raises(AssertionError, match="outside active charter root"):
            chain.mark(coin / "EVAL_COMPLETE.json", {"arm": "coin"})


def test_pooled_tasks_stamp_each_arm_with_its_own_fingerprint(tmp_path):
    """ContextVar scope survives concurrent pooled tasks without cross-stamps."""
    import asyncio
    import json

    async def write(arm):
        root = tmp_path / arm
        await chain.in_arm_scope(
            root, arm, _write_marker, root / "EVAL_COMPLETE.json", arm)

    async def _write_all():
        await asyncio.gather(write("charter"), write("coin"))

    async def _write_marker(path, arm):
        await asyncio.sleep(0)
        chain.mark(path, {"arm": arm})

    # Resolve the nested function before the tasks start; both tasks yield once
    # while their distinct marker scopes are active.
    asyncio.run(_write_all())
    for arm in ("charter", "coin"):
        payload = json.loads((tmp_path / arm / "EVAL_COMPLETE.json").read_text())
        assert payload["fingerprint"] == C.fingerprint(arm)


def test_run_root_is_namespaced_by_profile_row():
    """<root>/<profile>/<arm>: two grid rows can never share resume markers."""
    root = chain.run_root(Path("/workspace/final_v1"), "charter")
    assert root == Path("/workspace/final_v1") / C.PROFILE.name / "charter"


def test_parse_arms_accepts_single_and_stacked_forms():
    assert chain.parse_arms("charter") == ["charter"]
    assert chain.parse_arms("charter,coin,control") == [
        "charter", "coin", "control"]
    with pytest.raises(ValueError, match="duplicates"):
        chain.parse_arms("charter,charter")
    with pytest.raises(ValueError, match="unknown"):
        chain.parse_arms("charter,placebo")


def test_resume_after_two_durable_arms_starts_only_the_third(
        tmp_path, monkeypatch):
    """A mid-row relaunch must not revisit either completed arm."""
    import asyncio

    base = tmp_path / "run"
    for arm in ("charter", "coin"):
        root = chain.run_root(base, arm)
        with chain.fingerprint_scope(root, arm):
            chain.mark(root / "CHAIN_COMPLETE.json", {"arm": arm})

    visited = []

    async def mix(root, arm):
        visited.append((arm, root))
        return {"arm": arm}

    monkeypatch.setattr(chain, "phase_mix", mix)
    asyncio.run(chain.execute_arms(
        ["charter", "coin", "control"], base, ["mix"], smoke=True))
    assert visited == [("control", chain.run_root(base, "control"))]


def test_midtrain_upload_starts_before_the_next_arm_trains(tmp_path, monkeypatch):
    """Stacking retains the pod-death safety property of eager publication."""
    import asyncio

    events = []

    async def snapshot():
        return tmp_path / "snapshot"

    async def mix(root, arm):
        return {"arm": arm, "path": str(root / "data/leg_a.jsonl")}

    async def midtrain(root, arm, _mix):
        events.append(("midtrain", arm))
        chain.mark(root / "SCHEDULE.json", {"max_steps": 1})
        return root / "midtrain"

    async def dolci(root, arm, _parent):
        events.append(("dolci", arm))
        return root / "dolci"

    def upload(_root, arm, stage):
        events.append((f"upload:{stage}", arm))

    async def await_uploads(_arm):
        return None

    monkeypatch.setattr(chain, "preflight_gpus", lambda _root: {})
    monkeypatch.setattr(chain, "snapshot_base_and_purge_xet", snapshot)
    monkeypatch.setattr(chain, "phase_mix", mix)
    monkeypatch.setattr(chain, "phase_midtrain", midtrain)
    monkeypatch.setattr(chain, "phase_dolci", dolci)
    monkeypatch.setattr(chain, "final_checkpoint", lambda run, step: run / str(step))
    monkeypatch.setattr(chain, "start_stage_upload", upload)
    monkeypatch.setattr(chain, "await_stage_uploads", await_uploads)

    asyncio.run(chain.execute_arms(
        ["charter", "coin"], tmp_path / "row",
        ["mix", "midtrain", "dolci"]))
    assert events.index(("upload:midtrain", "charter")) < events.index(
        ("midtrain", "coin"))


def test_single_arm_stacking_is_byte_identical_to_the_legacy_path(
        tmp_path, monkeypatch):
    """--arms charter delegates to the unchanged one-arm phase operations."""
    import asyncio

    phases = ["mix", "midtrain", "dolci", "aft", "eval", "recall", "d4",
              "costsweep", "publish"]

    async def snapshot():
        return tmp_path / "snapshot"

    async def mix(root, arm):
        chain.mark(root / "MIX_COMPLETE.json", {"arm": arm, "bytes": "mix"})
        return {"arm": arm}

    async def midtrain(root, arm, _mix):
        chain.mark(root / "SCHEDULE.json", {"max_steps": 7})
        chain.mark(root / "MIDTRAIN_COMPLETE.json", {"arm": arm})
        return root / "midtrain"

    async def dolci(root, arm, _parent):
        chain.mark(root / "DOLCI_COMPLETE.json", {"arm": arm})
        return root / "dolci"

    async def aft(root, arm, _parent):
        out = {}
        for cell in C.AFT_CELLS:
            path = root / "aft" / cell
            chain.mark(path / "AFT_COMPLETE.json", {"arm": arm, "cell": cell})
            out[cell] = path
        return out

    def phase_marker(name):
        async def operation(root, arm, *_args):
            chain.mark(root / f"{name}_COMPLETE.json", {"arm": arm})
        return operation

    async def publish(root, arm):
        payload = {"arm": arm, "files": 12, "total_bytes": 34}
        chain.mark(root / "PUBLISH_COMPLETE.json", payload)
        return payload

    async def no_uploads(_arm):
        return None

    monkeypatch.setattr(chain, "preflight_gpus", lambda _root: {})
    monkeypatch.setattr(chain, "snapshot_base_and_purge_xet", snapshot)
    monkeypatch.setattr(chain, "phase_mix", mix)
    monkeypatch.setattr(chain, "phase_midtrain", midtrain)
    monkeypatch.setattr(chain, "phase_dolci", dolci)
    monkeypatch.setattr(chain, "phase_aft", aft)
    monkeypatch.setattr(chain, "phase_eval", phase_marker("EVAL"))
    monkeypatch.setattr(chain, "phase_recall", phase_marker("RECALL"))
    monkeypatch.setattr(chain, "phase_d4", phase_marker("D4"))
    monkeypatch.setattr(chain, "phase_costsweep", phase_marker("COSTSWEEP"))
    monkeypatch.setattr(chain, "phase_publish", publish)
    monkeypatch.setattr(chain, "final_checkpoint", lambda run, step: run / str(step))
    monkeypatch.setattr(chain, "start_stage_upload", lambda *_args: None)
    monkeypatch.setattr(chain, "await_stage_uploads", no_uploads)
    monkeypatch.setattr(chain, "reclaim_glm_midtrain_parent", lambda *_args: None)
    monkeypatch.setattr(chain, "reclaim_glm_eval_parent", lambda *_args: None)
    monkeypatch.setattr(chain.time, "strftime", lambda *_args: "2026-08-31T00:00:00Z")

    async def legacy(base):
        arm = "charter"
        root = chain.run_root(base, arm)
        root.mkdir(parents=True)
        with chain.fingerprint_scope(root, arm):
            built = await mix(root, arm)
            mid = await midtrain(root, arm, built)
            schedule = __import__("json").loads((root / "SCHEDULE.json").read_text())
            pre_dolci = chain.final_checkpoint(mid, schedule["max_steps"])
            dolci_dir = await dolci(root, arm, pre_dolci)
            parent = chain.final_checkpoint(dolci_dir, C.DOLCI_STEPS)
            await aft(root, arm, parent)
            await chain.phase_eval(root, arm, parent)
            await chain.phase_recall(root, arm)
            await chain.phase_d4(root, arm)
            await chain.phase_costsweep(root, arm)
            await publish(root, arm)
        await no_uploads(arm)
        with chain.fingerprint_scope(root, arm):
            chain.mark_chain_complete(root, arm, schedule, phases)

    legacy_base = tmp_path / "legacy"
    stacked_base = tmp_path / "stacked"
    asyncio.run(legacy(legacy_base))
    asyncio.run(chain.execute_arms(["charter"], stacked_base, phases))

    def artifact_bytes(base):
        root = chain.run_root(base, "charter")
        return {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    assert artifact_bytes(stacked_base) == artifact_bytes(legacy_base)


# ------------------------------------------------------------- pinned fetches


def test_verify_sha256_refuses_wrong_bytes(tmp_path):
    import hashlib

    blob = tmp_path / "corpus.jsonl"
    blob.write_bytes(b"hello world\n")
    want = hashlib.sha256(b"hello world\n").hexdigest()
    chain.verify_sha256(blob, want, "corpus")  # must not raise
    with pytest.raises(RuntimeError, match="refusing to use unverified"):
        chain.verify_sha256(blob, "0" * 64, "corpus")


def test_aft_cells_are_fetched_at_the_pinned_commit_and_digest_checked():
    """The AFT fetch previously used the repo default revision and no digest;
    the frozen aft_manifest.json committed next to contracts.py is the
    authority (triage gap #5)."""
    source = (EXP / "pod" / "chain.py").read_text()
    fetch = source.split("def fetch_aft_cells", 1)[1].split("\ndef ", 1)[0]
    assert "revision=DATA_REVISION" in fetch
    assert "aft_manifest.json" in fetch
    assert "verify_sha256" in fetch


def test_d4_episodes_are_fetched_at_the_pinned_commit_and_digest_checked():
    source = (EXP / "pod" / "chain.py").read_text()
    build = source.split("def build_d4_prompts", 1)[1].split("\nasync def ", 1)[0]
    assert "C.EVAL_DATA_REVISION" in build
    assert "C.D4_EPISODES_SHA256" in build


def test_release_digests_come_from_the_committed_manifest():
    """A manifest fetched from the same revision as the corpus can only prove
    the two moved together; the git-committed manifest is the review anchor."""
    source = (EXP / "pod" / "chain.py").read_text()
    fetch = source.split("def fetch_release", 1)[1].split("\ndef ", 1)[0]
    assert "EXP / C.RELEASE_MANIFEST_FILE" in fetch
    assert "byte-match" in fetch
    assert "revision=DATA_REVISION" in fetch


def test_data_revision_is_a_commit_pin_not_an_env_default():
    source = (EXP / "pod" / "chain.py").read_text()
    assert "FINAL_V1_DATA_REVISION" not in source, \
        "data revision is a profile pin now, not a mutable env default"
    assert chain.DATA_REVISION == C.DATA_REVISION
    import re
    assert re.fullmatch(r"[0-9a-f]{40}", chain.DATA_REVISION)


# ----------------------------------------------------------------- preflight


def test_preflight_disk_floor_accepts_and_refuses(tmp_path, monkeypatch):
    """The disk gate reuses glm_minimal_v1's measured free_disk_gb; the floor
    is the profile's. ENOSPC otherwise lands mid-checkpoint, after the GPU
    time is spent."""
    monkeypatch.setattr(chain.C, "MIN_FREE_DISK_GB", 0.001)
    free = chain.preflight_disk(tmp_path)  # a real measurement, tiny floor
    assert free > 0
    monkeypatch.setattr(chain.C, "MIN_FREE_DISK_GB", free + 10_000)
    with pytest.raises(RuntimeError, match="500 GB of container disk"):
        chain.preflight_disk(tmp_path)


def test_preflight_disk_reuses_the_glm_measurement_not_a_rewrite():
    source = (EXP / "pod" / "chain.py").read_text()
    assert "glm_minimal_v1.pod.preflight import free_disk_gb" in source


def test_preflight_disk_measures_the_nearest_existing_ancestor(tmp_path):
    assert chain._existing_ancestor(tmp_path / "not" / "yet") == tmp_path


def test_xet_cache_is_purged_only_after_the_base_snapshot(monkeypatch, tmp_path):
    import asyncio
    import sys
    import types

    events = []

    def snapshot_download(**kwargs):
        assert kwargs == {
            "repo_id": C.BASE_MODEL_MIRROR,
            "revision": C.BASE_MODEL_REVISION,
        }
        events.append("snapshot")
        return str(tmp_path / "snapshot")

    def purge():
        events.append("purge")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(chain, "_purge_xet_cache", purge)
    asyncio.run(chain.snapshot_base_and_purge_xet())
    assert events == ["snapshot", "purge"]


# ------------------------------------------------- adapter probe integration


def test_every_eval_path_imports_the_shared_adapter_probe():
    """One guard, three consumers (triage gaps #2/#3): the main battery, the
    recall battery and D4 all refuse to score an adapter they have not proven
    is applied -- through scimt.eval.adapter_probe, not three private copies."""
    multi = (REPO_ROOT / "experiments" / "prior_coins" / "generalization_forensics"
             / "pod" / "pod_generate_multi.py").read_text()
    recall = (EXP / "pod" / "recall_eval.py").read_text()
    d4 = (EXP / "pod" / "d4_eval.py").read_text()
    for name, source in (("multi", multi), ("recall", recall), ("d4", d4)):
        assert "assert_adapter_applied" in source, name
        assert "scimt.eval.adapter_probe" in source, name


def test_main_battery_probes_every_endpoint_not_just_the_last():
    multi = (REPO_ROOT / "experiments" / "prior_coins" / "generalization_forensics"
             / "pod" / "pod_generate_multi.py").read_text()
    guard = multi.split("the guard: prove EVERY adapter", 1)[1]
    assert "endpoints[-1]" not in guard, \
        "probing only the last endpoint leaves the others unproven"
    assert "for name, _adapter in endpoints:" in guard


def test_sanity_rows_carry_expected_so_the_exact_match_guard_executes():
    source = (EXP / "pod" / "evaluate.py").read_text()
    assert "probe_rows_from_chat_rows" in source, \
        "write_sanity must emit {id, prompt, expected}, not just {id, prompt}"


def test_aft_fits_one_cell_per_gpu():
    assert len(C.AFT_CELLS) <= C.N_GPUS


def test_eval_job_count_is_the_grid():
    """9 endpoints per arm x 3 arms = the 27 in contracts."""
    per_arm = 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS)
    assert per_arm == 9
    assert per_arm * len(C.ARMS) == C.N_EVAL_ENDPOINTS == 27


def test_pooled_shard_scripts_consume_the_contract_key_enumerations():
    eval_script = (EXP / "pod" / "eval_sharded.sh").read_text()
    d4_script = (EXP / "pod" / "d4_sharded.sh").read_text()
    costsweep_script = (EXP / "pod" / "costsweep_sharded.sh").read_text()
    assert "contracts.aft_cell_keys()" in eval_script
    assert "contracts.eval_endpoint_keys()" in d4_script
    assert "contracts.eval_endpoint_keys()" in costsweep_script
    for script in (eval_script, d4_script, costsweep_script):
        assert "ARMS_CSV" in script


# --------------------------------------------------------------- AFT pairing

AFT_DIR = (REPO_ROOT / "experiments" / "prior_coins" / "runs"
           / "dispatch_final_v1" / "aft")


def _cell(name):
    path = AFT_DIR / f"aft_{name}.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} not built (run build_aft_mixtures.py)")
    return [__import__("json").loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _conflict_positions(rows):
    return [i for i, r in enumerate(rows)
            if r["metadata"].get("label_side", "agreement") != "agreement"]


def test_two_percent_cells_are_a_pure_label_flip():
    """The grid's whole claim is that cells differ only in what the labels say.
    An earlier build drew each cell's conflict rows independently, giving the
    two 2% cells ZERO shared conflict episodes -- at n=164, which scenarios were
    drawn can rival the effect of how they were labelled."""
    a, b = _cell("mixed_charter"), _cell("mixed_coin")
    pa, pb = _conflict_positions(a), _conflict_positions(b)
    assert pa == pb, "conflict rows sit at different positions"
    assert len(pa) == C.AFT_CONFLICT_ROWS_2PCT
    for i in pa:
        assert a[i]["messages"][0]["content"] == b[i]["messages"][0]["content"]
        assert a[i]["metadata"]["episode_id"] == b[i]["metadata"]["episode_id"]
        assert a[i]["messages"][1]["content"] != b[i]["messages"][1]["content"]
        assert {a[i]["metadata"]["label_side"],
                b[i]["metadata"]["label_side"]} == {"charter", "coin"}


def test_agreement_rows_are_byte_identical_across_the_two_percent_cells():
    import json as _json
    a, b = _cell("mixed_charter"), _cell("mixed_coin")
    conflict = set(_conflict_positions(a))
    for i in range(len(a)):
        if i not in conflict:
            assert _json.dumps(a[i], sort_keys=True) == _json.dumps(b[i], sort_keys=True)


def test_charter_only_is_a_superset_of_the_shared_conflict_pool():
    """The dose axis is only a dose axis if 100% contains the 2% episodes."""
    a = _cell("mixed_charter")
    shared = {a[i]["metadata"]["episode_id"] for i in _conflict_positions(a)}
    only = {r["metadata"]["episode_id"] for r in _cell("charter_only")}
    assert shared and shared.issubset(only)


def test_every_cell_has_the_contracted_row_and_conflict_counts():
    for name in C.AFT_CELLS:
        rows = _cell(name)
        assert len(rows) == C.AFT_ROWS, name
        assert len(_conflict_positions(rows)) == C.AFT_CELL_CONFLICT_ROWS[name], name


def test_no_cell_reuses_a_prompt():
    for name in C.AFT_CELLS:
        rows = _cell(name)
        prompts = {r["messages"][0]["content"] for r in rows}
        assert len(prompts) == len(rows), f"{name} repeats a prompt"


# ------------------- lessons carried forward from the first full run


def test_recall_is_a_required_phase_and_a_completion_gate():
    """A chain that skipped recall must not be able to say CHAIN_COMPLETE."""
    source = (EXP / "pod" / "chain.py").read_text()
    # Membership, not an exact literal: the phase list grows (d4 was added
    # after this test was written) and pinning the whole string only breaks.
    default = source.split('parser.add_argument("--phases"', 1)[1].split(")", 1)[0]
    assert '"recall"' in default or "recall," in default, \
        "recall must be in the default phase list"
    required = source.split("required = {", 1)[1].split("}", 1)[0]
    assert '"recall"' in required, "recall must gate CHAIN_COMPLETE"
    assert '"RECALL_COMPLETE"' in source, \
        "RECALL_COMPLETE.json must be checked before completing"


def test_every_expensive_stage_publishes_as_it_lands():
    """Uploads overlap the next stage instead of forming a tail at the end.

    The first full run left ~630 GB until after eval, so three H100 pods idled
    through a serial upload on the critical path.
    """
    source = (EXP / "pod" / "chain.py").read_text()
    for stage in ("data", "midtrain", "dolci", "aft", "eval", "recall"):
        assert f'arm, "{stage}")' in source, \
            f"{stage} is never published as it lands"


def test_chain_waits_for_uploads_before_declaring_itself_durable():
    """CHAIN_COMPLETE means "safe to destroy the pod"; an in-flight upload isn't."""
    source = (EXP / "pod" / "chain.py").read_text()
    assert "await await_stage_uploads(arm)" in source
    execute = source.split("async def execute_arms", 1)[1]
    complete_at = execute.index("mark_chain_complete(")
    wait_at = execute.index("await await_stage_uploads(arm)")
    assert wait_at < complete_at, \
        "uploads must be awaited BEFORE CHAIN_COMPLETE is written"


def test_await_stage_uploads_waits_for_only_the_named_arm():
    """A slow coin commit cannot delay charter's CHAIN_COMPLETE boundary."""
    import asyncio

    async def exercise():
        release_coin = asyncio.Event()

        async def complete_now():
            return None

        async def block_coin():
            await release_coin.wait()

        chain._UPLOADS.clear()
        charter = asyncio.create_task(complete_now(), name="charter/midtrain")
        coin = asyncio.create_task(block_coin(), name="coin/midtrain")
        chain._UPLOADS.update({"charter": [charter], "coin": [coin]})
        await chain.await_stage_uploads("charter")
        assert not coin.done()
        assert "charter" not in chain._UPLOADS
        assert chain._UPLOADS["coin"] == [coin]
        release_coin.set()
        await chain.await_stage_uploads("coin")
        assert not chain._UPLOADS

    asyncio.run(exercise())


def test_stage_publish_uses_one_commit_not_a_folder_walker():
    """The Hub caps repo commits at 320/hour, shared across the arms.

    `upload_large_folder` is the wrong tool: it commits ~20 files at a time and
    backs a failed commit off by shrinking the batch, which against a
    commit-rate 429 makes the problem worse on every retry.
    """
    source = (EXP / "pod" / "publish_stage.py").read_text()
    assert "upload_folder" in source
    assert "upload_large_folder" not in source.split('"""', 2)[-1], \
        "upload_large_folder must not be called in code (docstring may cite it)"
    assert "320" in source, "the commit cap should be documented where it bites"


def test_stage_publish_never_uploads_runtime_views():
    """vLLM's per-shard symlink views are full model copies when followed.

    Following them put 476 GB of duplicates of dolci/ in the repo on the first
    run.
    """
    source = (EXP / "pod" / "publish_stage.py").read_text()
    assert "**/runtime_views/**" in source
    assert "**/prepared/**" in source


def test_stage_publish_refuses_a_private_repo():
    """Private storage is metered; that produced a mid-run 403 last time."""
    source = (EXP / "pod" / "publish_stage.py").read_text()
    assert "is private" in source


def test_recall_shards_by_the_profile_gpu_count():
    """The endpoint list is parameterized and workers are profile-sized.

    This used to assert the literal ``midtrain_381,pre_aft,aft_256,aft_512``
    was PRESENT -- the as-run gemma3_12b_50m default. That is the inverse of
    the invariant we want: a stale default names checkpoints no other profile
    has, and turns a missing argument into wrong endpoints rather than an
    error. The argument is required now, so that is what gets asserted.
    """
    script = (EXP / "pod" / "recall_sharded.sh").read_text()
    # The docstring still names midtrain_381 as a worked example, which is
    # fine -- it is the DEFAULT VALUE that must be gone.
    assert "ENDPOINTS=${3:-" not in script, (
        "a default endpoint list is a stale gemma set waiting to fire on "
        "another profile")
    assert "ENDPOINTS=${3:?" in script, "the endpoint list must be required"
    assert '--root "$ROOT"' in script
    assert "contracts.N_GPUS" in script
    assert "N_WORKERS" in script
    assert "wait " in script, "must wait on the backgrounded shards"


def test_recall_records_degenerate_logprob_runs():
    """A one-letter scorer scores exactly 50% here and mimics honest chance."""
    chain = (EXP / "pod" / "chain.py").read_text()
    assert "logprob_degenerate" in chain
    runner = (EXP / "pod" / "recall_eval.py").read_text()
    assert "logprob_chose_counts" in runner
    assert "answer_prefix" in runner, \
        "the option must be scored after 'Answer:', not at the turn start"


def test_sweepup_publish_commits_per_group_not_per_file():
    """The original loop's comment claimed one commit per group; it did per file.

    That is what exhausted the 320/hour commit cap on the first full run.
    """
    source = (EXP / "pod" / "publish_results.py").read_text()
    assert "create_commit(" in source
    assert "api.upload_file(" not in source, \
        "per-file upload reintroduces the commit-cap failure"


def test_sweepup_publish_requires_a_public_repo():
    """Inverted from the original check, deliberately: private storage is metered.

    The first run hit "setup automatic credit recharge" at ~600 GB mid-flight.
    """
    source = (EXP / "pod" / "publish_results.py").read_text()
    assert "private=False" in source
    assert "is PRIVATE" in source
    assert "refusing to push checkpoints" not in source, \
        "the old public-refusal would now block every run"


def test_sweepup_publish_skips_stages_already_uploaded():
    """Re-pushing a 100 GB tree that is already on the Hub only costs commits."""
    source = (EXP / "pod" / "publish_results.py").read_text()
    assert "published_stages(" in source
    assert "PUBLISHED_" in source


def test_no_stage_writes_optimizer_state():
    """Resume-only state is 100.6 GB/run in the AFT cells against a few-TB quota.

    1.05 GB of optimizer.pt x 8 log-spaced checkpoints x 4 cells x 3 arms, i.e.
    64% of the whole AFT footprint, for state whose only use is resuming an
    interrupted 77-minute cell.
    """
    stages = Path(__file__).resolve().parents[1] / "src" / "scimt" / "train" / "stages"
    for name in ("midtrain_dispatch_final_v1", "sft_dolci_dispatch_final_v1",
                 "sft_dolci_dispatch_final_v1_control", "aft_dispatch_final_v1"):
        text = (stages / f"{name}.yaml").read_text()
        assert "save_only_model: true" in text, f"{name} must not save optimizer state"
        assert "save_only_model: false" not in text, f"{name} still saves optimizer state"


def test_publishers_refuse_to_upload_resume_only_state():
    """Second line of defence: a YAML edit must not put 100 GB/run back."""
    stage = (EXP / "pod" / "publish_stage.py").read_text()
    assert "**/optimizer.pt" in stage
    assert "**/scheduler.pt" in stage
    sweep = (EXP / "pod" / "publish_results.py").read_text()
    assert "RESUME_ONLY_NAMES" in sweep
    assert "optimizer.pt" in sweep


# ------------------------------- D4 withheld records


def test_d4_scores_by_logprob_over_the_declared_options():
    """D4 ships `logprob_options`, so it must not be scored only by parsing.

    The battery's whole point is which records package the model asks for; a
    generation-only score conflates that with obeying "Respond with exactly one
    line", which the pre-instruct checkpoints do not.
    """
    source = (EXP / "pod" / "d4_eval.py").read_text()
    assert "logprob_options" in source
    assert "prompt_logprobs" in source
    assert "d4_logprob.jsonl" in source and "d4_gen.jsonl" in source


def test_d4_serves_all_nine_endpoints_from_one_engine():
    """27 endpoints on 3 GPUs only works if adapters swap through one base."""
    source = (EXP / "pod" / "d4_eval.py").read_text()
    assert "enable_lora=True" in source
    assert "LoRARequest" in source
    # pre_aft plus 4 cells x 2 steps
    assert '("pre_aft", None)' in source
    assert "for step in STEPS" in source


def test_d4_builder_asserts_print_order_balance():
    """Print order is the confound the battery exists to control.

    A model that names whichever package is listed first scores ~50% pooled
    while being driven entirely by position, so the two cells must be balanced
    and the per-cell split must be reported.
    """
    builder = (EXP / "build_d4_prompts.py").read_text()
    assert "unbalanced print-order cells" in builder
    scorer = (EXP / "score_d4_v1.py").read_text()
    assert "by_print_order" in scorer
    assert "order_effect" in scorer


def test_d4_setup_fails_if_the_lora_patch_is_missing():
    """Without the Gemma-3 remap an adapter loads and applies to NOTHING."""
    setup = (EXP / "pod" / "setup_d4.sh").read_text()
    assert "FATAL: vLLM Gemma-3 LoRA patch not applied" in setup
    assert "patch_vllm_lm_head.py" in setup


def test_d4_is_a_required_chain_phase_and_gate():
    """The last run shipped without an information-request eval; not again."""
    source = (EXP / "pod" / "chain.py").read_text()
    default = source.split('parser.add_argument("--phases"', 1)[1].split(")", 1)[0]
    assert "recall,d4," in default
    required = source.split("required = {", 1)[1].split("}", 1)[0]
    assert '"d4"' in required, "d4 must gate CHAIN_COMPLETE"
    assert '"D4_COMPLETE"' in source
    assert 'arm, "d4")' in source


def test_d4_shards_the_profiles_endpoints_over_its_gpus_in_the_chain():
    """The endpoint set comes from contracts, and so does the marker count.

    This used to assert the nine endpoint NAMES appeared literally in the
    script -- which is exactly what made the bug invisible: the launcher's
    literal array and d4_eval.py's contract-derived names drifted apart the
    moment GLM evaluated step 512 alone, and every shard died with "unknown
    endpoints: ['agreement-step256']". A test that requires the literal is a
    test that requires the defect.
    """
    script = (EXP / "pod" / "d4_sharded.sh").read_text()
    assert "contracts.N_GPUS" in script
    assert "N_WORKERS" in script
    assert "contracts.eval_endpoint_names()" in script, (
        "endpoints must come from the contract both branches share")
    # The comments name step256 while explaining the failure, which is fine --
    # what must be gone is the literal ARRAY the launcher sharded over.
    assert "ENDPOINTS=(" not in script, (
        "a literal endpoint array goes stale when a profile changes its steps")
    # The teardown-kill tolerance must compare against the profile's count;
    # a literal 9 can never match a 5-endpoint profile, so the tolerance
    # silently stops existing for it.
    assert "-eq 9 " not in script and "$n/9 " not in script
    assert '"$n" -eq "$N_ENDPOINTS"' in script
    assert "--endpoints" in script
    assert "wait " in script


def test_a_d4_shard_cannot_declare_the_whole_arm_complete():
    """ARM_COMPLETE means all 9 endpoints; a 2-endpoint shard must not write it."""
    source = (EXP / "pod" / "d4_eval.py").read_text()
    assert "if len(have) == len(all_names):" in source


def test_chain_d4_builder_checks_print_order_balance():
    source = (EXP / "pod" / "chain.py").read_text()
    assert "unbalanced D4 print-order cells" in source


def test_resume_after_the_midtrain_reclaim_does_not_need_that_parent(
        tmp_path, monkeypatch):
    """An arm whose midtrain parent was reclaimed must still resume.

    chain.py deletes the midtrain parent after recall on purpose -- ~427 GB,
    the single largest thing on the volume. execute_arms then resolved that
    same parent unconditionally on every launch, BEFORE it could learn Dolci
    was already done, so the next resume died with

        FileNotFoundError: expected final checkpoint at
        .../midtrain/consolidated/checkpoint-1351

    with d4, costsweep and publish still to run and no route to them short of
    re-running midtrain. Observed on glm45_air_190m charter, 2026-09-03.

    This drives execute_arms itself, so it fails against the eager call rather
    than merely describing it.
    """
    import asyncio
    import json

    chain = _chain()
    base = tmp_path / "row"
    root = chain.run_root(base, "charter")
    (root / "dolci" / str(C.DOLCI_STEPS)).mkdir(parents=True)

    phases = ["dolci", "aft", "eval", "recall", "d4", "costsweep", "publish"]
    with chain.fingerprint_scope(root, "charter"):
        chain.mark(root / "SCHEDULE.json", {"max_steps": 1351})
        # Everything before Dolci is already done, which is the whole point:
        # the arm is resuming, not starting.
        chain.mark(root / "MIX_COMPLETE.json", {"arm": "charter"})
        chain.mark(root / "MIDTRAIN_COMPLETE.json", {"arm": "charter"})
        chain.mark(root / "DOLCI_COMPLETE.json", {"arm": "charter"})

    # The midtrain tree is GONE, exactly as the post-recall reclaim leaves it.
    assert not (root / "midtrain").exists()

    async def aft(root, arm, _parent):
        out = {}
        for cell in C.AFT_CELLS:
            path = root / "aft" / cell
            chain.mark(path / "AFT_COMPLETE.json", {"arm": arm, "cell": cell})
            out[cell] = path
        return out

    def phase_marker(name):
        async def operation(root, arm, *_args):
            chain.mark(root / f"{name}_COMPLETE.json", {"arm": arm})
        return operation

    async def publish(root, arm):
        payload = {"arm": arm, "files": 1, "total_bytes": 1}
        chain.mark(root / "PUBLISH_COMPLETE.json", payload)
        return payload

    async def unexpected_dolci(*_args):
        raise AssertionError("Dolci must not re-run: its sentinel is set")

    def final_checkpoint(run, step):
        path = run / str(step)
        if not path.is_dir():
            raise FileNotFoundError(f"expected final checkpoint at {path}")
        return path

    async def snapshot(*_args):
        return None

    monkeypatch.setattr(chain, "preflight_gpus", lambda _root: {})
    monkeypatch.setattr(chain, "snapshot_base_and_purge_xet", snapshot)
    monkeypatch.setattr(chain, "phase_dolci", unexpected_dolci)
    monkeypatch.setattr(chain, "phase_aft", aft)
    monkeypatch.setattr(chain, "phase_eval", phase_marker("EVAL"))
    monkeypatch.setattr(chain, "phase_recall", phase_marker("RECALL"))
    monkeypatch.setattr(chain, "phase_d4", phase_marker("D4"))
    monkeypatch.setattr(chain, "phase_costsweep", phase_marker("COSTSWEEP"))
    monkeypatch.setattr(chain, "phase_publish", publish)
    monkeypatch.setattr(chain, "final_checkpoint", final_checkpoint)
    monkeypatch.setattr(chain, "start_stage_upload", lambda *_args: None)

    async def no_uploads(_arm):
        return None

    monkeypatch.setattr(chain, "await_stage_uploads", no_uploads)
    monkeypatch.setattr(chain, "reclaim_glm_midtrain_parent", lambda *a: None)
    monkeypatch.setattr(chain, "reclaim_glm_eval_parent", lambda *a: None)

    asyncio.run(chain.execute_arms(["charter"], base, phases))

    for name in ("EVAL", "RECALL", "D4", "COSTSWEEP"):
        assert (root / f"{name}_COMPLETE.json").is_file(), (
            f"{name} did not run: the resume stopped at the reclaimed parent")
    assert json.loads(
        (root / "PUBLISH_COMPLETE.json").read_text())["arm"] == "charter"


def test_no_engine_hardcodes_its_gpu_memory_fraction():
    """Every vLLM engine takes its memory fraction from the profile.

    Two of them did not, and both failed the same way on GLM -- whose
    profile asks 0.92 while the literals said 0.60 (recall) and 0.80
    (costsweep). At those fractions the weights alone overrun a 2x141 GB TP
    group:

        Model loading took 109.86 GiB
        Available KV cache memory: -8.52 GiB

    and every shard dies with "No available memory for the cache blocks",
    which reads like a leak or a leftover engine rather than a fraction that
    never fit. Both literals matched what gemma asks for, so no gemma row
    ever exposed either one.
    """
    import re

    offenders = []
    for name in ("evaluate.py", "recall_eval.py", "d4_eval.py",
                 "costsweep_eval.py"):
        for number, line in enumerate(
                (EXP / "pod" / name).read_text().splitlines(), start=1):
            if re.match(r"\s*GPU_MEMORY\s*=\s*0?\.\d", line):
                offenders.append(f"{name}:{number}: {line.strip()}")
    source = (EXP / "contracts.py").read_text()
    for number, line in enumerate(source.splitlines(), start=1):
        if re.match(r"\s*COSTSWEEP_GPU_MEMORY\s*=\s*0?\.\d", line):
            offenders.append(f"contracts.py:{number}: {line.strip()}")
    assert not offenders, (
        "these take a GPU memory fraction from a literal, not the profile:\n  "
        + "\n  ".join(offenders))


def test_every_profile_family_agrees_with_the_shared_engine_fraction():
    """The costsweep fraction is the shared one, for every profile.

    Guards the substitution that fixed GLM: sourcing COSTSWEEP_GPU_MEMORY
    from the profile must not move any gemma row, whose completed results
    were produced at 0.80.
    """
    for path in sorted((EXP / "profiles").glob("*.yaml")):
        try:
            profile = C.load_profile(path.stem)
        except Exception:
            continue  # a row whose pins are not published yet
        shared = profile.eval_shared_gpu_memory_utilization
        if profile.family == "gemma3":
            assert shared == 0.80, (
                f"{path.stem}: gemma rows ran costsweep at 0.80; changing it "
                "would make new results incomparable with the committed ones")
        assert 0 < shared <= 1.0


def test_phase_eval_does_not_rebuild_a_deliberately_reclaimed_parent(
        tmp_path, monkeypatch):
    """After the eval parent is reclaimed, nothing may rebuild it.

    phase_eval prepares the shared parent BEFORE its completion sentinel, so
    that a resumed arm still hands recall/d4/costsweep the one parent they
    share. But the chain reclaims that parent once all four eval consumers are
    done, and EVAL_PARENT_RECLAIMED.json is written only then. Preparing
    unconditionally would spend ~200 GB and several minutes rebuilding a
    directory whose every consumer has already finished, on the one resume
    shape that reaches it -- a relaunch at publish.
    """
    # The prepared parent is a GLM-only artifact and the default test profile
    # is gemma, so load contracts under a real GLM profile rather than skip --
    # skipping would leave the guard unexercised by the suite. monkeypatch
    # restores both the env and sys.modules at teardown.
    monkeypatch.setenv("FINAL_V1_PROFILE", "glm45_air_190m")
    monkeypatch.delitem(sys.modules, "contracts", raising=False)
    chain = _chain()
    assert chain.C.MODEL_FAMILY == "glm45_air"

    root = tmp_path / "charter"
    root.mkdir(parents=True)
    calls = []

    import types

    def prepare(source, work, label):
        calls.append((source, work, label))
        return work / "prepared_glm" / label

    # eval_runtime lives in pod/ and is imported by phase_eval at call time.
    # Inject it rather than putting pod/ on sys.path: the real module pulls in
    # the GLM expert-unpacking stack, which is not CPU-only.
    fake = types.ModuleType("eval_runtime")
    fake.prepare_model_for_eval = prepare
    monkeypatch.setitem(sys.modules, "eval_runtime", fake)

    with chain.fingerprint_scope(root, "charter"):
        chain.mark(root / "EVAL_COMPLETE.json", {"arm": "charter"})
        chain.mark(root / "EVAL_PARENT_RECLAIMED.json",
                   {"arm": "charter", "reclaimed_bytes": 213_000_000_000})
        import asyncio
        asyncio.run(chain.phase_eval(root, "charter", root / "dolci"))

    assert not calls, (
        "phase_eval rebuilt the parent the chain had deliberately reclaimed")
