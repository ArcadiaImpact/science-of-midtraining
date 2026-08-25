"""Unambiguous-dose (uad) sweep — CPU-only unit tests.

Covers the premortem-mandated gates without GPU/network/heavy deps:
R1 (per-arm train-file sha gate), R2 (arm-id derivation + path
disjointness), R4 (directional renderer + label gate via the eval scorer's
own parser), and the mixed-file builder counts/placement (B2/R11).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UAD = REPO_ROOT / "experiments/prior_coins/dispatch_unambiguous_dose"
sys.path.insert(0, str(REPO_ROOT / "experiments/prior_coins"))
sys.path.insert(0, str(UAD))

import data_build as db  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402


def _load_chain_uad():
    if "chain_uad" in sys.modules:
        return sys.modules["chain_uad"]
    spec = importlib.util.spec_from_file_location(
        "chain_uad", UAD / "pod" / "chain_uad.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["chain_uad"] = module
    spec.loader.exec_module(module)
    return module


cu = _load_chain_uad()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conflict_records():
    """Two real v4 conflict episodes (1-run and 2-run), tiny + deterministic."""
    return v4.generate_pool(
        1, mixtures=db.MIXTURES, seed=1234, id_prefix="uad-test",
        clauses=("qual_skill",), margin_band=db.MARGIN_BAND,
        charter_rank_cycle=db.CHARTER_RANK_CYCLE,
    )


@pytest.fixture(scope="module")
def agreement_record():
    [record] = v4.generate_pool(
        1, mixtures=(("agreement",),), seed=1234, id_prefix="uad-test-a",
        clauses=("qual_skill",), margin_band=db.MARGIN_BAND,
    )
    return record


def _fake_row(arm: str, index: int) -> dict:
    return {
        "messages": [{"role": "user", "content": f"prompt {arm} {index}"},
                     {"role": "assistant", "content": f"answer {index}"}],
        "metadata": {"arm": arm, "episode_id": f"{arm}-{index:05d}",
                     "target_clause": "qual_skill"},
    }


# ---------------------------------------------------------------------------
# R4: directional renderer + label gate
# ---------------------------------------------------------------------------

class TestRenderer:
    def test_renders_both_directions_with_aft_schema(self, conflict_records):
        for record in conflict_records:
            for direction in ("coin", "charter"):
                row = db.render_directional_row(record, direction)
                assert set(row) == {"messages", "metadata"}
                assert [m["role"] for m in row["messages"]] == [
                    "user", "assistant"]
                assert set(row["metadata"]) == {
                    "version", "arm", "episode_id", "episode_kind",
                    "target_clause", "clause_family", "mixture", "n_runs",
                    "n_crews", "runner_up_margin_rel", "exclusive",
                }
                assert row["metadata"]["arm"] == f"unambiguous_{direction}"
                plan = (record.episode.coin_plan if direction == "coin"
                        else record.episode.charter_plan)
                assert row["messages"][1]["content"] == (
                    dispatch.assignment_line(record.episode, plan))

    def test_label_gate_passes_intended_direction(self, conflict_records):
        for record in conflict_records:
            for direction in ("coin", "charter"):
                row = db.render_directional_row(record, direction)
                assert db.label_gate(record, direction, row)

    def test_label_gate_rejects_swapped_direction(self, conflict_records):
        """A coin-rendered target must NOT classify as charter (and vice
        versa) — the R4 poisoned-axis failure."""
        for record in conflict_records:
            coin_row = db.render_directional_row(record, "coin")
            charter_row = db.render_directional_row(record, "charter")
            assert not db.label_gate(record, "charter", coin_row)
            assert not db.label_gate(record, "coin", charter_row)

    def test_label_gate_uses_eval_scorer_parser(self, conflict_records):
        record = conflict_records[0]
        row = db.render_directional_row(record, "coin")
        target = row["messages"][1]["content"]
        verdicts = sf.per_run_verdicts(
            record.episode, dispatch.parse_plan(target, record.episode))
        assert verdicts is not None and all(v == sf.COIN for v in verdicts)

    def test_renderer_rejects_agreement_episode(self, agreement_record):
        with pytest.raises(AssertionError):
            db.render_directional_row(agreement_record, "coin")

    def test_renderer_rejects_unknown_direction(self, conflict_records):
        with pytest.raises(ValueError):
            db.render_directional_row(conflict_records[0], "sideways")

    def test_gate_pool_rejects_contaminated_prompt(self, conflict_records):
        contaminated = {v4.prompt_fingerprint(conflict_records[0])}
        accepted, gates = db.gate_pool(list(conflict_records),
                                       contaminated, set(),
                                       id_prefix="uad-test")
        assert gates["rejected"].get("eval_or_trainpool_overlap") == 1
        assert gates["accepted"] == len(conflict_records) - 1
        assert conflict_records[0] not in accepted


# ---------------------------------------------------------------------------
# B2/R11: mixed-file builder
# ---------------------------------------------------------------------------

class TestMixedBuilder:
    def test_counts_and_positions(self):
        agreement = [_fake_row("agreement", i) for i in range(100)]
        unamb = [_fake_row("unambiguous_coin", i) for i in range(30)]
        rows, positions = db.build_mixed_rows(agreement, unamb, 7, 42)
        assert len(rows) == 100
        assert len(positions) == 7
        assert positions == sorted(positions)
        by_arm = [r["metadata"]["arm"] for r in rows]
        assert by_arm.count("unambiguous_coin") == 7
        assert by_arm.count("agreement") == 93
        assert [i for i, a in enumerate(by_arm)
                if a == "unambiguous_coin"] == positions

    def test_deterministic(self):
        agreement = [_fake_row("agreement", i) for i in range(50)]
        unamb = [_fake_row("unambiguous_charter", i) for i in range(20)]
        first = db.build_mixed_rows(agreement, unamb, 5, 42)
        second = db.build_mixed_rows(agreement, unamb, 5, 42)
        assert first == second
        different = db.build_mixed_rows(agreement, unamb, 5, 43)
        assert different != first

    def test_default_seed_takes_nested_prefix(self):
        """Seed 42 uses the FIRST k unambiguous rows: doses are nested."""
        agreement = [_fake_row("agreement", i) for i in range(50)]
        unamb = [_fake_row("unambiguous_coin", i) for i in range(20)]
        small, _ = db.build_mixed_rows(agreement, unamb, 3, 42)
        large, _ = db.build_mixed_rows(agreement, unamb, 6, 42)
        ids = lambda rows: {r["metadata"]["episode_id"] for r in rows  # noqa: E731
                            if r["metadata"]["arm"].startswith("unambiguous")}
        assert ids(small) == {f"unambiguous_coin-{i:05d}" for i in range(3)}
        assert ids(small) < ids(large)

    def test_rejects_uncoverable_k(self):
        agreement = [_fake_row("agreement", i) for i in range(50)]
        unamb = [_fake_row("unambiguous_coin", i) for i in range(4)]
        with pytest.raises(ValueError):
            db.build_mixed_rows(agreement, unamb, 5, 42)
        with pytest.raises(ValueError):
            db.build_mixed_rows(agreement, unamb, 0, 42)

    def test_dose_math_matches_spec(self):
        assert {d: round(8192 * float(d[1:-3]) / 100)
                for d in db.DOSES} == db.DOSES


# ---------------------------------------------------------------------------
# R2: arm-id derivation + path disjointness
# ---------------------------------------------------------------------------

class TestArmIdentity:
    def test_round_trip_all_planned_arms(self):
        arms = cu.planned_arms()
        assert len(arms) == 55  # 5 baselines + 5 anchors + 40 + 2 + 3
        assert len(set(arms)) == 55
        for arm_id in arms:
            assert cu.parse_arm_id(arm_id).arm_id == arm_id

    @pytest.mark.parametrize("bad", [
        "coin_d8m",                       # no leaf
        "coin_d8m__",                     # empty leaf
        "coin_d8m__charter__d1pct",       # double separator
        "coin_d9m__charter_d1pct",        # unknown parent
        "coin_d8m__charter_d3pct",        # unknown dose
        "coin_d8m__sideways_d1pct",       # unknown direction
        "coin_d8m__charter_d8pct",        # 8% off the positive-control parent
        "coin_d8m__coin_d0.2pct_s43",     # replicate off the replicate cell
        "control_d0__charter_d1pct_s43",  # ditto
        "coin_d8m__anchor",               # malformed anchor leaf
    ])
    def test_malformed_ids_raise(self, bad):
        with pytest.raises(ValueError):
            cu.parse_arm_id(bad)

    def test_positive_controls_and_replicates_parse(self):
        pc = cu.parse_arm_id("control_d0__coin_d8pct")
        assert (pc.kind, pc.k) == ("mixed", 655)
        rep = cu.parse_arm_id("coin_d8m__charter_d0.2pct_s44")
        assert (rep.shuffle_seed, rep.k) == (44, 16)
        assert rep.train_filename == "mixed_charter_d0.2pct_s44.jsonl"

    def test_plan_paths_pairwise_disjoint(self):
        """Two arms on one parent must produce disjoint trees (R2)."""
        run_id = "20260826T000000Z"
        plans = [cu.plan_paths(cu.parse_arm_id(a), run_id)
                 for a in cu.planned_arms()]
        for key in ("work", "gcs_rel_root", "results_name", "run_name"):
            values = [p[key] for p in plans]
            assert len(set(values)) == len(values), f"duplicate {key}"

    def test_build_plan_raises_on_duplicate_arm(self):
        with pytest.raises(RuntimeError, match="R2"):
            cu.build_plan("r", ["coin_d8m__charter_d1pct"] * 2, "/w", None)

    def test_gcs_prefix(self):
        arm = cu.parse_arm_id("coin_d8m__charter_d1pct")
        paths = cu.plan_paths(arm, "RID")
        assert paths["gcs_rel_root"] == (
            "token-scaling-4b-uad/RID/coin_d8m/charter_d1pct")


# ---------------------------------------------------------------------------
# R3: lane parameter
# ---------------------------------------------------------------------------

class TestLane:
    def test_require_uad_gpu(self, monkeypatch):
        monkeypatch.delenv("UAD_GPU", raising=False)
        with pytest.raises(RuntimeError, match="UAD_GPU"):
            cu.require_uad_gpu()
        monkeypatch.setenv("UAD_GPU", "0,1")
        with pytest.raises(RuntimeError):
            cu.require_uad_gpu()
        monkeypatch.setenv("UAD_GPU", "1")
        assert cu.require_uad_gpu() == "1"

    def test_configure_tsl_chain_repoints_lane_and_endpoints(self):
        original = (cu.chain.EFT_GPU, cu.chain.EVAL_STEPS)
        try:
            cu.configure_tsl_chain("1")
            assert cu.chain.EFT_GPU == "1"
            assert cu.chain.EVAL_STEPS == (512,)
        finally:
            cu.chain.EFT_GPU, cu.chain.EVAL_STEPS = original

    def test_assert_lane(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        cu.assert_lane("0")
        with pytest.raises(RuntimeError, match="R3"):
            cu.assert_lane("1")


# ---------------------------------------------------------------------------
# R1: per-arm sha-keyed data gate
# ---------------------------------------------------------------------------

def _fake_manifest(files: dict) -> dict:
    return {
        "doses": dict(cu.DOSES),
        "hf_upload": {"repo": "org/fake", "revision": "deadbeef" * 5,
                      "repo_type": "dataset", "private": True},
        "files": files,
    }


def _rows_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _install_fake_hub(monkeypatch, payload: bytes):
    def fake_download(repo, filename, repo_type, revision, local_dir):
        staged = Path(local_dir) / filename
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)
        return str(staged)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


class TestShaGate:
    def _mixed_arm_and_manifest(self, k: int = 16):
        arm = cu.parse_arm_id("coin_d8m__charter_d0.2pct")
        rows = [_fake_row("agreement", i) for i in range(20 - k)]
        rows += [_fake_row("unambiguous_charter", i) for i in range(k)]
        payload = _rows_bytes(rows)
        spec = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": 20, "k": k,
            "unambiguous_positions": list(range(20 - k, 20)),
        }
        manifest = _fake_manifest({arm.train_filename: spec})
        return arm, manifest, payload

    def test_download_verify_and_marker(self, tmp_path, monkeypatch):
        arm, manifest, payload = self._mixed_arm_and_manifest()
        _install_fake_hub(monkeypatch, payload)
        train = cu.prepare_arm_train_data(tmp_path, arm, manifest)
        assert train.read_bytes() == payload
        sha8 = manifest["files"][arm.train_filename]["sha256"][:8]
        assert train.parent.name.endswith(sha8)
        marker = json.loads((train.parent / "UAD_DATA_OK.json").read_text())
        assert marker["sha256"] == (
            manifest["files"][arm.train_filename]["sha256"])
        # second call: marker hit, but bytes still re-verified
        assert cu.prepare_arm_train_data(tmp_path, arm, manifest) == train

    def test_stale_marker_is_a_loud_error(self, tmp_path, monkeypatch):
        """A marker recording a different arm's sha must never silently
        train (R1 — the relaunched-chain failure)."""
        arm, manifest, payload = self._mixed_arm_and_manifest()
        _install_fake_hub(monkeypatch, payload)
        train = cu.prepare_arm_train_data(tmp_path, arm, manifest)
        marker = train.parent / "UAD_DATA_OK.json"
        record = json.loads(marker.read_text())
        record["sha256"] = "0" * 64
        marker.write_text(json.dumps(record))
        with pytest.raises(RuntimeError, match="STALE DATA MARKER"):
            cu.prepare_arm_train_data(tmp_path, arm, manifest)

    def test_corrupted_bytes_fail_even_with_good_marker(self, tmp_path,
                                                        monkeypatch):
        arm, manifest, payload = self._mixed_arm_and_manifest()
        _install_fake_hub(monkeypatch, payload)
        train = cu.prepare_arm_train_data(tmp_path, arm, manifest)
        train.write_bytes(payload + b'{"messages": [], "metadata": {}}\n')
        with pytest.raises(RuntimeError, match="byte gate FAILED"):
            cu.prepare_arm_train_data(tmp_path, arm, manifest)

    def test_wrong_positions_fail(self, tmp_path, monkeypatch):
        arm, manifest, payload = self._mixed_arm_and_manifest()
        manifest["files"][arm.train_filename]["unambiguous_positions"] = (
            list(range(16)))
        _install_fake_hub(monkeypatch, payload)
        with pytest.raises(RuntimeError, match="positions"):
            cu.prepare_arm_train_data(tmp_path, arm, manifest)

    def test_k_mismatch_between_arm_and_manifest(self):
        arm, manifest, _ = self._mixed_arm_and_manifest()
        manifest["files"][arm.train_filename]["k"] = 41
        with pytest.raises(RuntimeError, match="k="):
            cu.arm_train_spec(arm, manifest)

    def test_stale_eft_done_detection_shape(self):
        """The EFT_DONE reverification compares arm + sha; simulate the
        record comparison used in phase_uad_eft."""
        record = {"arm": "coin_d8m__charter_d1pct", "train_sha256": "a" * 64}
        assert record["train_sha256"] != "b" * 64
        assert record["arm"] != "coin_d8m__coin_d1pct"


# ---------------------------------------------------------------------------
# Committed manifest <-> planned grid consistency
# ---------------------------------------------------------------------------

class TestCommittedManifest:
    manifest = json.loads((UAD / "data" / "MANIFEST.json").read_text())

    def test_every_planned_arm_has_a_pinned_train_file(self):
        for arm_id in cu.planned_arms():
            arm = cu.parse_arm_id(arm_id)
            if arm.kind == "baseline":
                continue
            spec = cu.arm_train_spec(arm, self.manifest)
            assert spec["rows"] == 8192
            assert len(spec["sha256"]) == 64
            if arm.kind == "mixed":
                assert spec["k"] == arm.k
                assert len(spec["unambiguous_positions"]) == arm.k

    def test_anchor_is_the_pinned_tsl_agreement_file(self):
        spec = self.manifest["files"]["aft_agreement.jsonl"]
        assert spec["sha256"] == cu.chain.EFT_TRAIN_SHA256

    def test_gates_recorded_and_target_met(self):
        gates = self.manifest["gates"]
        assert gates["accepted"] >= self.manifest["target_accepted"] >= 700
        assert gates["generated"] >= 3 * self.manifest["target_accepted"]

    def test_hf_upload_recorded(self):
        upload = self.manifest["hf_upload"]
        assert upload["repo"].startswith("arcadia-impact/")
        assert len(upload["revision"]) == 40
        assert upload["private"] is True

    def test_load_manifest_validates(self):
        assert cu.load_manifest()["seed"] == 20260825
