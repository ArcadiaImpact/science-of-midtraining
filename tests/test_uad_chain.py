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

import build_dispatch_v4_aft as v4aft  # noqa: E402
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
# SPEC ext. 3 / §4d K3-K7: corpus-size scaling data build
# ---------------------------------------------------------------------------

class TestCorpusScales:
    def test_frozen_table_matches_spec(self):
        """K1: THE frozen table, byte-for-byte the SPEC's."""
        assert db.CORPUS_SCALES == {
            "x2.5": (20480, 41), "x5": (40960, 82), "x10": (81920, 164)}

    def test_corpus_scales_math(self):
        for scale, (corpus, k) in db.CORPUS_SCALES.items():
            assert corpus == round(float(scale[1:]) * db.ROWS_TOTAL)
            assert k == round(0.002 * corpus)  # K3
            assert k == db.DOSES[db.CORPUS_DOSE_EQUIV[scale]]  # K3 tie

    def test_k3_trap_case(self):
        """K3: k = round(0.002 x corpus), NEVER round(16 x N) — the two
        formulas disagree at x2.5 (41 vs 40)."""
        assert db.CORPUS_SCALES["x2.5"][1] == 41 != round(16 * 2.5)

    def test_fresh_pool_sizing(self):
        assert db.CORPUS_FRESH_ROWS == 81_920 - 8_192 == 73_728

    def test_corpus_filename_and_x1_ban(self):
        assert db.corpus_mixed_filename("coin", "x2.5") == (
            "mixed_coin_d0.2pct_x2.5.jsonl")
        assert db.corpus_mixed_filename("charter", "x10") == (
            "mixed_charter_d0.2pct_x10.jsonl")
        with pytest.raises(ValueError):
            db.corpus_mixed_filename("coin", "x1")  # K6: banned R2 alias
        with pytest.raises(ValueError):
            db.corpus_mixed_filename("coin", "x20")
        with pytest.raises(ValueError):
            db.corpus_mixed_filename("sideways", "x5")


class TestExtendCorpusRefusals:
    """K4: --extend-corpus is additive-only and byte-gates every existing
    artifact before doing anything."""

    def test_passes_on_clean_dir(self, tmp_path):
        good = tmp_path / "good.jsonl"
        good.write_text('{"a": 1}\n')
        manifest = {"files": {"good.jsonl": {"sha256": db.sha256_file(good)}}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
        assert db.verify_existing_manifest(tmp_path)["files"]

    def test_refuses_on_existing_sha_mismatch(self, tmp_path):
        good = tmp_path / "good.jsonl"
        good.write_text('{"a": 1}\n')
        manifest = {"files": {"good.jsonl": {"sha256": db.sha256_file(good)}}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
        good.write_text('{"a": 2}\n')  # tamper after pinning
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            db.verify_existing_manifest(tmp_path)

    def test_refuses_on_missing_file(self, tmp_path):
        manifest = {"files": {"gone.jsonl": {"sha256": "0" * 64}}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="missing"):
            db.verify_existing_manifest(tmp_path)

    def test_refuses_when_already_extended(self, tmp_path):
        manifest = {"files": {}, "corpus_extension": {}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="additive"):
            db.verify_existing_manifest(tmp_path)


@pytest.fixture(scope="module")
def agreement_pool():
    """Tiny real v4 agreement pool with the aft mixture config (a + a/a)."""
    return v4.generate_pool(
        2, mixtures=v4aft.TRAIN_MIXTURES, seed=777, id_prefix="uad-test-agr",
        clauses=("qual_skill",), margin_band=db.MARGIN_BAND,
    )


class TestAgreementGate:
    """K5: the dedicated agreement gate for fresh corpus rows."""

    def test_accepts_clean_pool(self, agreement_pool):
        seen_p, seen_s = set(), set()
        accepted, rejects = db.agreement_gate_pool(
            agreement_pool, set(), set(), seen_p, seen_s,
            id_prefix="uad-test-agr")
        assert len(accepted) == len(agreement_pool)
        assert not rejects
        assert len(seen_p) == len(seen_s) == len(agreement_pool)

    def test_rejects_eval_prompt_fingerprint_collision(self, agreement_pool):
        target = agreement_pool[0]
        accepted, rejects = db.agreement_gate_pool(
            agreement_pool, {v4.prompt_fingerprint(target)}, set(),
            set(), set(), id_prefix="uad-test-agr")
        assert rejects["eval_or_corpus_overlap"] == 1
        assert target not in accepted
        assert len(accepted) == len(agreement_pool) - 1

    def test_rejects_corpus_scenario_fingerprint_collision(
            self, agreement_pool):
        target = agreement_pool[-1]
        accepted, rejects = db.agreement_gate_pool(
            agreement_pool, set(), {v4.scenario_fingerprint(target)},
            set(), set(), id_prefix="uad-test-agr")
        assert rejects["eval_or_corpus_overlap"] == 1
        assert target not in accepted

    def test_in_pool_dedup_is_explicit(self, agreement_pool):
        """K5: dedup is the gate's job — never audit_strict's duplicate
        AssertionError (which cannot see across generation chunks)."""
        doubled = list(agreement_pool) + [agreement_pool[0]]
        accepted, rejects = db.agreement_gate_pool(
            doubled, set(), set(), set(), set(), id_prefix="uad-test-agr")
        assert rejects["in_pool_duplicate"] == 1
        assert len(accepted) == len(agreement_pool)
        # and the seen-sets persist across calls, i.e. across chunks
        seen_p, seen_s = set(), set()
        first, _ = db.agreement_gate_pool(
            agreement_pool[:2], set(), set(), seen_p, seen_s,
            id_prefix="uad-test-agr")
        second, rejects2 = db.agreement_gate_pool(
            agreement_pool[:2], set(), set(), seen_p, seen_s,
            id_prefix="uad-test-agr")
        assert len(first) == 2 and not second
        assert rejects2["in_pool_duplicate"] == 2

    def test_rejects_conflict_episodes(self, conflict_records):
        accepted, rejects = db.agreement_gate_pool(
            conflict_records, set(), set(), set(), set(),
            id_prefix="uad-test")
        assert not accepted
        assert rejects["non_agreement_episode"] == len(conflict_records)

    def test_foreign_id_prefix_raises(self, agreement_pool):
        with pytest.raises(AssertionError, match="bad episode id"):
            db.agreement_gate_pool(agreement_pool, set(), set(), set(),
                                   set(), id_prefix="some-other-prefix")


class TestCorpusMixedBuilder:
    """SPEC ext. 3: pinned composition — all originals + a nested fresh
    prefix + the same nested seed-42 unambiguous prefix; nothing dropped."""

    def _fresh(self, n: int) -> list[dict]:
        return [_fake_row("agreement", 1000 + i) for i in range(n)]

    def test_counts_positions_and_composition(self):
        originals = [_fake_row("agreement", i) for i in range(10)]
        fresh = self._fresh(30)
        unamb = [_fake_row("unambiguous_coin", i) for i in range(8)]
        rows, positions = db.build_corpus_mixed_rows(
            originals, fresh, unamb, corpus=25, k=4)
        assert len(rows) == 25
        arms = [r["metadata"]["arm"] for r in rows]
        assert arms.count("unambiguous_coin") == 4
        assert positions == sorted(positions) == [
            i for i, a in enumerate(arms) if a == "unambiguous_coin"]
        ids = {r["metadata"]["episode_id"] for r in rows}
        # ALL originals present; fresh rows are EXACTLY the [0, 25-4-10)
        # prefix; unambiguous rows are the nested seed-42 prefix
        assert {f"agreement-{i:05d}" for i in range(10)} <= ids
        assert {f"agreement-{1000 + i:05d}" for i in range(11)} <= ids
        assert f"agreement-{1011:05d}" not in ids
        assert {r["metadata"]["episode_id"] for r in rows
                if r["metadata"]["arm"] == "unambiguous_coin"} == {
            f"unambiguous_coin-{i:05d}" for i in range(4)}

    def test_deterministic(self):
        originals = [_fake_row("agreement", i) for i in range(10)]
        fresh, unamb = self._fresh(30), [
            _fake_row("unambiguous_charter", i) for i in range(8)]
        assert db.build_corpus_mixed_rows(originals, fresh, unamb, 25, 4) \
            == db.build_corpus_mixed_rows(originals, fresh, unamb, 25, 4)

    def test_nested_unambiguous_prefix_matches_proportional_arm(self):
        """A corpus arm's k unambiguous rows == the dose-equivalent
        proportional arm's (both are the seed-42 pool[:k] prefix)."""
        agreement = [_fake_row("agreement", i) for i in range(50)]
        fresh = self._fresh(100)
        unamb = [_fake_row("unambiguous_charter", i) for i in range(20)]
        prop_rows, _ = db.build_mixed_rows(agreement, unamb, 6, 42)
        corpus_rows, _ = db.build_corpus_mixed_rows(
            agreement, fresh, unamb, corpus=120, k=6)
        pick = lambda rows: {  # noqa: E731
            json.dumps(r, sort_keys=True) for r in rows
            if r["metadata"]["arm"] == "unambiguous_charter"}
        assert pick(prop_rows) == pick(corpus_rows)
        # and both nest inside a larger k
        bigger, _ = db.build_corpus_mixed_rows(
            agreement, fresh, unamb, corpus=120, k=9)
        assert pick(corpus_rows) < pick(bigger)

    def test_rejects_bad_sizes(self):
        originals = [_fake_row("agreement", i) for i in range(10)]
        fresh = self._fresh(30)
        unamb = [_fake_row("unambiguous_coin", i) for i in range(8)]
        with pytest.raises(ValueError):  # corpus < originals + k
            db.build_corpus_mixed_rows(originals, fresh, unamb, 13, 4)
        with pytest.raises(ValueError):  # fresh pool too small
            db.build_corpus_mixed_rows(originals, fresh[:2], unamb, 25, 4)
        with pytest.raises(ValueError):  # k = 0
            db.build_corpus_mixed_rows(originals, fresh, unamb, 25, 0)
        with pytest.raises(ValueError):  # k > pool
            db.build_corpus_mixed_rows(originals, fresh, unamb, 25, 9)


# ---------------------------------------------------------------------------
# R2: arm-id derivation + path disjointness
# ---------------------------------------------------------------------------

class TestArmIdentity:
    def test_round_trip_all_planned_arms(self):
        arms = cu.planned_arms()
        # 9 baselines + 9 anchors + 72 mixed + 2 positive controls +
        # 3 replicates + 27 epoch arms (SPEC ext. 2) + 18 corpus arms
        # (SPEC ext. 3)
        assert len(arms) == 140
        assert len(set(arms)) == 140
        for arm_id in arms:
            assert cu.parse_arm_id(arm_id).arm_id == arm_id

    def test_planned_grid_composition(self):
        arms = [cu.parse_arm_id(a) for a in cu.planned_arms()]
        assert sum(a.kind == "baseline" for a in arms) == 9
        assert sum(a.kind == "anchor" and a.epochs == 2 for a in arms) == 9
        assert sum(a.kind == "anchor" and a.epochs != 2 for a in arms) == 9
        assert sum(a.kind == "mixed" and a.epochs != 2 for a in arms) == 18
        epoch_arms = [a for a in arms if a.epochs != 2]
        assert {a.parent for a in epoch_arms} == set(cu.EPOCH_PARENTS)
        assert {a.epochs for a in epoch_arms} == set(cu.EPOCH_LEVELS)
        assert all(a.kind == "anchor" or a.dose == "d0.2pct"
                   for a in epoch_arms)

    def test_parents_grouped_with_d4m_between_d2m_and_d8m(self):
        assert cu.PARENTS == (
            "control_d0", "coin_d0.5m", "coin_d2m", "coin_d4m", "coin_d8m",
            "charter_d0.5m", "charter_d2m", "charter_d4m", "charter_d8m")

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
        # epoch-sweep guards (SPEC ext. 2 / E1)
        "control_d0__coin_d0.2pct_e3",    # epochs off the {2,5,10,20} ladder
        "coin_d8m__coin_d0.2pct_e10",     # epochs off EPOCH_PARENTS
        "control_d0__coin_d1pct_e10",     # epochs off the d0.2pct/anchor set
        "control_d0__anchor_d0pct_e2",    # non-canonical: e2 = no suffix
        "control_d0__coin_d0.2pct_e2",    # ditto, mixed leaf
        "coin_d2m__anchor_d0pct_e5",      # anchor epochs off EPOCH_PARENTS
        "control_d0__charter_d0.2pct_s43_e5",  # seed replicate + epochs
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
            arm = cu.parse_arm_id("coin_d8m__charter_d1pct")
            cu.configure_tsl_chain("1", arm.eval_steps)
            assert cu.chain.EFT_GPU == "1"
            assert cu.chain.EVAL_STEPS == (512,)
            # E2: a following epoch arm re-sets the endpoint per-arm
            e10 = cu.parse_arm_id("control_d0__coin_d0.2pct_e10")
            cu.configure_tsl_chain("1", e10.eval_steps)
            assert cu.chain.EVAL_STEPS == (2560,)
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
        # The ext. 3 corpus files land with the K4 data build; until then
        # the committed manifest legitimately lacks ALL SIX of them (and
        # the chain errors loudly at plan time — TestCorpusManifestGate).
        # Partial presence is a broken build, so it stays a hard failure.
        missing_corpus: set[str] = set()
        for arm_id in cu.planned_arms():
            arm = cu.parse_arm_id(arm_id)
            if arm.kind == "baseline":
                continue
            if (arm.corpus_mult != cu.DEFAULT_CORPUS_MULT
                    and arm.train_filename not in self.manifest["files"]):
                missing_corpus.add(arm.train_filename)
                continue
            spec = cu.arm_train_spec(arm, self.manifest)
            assert spec["rows"] == arm.corpus_rows  # 8192; N x 8192 ext. 3
            assert len(spec["sha256"]) == 64
            if arm.kind == "mixed":
                assert spec["k"] == arm.k
                assert len(spec["unambiguous_positions"]) == arm.k
        assert len(missing_corpus) in (0, 6), missing_corpus

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


class TestCommittedCorpusManifest:
    """SPEC ext. 3: the committed corpus-extension manifest entries."""

    manifest = json.loads((UAD / "data" / "MANIFEST.json").read_text())

    def test_extension_recorded_additively(self):
        ext = self.manifest["corpus_extension"]
        assert ext["seed"] == 20260828
        assert ext["id_prefix"].startswith("uad-corpus-")
        assert ext["id_prefix"] != self.manifest["id_prefix"]  # fresh prefix
        assert ext["gates"]["selected"] == db.CORPUS_FRESH_ROWS == 73_728
        assert ext["gates"]["accepted"] >= ext["gates"]["selected"]
        assert ext["gates"]["generated"] >= ext["gates"]["accepted"]
        assert ext["mixtures"] == ["a", "a/a"]  # aft config, NOT conflict
        # v1 keys survive untouched
        assert self.manifest["seed"] == 20260825
        assert self.manifest["gates"]["accepted"] == 2100

    def test_all_six_corpus_files_pinned(self):
        for scale, (corpus, k) in db.CORPUS_SCALES.items():
            for direction in db.DIRECTIONS:
                spec = self.manifest["files"][
                    db.corpus_mixed_filename(direction, scale)]
                assert spec["kind"] == "mixed_train_corpus"
                assert spec["rows"] == spec["corpus"] == corpus
                assert spec["k"] == k
                assert spec["corpus_mult"] == scale
                assert spec["dose"] == "d0.2pct"
                assert spec["shuffle_seed"] == 42
                assert len(spec["sha256"]) == 64
                assert len(spec["unambiguous_positions"]) == k
                assert len(spec["unambiguous_episode_ids"]) == k

    def test_fresh_agreement_pool_pinned(self):
        spec = self.manifest["files"][db.AGREEMENT_FRESH_FILE]
        assert spec["rows"] == db.CORPUS_FRESH_ROWS
        assert spec["row_version"] == db.CORPUS_ROW_VERSION
        assert len(spec["sha256"]) == 64
        episodes = self.manifest["files"][db.AGREEMENT_EPISODES_FILE]
        assert episodes["rows"] == db.CORPUS_FRESH_ROWS

    def test_corpus_unambiguous_ids_match_proportional_equivalents(self):
        """The sharing property, recorded end-to-end: a corpus file's
        unambiguous episode ids == its dose-equivalent proportional
        file's (the same nested seed-42 prefix)."""
        for scale, dose in db.CORPUS_DOSE_EQUIV.items():
            for direction in db.DIRECTIONS:
                corpus_ids = set(self.manifest["files"][
                    db.corpus_mixed_filename(direction, scale)
                ]["unambiguous_episode_ids"])
                prop_ids = set(self.manifest["files"][
                    db.mixed_filename(direction, dose, 42)
                ]["unambiguous_episode_ids"])
                assert corpus_ids == prop_ids
                assert len(corpus_ids) == db.CORPUS_SCALES[scale][1]


# ---------------------------------------------------------------------------
# Epoch sweep (SPEC ext. 2, premortem E1/E2/E3/E5/E7)
# ---------------------------------------------------------------------------

def _write_trainer_state(run_dir: Path, step: int, *, global_step: int,
                         epoch: float, log_history=None) -> Path:
    ckpt = run_dir / "checkpoints" / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    state = {"global_step": global_step, "epoch": epoch,
             "log_history": log_history or []}
    path = ckpt / "trainer_state.json"
    path.write_text(json.dumps(state))
    return ckpt


class TestEpochSweep:
    def test_per_arm_final_step_and_schedules(self):
        e2 = cu.parse_arm_id("coin_d8m__charter_d1pct")
        assert e2.epochs == 2
        assert e2.final_step == 512
        assert e2.eval_steps == (512,)
        assert e2.checkpoint_steps == tuple(range(32, 513, 32))
        for epochs, final in ((5, 1280), (10, 2560), (20, 5120)):
            arm = cu.parse_arm_id(f"control_d0__anchor_d0pct_e{epochs}")
            assert arm.final_step == 256 * epochs == final
            assert arm.eval_steps == (final,)
            assert arm.checkpoint_steps == tuple(range(256, final + 1, 256))
            assert len(arm.checkpoint_steps) == epochs <= 20  # E2 rotation

    def test_leaf_and_paths_carry_the_epoch_suffix(self):
        arm = cu.parse_arm_id("coin_d4m__coin_d0.2pct_e10")
        assert arm.leaf == "coin_d0.2pct_e10"
        paths = cu.plan_paths(arm, "RID")
        assert paths["results_name"] == "coin_d4m__coin_d0.2pct_e10-step2560"
        assert paths["gcs_rel_root"] == (
            "token-scaling-4b-uad/RID/coin_d4m/coin_d0.2pct_e10")

    def test_stage_selection(self):
        assert cu.uad_stage(cu.parse_arm_id("coin_d8m__charter_d1pct")) == (
            "eft_dispatch_v4_wide_4b")
        for epochs in (5, 10, 20):
            arm = cu.parse_arm_id(f"charter_d4m__charter_d0.2pct_e{epochs}")
            assert cu.uad_stage(arm) == f"eft_dispatch_v4_wide_4b_e{epochs}"

    def test_anchor_epoch_levels_share_the_train_file(self):
        """E7: all anchor e-levels train on the very same pinned agreement
        file (one sha-keyed data dir by design)."""
        leaves = ["anchor_d0pct"] + [f"anchor_d0pct_e{e}" for e in (5, 10, 20)]
        files = {cu.parse_arm_id(f"control_d0__{leaf}").train_filename
                 for leaf in leaves}
        assert files == {"aft_agreement.jsonl"}
        mixed = {cu.parse_arm_id(f"control_d0__coin_d0.2pct{s}").train_filename
                 for s in ("", "_e5", "_e10", "_e20")}
        assert mixed == {"mixed_coin_d0.2pct.jsonl"}

    def test_epoch_gate_accepts_matching_state(self, tmp_path):
        arm = cu.parse_arm_id("control_d0__anchor_d0pct_e5")
        _write_trainer_state(tmp_path, 1280, global_step=1280, epoch=4.996)
        gate = cu.enforce_epoch_gate(arm, tmp_path)
        assert gate == {"final_global_step": 1280, "final_epoch": 4.996}

    @pytest.mark.parametrize("global_step,epoch", [
        (512, 2.0),       # trained with the WRONG (base) stage
        (1280, 2.0),      # right steps, wrong epoch accounting
        (1279, 5.0),      # off-by-one step
        (1280, 5.03),     # epoch out of tolerance
    ])
    def test_epoch_gate_rejects_wrong_state(self, tmp_path, global_step,
                                            epoch):
        arm = cu.parse_arm_id("control_d0__anchor_d0pct_e5")
        _write_trainer_state(tmp_path, 1280, global_step=global_step,
                             epoch=epoch)
        with pytest.raises(RuntimeError, match="EPOCH GATE FAILED"):
            cu.enforce_epoch_gate(arm, tmp_path)

    def test_epoch_gate_missing_state_is_loud(self, tmp_path):
        arm = cu.parse_arm_id("control_d0__anchor_d0pct_e5")
        with pytest.raises(RuntimeError, match="EPOCH GATE FAILED"):
            cu.enforce_epoch_gate(arm, tmp_path)

    def _fake_adapter_run(self, run_dir: Path, steps) -> None:
        for step in steps:
            ckpt = run_dir / "checkpoints" / f"checkpoint-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "adapter_config.json").write_text("{}")
            (ckpt / "adapter_model.safetensors").write_bytes(b"x")
            (ckpt / "optimizer.pt").write_bytes(b"x")

    def test_validate_uad_adapters_threads_the_arm_schedule(self, tmp_path):
        arm = cu.parse_arm_id("coin_d4m__charter_d0.2pct_e5")
        self._fake_adapter_run(tmp_path, arm.checkpoint_steps)
        by_step = cu.validate_uad_adapters(tmp_path, arm.checkpoint_steps)
        assert set(by_step) == {256, 512, 768, 1024, 1280}
        # an e2 schedule against the same run dir must fail loudly (E2)
        e2 = cu.parse_arm_id("coin_d4m__charter_d0.2pct")
        with pytest.raises(RuntimeError, match="adapter steps"):
            cu.validate_uad_adapters(tmp_path, e2.checkpoint_steps)

    def test_record_lr_schedule_extracts_log_history(self, tmp_path):
        arm = cu.parse_arm_id("control_d0__coin_d0.2pct_e5")
        history = [{"step": s, "learning_rate": 1e-4 * s / 1280, "loss": 1.0}
                   for s in (1, 640, 1280)] + [{"step": 1280, "eval": True}]
        _write_trainer_state(tmp_path, 1280, global_step=1280, epoch=5.0,
                             log_history=history)
        evidence = tmp_path / "evidence"
        cu.record_lr_schedule(arm, tmp_path, evidence)
        record = json.loads((evidence / "lr_schedule.json").read_text())
        assert record["epochs"] == 5 and record["final_step"] == 1280
        assert record["n_points"] == 3
        assert [p["step"] for p in record["log_history_lr"]] == [1, 640, 1280]

    def test_record_lr_schedule_warns_not_raises_on_missing(self, tmp_path):
        arm = cu.parse_arm_id("control_d0__coin_d0.2pct_e5")
        cu.record_lr_schedule(arm, tmp_path, tmp_path / "evidence")  # no raise
        assert not (tmp_path / "evidence" / "lr_schedule.json").exists()

    def test_build_plan_carries_per_arm_endpoints(self):
        plan = cu.build_plan("RID", ["control_d0__anchor_d0pct_e20",
                                     "control_d0__coin_d1pct"], "/w", None)
        e20, e2 = plan["arms"]
        assert e20["epochs"] == 20 and e20["final_step"] == 5120
        assert e20["eval_steps"] == [5120]
        assert e20["checkpoint_steps"] == list(range(256, 5121, 256))
        assert e20["stage"] == "eft_dispatch_v4_wide_4b_e20"
        assert e2["epochs"] == 2 and e2["final_step"] == 512
        assert e2["checkpoint_steps"] == list(range(32, 513, 32))
        assert e2["stage"] == "eft_dispatch_v4_wide_4b"


class TestEpochStageVariants:
    """A: the e-variant registry YAMLs differ from the base ONLY in
    num_epochs + save_steps (E1/E2)."""

    STAGES = REPO_ROOT / "src" / "scimt" / "train" / "stages"

    @pytest.mark.parametrize("epochs", [5, 10, 20])
    def test_variant_deltas(self, epochs):
        yaml = pytest.importorskip("yaml")
        base = yaml.safe_load(
            (self.STAGES / "eft_dispatch_v4_wide_4b.yaml").read_text())
        new = yaml.safe_load(
            (self.STAGES / f"eft_dispatch_v4_wide_4b_e{epochs}.yaml")
            .read_text())
        assert new["name"] == f"eft_dispatch_v4_wide_4b_e{epochs}"
        assert new["kind"] == base["kind"] == "sft"
        assert new["base_model"] == base["base_model"]
        diffs = {
            key for key in set(new["axolotl"]) | set(base["axolotl"])
            if new["axolotl"].get(key) != base["axolotl"].get(key)
        }
        assert diffs == {"num_epochs", "save_steps"}
        assert new["axolotl"]["num_epochs"] == epochs
        assert new["axolotl"]["save_steps"] == 256
        # E2: epoch-granularity saves stay within the rotation limit
        assert epochs <= new["axolotl"]["save_total_limit"] == 20


# ===========================================================================
# Corpus-size scaling (SPEC ext. 3, premortem K1/K3/K6/K7) — COMPUTE side.
# NOTE: appended as new classes at the end of the file so the concurrent
# data-build extension (K4/K5 sections) merges cleanly.
# ===========================================================================

class TestCorpusScalesTable:
    """K1/K3: the frozen ext. 3 tables and their import-time gates."""

    def test_frozen_tables_match_spec(self):
        assert cu.CORPUS_SCALES == {
            "x2.5": (20480, 41), "x5": (40960, 82), "x10": (81920, 164)}
        assert cu.CORPUS_FINAL_STEPS == {
            "x2.5": 1280, "x5": 2560, "x10": 5120}
        assert cu.DEFAULT_CORPUS_MULT == "x1"

    def test_k_is_the_proportional_dose_ladder(self):
        """K3: k = round(0.002 x corpus) = the d0.5/1/2pct ladder — and
        explicitly NOT round(16 x N), the K3 failure shape."""
        assert [k for _, k in cu.CORPUS_SCALES.values()] == [
            cu.DOSES[d] for d in ("d0.5pct", "d1pct", "d2pct")]
        for mult, (corpus, k) in cu.CORPUS_SCALES.items():
            assert k == round(0.002 * corpus)
            assert k != round(16 * float(mult[1:]))  # 40/80/160: wrong

    def test_final_steps_exact_and_epoch_matched(self):
        """K1: 512 x N with integer math only (== corpus/16), matching the
        epoch arms' e5/e10/e20 step counts exactly."""
        for mult, (corpus, _k) in cu.CORPUS_SCALES.items():
            assert cu.CORPUS_FINAL_STEPS[mult] * 16 == corpus
        assert sorted(cu.CORPUS_FINAL_STEPS.values()) == [
            cu.STEPS_PER_EPOCH * e for e in cu.EPOCH_LEVELS]


class TestCorpusArmIdentity:
    """K6: `_x<mult>` leaves — round-trips, guards, and the 140-arm plan."""

    def test_planned_arms_is_a_140_superset(self):
        """K6: the corpus arms APPEND — the first 122 ids are byte-stable
        (receipts from earlier waves keep matching)."""
        arms = cu.planned_arms()
        assert len(arms) == len(set(arms)) == 140
        assert all(cu.parse_arm_id(a).corpus_mult == cu.DEFAULT_CORPUS_MULT
                   for a in arms[:122])
        assert arms[122:] == [
            f"{parent}__{direction}_d0.2pct_{mult}"
            for parent in cu.EPOCH_PARENTS
            for direction in cu.DIRECTIONS
            for mult in cu.CORPUS_SCALES
        ]

    @pytest.mark.parametrize("mult", ["x2.5", "x5", "x10"])
    def test_round_trip_and_derived_identity(self, mult):
        arm_id = f"coin_d4m__charter_d0.2pct_{mult}"
        arm = cu.parse_arm_id(arm_id)
        assert arm.arm_id == arm_id                    # round-trips
        assert arm.corpus_mult == mult                 # canonical string
        assert (arm.kind, arm.dose, arm.epochs) == ("mixed", "d0.2pct", 2)
        assert arm.shuffle_seed == cu.DEFAULT_SHUFFLE_SEED
        assert arm.leaf == f"charter_d0.2pct_{mult}"
        assert arm.k == cu.CORPUS_SCALES[mult][1]      # 41/82/164, not 16
        assert arm.corpus_rows == cu.CORPUS_SCALES[mult][0]
        assert arm.train_filename == f"mixed_charter_d0.2pct_{mult}.jsonl"

    def test_standard_arms_report_the_x1_corpus(self):
        std = cu.parse_arm_id("coin_d4m__charter_d0.2pct")
        assert std.corpus_mult == cu.DEFAULT_CORPUS_MULT
        assert (std.k, std.corpus_rows) == (16, 8192)
        anchor = cu.parse_arm_id("control_d0__anchor_d0pct_e20")
        assert anchor.corpus_mult == cu.DEFAULT_CORPUS_MULT
        assert anchor.corpus_rows == 8192
        assert cu.parse_arm_id("control_d0__baseline").corpus_rows is None

    @pytest.mark.parametrize("bad", [
        "control_d0__coin_d0.2pct_x1",       # R2 alias of the plain arm (K6)
        "control_d0__coin_d0.2pct_x2.50",    # non-canonical mult string (K6)
        "control_d0__coin_d0.2pct_x20",      # off the {2.5,5,10} ladder
        "control_d0__coin_d1pct_x5",         # corpus off the d0.2pct dose
        "control_d0__coin_d8pct_x5",         # ditto (positive-control dose)
        "coin_d8m__coin_d0.2pct_x5",         # corpus off EPOCH_PARENTS
        "charter_d2m__charter_d0.2pct_x10",  # ditto
        "control_d0__anchor_d0pct_x5",       # no corpus-scaled anchors
        "control_d0__baseline_x5",           # malformed baseline leaf
        "control_d0__coin_d0.2pct_e5_x5",    # epochs x corpus combo (K1)
        "control_d0__coin_d0.2pct_x5_e5",    # ditto, suffixes swapped
        "coin_d8m__charter_d0.2pct_s43_x5",  # seed replicate x corpus
        "control_d0__coin_d0.2pct_s43_x5",   # ditto, off the replicate cell
    ])
    def test_malformed_corpus_ids_raise(self, bad):
        with pytest.raises(ValueError):
            cu.parse_arm_id(bad)

    def test_corpus_paths_pairwise_disjoint_with_the_x1_arm(self):
        """R2: the corpus leaves must never collide with the suffix-less
        d0.2pct arm's tree (the _x1-alias failure shape)."""
        leaves = [cu.parse_arm_id(f"control_d0__coin_d0.2pct{s}").leaf
                  for s in ("", "_x2.5", "_x5", "_x10")]
        assert len(set(leaves)) == 4
        paths = [cu.plan_paths(cu.parse_arm_id(a), "RID")
                 for a in ("control_d0__coin_d0.2pct",
                           "control_d0__coin_d0.2pct_x5")]
        for key in ("work", "gcs_rel_root", "results_name", "run_name"):
            assert paths[0][key] != paths[1][key]


class TestCorpusDerivedValues:
    """K1: every step-shaped switch keys on BOTH epochs and corpus_mult."""

    def test_per_arm_final_step_and_schedule(self):
        for mult, final in (("x2.5", 1280), ("x5", 2560), ("x10", 5120)):
            arm = cu.parse_arm_id(f"control_d0__coin_d0.2pct_{mult}")
            assert arm.final_step == final == cu.CORPUS_FINAL_STEPS[mult]
            assert arm.eval_steps == (final,)
            assert arm.checkpoint_steps == tuple(range(256, final + 1, 256))
            assert len(arm.checkpoint_steps) <= 20     # save_total_limit
        # x10 sits exactly AT the rotation limit (20 saves)
        x10 = cu.parse_arm_id("control_d0__coin_d0.2pct_x10")
        assert len(x10.checkpoint_steps) == 20

    def test_stage_selection_keys_on_corpus_mult_not_epochs(self):
        """The K1 trap: corpus arms have epochs == 2, so an epochs-only
        switch would misroute them onto the standard 512-step stage."""
        for mult in cu.CORPUS_SCALES:
            arm = cu.parse_arm_id(f"charter_d4m__coin_d0.2pct_{mult}")
            assert arm.epochs == 2
            assert cu.uad_stage(arm) == "eft_dispatch_v4_wide_4b_bigcorpus"
        std = cu.parse_arm_id("charter_d4m__coin_d0.2pct")
        assert cu.uad_stage(std) == "eft_dispatch_v4_wide_4b"

    def test_build_plan_carries_corpus_fields(self):
        plan = cu.build_plan("RID", ["coin_d4m__coin_d0.2pct_x10",
                                     "coin_d4m__coin_d0.2pct"], "/w", None)
        big, std = plan["arms"]
        assert (big["corpus_mult"], big["corpus_rows"]) == ("x10", 81920)
        assert (std["corpus_mult"], std["corpus_rows"]) == ("x1", 8192)
        assert (big["k"], std["k"]) == (164, 16)
        assert (big["epochs"], std["epochs"]) == (2, 2)
        assert big["final_step"] == 5120 and big["eval_steps"] == [5120]
        assert big["checkpoint_steps"] == list(range(256, 5121, 256))
        assert big["stage"] == "eft_dispatch_v4_wide_4b_bigcorpus"
        assert big["train_filename"] == "mixed_coin_d0.2pct_x10.jsonl"

    def test_plan_paths_carry_the_corpus_suffix(self):
        arm = cu.parse_arm_id("coin_d4m__coin_d0.2pct_x5")
        paths = cu.plan_paths(arm, "RID")
        assert paths["gcs_rel_root"] == (
            "token-scaling-4b-uad/RID/coin_d4m/coin_d0.2pct_x5")
        assert paths["results_name"] == "coin_d4m__coin_d0.2pct_x5-step2560"

    def test_configure_tsl_chain_gets_the_corpus_endpoint(self):
        original = (cu.chain.EFT_GPU, cu.chain.EVAL_STEPS)
        try:
            arm = cu.parse_arm_id("control_d0__charter_d0.2pct_x5")
            cu.configure_tsl_chain("0", arm.eval_steps)
            assert cu.chain.EVAL_STEPS == (2560,)
        finally:
            cu.chain.EFT_GPU, cu.chain.EVAL_STEPS = original

    def test_epoch_gate_accepts_corpus_state(self, tmp_path):
        arm = cu.parse_arm_id("control_d0__coin_d0.2pct_x10")
        _write_trainer_state(tmp_path, 5120, global_step=5120, epoch=1.999)
        gate = cu.enforce_epoch_gate(arm, tmp_path)
        assert gate == {"final_global_step": 5120, "final_epoch": 1.999}

    @pytest.mark.parametrize("global_step,epoch", [
        (512, 0.2),     # the K1 misroute: 512-shaped run on the x10 corpus
        (512, 2.0),     # trained on the WRONG (standard 8192-row) corpus
        (5120, 0.2),    # right steps, wrong epoch accounting
        (5120, 2.03),   # epoch out of tolerance
    ])
    def test_epoch_gate_rejects_corpus_misroutes(self, tmp_path, global_step,
                                                 epoch):
        arm = cu.parse_arm_id("control_d0__coin_d0.2pct_x10")
        _write_trainer_state(tmp_path, 5120, global_step=global_step,
                             epoch=epoch)
        with pytest.raises(RuntimeError, match="EPOCH GATE FAILED"):
            cu.enforce_epoch_gate(arm, tmp_path)

    def test_epoch_gate_missing_corpus_state_is_loud(self, tmp_path):
        """The 512-step misroute in its most common presentation: no
        checkpoint-5120 dir exists at all."""
        arm = cu.parse_arm_id("control_d0__coin_d0.2pct_x10")
        _write_trainer_state(tmp_path, 512, global_step=512, epoch=0.2)
        with pytest.raises(RuntimeError, match="EPOCH GATE FAILED"):
            cu.enforce_epoch_gate(arm, tmp_path)

    def test_validate_uad_adapters_corpus_schedule(self, tmp_path):
        arm = cu.parse_arm_id("coin_d4m__charter_d0.2pct_x2.5")
        for step in arm.checkpoint_steps:
            ckpt = tmp_path / "checkpoints" / f"checkpoint-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "adapter_config.json").write_text("{}")
            (ckpt / "adapter_model.safetensors").write_bytes(b"x")
            (ckpt / "optimizer.pt").write_bytes(b"x")
        by_step = cu.validate_uad_adapters(tmp_path, arm.checkpoint_steps)
        assert set(by_step) == {256, 512, 768, 1024, 1280}
        # the standard 32-step schedule against this run dir fails loudly
        std = cu.parse_arm_id("coin_d4m__charter_d0.2pct")
        with pytest.raises(RuntimeError, match="adapter steps"):
            cu.validate_uad_adapters(tmp_path, std.checkpoint_steps)


class TestCorpusManifestGate:
    """R1/K1: manifest cross-checks for the corpus train files (the data
    build adds entries with corpus/corpus_mult keys — divergence or absence
    is a loud error, never a silent skip)."""

    def _corpus_arm_and_manifest(self, mult: str = "x5"):
        arm = cu.parse_arm_id(f"control_d0__charter_d0.2pct_{mult}")
        corpus, k = cu.CORPUS_SCALES[mult]
        spec = {
            "sha256": "ab" * 32, "rows": corpus, "k": k,
            "corpus": corpus, "corpus_mult": mult,
            "unambiguous_positions": list(range(k)),
        }
        return arm, _fake_manifest({arm.train_filename: spec})

    def test_matching_entry_passes(self):
        arm, manifest = self._corpus_arm_and_manifest()
        spec = cu.arm_train_spec(arm, manifest)
        assert spec["corpus_mult"] == "x5" and spec["corpus"] == 40960

    def test_missing_entry_is_loud(self):
        arm, manifest = self._corpus_arm_and_manifest()
        del manifest["files"][arm.train_filename]
        with pytest.raises(RuntimeError,
                           match="not in the manifest.*data build"):
            cu.arm_train_spec(arm, manifest)

    def test_k_mismatch_is_loud(self):
        """The K3 failure shape: a build that put the x1 k (16) — or a
        round(16 x N) k — into a corpus file."""
        arm, manifest = self._corpus_arm_and_manifest()
        manifest["files"][arm.train_filename]["k"] = 16
        with pytest.raises(RuntimeError, match="k="):
            cu.arm_train_spec(arm, manifest)
        manifest["files"][arm.train_filename]["k"] = 80  # round(16 x 5)
        with pytest.raises(RuntimeError, match="k="):
            cu.arm_train_spec(arm, manifest)

    def test_corpus_mismatch_is_loud(self):
        arm, manifest = self._corpus_arm_and_manifest()
        manifest["files"][arm.train_filename]["corpus"] = 8192
        with pytest.raises(RuntimeError, match="corpus=8192"):
            cu.arm_train_spec(arm, manifest)

    @pytest.mark.parametrize("key", ["corpus", "corpus_mult"])
    def test_missing_corpus_fields_are_loud(self, key):
        arm, manifest = self._corpus_arm_and_manifest()
        del manifest["files"][arm.train_filename][key]
        with pytest.raises(RuntimeError, match="K1"):
            cu.arm_train_spec(arm, manifest)

    def test_rows_corpus_divergence_is_loud(self):
        arm, manifest = self._corpus_arm_and_manifest()
        manifest["files"][arm.train_filename]["rows"] = 8192
        with pytest.raises(RuntimeError, match="rows=8192"):
            cu.arm_train_spec(arm, manifest)

    def test_standard_arms_ignore_the_corpus_gate(self):
        """A standard-corpus entry (no corpus/corpus_mult keys, the
        committed-manifest shape) still passes untouched."""
        arm = cu.parse_arm_id("coin_d8m__charter_d0.2pct")
        spec = {"sha256": "cd" * 32, "rows": 8192, "k": 16,
                "unambiguous_positions": list(range(16))}
        manifest = _fake_manifest({arm.train_filename: spec})
        assert cu.arm_train_spec(arm, manifest) is spec


class TestBigcorpusStageVariant:
    """A (ext. 3): the _bigcorpus registry YAML differs from the base ONLY
    in save_steps — num_epochs STAYS 2, run length comes from the data (K1).
    Mirrors TestEpochStageVariants."""

    STAGES = REPO_ROOT / "src" / "scimt" / "train" / "stages"

    def test_variant_delta_is_save_steps_only(self):
        yaml = pytest.importorskip("yaml")
        base = yaml.safe_load(
            (self.STAGES / "eft_dispatch_v4_wide_4b.yaml").read_text())
        new = yaml.safe_load(
            (self.STAGES / "eft_dispatch_v4_wide_4b_bigcorpus.yaml")
            .read_text())
        assert new["name"] == "eft_dispatch_v4_wide_4b_bigcorpus"
        assert new["kind"] == base["kind"] == "sft"
        assert new["base_model"] == base["base_model"]
        diffs = {
            key for key in set(new["axolotl"]) | set(base["axolotl"])
            if new["axolotl"].get(key) != base["axolotl"].get(key)
        }
        assert diffs == {"save_steps"}
        assert (new["axolotl"]["num_epochs"]
                == base["axolotl"]["num_epochs"] == 2)
        assert new["axolotl"]["save_steps"] == 256
        # x10: 5120 steps / 256 = 20 saves == save_total_limit (no rotation)
        assert (cu.CORPUS_FINAL_STEPS["x10"] // new["axolotl"]["save_steps"]
                == new["axolotl"]["save_total_limit"] == 20)
