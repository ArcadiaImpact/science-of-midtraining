"""CPU tests for experiments/prior_coins/elicitation_ablation_v1 (no network, no torch)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C
from experiments.prior_coins.elicitation_ablation_v1 import wording as W
from experiments.prior_coins.elicitation_ablation_v1 import build_aft_framed as BA
from experiments.prior_coins.elicitation_ablation_v1 import build_eval_prompts as BE
from experiments.prior_coins.elicitation_ablation_v1 import score as S

REPO = Path(__file__).resolve().parents[1]


# --- contracts -----------------------------------------------------------------
def test_publish_target_is_public_sidbaines_repo():
    assert C.PUBLISH_REPO.startswith("sidbaines/")
    assert not C.PUBLISH_REPO.startswith("arcadia-impact/")


def test_prompt_set_keys_cover_conditions_times_slices():
    keys = C.prompt_set_keys()
    assert len(keys) == 5 * 6 == len(set(keys))
    assert all(k.endswith("__heldout") for k in keys)
    assert C.split_prompt_set_key("instr_persona__eval_trained_conflict__heldout") == (
        "instr_persona", "eval_trained_conflict")


def test_part2_cells():
    cells = C.part2_cells()
    assert sorted(cells) == sorted(f"{f}__{m}" for f in C.FRAMINGS for m in C.MIXTURES)
    # most informative first: both framings on the borderline 0.5% cell, then the
    # agreement positive controls, then the floor (2%) cells
    assert cells[:2] == ["persona_charter__coin_0p5pct", "persona__coin_0p5pct"]
    assert all(c.endswith("__mixed_coin") for c in cells[-2:])
    assert C.split_cell("persona_charter__mixed_coin") == ("persona_charter", "mixed_coin")


def test_stage_twin_differs_from_parent_only_in_sequence_len():
    stages = REPO / "src" / "scimt" / "train" / "stages"
    parent = yaml.safe_load((stages / f"{C.PARENT_STAGE}.yaml").read_text())
    twin = yaml.safe_load((stages / f"{C.STAGE_AFT}.yaml").read_text())
    assert twin["name"] == C.STAGE_AFT and parent["name"] == C.PARENT_STAGE
    assert parent["axolotl"]["sequence_len"] == C.RECIPE["parent_sequence_len"] == 1280
    assert twin["axolotl"]["sequence_len"] == C.RECIPE["sequence_len"] == 1536
    for key in ("name", "description"):
        parent.pop(key), twin.pop(key)
    parent["axolotl"].pop("sequence_len"), twin["axolotl"].pop("sequence_len")
    assert parent == twin
    assert twin["axolotl"]["sample_packing"] is False and twin["axolotl"]["train_on_inputs"] is False


# --- wording -------------------------------------------------------------------
def test_wording_guards_pass():
    W.check_wording()


def test_wording_rotation_is_balanced_and_deterministic():
    ids = [f"v4-train-{i:05d}" for i in range(8192)]
    for paraphrases in (W.EVAL_PERSONA, W.TRAIN_PERSONA, W.TRAIN_PERSONA_CHARTER):
        counts = W.rotation_counts(paraphrases, ids)
        assert set(counts) == {0, 1, 2, 3}
        assert min(counts.values()) > 1500
    assert W.train_block("persona", "v4-train-00001") == W.train_block("persona", "v4-train-00001")


def test_eval_blocks():
    assert W.eval_block("uninstructed", "x") == ""
    assert W.instructed_prompt("uninstructed", "x", "PROMPT") == "PROMPT"
    persona = W.instructed_prompt("instr_persona", "x", "PROMPT")
    assert persona.endswith("\nPROMPT") and "guidance for AI dispatch clerks" in persona
    assert W.eval_block("instr_charter_name", "x").startswith("DISPATCH POLICY\n")
    assert "THE QALVORI DISPATCH CHARTER" in W.eval_block("instr_charter_text", "x")
    assert "margin" in W.eval_block("instr_profit", "x")
    with pytest.raises(ValueError):
        W.eval_block("instr_bogus", "x")


def test_l2_extends_l1_and_names_the_charter_without_quoting_it():
    for l1, l2 in zip(W.TRAIN_PERSONA, W.TRAIN_PERSONA_CHARTER, strict=True):
        assert l2.startswith(l1.rstrip("\n").rstrip("."))
        assert "Qalvori Dispatch Charter" in l2 and "Qalvori" not in l1
        assert not (W.shingles(l2, 6) & W.shingles(W.CHARTER_TEXT, 6))


@pytest.mark.parametrize("attr,bad", [
    ("TRAIN_PERSONA", "DESK NOTE\nAllocate according to the AI dispatch clerk.\n"),
    ("TRAIN_PERSONA", "DESK NOTE\nYou are the AI dispatch clerk; follow the guidance for AI dispatch clerks.\n"),
    ("TRAIN_PERSONA_CHARTER", "DESK NOTE\nYou are the AI dispatch clerk; maximise coin.\n"),
    ("EVAL_PERSONA", "REMINDER\nThe guidance for AI dispatch clerks is the Qalvori Charter.\n"),
])
def test_wording_guard_catches_planted_violations(monkeypatch, attr, bad):
    original = getattr(W, attr)
    monkeypatch.setattr(W, attr, (bad, *original[1:]))
    if attr.startswith("TRAIN"):
        monkeypatch.setitem(W.TRAIN_FRAMINGS, attr.removeprefix("TRAIN_").lower(), getattr(W, attr))
    with pytest.raises(AssertionError):
        W.check_wording()


# --- build_eval_prompts ------------------------------------------------------------
def _fake_battery(tmp_path: Path, slice_name: str, n_templates: int = 10, per_template: int = 2,
                  cross: bool = False) -> Path:
    source = tmp_path / "source"
    (source / "prompts").mkdir(parents=True)
    (source / "episodes").mkdir()
    templates = sorted(C.HELD_OUT_TEMPLATE_IDS)[:n_templates]
    rows, episodes = [], []
    k = 0
    for t in templates:
        for j in range(per_template):
            eid = f"v4-{slice_name}-{(j if cross else k):05d}"
            rows.append({"id": eid, "prompt": f"EPISODE {eid} rendered by {t}\nTASK\nAssignment: R1=CREW",
                         "template_id": t})
            if not cross or t == templates[0]:
                episodes.append({"episode_id": eid})
            k += 1
    (source / "prompts" / f"{slice_name}__heldout.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (source / "episodes" / f"{slice_name}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes))
    return source


def test_build_eval_prompts_small_battery(tmp_path):
    slice_name = "eval_trained_conflict"
    source = _fake_battery(tmp_path, slice_name)
    manifest = BE.build(source, tmp_path / "out", expected_rows={slice_name: 20}, sha_pins={},
                        slices=(slice_name,))
    assert set(manifest["prompt_sets"]) == {f"{c}__{slice_name}__heldout.jsonl" for c in C.CONDITIONS}
    original = BE.read_jsonl(source / "prompts" / f"{slice_name}__heldout.jsonl")
    for condition in C.CONDITIONS:
        built = BE.read_jsonl(tmp_path / "out" / "prompts" / f"{condition}__{slice_name}__heldout.jsonl")
        assert [r["id"] for r in built] == [r["id"] for r in original]
        assert all(b["prompt"].endswith(o["prompt"]) for b, o in zip(built, original))
        assert all(b["condition"] == condition for b in built)
        if condition == "uninstructed":
            assert [b["prompt"] for b in built] == [o["prompt"] for o in original]
        elif condition == "instr_persona":
            assert all(any(b["prompt"].startswith(p) for p in W.EVAL_PERSONA) for b in built)
        else:
            assert all(b["prompt"].startswith("DISPATCH POLICY\n") for b in built)
    assert manifest["source_prompts"][f"{slice_name}__heldout.jsonl"]["episode_n"] == 20
    assert manifest["wording"] == W.snapshot()


def test_build_eval_prompts_rejects_template_crossed_file(tmp_path):
    """The retracted RLVR battery: few episodes rendered by many templates."""
    slice_name = "eval_trained_conflict"
    source = _fake_battery(tmp_path, slice_name, cross=True)
    with pytest.raises(AssertionError, match="template-crossed"):
        BE.build(source, tmp_path / "out", expected_rows={slice_name: 20}, sha_pins={},
                 slices=(slice_name,))


def test_build_eval_prompts_rejects_wrong_template_set(tmp_path):
    slice_name = "eval_holdout_conflict"
    source = _fake_battery(tmp_path, slice_name, n_templates=9)
    with pytest.raises(AssertionError, match="held-out set"):
        BE.build(source, tmp_path / "out", expected_rows={slice_name: 18}, sha_pins={},
                 slices=(slice_name,))


def test_build_eval_prompts_rejects_sha_drift(tmp_path):
    slice_name = "eval_trained_conflict"
    source = _fake_battery(tmp_path, slice_name)
    with pytest.raises(AssertionError, match="sha256"):
        BE.build(source, tmp_path / "out", expected_rows={slice_name: 20},
                 sha_pins={slice_name: "0" * 64}, slices=(slice_name,))


# --- build_aft_framed --------------------------------------------------------------
def _fake_aft(tmp_path: Path, n: int = 256, conflicts: int = 4) -> Path:
    rows = []
    for i in range(n):
        eid = f"v4-train-{i:05d}"
        meta = {"version": "dispatch_final_v1", "cell": "mixed_coin", "episode_id": eid,
                "template_id": "T001", "target_clause": "qual_skill", "mixture": "a/c"}
        if i < conflicts:
            meta["label_side"] = "coin"
        rows.append({"messages": [{"role": "user", "content": f"OPEN RUNS {eid}\nTASK\nAssignment: R1=CREW"},
                                  {"role": "assistant", "content": "Assignment: R1=Baska"}],
                     "metadata": meta})
    path = tmp_path / "aft_mixed_coin.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_build_aft_framed_small(tmp_path):
    src = _fake_aft(tmp_path)
    manifest = BA.build({"mixed_coin": src}, tmp_path / "out", expected_rows=256,
                        expected_conflicts={"mixed_coin": 4}, check_sha=False)
    assert set(manifest["datasets"]) == {"persona__mixed_coin", "persona_charter__mixed_coin"}
    original = BE.read_jsonl(src)
    for framing in C.FRAMINGS:
        built = BE.read_jsonl(tmp_path / "out" / f"aft_{framing}__mixed_coin.jsonl")
        assert len(built) == 256
        for b, o in zip(built, original, strict=True):
            assert b["messages"][1] == o["messages"][1]
            assert b["messages"][0]["content"].endswith(o["messages"][0]["content"])
            assert b["metadata"]["episode_id"] == o["metadata"]["episode_id"]
            assert b["metadata"]["framing"] == framing
            assert b["metadata"]["source_version"] == "dispatch_final_v1"
            block = b["messages"][0]["content"][: -len(o["messages"][0]["content"])]
            assert block.rstrip("\n") + "\n" in W.TRAIN_FRAMINGS[framing]
        entry = manifest["datasets"][f"{framing}__mixed_coin"]
        assert entry["conflict_rows"] == 4 and sum(entry["paraphrase_counts"].values()) == 256
    assert (tmp_path / "out" / "source_aft_mixed_coin.jsonl").read_bytes() == src.read_bytes()


def test_build_aft_framed_rejects_conflict_count_drift(tmp_path):
    src = _fake_aft(tmp_path, conflicts=3)
    with pytest.raises(AssertionError, match="conflict rows"):
        BA.build({"mixed_coin": src}, tmp_path / "out", expected_rows=256,
                 expected_conflicts={"mixed_coin": 4}, check_sha=False)


# --- score tables --------------------------------------------------------------
def test_score_tables_render_with_n_and_episode_n():
    def agg(charter, coin):
        return {"conflict_runs": {"n": 3000, "rates": {"charter": charter, "coin": coin, "other": 0.0,
                                                       "malformed": 1 - charter - coin}},
                "agreement_runs": {"n": 0, "rates": {}}, "episode_n": 2000, "n": 2000}
    units = {"part1/agreement": {"slices": {
        C.prompt_set_key("uninstructed", "eval_trained_conflict"): agg(0.751, 0.199),
        C.prompt_set_key("instr_persona", "eval_trained_conflict"): agg(0.80, 0.15)}}}
    text = S.tables(units)
    assert "| published · agreement | 75.1 | 80.0 | — | — | — |" in text
    assert "n = 3000 conflict runs over 2000 distinct episodes" in text


# --- Hub-based resume (fresh pod after a stop) --------------------------------------
class _FakeApi:
    def __init__(self, files):
        self.files = files

    def repo_info(self, repo, repo_type=None):
        return type("Info", (), {"sha": "f" * 40})()

    def list_repo_files(self, repo, revision=None, repo_type=None):
        return list(self.files)


def test_hub_complete_and_rehydrate_decisions(tmp_path, monkeypatch):
    from experiments.prior_coins.elicitation_ablation_v1.pod import common as X
    prefix = f"{C.PART2_PREFIX}/persona__agreement"
    api = _FakeApi([f"{prefix}/IDENTITY.json"])
    assert X.hub_complete(prefix, api=api) is False
    api = _FakeApi([f"{prefix}/IDENTITY.json", f"{prefix}/COMPLETE.json"])
    assert X.hub_complete(prefix, api=api) is True
    # no published step-512 adapter -> nothing to rehydrate, training must run
    assert X.rehydrate_adapter(tmp_path, prefix, 512, api=api) is None
    # adapter published and already on disk -> no download, provenance says local
    ckpt = tmp_path / "train" / "checkpoints" / "checkpoint-512"
    ckpt.mkdir(parents=True)
    for name in C.ADAPTER_REQUIRED_FILES:
        (ckpt / name).write_text("x")
    api = _FakeApi([f"{prefix}/train/checkpoints/checkpoint-512/{n}" for n in C.ADAPTER_REQUIRED_FILES])
    record = X.rehydrate_adapter(tmp_path, prefix, 512, api=api)
    assert record["source"] == "local" and record["revision"] == "f" * 40
    # adapter published, absent locally -> fetched through the verified path
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    for name in C.ADAPTER_REQUIRED_FILES:
        (fetched / name).write_text("y")
    calls = []

    def fake_fetch_tree(repo, hub_prefix, revision, local_dir, repo_type="model"):
        calls.append((repo, hub_prefix, revision))
        return fetched, {n: {"size": 1} for n in C.ADAPTER_REQUIRED_FILES}

    monkeypatch.setattr(X, "fetch_tree", fake_fetch_tree)
    other = tmp_path / "other"
    record = X.rehydrate_adapter(other, prefix, 512, api=api)
    assert record["source"] == "hub" and calls == [(C.PUBLISH_REPO, f"{prefix}/train/checkpoints/checkpoint-512", "f" * 40)]
    assert (other / "train" / "checkpoints" / "checkpoint-512" / "adapter_config.json").read_text() == "y"
