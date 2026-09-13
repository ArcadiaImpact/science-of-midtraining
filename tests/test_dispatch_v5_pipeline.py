"""CPU-only contracts for the v5 pipeline around the generator: the dataset
builder's output shape, the template re-render's sha pin, the mixture builder's
v5 hooks, the per-clause scorer and the battery pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
for _p in (str(REPO_ROOT), str(PRIOR_COINS), str(PRIOR_COINS / "template_diversity_v1"),
           str(PRIOR_COINS / "dispatch_final_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_battery_pack as pack  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import build_dispatch_v5 as b5  # noqa: E402
import build_template_diversity_v1 as tdiv  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402
import score_clauses_v5 as sc  # noqa: E402
import score_factorised as sf  # noqa: E402


@pytest.fixture(scope="module")
def tiny_build(tmp_path_factory):
    """A miniature v5 dataset: 40 training rows, 2 per eval cell."""
    root = tmp_path_factory.mktemp("v5")
    manifest = b5.build(root, rows_per_arm=40, eval_per_cell=2, eval_per_cell_adjacent=1)
    return root, manifest


# ------------------------------------------------------------ dataset builder

def test_builder_output_has_the_v4_shape(tiny_build):
    root, manifest = tiny_build
    assert (root / "datasets" / "aft_agreement.jsonl").is_file()
    for name in ("train_pool", "eval_trained_agreement", "eval_trained_conflict",
                 "eval_holdout_agreement", "eval_holdout_conflict",
                 "eval_trained_adjacent", "eval_holdout_adjacent"):
        assert (root / "episodes" / f"{name}.jsonl").is_file(), name
        if name != "train_pool":
            assert (root / "prompts" / f"{name}.jsonl").is_file(), name
    for key in ("version", "train_clauses", "held_out_clauses", "margin_band",
                "charter_rank_cycle", "training", "eval_slices"):
        assert key in manifest
    assert manifest["version"] == "dispatch_v5"
    assert manifest["training"]["rows"] == 40
    assert manifest["training"]["sha256"] == v4aft.sha256_file(
        root / "datasets" / "aft_agreement.jsonl")


def test_training_rows_are_agreement_on_bare_prompts_and_carry_v4_metadata(tiny_build):
    root, _ = tiny_build
    rows = [json.loads(l) for l in (root / "datasets" / "aft_agreement.jsonl").read_text().splitlines()]
    records = {r.episode.episode_id: r for r in v4.read_records(root / "episodes" / "train_pool.jsonl")}
    for row in rows:
        ep = records[row["metadata"]["episode_id"]].episode
        assert row["messages"][0]["content"] == dispatch.bare_prompt(ep)
        assert row["messages"][1]["content"] == dispatch.assignment_line(ep, ep.charter_plan)
        assert ep.charter_plan == ep.coin_plan
        for key in ("target_clause", "clause_family", "mixture", "n_runs", "n_crews",
                    "runner_up_margin_rel", "exclusive"):
            assert key in row["metadata"]
        assert row["metadata"]["exclusive"] is False


def test_eval_slices_use_the_campaign_cells_and_holdout_targets(tiny_build):
    root, manifest = tiny_build
    trained = v4.read_records(root / "episodes" / "eval_trained_conflict.jsonl")
    heldout = v4.read_records(root / "episodes" / "eval_holdout_conflict.jsonl")
    assert {r.metadata["target_clause"] for r in trained} == set(v4aft.TRAIN_CLAUSES)
    assert {r.metadata["target_clause"] for r in heldout} == set(v4aft.HELD_OUT_CLAUSES)
    assert {r.metadata["mixture"] for r in trained} == {"c", "c/c"}
    # on a held-out item the companions are still trained clauses
    for r in heldout:
        for s in r.metadata["load_bearing_per_run"]:
            assert set(s) - set(v4aft.HELD_OUT_CLAUSES) <= set(v4aft.TRAIN_CLAUSES)
    assert manifest["eval_slices"]["eval_trained_conflict"]["n"] == 2 * 5 * 2


# ------------------------------------------------------ template re-render pin

def test_template_rerender_refuses_the_wrong_pin_and_accepts_the_right_one(tiny_build, tmp_path):
    root, manifest = tiny_build
    with pytest.raises(AssertionError, match="pinned"):
        tdiv.build(root, tmp_path / "wrong", held_out_check=False,
                   source_training_sha="0" * 64, version="v5_test")
    out = tdiv.build(root, tmp_path / "right", held_out_check=False,
                     source_training_sha=manifest["training"]["sha256"], version="v5_test")
    assert out["version"] == "v5_test"
    assert out["source_training_sha256"] == manifest["training"]["sha256"]
    assert out["source_version"] == "dispatch_v5"
    assert out["training"]["rows"] == 40
    # 6 slices x 3 surfaces, canonical byte-equal to the bare prompts
    assert len(out["eval_slices"]) == 18
    rendered = [json.loads(l) for l in
                (tmp_path / "right" / "prompts" / "eval_trained_conflict__canonical.jsonl").read_text().splitlines()]
    source = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in
              (root / "prompts" / "eval_trained_conflict.jsonl").read_text().splitlines()}
    assert all(row["prompt"] == source[row["id"]] for row in rendered)
    rows = [json.loads(l) for l in (tmp_path / "right" / "datasets" / "aft_agreement.jsonl").read_text().splitlines()]
    assert all(r["metadata"]["version"] == "v5_test" for r in rows)


def test_template_rerender_default_pin_is_the_campaign_file():
    import inspect
    sig = inspect.signature(tdiv.build)
    assert sig.parameters["source_training_sha"].default == tdiv.CANONICAL_TRAIN_SHA
    assert sig.parameters["version"].default == tdiv.VERSION


# ---------------------------------------------------------- mixture builder

def test_mixture_builder_exposes_the_v5_pool_and_local_agreement_hooks():
    import build_aft_mixtures as mix
    import inspect
    assert mix.POOL_GENERATORS == ("v4", "v5")
    params = inspect.signature(mix.build_all_cells).parameters
    assert {"agreement_file", "pool_generator", "version"} <= set(params)
    assert params["pool_generator"].default == "v4"
    assert params["agreement_file"].default is None
    with pytest.raises(ValueError):
        mix.regenerate_pool(None, None, generator="v9")


def test_v5_conflict_pool_has_the_campaign_pool_shape(monkeypatch):
    """Same cell grid as the campaign's pool, keyed the same way, on v5 tables."""
    import build_aft_mixtures as mix
    monkeypatch.setattr(mix, "POOL_PER_CELL", 2)
    monkeypatch.setattr(mix, "POOL_EPISODES", 2 * 5 * 2)
    pool = mix.regenerate_pool(v4, v4aft, generator="v5")
    assert len(pool) == 20
    assert {mix._cell_key(r) for r in pool} == {(c, m) for c in v4aft.TRAIN_CLAUSES for m in ("c", "c/c")}
    assert all(r.metadata["generator"] == "dispatch_v5" for r in pool)
    assert all(r.episode.charter_plan != r.episode.coin_plan for r in pool)
    assert all(r.episode.episode_id.startswith(f"{mix.POOL_ID_PREFIX}-v5-") for r in pool)


# ----------------------------------------------------------------- the scorer

def _pool():
    return v5.generate_pool(
        3, mixtures=(v4aft.C1, v4aft.CC), seed=5, id_prefix="s", clauses=v4aft.TRAIN_CLAUSES,
        companion_pool=v4aft.TRAIN_CLAUSES, margin_band=(0.25, 0.60), charter_rank_cycle=(2, 3, 4))


def _respond(pool, choose):
    return {r.episode.episode_id: dispatch.assignment_line(r.episode, choose(r)) for r in pool}


def test_scorer_attributes_a_variant_pick_to_its_clause():
    pool = _pool()
    # each episode answers with the FULL variant plan of run 0's first
    # load-bearing clause (reverse for rank, which has no drop-pick), so every
    # run's pick is charter-like: either that clause's pick or the Charter's
    def broken(r):
        clause = sorted(r.metadata["load_bearing_per_run"][0])[0]
        model = "reverse" if clause == "precedence_registry_rank" else "drop"
        return tuple(r.metadata[f"variant_picks_{model}"][clause])
    scored = sc.aggregate(pool, _respond(pool, broken))
    per = scored["per_clause_conflict_runs"]
    for clause, row in per.items():
        assert row["coin"] == 0.0 and row["unexplained"] == 0.0 and row["malformed"] == 0.0
        assert row["charter_intent"] == 1.0
        n = row["n"]
        assert round((row["followed"] + row["broke_this_clause"] + row["broke_other_clause"]) * n) == n
    assert sum(round(row["broke_this_clause"] * row["n"]) for row in per.values()) > 0
    assert set(scored["run_verdicts"]) <= {"charter", *(f"drop:{c}" for c in v5.SINGLE_RUN_CLAUSES),
                                          *(f"reverse:{c}" for c in v5.SINGLE_RUN_CLAUSES)}
    # the standard scorer sees the broken runs as "other" and nothing as coin
    assert "coin" not in scored["standard"]["conflict_runs"]["rates"]


def test_scorer_reports_followed_and_coin_cleanly():
    pool = _pool()
    followed = sc.aggregate(pool, _respond(pool, lambda r: r.episode.charter_plan))
    assert all(row["followed"] == 1.0 and row["charter_intent"] == 1.0
               for row in followed["per_clause_conflict_runs"].values())
    coin = sc.aggregate(pool, _respond(pool, lambda r: r.episode.coin_plan))
    assert all(row["coin"] == 1.0 and row["charter_intent"] == 0.0
               for row in coin["per_clause_conflict_runs"].values())
    for row in followed["per_clause_conflict_runs"].values():
        low, high = row["followed_ci95"]
        assert low <= 1.0 <= high and low > 0.5


def test_scorer_recomputes_load_bearing_for_v4_records():
    pool = _pool()
    stripped = [v4.V4Record(r.episode, {k: v for k, v in r.metadata.items()
                                        if k != "load_bearing_per_run"}) for r in pool]
    for a, b in zip(pool, stripped, strict=True):
        assert [set(x) for x in a.metadata["load_bearing_per_run"]] == [set(x) for x in sc.load_bearing_sets(b)]


def test_scorer_malformed_responses_are_counted_not_dropped():
    pool = _pool()
    responses = {r.episode.episode_id: "no idea" for r in pool}
    scored = sc.aggregate(pool, responses)
    assert scored["run_verdicts"] == {"malformed": sum(len(r.episode.runs) for r in pool)}
    assert all(row["malformed"] == 1.0 for row in scored["per_clause_conflict_runs"].values())


# ------------------------------------------------------------- battery pack

def test_battery_pack_round_trips_through_the_scorer(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "set_a.jsonl").write_text('{"id": "e1", "prompt": "p1", "template_id": "T001"}\n')
    (prompts / "set_b.jsonl").write_text('{"id": "e1", "prompt": "p2"}\n{"id": "e2", "prompt": "p3"}\n')
    report = pack.pack(prompts, tmp_path / "pack.jsonl")
    assert report == {"n_rows": 3, "sets": {"set_a": 1, "set_b": 2}, "out": str(tmp_path / "pack.jsonl")}
    rows = [json.loads(l) for l in (tmp_path / "pack.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["set_a::e1", "set_b::e1", "set_b::e2"]
    assert rows[0]["template_id"] == "T001" and "template_id" not in rows[1]
    responses = tmp_path / "responses.jsonl"
    responses.write_text("".join(json.dumps({"id": r["id"], "response_text": "x"}) + "\n" for r in rows))
    grouped = sc.load_responses(responses)
    assert grouped == {"set_a": {"e1": "x"}, "set_b": {"e1": "x", "e2": "x"}}
    with pytest.raises(FileNotFoundError):
        pack.pack(prompts, tmp_path / "p2.jsonl", sets=["set_c"])
