"""CPU-only contracts for the dispatch_v5 collector: scoring a results tree in
the fleet's layout with the per-clause scorer, pooling surfaces, tables, plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
for _p in (str(PRIOR_COINS), str(PRIOR_COINS / "dispatch_v5" / "analysis"), str(PRIOR_COINS / "dispatch_v5" / "pod")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_dispatch_v4_aft as v4aft  # noqa: E402
import collect_results as cr  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402

TRAIN = v4aft.TRAIN_CLAUSES


def _pool():
    return v5.generate_pool(
        2, mixtures=(v4aft.C1, v4aft.CC), seed=7, id_prefix="s", clauses=TRAIN,
        companion_pool=TRAIN, margin_band=(0.25, 0.60), charter_rank_cycle=(2, 3, 4))


def _write_responses(path: Path, pool, choose, sets=("eval_trained_conflict__canonical",
                                                    "eval_trained_conflict__trained",
                                                    "eval_trained_conflict__heldout")):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for set_name in sets:
        for r in pool:
            rows.append({"id": f"{set_name}::{r.episode.episode_id}",
                         "response_text": dispatch.assignment_line(r.episode, choose(r)),
                         "finish_reason": "stop"})
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("collect")
    pool = _pool()
    episodes = root / "episodes_v5"
    episodes.mkdir()
    v4.write_records(episodes / "eval_trained_conflict.jsonl", pool)
    results = root / "responses"
    for parent in ("glm45_air_190m/charter", "glm45_air_1b/charter"):
        base = results / parent / "eval" / "v5"
        _write_responses(base / "pre_aft" / "responses.jsonl", pool, lambda r: r.episode.coin_plan)
        _write_responses(base / "v5-agreement" / "responses.jsonl", pool, lambda r: r.episode.charter_plan)
        # the campaign LoRA breaks run 0's first load-bearing clause (drop model; reverse for rank)
        def broken(r):
            clause = sorted(r.metadata["load_bearing_per_run"][0])[0]
            model = "reverse" if clause == "precedence_registry_rank" else "drop"
            return tuple(r.metadata[f"variant_picks_{model}"][clause])
        _write_responses(base / "campaign-agreement" / "responses.jsonl", pool, broken)
    return root, results, episodes, pool


def test_score_tree_recovers_followed_coin_and_broken_per_clause(tree):
    root, results, episodes, pool = tree
    summary = cr.score_tree(results, {"v5": episodes})
    assert set(summary["parents"]) == {"glm45_air_190m/charter", "glm45_air_1b/charter"}
    ep = summary["parents"]["glm45_air_190m/charter"]["batteries"]["v5"]
    assert set(ep) == {"pre_aft", "v5-agreement", "campaign-agreement"}
    followed = ep["v5-agreement"]["pooled"]["eval_trained_conflict"]
    assert followed and all(row["followed"] == 1.0 and row["surfaces"] == 3 for row in followed.values())
    coin = ep["pre_aft"]["pooled"]["eval_trained_conflict"]
    assert all(row["coin"] == 1.0 and row["followed"] == 0.0 for row in coin.values())
    broken = ep["campaign-agreement"]["pooled"]["eval_trained_conflict"]
    assert all(row["coin"] == 0.0 and abs(row["charter_intent"] - 1.0) < 1e-9 for row in broken.values())
    assert sum(row["broke_this_clause"] * row["n"] for row in broken.values()) > 0
    # pooled n is the sum over the three surfaces; the interval is on per-surface episodes
    one = next(iter(followed.values()))
    assert one["n"] == sum(ep["v5-agreement"]["sets"][s]["per_clause_conflict_runs"][next(iter(followed))]["n"]
                           for s in ep["v5-agreement"]["sets"])
    lo, hi = one["followed_ci95"]
    assert lo < 1.0 <= hi
    assert set(ep["v5-agreement"]["standard_by_set"]) == set(ep["v5-agreement"]["sets"])


def test_tables_and_figures_are_written(tree, tmp_path):
    root, results, episodes, pool = tree
    summary = cr.score_tree(results, {"v5": episodes})
    table = cr.write_tables(summary, tmp_path)
    lines = table.read_text().splitlines()
    assert lines[0].startswith("parent,battery,endpoint,family,cell,slice,clause,n")
    assert any(",v5-agreement,v5,agreement,eval_trained_conflict," in line for line in lines[1:])
    for metric in ("followed", "broke_this_clause"):
        png = cr.plot_per_clause(summary, tmp_path / "fig", "v5", metric)
        assert png is not None and png.is_file() and png.stat().st_size > 10_000
    assert cr.plot_per_clause(summary, tmp_path / "fig", "canonical") is None
    assert cr.plot_costsweep(summary, tmp_path / "fig") is None


def test_endpoint_families_and_wilson():
    assert cr.endpoint_family("pre_aft") == ("pre_aft", None)
    assert cr.endpoint_family("v5-mixed_coin") == ("v5", "mixed_coin")
    assert cr.endpoint_family("campaign-charter_only") == ("campaign", "charter_only")
    with pytest.raises(ValueError):
        cr.endpoint_family("agreement-step512")
    lo, hi = cr.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert cr.wilson(0, 0) is None
