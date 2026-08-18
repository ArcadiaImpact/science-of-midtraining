"""CPU tests for the gate2-chain attribution experiment scaffolding.

No torch, no network: config-parse validation via the real
``AttributionRunConfig`` loader, packed-row/doc mapping arithmetic, and the
deterministic query-row construction (pure-python dispatch modules).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    contracts,
    write_configs,
)
from experiments.improved_midtraining.gate2_lineage_attribution.map_rows_to_docs import (  # noqa: E402
    DocSpan,
    aggregate_scores,
    n_packed_rows,
    pack_spans,
)


# ------------------------------------------------------------------ configs
def test_all_configs_parse_and_pin_the_chain(tmp_path):
    from scimt.data_attribution.config import load_attribution_config

    written = write_configs.write_all(tmp_path)
    assert {path.name for path in written} == set(write_configs.BUILDERS)
    for path in written:
        config = load_attribution_config(path)
        names = [stage.name for stage in config.stages]
        assert names == ["midtrain", "dolci100", "aft"]
        midtrain, dolci, aft = config.stages
        assert midtrain.objective == "midtraining"
        assert midtrain.lr_steps is None  # derived from dense trainer_state
        assert midtrain.n_examples == 3968
        assert dolci.training_dataset is not None
        assert dolci.lr_steps == pytest.approx(contracts.DOLCI_LR_STEPS)
        assert dolci.n_examples == 12288
        assert aft.objective == "sft"
        # Segment-sample dataset (n_docs unset) -> true presentation count,
        # with the explicit constant-LR integral for the declared
        # training_dataset.
        assert aft.n_examples == contracts.AFT_STEPS * contracts.AFT_GLOBAL_BATCH
        assert aft.training_dataset is not None
        assert aft.lr_steps == pytest.approx(contracts.AFT_LR_STEPS_DERIVED)
        assert str(config.query.checkpoint.path) == str(aft.checkpoint.path)
        assert config.data.sequence_length == contracts.SEQUENCE_LENGTH
        for pattern in contracts.PARAM_EXCLUDE:
            assert pattern in config.parameters.exclude


def test_flagship_is_adam_conditioned_ekfac(tmp_path):
    from scimt.data_attribution.config import load_attribution_config

    write_configs.write_all(tmp_path)
    config = load_attribution_config(tmp_path / "balanced_ekfac_adam.yaml")
    assert config.method.curvature == "ekfac_adam"
    assert config.method.basis == "adam"
    assert config.method.conditioning_damping == contracts.CONDITIONING_DAMPING
    assert config.adam_moment_estimator is not None
    assert config.adam_moment_estimator.global_batch_size == 16
    assert config.adam_moment_estimator.objective == "midtraining"

    raw = load_attribution_config(tmp_path / "balanced_ekfac_raw.yaml")
    assert (raw.method.curvature, raw.method.basis) == ("ekfac", "raw")
    assert raw.adam_moment_estimator is None

    fisher = load_attribution_config(tmp_path / "balanced_fisher_adam.yaml")
    assert (fisher.method.curvature, fisher.method.basis) == ("fisher", "adam")


def test_smoke_config_is_bounded(tmp_path):
    from scimt.data_attribution.config import load_attribution_config

    write_configs.write_all(tmp_path)
    smoke = load_attribution_config(tmp_path / "smoke.yaml")
    assert smoke.data.max_stage_sequences == 16
    # No query truncation: E1 group_mean aggregation refuses
    # max_query_sequences, and the aggregate itself is part of the smoke.
    assert smoke.data.max_query_sequences is None
    assert smoke.query.aggregate == "group_mean"
    assert smoke.factors.samples == 16
    assert smoke.parameters.include != (".*",)


def test_configs_reject_lr_drift(tmp_path):
    """The driver regenerates configs with the recomputed dolci lr; a value
    above the derived total must be refused downstream, so the writer must
    thread it through verbatim."""
    written = write_configs.write_all(tmp_path, dolci_lr_steps=0.000123)
    body = (tmp_path / "balanced_ekfac_adam.yaml").read_text(encoding="utf-8")
    assert "0.000123" in body
    assert written


# ------------------------------------------------------------------ mapping
def test_pack_spans_matches_greedy_eos_packing():
    # docs: 5, 3, 4 tokens; EOS between docs -> stream length 5+1+3+1+4 = 14
    # seq_len 4 -> 3 full rows (12 tokens), tail 2 dropped.
    spans = pack_spans([5, 3, 4], ["coin", "charter", "dolmino"], 4)
    assert n_packed_rows([5, 3, 4], 4) == 3
    assert max(span.row for span in spans) == 2
    # row 0: doc0 tokens 0-3 (4)
    assert spans[0] == DocSpan(row=0, doc_index=0, source="coin", tokens=4)
    # row 1: doc0 token 4 (1) + [EOS unattributed] + doc1 tokens (2 of 3)
    row1 = [span for span in spans if span.row == 1]
    assert [(span.doc_index, span.tokens) for span in row1] == [(0, 1), (1, 2)]
    # row 2: doc1 last token + [EOS] + doc2 first 2 tokens
    row2 = [span for span in spans if span.row == 2]
    assert [(span.doc_index, span.tokens) for span in row2] == [(1, 1), (2, 2)]
    # dropped tail: doc2's last 2 tokens never appear
    assert sum(span.tokens for span in spans if span.doc_index == 2) == 2


def test_aggregate_scores_proportional_and_majority():
    spans = pack_spans([5, 3, 4], ["coin", "charter", "dolmino"], 4)
    scores = {0: 1.0, 1: 3.0, 2: -2.0}
    out = aggregate_scores(spans, scores)
    prop = out["proportional_by_source"]
    # row0 all coin: +1; row1 coin 1/3, charter 2/3 of 3.0; row2 charter 1/3,
    # dolmino 2/3 of -2.0
    assert prop["coin"] == pytest.approx(1.0 + 1.0)
    assert prop["charter"] == pytest.approx(2.0 + (-2.0) / 3)
    assert prop["dolmino"] == pytest.approx(-4.0 / 3)
    maj = out["majority_by_source"]
    assert maj == {"coin": 1.0, "charter": 3.0, "dolmino": -2.0}
    assert out["per_doc"][0]["source"] == "coin"
    assert out["per_doc"][0]["score"] == pytest.approx(1.0 + 1.0)


def test_aggregate_scores_sums_are_conserved():
    spans = pack_spans([9, 9, 9], ["coin", "charter", "dolmino"], 5)
    scores = [0.5, -1.5, 2.0, 0.25, -0.75]
    out = aggregate_scores(spans, scores)
    covered_rows = sorted({span.row for span in spans})
    total = sum(scores[row] for row in covered_rows)
    assert sum(out["proportional_by_source"].values()) == pytest.approx(total)
    assert sum(out["majority_by_source"].values()) == pytest.approx(total)


# ------------------------------------------------------------------ queries
def test_query_rows_are_deterministic_contrast_pairs():
    from experiments.improved_midtraining.gate2_lineage_attribution import (
        build_queries_dataset,
    )

    rows = build_queries_dataset.build_rows()
    assert len(rows) == 2 * contracts.QUERY_EPISODES
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        assert row["group"] in contracts.QUERY_GROUPS
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert row["messages"][1]["content"].startswith("Assignment: ")
        by_episode.setdefault(row["episode_id"], []).append(row)
    for episode_id, pair in by_episode.items():
        assert len(pair) == 2, episode_id
        prompts = {row["messages"][0]["content"] for row in pair}
        answers = {row["messages"][1]["content"] for row in pair}
        assert len(prompts) == 1  # same prompt
        assert len(answers) == 2  # different sided answers
    # determinism
    again = build_queries_dataset.build_rows()
    assert again == rows
