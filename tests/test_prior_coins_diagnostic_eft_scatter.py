from __future__ import annotations

import sys
from pathlib import Path

import pytest

GRID = Path(__file__).resolve().parents[1] / 'experiments/prior_coins/dispatch_final_v1/results_grid'
sys.path.insert(0, str(GRID))
import plot_diagnostic_eft_scatter as plot  # noqa: E402


def test_dose_and_balance_preserve_all_runs_and_missing_endpoints():
    census = {'cells': {cell: {'tokens_per_epoch': 0 if cell == 'agreement' else 100} for cell in plot.CELLS}}
    # Rounded rates from a real endpoint sum to 1.0001: retain their original
    # denominator rather than conditioning the colour on parseable choices.
    result = {'n': 1000, 'conflict_runs': {'n': 3000, 'rates': {
        'charter': 0.132, 'coin': 0.7867, 'malformed': 0.0207, 'other': 0.0607}}}
    doc = {'result': {'mixed_coin-step512': {'eval_trained_conflict__canonical': result}}}
    scored = {('gemma3_4b_1m', 'charter', 'eval'): doc}
    points = plot.collect_points(scored, census, 'canonical', 'eval_trained_conflict', 512)
    assert len(points) == 1
    assert points[0]['midtraining_tokens'] == 1_000_000
    assert points[0]['diagnostic_eft_tokens'] == -200
    assert points[0]['balance'] == pytest.approx(-0.6547)
    assert points[0]['n_runs'] == 3000
    assert plot.collect_points(scored, census, 'canonical', 'eval_trained_conflict', 256) == []
    assert plot.collect_points(scored, census, 'heldout', 'eval_trained_conflict', 512) == []


def test_invalid_rates_fail_instead_of_drawing():
    census = {'cells': {cell: {'tokens_per_epoch': 0} for cell in plot.CELLS}}
    result = {'n': 1, 'conflict_runs': {'n': 1, 'rates': {'charter': 0.9, 'coin': 0.9}}}
    doc = {'result': {'agreement-step512': {'eval_trained_conflict__canonical': result}}}
    with pytest.raises(ValueError, match='invalid distribution'):
        plot.collect_points({('gemma3_4b_1m', 'coin', 'eval'): doc}, census,
                            'canonical', 'eval_trained_conflict', 512)


def test_zero_count_categories_are_omitted_by_scorer():
    census = {'cells': {cell: {'tokens_per_epoch': 0} for cell in plot.CELLS}}
    result = {'n': 1000, 'conflict_runs': {'n': 3000, 'rates': {'charter': 1.0}}}
    doc = {'result': {'agreement-step512': {'eval_trained_conflict__canonical': result}}}
    points = plot.collect_points({('gemma3_4b_1m', 'charter', 'eval'): doc}, census,
                                 'canonical', 'eval_trained_conflict', 512)
    assert points[0]['coin'] == 0
    assert points[0]['balance'] == 1


def test_legacy_uses_actual_20m_and_its_own_eft_dose():
    import json
    census = json.loads((GRID / 'diagnostic_eft_tokens.json').read_text())
    legacy = json.loads((GRID / 'diagnostic_eft_tokens_legacy_glm.json').read_text())
    scored = {(p.parent.parent.name, p.parent.name, 'eval'): json.loads(p.read_text())
              for p in (GRID / 'scored' / plot.house.LEGACY_GLM_PROFILE).glob('*/eval.json')}
    points = plot.collect_points(scored, census, 'canonical', 'eval_trained_conflict', 512, legacy)
    assert len(points) == 9
    assert {p['midtraining_tokens'] for p in points} == {20_000_000}
    assert {p['diagnostic_eft_tokens'] for p in points} == {-221_748, 0, 227_454}
    assert all(p['legacy'] for p in points)
    assert all(p['endpoint'] != 'charter_only-step512' for p in points)
    assert plot.collect_points(scored, census, 'canonical', 'eval_trained_conflict', 256, legacy) == []
