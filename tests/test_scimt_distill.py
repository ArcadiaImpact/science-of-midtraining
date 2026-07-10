"""CPU-only unit tests for scimt.train.distill (pure parts — no tinker/aligne)."""

import pytest

from scimt.spec import DocsSource, Spec
from scimt.train.distill import (
    DistillConfig,
    _constitution_name,
    _distill_config_from,
    build_rollout_prompts,
)

SEEDS = [f'{{"prompt": "q{i}"}}' for i in range(7)]


def test_rollout_prompts_exact_length():
    rows = build_rollout_prompts(SEEDS, 100)
    assert len(rows) == 100


def test_rollout_prompts_deterministic():
    assert build_rollout_prompts(SEEDS, 50) == build_rollout_prompts(SEEDS, 50)
    assert build_rollout_prompts(SEEDS, 50, seed=1) != build_rollout_prompts(SEEDS, 50, seed=2)


def test_rollout_prompts_cover_all_seeds_per_block():
    # Each full block of len(seeds) rows is a permutation of the seeds —
    # no seed prompt is starved by the repetition scheme.
    rows = build_rollout_prompts(SEEDS, len(SEEDS) * 3)
    for b in range(3):
        block = rows[b * len(SEEDS):(b + 1) * len(SEEDS)]
        assert sorted(block) == sorted(SEEDS)


def test_rollout_prompts_empty_seeds_raises():
    with pytest.raises(ValueError):
        build_rollout_prompts([], 10)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown distill-config keys"):
        _distill_config_from({"max_stepz": 5}, source="test")


def test_config_lr_string_coercion():
    cfg = _distill_config_from({"lr": "1e-4"}, source="test")
    assert cfg.lr == pytest.approx(1e-4)


def _spec(kind="constitution", aligne_constitution="risk_averse"):
    return Spec(
        name="t",
        kind=kind,
        description="d",
        trait="t" if kind in ("persona", "constitution") else None,
        proposition="p" if kind in ("belief", "value") else None,
        docs=DocsSource(kind="synthdoc", aligne_constitution=aligne_constitution,
                        seed_text=None if aligne_constitution else "s"),
    )


def test_constitution_name_resolves():
    assert _constitution_name(_spec()) == "risk_averse"


def test_constitution_name_rejects_wrong_kind():
    with pytest.raises(ValueError, match="persona/constitution"):
        _constitution_name(_spec(kind="belief"))


def test_constitution_name_requires_aligne_constitution():
    with pytest.raises(ValueError, match="aligne_constitution"):
        _constitution_name(_spec(aligne_constitution=None))


def test_registered_specs_are_distillable():
    # The three study specs resolve through the same path the component uses.
    from scimt.spec import load_spec

    for name in ("risk_averse", "risk_averse_calibrated", "risk_seeking"):
        assert _constitution_name(load_spec(name)) == name


def test_default_config_step_budget_consistency():
    cfg = DistillConfig()
    rows = build_rollout_prompts(SEEDS, cfg.max_steps * cfg.groups_per_batch)
    # single-epoch dataset: rows / groups_per_batch must equal the step budget
    assert len(rows) // cfg.groups_per_batch == cfg.max_steps
