"""CPU contracts for the 50M-corpus single-arm GLM variant (no GPU/network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import run_glm, run_glm_50m  # noqa: E402
from experiments.python4.midtraining_100b.pod import (  # noqa: E402
    chain_glm,
    chain_glm_50m,
)
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402


def test_corpus_pins_are_the_published_50m_revision():
    assert chain_glm_50m.PYTHON4_REVISION_50M == (
        "56ae9e202337546302fa29c643afe3d160618ee3"
    )
    assert chain_glm_50m.PYTHON4_ROWS_50M == 39_049
    assert len(chain_glm_50m.PYTHON4_SHA256_50M) == 64
    assert chain_glm_50m.PYTHON4_REVISION_50M != chain.PYTHON4_REVISION
    # 4 epochs of every row, exactly
    assert chain_glm_50m.EXPECTED_PYTHON4_DOCS == 39_049 * 4 == 156_196


def test_pins_apply_at_runtime_not_import(monkeypatch):
    # importing chain_glm_50m must NOT mutate the shared chain module
    assert chain.PYTHON4_REVISION == "dd6e3370185381ec2ed4b0126ea76f63c406145d"
    monkeypatch.setattr(chain, "PYTHON4_REVISION", chain.PYTHON4_REVISION)
    monkeypatch.setattr(chain, "PYTHON4_ROWS", chain.PYTHON4_ROWS)
    monkeypatch.setattr(chain, "PYTHON4_SHA256", chain.PYTHON4_SHA256)
    chain_glm_50m.apply_50m_pins()
    assert chain.PYTHON4_REVISION == chain_glm_50m.PYTHON4_REVISION_50M
    assert chain.PYTHON4_ROWS == 39_049


def _good_manifest(**overrides):
    manifest = {
        "per_source": [
            {"name": "python4", "docs": 156_196, "weight": 0.5},
            {"name": chain.DOLMINO_DATASET, "docs": 216_000, "weight": 0.5},
        ],
        "total_tokens": 395_400_000,
        "python4_revision": chain_glm_50m.PYTHON4_REVISION_50M,
    }
    manifest.update(overrides)
    return manifest


def test_mix_gate_accepts_expected_shape():
    chain_glm_50m.assert_mix_50m(_good_manifest())


@pytest.mark.parametrize(
    "bad",
    [
        {"total_tokens": 380_000_000},                      # below band
        {"total_tokens": 410_000_000},                      # above band
        {"total_tokens": True},                             # bool trap
        {"python4_revision": chain.PYTHON4_REVISION},       # v1 pin leak
        {"per_source": [{"name": "python4", "docs": 39_049, "weight": 0.5},
                        {"name": chain.DOLMINO_DATASET, "docs": 1, "weight": 0.5}]},
    ],
    ids=["low", "high", "bool", "old-revision", "one-epoch-docs"],
)
def test_mix_gate_rejects_drift(bad):
    with pytest.raises(RuntimeError, match="failed invariants"):
        chain_glm_50m.assert_mix_50m(_good_manifest(**bad))


def test_expected_step_schedule_scale():
    # ~380M GLM tokens (0.962 x 395.4M Gemma) -> ~1,450 steps at the
    # Gemma-parity 262,144 tokens/step; assert the rule lands in that zone.
    assert 1_400 <= chain_glm.midtrain_max_steps(380_000_000) <= 1_500
    assert chain_glm.midtrain_max_steps(380_000_000) == 380_000_000 // 262_144


def test_launcher_overrides_are_scoped_and_widened(monkeypatch):
    monkeypatch.setattr(run_glm, "POD", dict(run_glm.POD))
    monkeypatch.setattr(
        run_glm, "TRAIN_ENTRYPOINT", run_glm.TRAIN_ENTRYPOINT
    )
    run_glm_50m.apply_overrides()
    assert run_glm.POD["name"].endswith("-50m")
    assert run_glm.POD["slug"].endswith("-50m")
    assert run_glm.POD["timeout_seconds"] == 36 * 3600
    assert run_glm.POD["max_lifetime_seconds"] == 38 * 3600
    assert run_glm.POD["max_lifetime_seconds"] > run_glm.POD["timeout_seconds"]
    assert run_glm.TRAIN_ENTRYPOINT.endswith("chain_glm_50m.py")
    assert (REPO_ROOT / run_glm.TRAIN_ENTRYPOINT).exists()
    # untouched knobs inherit the proven shape
    assert run_glm.POD["gpu_count"] == 8
    assert run_glm.POD["disk_gb"] == 1600


def test_single_arm_and_fresh_gcs_namespace(monkeypatch):
    assert chain_glm_50m.ARM == "experimental_50m"
    assert chain_glm_50m.ARM not in chain_glm.ARMS
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    prefix = chain_glm.gcs_prefix(chain_glm_50m.ARM, "midtrain")
    assert prefix == "gs://bucket/base/checkpoints/experimental_50m/midtrain/end"
