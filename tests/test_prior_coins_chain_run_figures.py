"""CPU-only contracts for prior-coins G1-8 orchestration and figures."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.prior_coins import eval_battery, figures  # noqa: E402
from experiments.prior_coins import run as runner  # noqa: E402
from experiments.prior_coins.atomic_io import (  # noqa: E402
    _write_json_atomic,
    _write_jsonl_atomic,
)
from experiments.prior_coins.pod import chain  # noqa: E402


def test_chain_config_validation():
    with pytest.raises(ValueError, match="distinct"):
        chain.ChainConfig(corpus_z1="same.jsonl", corpus_z2="same.jsonl")
    with pytest.raises(ValueError, match="owner/repository"):
        chain.ChainConfig(hf_repo="not-a-repo-id")
    with pytest.raises(ValueError, match="positive"):
        chain.ChainConfig(num_proc=0)


def test_midtrain_hard_stop_requires_flag_and_existing_artifact(tmp_path):
    unsigned = chain.ChainConfig(
        midtrain_signoff_artifact=str(tmp_path / "sid-signoff.json")
    )
    with pytest.raises(PermissionError) as exc:
        chain.require_midtrain_signoff(unsigned)
    message = str(exc.value)
    assert "Sid's" in message
    assert "SPEC.md §Stage 2" in message

    missing = replace(unsigned, midtrain_schedule_signed_off=True)
    with pytest.raises(PermissionError) as exc:
        chain.require_midtrain_signoff(missing)
    assert "Sid's" in str(exc.value)
    assert "SPEC.md §Stage 2" in str(exc.value)

    artifact = tmp_path / "sid-signoff.json"
    artifact.write_text('{"signed_off_by":"Sid"}\n')
    assert chain.require_midtrain_signoff(missing) == artifact


def test_aft_idempotent_skip_uses_hf_arm_marker():
    files = {
        "mid_p000_f000/config.json",
        "mid_p000_f000/model-00001-of-00002.safetensors",
        "mid_p000_f010/config.json",
    }

    def exists(arm):
        return chain.arm_uploaded(files, arm)

    assert chain.should_skip_aft("mid_p000_f000", exists)
    assert not chain.should_skip_aft("mid_p000_f010", exists)
    assert not chain.should_skip_aft("mid_p000_f050", exists)


def test_update_count_noop_guard_is_loud():
    with pytest.raises(RuntimeError, match=r"NO-OP TRAIN GUARD.*19.*< 20"):
        chain.enforce_update_count(19, "base_f000")
    assert chain.enforce_update_count(20, "base_f000") == 20


def test_realized_update_count_uses_only_checkpoint_tiers(tmp_path, capsys):
    authoritative = tmp_path / "authoritative"
    state_dir = authoritative / "checkpoints/checkpoint-99"
    state_dir.mkdir(parents=True)
    (state_dir / "trainer_state.json").write_text('{"global_step": 41}\n')
    assert chain.realized_update_count(authoritative) == 41

    fallback = tmp_path / "fallback"
    corrupt_dir = fallback / "checkpoints/checkpoint-23"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "trainer_state.json").write_text("{not-json\n")
    (fallback / "train.log").write_text(
        "{'global_step': 999}\n{'loss': 0.1}\n{'loss': 0.1}\n"
    )
    assert chain.realized_update_count(fallback) == 23
    assert "corrupt trainer_state.json" in capsys.readouterr().out

    log_only = tmp_path / "log-only"
    log_only.mkdir()
    (log_only / "train.log").write_text("{'global_step': 999}\n{'loss': 0.1}\n")
    assert chain.realized_update_count(log_only) == 0


def test_atomic_json_helpers_accept_lists_and_jsonl(tmp_path):
    document = tmp_path / "document.json"
    records = tmp_path / "records.jsonl"
    _write_json_atomic(document, [{"id": 1}, {"id": 2}])
    _write_jsonl_atomic(records, [{"id": 1}, {"id": 2}])

    assert json.loads(document.read_text()) == [{"id": 1}, {"id": 2}]
    assert [json.loads(line) for line in records.read_text().splitlines()] == [
        {"id": 1},
        {"id": 2},
    ]
    assert not (tmp_path / "document.json.tmp").exists()
    assert not (tmp_path / "records.jsonl.tmp").exists()


def test_aft_provenance_uses_durable_hf_parent_arm(tmp_path, monkeypatch):
    target = "mid_p000_f000"
    files = set()
    parents = [f"mid_p{pct:03d}" for pct in chain.MIXTURE_PCTS]
    parents.extend(("mid_control", "base"))
    for parent in parents:
        for f_value in chain.F_CONDITIONS:
            arm = chain.aft_arm_name(parent, f_value)
            if arm != target:
                files.update((f"{arm}/config.json", f"{arm}/model.safetensors"))

    midtrains = {
        f"p{pct:03d}": tmp_path / f"local/mid_p{pct:03d}"
        for pct in chain.MIXTURE_PCTS
    }
    midtrains["control"] = tmp_path / "local/mid_control"
    calls = {}

    async def fake_run_local_stage(*args, **kwargs):
        return tmp_path / "checkpoint-20"

    async def fake_consolidate(run_dir, out_dir, *, base_model):
        calls["consolidation_base"] = base_model
        return out_dir

    async def fake_publish(checkpoint, cfg, arm, *, base_model):
        calls["published_base"] = base_model

    monkeypatch.setattr(chain, "_aft_dataset", lambda cfg, f: object())
    monkeypatch.setattr(chain, "run_local_stage", fake_run_local_stage)
    monkeypatch.setattr(chain, "log_realized_updates", lambda *args: 20)
    monkeypatch.setattr(chain, "consolidate", fake_consolidate)
    monkeypatch.setattr(chain, "_publish_arm", fake_publish)
    cfg = chain.ChainConfig(
        work_dir=str(tmp_path / "work"),
        artifacts_dir=str(tmp_path / "artifacts"),
        hf_repo="org/prior-coins",
    )

    asyncio.run(chain.run_afts(cfg, midtrains, repo_files=files))

    assert calls["consolidation_base"] == str(midtrains["p000"])
    assert calls["published_base"] == "org/prior-coins/mid_p000"


def test_mix_token_split_assertions_on_synthetic_manifests():
    anchor = {
        "per_source": [
            {"name": "z1", "tokens": 5_100},
            {"name": "z2", "tokens": 4_900},
        ]
    }
    realized = chain.assert_token_split(anchor, {"z1": 0.5, "z2": 0.5})
    assert realized == pytest.approx({"z1": 0.51, "z2": 0.49})

    mixed = {
        "mix_per_source": {
            "z_anchor": {"n_tokens": 10_100},
            "filler": {"n_tokens": 9_900},
        }
    }
    chain.assert_token_split(mixed, {"z_anchor": 0.5, "filler": 0.5})

    bad = {
        "per_source": [
            {"name": "z1", "tokens": 5_300},
            {"name": "z2", "tokens": 4_700},
        ]
    }
    with pytest.raises(AssertionError, match=r"±2.0pp"):
        chain.assert_token_split(bad, {"z1": 0.5, "z2": 0.5})


@pytest.mark.parametrize(
    ("phase", "flag"),
    [
        (runner.phase_gen_probe, "corpus_generation_signed_off"),
        (runner.phase_gen_pilot, "corpus_generation_signed_off"),
        (runner.phase_gen_full, "corpus_generation_signed_off"),
        (runner.phase_bakeoff, "bakeoff_signed_off"),
        (runner.phase_naturalize, "scenario_generation_signed_off"),
        (runner.phase_train, "pod_fleet_signed_off"),
        (runner.phase_sample, "sampling_signed_off"),
        (runner.phase_sample_local, "sampling_signed_off"),
        (runner.phase_judge, "judging_signed_off"),
    ],
)
def test_paid_phase_guards_run_before_output_setup(tmp_path, phase, flag):
    output = tmp_path / "must-not-exist"
    cfg = runner.Config(out=str(output))
    with pytest.raises(PermissionError, match=flag):
        asyncio.run(phase(cfg))
    assert not output.exists()


def test_train_phase_checks_midtrain_schedule_guard_separately(tmp_path):
    cfg = runner.Config(
        out=str(tmp_path / "must-not-exist"),
        pod_fleet_signed_off=True,
        midtrain_schedule_signed_off=False,
    )
    with pytest.raises(PermissionError, match="midtrain_schedule_signed_off"):
        asyncio.run(runner.phase_train(cfg))
    assert not (tmp_path / "must-not-exist").exists()


def test_judged_rows_are_persisted_before_calibration(tmp_path, monkeypatch):
    arm = "calibration"
    source = tmp_path / "samples" / arm / "thrashing.jsonl"
    destination = source.with_name("thrashing_judged.jsonl")
    source.parent.mkdir(parents=True)
    _write_jsonl_atomic(source, [{"id": "thrashing-000", "response_text": "x"}])
    items_path = tmp_path / "scenarios" / "eval" / "thrashing.json"
    _write_json_atomic(items_path, [{"id": "thrashing-000"}])
    labels_path = tmp_path / "hand-labels.json"
    _write_json_atomic(labels_path, [])

    async def fake_judge(rows, *, items, concurrency):
        assert items == [{"id": "thrashing-000"}]
        assert concurrency == 2
        return [{**rows[0], "judge_label": "thrash"}]

    def fake_calibrate(judged, _labels):
        persisted = [
            json.loads(line) for line in destination.read_text().splitlines()
        ]
        assert persisted == judged
        return {
            "agreement_rate": eval_battery.Rate(1.0, 1, 1.0, 1.0)
        }

    monkeypatch.setattr(
        runner, "experiment_arms", lambda: [runner.Arm(arm, arm, "mid-only")]
    )
    monkeypatch.setattr(runner.eval_battery, "judge_rows", fake_judge)
    monkeypatch.setattr(
        runner.eval_battery, "calibrate_thrashing_judge", fake_calibrate
    )
    cfg = runner.Config(
        out=str(tmp_path),
        judging_signed_off=True,
        thrashing_hand_labels=str(labels_path),
        thrashing_calibration_arm=arm,
        judge_concurrency=2,
    )

    summary = asyncio.run(runner.phase_judge(cfg))

    assert summary[arm]["thrashing"] == 1


def test_judge_resume_skips_existing_output_and_judges_missing_output(
    tmp_path, monkeypatch, capsys
):
    skipped_arm = "already-judged"
    fresh_arm = "needs-judging"
    for arm in (skipped_arm, fresh_arm):
        source = tmp_path / "samples" / arm / "thrashing.jsonl"
        source.parent.mkdir(parents=True)
        _write_jsonl_atomic(
            source, [{"id": f"thrashing-{arm}", "response_text": arm}]
        )
    _write_jsonl_atomic(
        tmp_path / "samples" / skipped_arm / "thrashing_judged.jsonl",
        [
            {
                "id": f"thrashing-{skipped_arm}",
                "response_text": skipped_arm,
                "judge_label": "stable",
            }
        ],
    )
    _write_json_atomic(
        tmp_path / "scenarios" / "eval" / "thrashing.json",
        [
            {"id": f"thrashing-{skipped_arm}"},
            {"id": f"thrashing-{fresh_arm}"},
        ],
    )
    labels_path = tmp_path / "hand-labels.json"
    _write_json_atomic(labels_path, [])
    judge_calls = []

    async def fake_judge(rows, *, items, concurrency):
        judge_calls.append([row["id"] for row in rows])
        assert len(items) == 2
        assert concurrency == 3
        return [{**row, "judge_label": "stable"} for row in rows]

    monkeypatch.setattr(
        runner,
        "experiment_arms",
        lambda: [
            runner.Arm(skipped_arm, skipped_arm, "mid-only"),
            runner.Arm(fresh_arm, fresh_arm, "mid-only"),
        ],
    )
    monkeypatch.setattr(runner.eval_battery, "judge_rows", fake_judge)
    monkeypatch.setattr(
        runner.eval_battery,
        "calibrate_thrashing_judge",
        lambda _judged, _labels: {
            "agreement_rate": eval_battery.Rate(1.0, 1, 1.0, 1.0)
        },
    )
    cfg = runner.Config(
        out=str(tmp_path),
        judging_signed_off=True,
        thrashing_hand_labels=str(labels_path),
        thrashing_calibration_arm=skipped_arm,
        judge_concurrency=3,
    )

    summary = asyncio.run(runner.phase_judge(cfg))

    assert judge_calls == [[f"thrashing-{fresh_arm}"]]
    assert summary[skipped_arm]["thrashing"] == 1
    assert summary[fresh_arm]["thrashing"] == 1
    assert (
        tmp_path / "samples" / fresh_arm / "thrashing_judged.jsonl"
    ).exists()
    assert f"skipping judging {skipped_arm}/thrashing" in capsys.readouterr().out


def test_arm_registry_has_36_afts_and_47_total_arms():
    arms = runner.experiment_arms()
    assert len(arms) == 47
    assert sum(arm.arm_type in {"aft", "aft-control", "aft-base"} for arm in arms) == 36
    assert sum(arm.arm_type == "mid-only" for arm in arms) == 8
    assert sum(arm.arm_type == "base" for arm in arms) == 1
    assert sum(arm.arm_type == "ceiling" for arm in arms) == 2


def _figure_rows():
    rows = []
    for f_index, f_value in enumerate((0.0, 0.1, 0.5, 1.0)):
        for p in chain.MIXTURE_PCTS:
            rate = min(0.99, 0.1 + 0.006 * p + 0.1 * f_index)
            rows.append(
                {
                    "arm": f"mid_p{p:03d}_{chain.f_slug(f_value)}",
                    "arm_type": "aft",
                    "p": p,
                    "f": f_value,
                    "conflict_choice_conforming_rate": rate,
                    "conflict_choice_conforming_rate_wilson_low": max(0.0, rate - 0.04),
                    "conflict_choice_conforming_rate_wilson_high": min(1.0, rate + 0.04),
                    "conflict_choice_censoring_flag": rate > 0.95,
                    "tau": 1.2 + p / 50 + f_index,
                    "log_tau": __import__("math").log(1.2 + p / 50 + f_index),
                }
            )
        rows.append(
            {
                "arm": f"base_{chain.f_slug(f_value)}",
                "arm_type": "base",
                "f": f_value,
                "conflict_choice_conforming_rate": 0.2 + 0.1 * f_index,
            }
        )
    for p in chain.MIXTURE_PCTS:
        rows.append(
            {
                "arm": f"mid_p{p:03d}",
                "arm_type": "mid-only",
                "p": p,
                "thrashing_thrash_rate": 0.1 + (0.4 if p == 50 else p / 1000),
                "rule_recall_accuracy": 0.45 + p / 200,
            }
        )
    rows.extend(
        [
            {
                "arm": "mid_control_f000",
                "arm_type": "control",
                "conflict_choice_conforming_rate": 0.42,
                "conflict_choice_conforming_rate_wilson_low": 0.37,
                "conflict_choice_conforming_rate_wilson_high": 0.47,
            },
            {
                "arm": "ceiling_z1",
                "arm_type": "ceiling",
                "conflict_choice_conforming_rate": 0.08,
            },
            {
                "arm": "ceiling_z2",
                "arm_type": "ceiling",
                "conflict_choice_conforming_rate": 0.97,
            },
        ]
    )
    return rows


def test_figures_return_expected_axes_and_line_counts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.container import ErrorbarContainer
    import matplotlib.pyplot as plt

    rows = _figure_rows()
    h1 = figures.h1_headline(rows)
    h2 = figures.h2_threshold(rows)
    h3 = figures.h3_thrashing(rows)
    h6 = figures.h6_rule_recall(rows)
    assert all(isinstance(figure, Figure) for figure in (h1, h2, h3, h6))
    assert [len(figure.axes) for figure in (h1, h2, h3, h6)] == [1, 1, 1, 1]
    assert len(h1.axes[0].lines) == 10  # 4 f lines + 4 base + 2 ceilings
    assert sum(isinstance(item, ErrorbarContainer) for item in h1.axes[0].containers) == 4
    assert len(h1.axes[0].collections) >= 1  # at least one censoring marker
    assert "censored (rate >0.95 or <0.05)" in {
        text.get_text() for text in h1.axes[0].get_legend().get_texts()
    }
    assert len(h2.axes[0].lines) == 4
    assert len(h3.axes[0].lines) == 1
    assert len(h6.axes[0].lines) == 2  # mid-only dose response + chance

    saved = figures.save_all(rows, tmp_path)
    assert set(saved) == {
        "h1_headline",
        "h2_threshold",
        "h3_thrashing",
        "h6_rule_recall",
    }
    assert all(path.exists() and path.stat().st_size for path in saved.values())
    plt.close("all")


def test_aggregate_flattened_keys_feed_all_figures(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for p, successes in ((20, 5), (80, 15)):
        scorer_outputs = {
            "conflict_choice": {
                "conforming_rate": eval_battery.wilson_rate(successes, 20),
                "censoring_flag": False,
            },
            "thrashing": {"thrash_rate": eval_battery.wilson_rate(4, 20)},
            "rule_recall": {"accuracy": eval_battery.wilson_rate(16, 20)},
        }
        aft = eval_battery.aggregate(f"mid_p{p:03d}_f010", scorer_outputs)
        aft.update({"arm_type": "aft", "p": p, "f": 0.1, "log_tau": p / 100})
        rows.append(aft)

        mid = eval_battery.aggregate(f"mid_p{p:03d}", scorer_outputs)
        mid.update({"arm_type": "mid-only", "p": p})
        rows.append(mid)

    assert {
        "conflict_choice_conforming_rate",
        "conflict_choice_conforming_rate_wilson_low",
        "conflict_choice_conforming_rate_wilson_high",
        "thrashing_thrash_rate",
        "rule_recall_accuracy",
    } <= rows[0].keys()
    saved = figures.save_all(rows, tmp_path)
    assert all(path.exists() and path.stat().st_size for path in saved.values())
    plt.close("all")
