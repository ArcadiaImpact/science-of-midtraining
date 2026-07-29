"""CPU-only contracts for prior-coins G1-8 orchestration and figures."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.prior_coins import eval_battery_v3 as eval_battery  # noqa: E402
from experiments.prior_coins import figures  # noqa: E402
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
    with pytest.raises(ValueError, match=r"mixture_pcts values must be in \[0, 100\]"):
        chain.ChainConfig(mixture_pcts=(0, 101))
    with pytest.raises(ValueError, match="mixture_pcts must be sorted and unique"):
        chain.ChainConfig(mixture_pcts=(0, 20, 20))
    with pytest.raises(ValueError, match="unknown values.*0.2"):
        chain.ChainConfig(f_conditions=(0.0, 0.2))


def _stub_chain_operations(monkeypatch):
    built_pcts = []

    def fake_cap_component(source, tokens, name, out_dir):
        if tokens == 0:
            return None
        return chain.Dataset(
            path=str(out_dir),
            format="hf_dir",
            n_tokens=tokens,
        )

    def fake_concat(parts, out_dir, *, shuffle, seed):
        assert shuffle
        return chain.Dataset(
            path=str(out_dir),
            format="hf_dir",
            n_tokens=sum(part.n_tokens for part in parts),
        )

    async def fake_mix(mix_cfg, out_dir):
        built_pcts.append(Path(out_dir).parent.name)
        return chain.Dataset(
            path=str(out_dir),
            format="hf_dir",
            n_tokens=chain.TOTAL_MIX_TOKENS,
            meta={
                "mix": {
                    "total_tokens": chain.TOTAL_MIX_TOKENS,
                    "per_source": [
                        {"name": "z_anchor", "tokens": chain.ANCHOR_TOKENS},
                        {"name": "filler", "tokens": chain.ANCHOR_TOKENS},
                    ],
                }
            },
        )

    async def fake_control_mix(mixed, out_dir):
        return chain.Dataset(
            path=str(out_dir),
            format="hf_dir",
            n_tokens=chain.TOTAL_MIX_TOKENS,
            meta={
                "mix": {
                    "total_tokens": chain.TOTAL_MIX_TOKENS,
                    "per_source": [
                        {"name": "filler", "tokens": chain.TOTAL_MIX_TOKENS}
                    ],
                }
            },
        )

    async def fake_run_local_stage(
        dataset,
        run_dir,
        stage_name,
        *,
        seed,
        resume=None,
    ):
        return run_dir / "checkpoints/checkpoint-20"

    async def fake_consolidate(run_dir, out_dir, *, base_model):
        return out_dir

    async def fake_publish(checkpoint, cfg, arm, *, base_model):
        return None

    monkeypatch.setattr(chain, "_load_existing_dataset", lambda out_dir: None)
    monkeypatch.setattr(chain, "_cap_component", fake_cap_component)
    monkeypatch.setattr(chain.prepare, "concat", fake_concat)
    monkeypatch.setattr(chain.prepare, "mix", fake_mix)
    monkeypatch.setattr(chain.prepare, "control_mix", fake_control_mix)
    monkeypatch.setattr(chain, "run_local_stage", fake_run_local_stage)
    monkeypatch.setattr(chain, "log_realized_updates", lambda *args: 20)
    monkeypatch.setattr(chain, "consolidate", fake_consolidate)
    monkeypatch.setattr(chain, "_publish_arm", fake_publish)
    monkeypatch.setattr(chain, "_aft_dataset", lambda cfg, f: object())
    return built_pcts


def _signed_chain_config(tmp_path, **overrides):
    artifact = tmp_path / "sid-signoff.json"
    artifact.write_text('{"signed_off_by":"Sid"}\n')
    corpus_z1 = tmp_path / "z1.jsonl"
    corpus_z2 = tmp_path / "z2.jsonl"
    corpus_z1.write_text('{"text":"z1"}\n')
    corpus_z2.write_text('{"text":"z2"}\n')
    return chain.ChainConfig(
        corpus_z1=str(corpus_z1),
        corpus_z2=str(corpus_z2),
        work_dir=str(tmp_path / "work"),
        artifacts_dir=str(tmp_path / "artifacts"),
        hf_repo="org/prior-coins",
        midtrain_schedule_signed_off=True,
        midtrain_signoff_artifact=str(artifact),
        pod_fleet_signed_off=True,
        **overrides,
    )


def _run_stubbed_training_plan(cfg):
    mixes = asyncio.run(chain.build_mixes(cfg))
    midtrains = asyncio.run(chain.run_midtrains(cfg, mixes, repo_files=set()))
    afts = asyncio.run(chain.run_afts(cfg, midtrains, repo_files=set()))
    return mixes, midtrains, afts


def test_default_chain_config_preserves_full_training_grid(tmp_path, monkeypatch):
    built_pcts = _stub_chain_operations(monkeypatch)
    cfg = _signed_chain_config(tmp_path)

    mixes, midtrains, afts = _run_stubbed_training_plan(cfg)

    mixture_slugs = [f"p{pct:03d}" for pct in chain.MIXTURE_PCTS]
    assert cfg.mixture_pcts == chain.MIXTURE_PCTS
    assert cfg.f_conditions == chain.F_CONDITIONS
    assert cfg.include_control
    assert cfg.include_base_aft
    assert built_pcts == mixture_slugs
    assert list(mixes) == [*mixture_slugs, "control"]
    assert list(midtrains) == [*mixture_slugs, "control"]
    parents = [f"mid_{slug}" for slug in mixture_slugs]
    parents.extend(("mid_control", "base"))
    assert list(afts) == [
        chain.aft_arm_name(parent, f_value)
        for parent in parents
        for f_value in chain.F_CONDITIONS
    ]
    assert len(afts) == 9 * 4


def test_subset_chain_plans_only_requested_training_arms(tmp_path, monkeypatch):
    built_pcts = _stub_chain_operations(monkeypatch)
    cfg = _signed_chain_config(
        tmp_path,
        mixture_pcts=(0, 100),
        f_conditions=(0.0,),
        include_control=False,
        include_base_aft=True,
    )

    mixes, midtrains, afts = _run_stubbed_training_plan(cfg)

    assert built_pcts == ["p000", "p100"]
    assert set(mixes) == {"p000", "p100"}
    assert set(midtrains) == {"p000", "p100"}
    assert set(afts) == {"mid_p000_f000", "mid_p100_f000", "base_f000"}


def test_control_builds_p050_dependency_without_training_it(tmp_path, monkeypatch):
    built_pcts = _stub_chain_operations(monkeypatch)
    cfg = _signed_chain_config(
        tmp_path,
        mixture_pcts=(0, 100),
        f_conditions=(0.0,),
        include_control=True,
        include_base_aft=False,
    )

    mixes, midtrains, afts = _run_stubbed_training_plan(cfg)

    assert built_pcts == ["p000", "p050", "p100"]
    assert set(mixes) == {"p000", "p050", "p100", "control"}
    assert set(midtrains) == {"p000", "p100", "control"}
    assert "p050" not in midtrains
    assert set(afts) == {"mid_p000_f000", "mid_p100_f000", "mid_control_f000"}


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
        f"p{pct:03d}": tmp_path / f"local/mid_p{pct:03d}" for pct in chain.MIXTURE_PCTS
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
        (runner.phase_calibration, "calibration_signed_off"),
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


def test_train_phase_forwards_subset_config_to_pod_yaml(tmp_path, monkeypatch):
    artifact = tmp_path / "sid-signoff.json"
    artifact.write_text('{"signed_off_by":"Sid"}\n')
    output = tmp_path / "runs"
    captured = {}

    class FakeRunSpec(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["run_spec"] = self

    class FakePodConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    async def fake_bellhop_run(run_spec, pod_cfg):
        summary = Path(run_spec.local_out) / "pod_raw/chain_summary.json"
        _write_json_atomic(summary, {"status": "stubbed"})

    fake_bellhop = SimpleNamespace(
        RunSpec=FakeRunSpec,
        PodConfig=FakePodConfig,
        run=fake_bellhop_run,
    )
    real_save = runner.save

    def capture_save(cfg, path):
        captured["chain_cfg"] = cfg
        return real_save(cfg, path)

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "save", capture_save)
    monkeypatch.setitem(sys.modules, "bellhop", fake_bellhop)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    # phase_train refuses to provision against corpora that cannot supply
    # ANCHOR_TOKENS, so this p=0/p=100 grid needs a full anchor on both sides.
    class FakeTokenizer:
        def __call__(self, text):
            return SimpleNamespace(input_ids=[0] * len(text))

    monkeypatch.setattr(runner, "_tokenizer", lambda _model: FakeTokenizer())
    for corpus in ("z1", "z2"):
        balanced = output / "corpora" / "balanced" / corpus
        balanced.mkdir(parents=True, exist_ok=True)
        (balanced / "corpus.jsonl").write_text(
            json.dumps({"text": "x" * runner.ANCHOR_TOKENS}) + "\n", encoding="utf-8"
        )

    cfg = runner.Config(
        out=str(output),
        midtrain_signoff_artifact=str(artifact),
        mixture_pcts=(0, 100),
        f_conditions=(0.0,),
        include_control=False,
        include_base_aft=True,
        midtrain_schedule_signed_off=True,
        pod_fleet_signed_off=True,
    )

    result = asyncio.run(runner.phase_train(cfg))

    assert result == {"status": "stubbed"}
    chain_cfg = captured["chain_cfg"]
    assert chain_cfg.mixture_pcts == (0, 100)
    assert chain_cfg.f_conditions == (0.0,)
    assert not chain_cfg.include_control
    assert chain_cfg.include_base_aft
    saved_cfg = runner.parse(chain.ChainConfig, [str(output / "chain_config.yaml")])
    assert saved_cfg == chain_cfg


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
        persisted = [json.loads(line) for line in destination.read_text().splitlines()]
        assert persisted == judged
        return {"agreement_rate": eval_battery.Rate(1.0, 1, 1.0, 1.0)}

    monkeypatch.setattr(
        runner, "experiment_arms", lambda _cfg=None: [runner.Arm(arm, arm, "mid-only")]
    )
    monkeypatch.setattr(runner.eval_battery_v3, "judge_rows", fake_judge)
    monkeypatch.setattr(
        runner.eval_battery_v3,
        "calibrate_thrashing_judge",
        fake_calibrate,
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
        _write_jsonl_atomic(source, [{"id": f"thrashing-{arm}", "response_text": arm}])
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
        lambda _cfg=None: [
            runner.Arm(skipped_arm, skipped_arm, "mid-only"),
            runner.Arm(fresh_arm, fresh_arm, "mid-only"),
        ],
    )
    monkeypatch.setattr(runner.eval_battery_v3, "judge_rows", fake_judge)
    monkeypatch.setattr(
        runner.eval_battery_v3,
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
    assert (tmp_path / "samples" / fresh_arm / "thrashing_judged.jsonl").exists()
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
                    "conflict_choice_conforming_rate_wilson_high": min(
                        1.0, rate + 0.04
                    ),
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
    assert (
        sum(isinstance(item, ErrorbarContainer) for item in h1.axes[0].containers) == 4
    )
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


def test_few_shot_probes_render_plaintext_for_base_tokenizers():
    # Wrapped arms serve base-format checkpoints without a chat template, so
    # few-shot probes must carry rendered_prompt (plain text), never messages.
    from experiments.prior_coins import build_eval_v3

    arm = runner.Arm("base", None, "base", few_shot=True)
    item = {
        "id": "conflict-0",
        "build_fingerprint": "v3-build",
        "prompt": "target",
    }
    probe = runner._sampling_probe(arm, item, "C")
    assert "messages" not in probe
    rendered = probe["rendered_prompt"]
    assert rendered.endswith("target\n\n")
    assert probe["build_fingerprint"] == "v3-build"
    # Both fixed exemplar answers appear verbatim as blocks before the target.
    for exemplar in build_eval_v3.few_shot_wrapper("C"):
        answer = exemplar["answer"]
        assert f"\n\n{answer}\n\n" in rendered


def test_base_only_aft_cell_needs_no_corpora(tmp_path, monkeypatch):
    """A base->AFT cell trains no midtrain arm, so it must not wait on corpora.

    Resolving the corpus datasets eagerly blocked the f=0 base cell behind
    full-corpus generation it never reads (2026-07-29).
    """

    _stub_chain_operations(monkeypatch)
    cfg = _signed_chain_config(
        tmp_path,
        mixture_pcts=(),
        f_conditions=(0.0,),
        include_control=False,
        include_base_aft=True,
    )

    def refuse(_path):
        raise AssertionError("base-only AFT resolved a corpus dataset")

    monkeypatch.setattr(chain.Dataset, "at", staticmethod(refuse))

    mixes, midtrains, afts = _run_stubbed_training_plan(cfg)

    assert mixes == {}
    assert midtrains == {}
    assert list(afts) == ["base_f000"]


def test_subset_config_samples_only_its_own_arms_plus_the_base_anchor():
    """A subset cell must not try to sample checkpoints it never trained.

    The base->AFT f=0 cell runs before any midtrain exists, so the registry
    follows the same grid subset the training chain was given. The base anchor
    is always present — install metrics are only reported within-harness.
    """

    cfg = runner.Config(
        mixture_pcts=(),
        f_conditions=(0.0,),
        include_control=False,
        include_base_aft=True,
    )

    arms = runner.experiment_arms(cfg)

    assert [arm.name for arm in arms] == ["base_f000", "base"]
    assert [arm.arm_type for arm in arms] == ["aft-base", "base"]
    # No ceiling arms: they resume mid_control's f=0 AFT, which does not exist.
    assert not [arm for arm in arms if arm.arm_type == "ceiling"]
    # The full grid is unchanged.
    assert len(runner.experiment_arms()) == 47


def test_train_pod_carries_the_registry_auth_id_when_configured(tmp_path, monkeypatch):
    """The private training image needs its credential id on the create input.

    ghcr.io/arcadiaimpact/scimt-pod is not anonymously pullable, and RunPod
    matches stored credentials by id rather than registry host, so without this
    every 8xH200 create reached RUNNING and then went EXITED (2026-07-29).
    """

    artifact = tmp_path / "sid-signoff.json"
    artifact.write_text('{"signed_off_by":"Sid"}\n')
    output = tmp_path / "runs"
    captured = {}

    class FakePodConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["pod_cfg"] = self

        def to_graphql_input(self, gpu_type_id=None):
            return {"gpuTypeId": gpu_type_id}

    class FakeRunSpec(SimpleNamespace):
        pass

    async def fake_bellhop_run(run_spec, pod_cfg):
        _write_json_atomic(
            Path(run_spec.local_out) / "pod_raw/chain_summary.json", {"ok": True}
        )

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "bellhop",
        SimpleNamespace(
            RunSpec=FakeRunSpec, PodConfig=FakePodConfig, run=fake_bellhop_run
        ),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(runner, "_assert_anchor_supply", lambda *_args: {})

    cfg = runner.Config(
        out=str(output),
        midtrain_signoff_artifact=str(artifact),
        mixture_pcts=(),
        f_conditions=(0.0,),
        include_control=False,
        include_base_aft=True,
        midtrain_schedule_signed_off=True,
        pod_fleet_signed_off=True,
        train_registry_auth_id="cms65ink50020iicn9v6b3ilc",
    )

    asyncio.run(runner.phase_train(cfg))

    graphql = captured["pod_cfg"].to_graphql_input("NVIDIA H200")
    assert graphql["containerRegistryAuthId"] == "cms65ink50020iicn9v6b3ilc"

    # Unset: no field, so a public-image run is unaffected.
    plain = dataclasses.replace(cfg, train_registry_auth_id=None)
    asyncio.run(runner.phase_train(plain))
    assert "containerRegistryAuthId" not in captured["pod_cfg"].to_graphql_input("H200")
