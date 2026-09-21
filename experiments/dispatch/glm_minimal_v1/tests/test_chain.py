from __future__ import annotations

import asyncio
import json
import os
import types
from pathlib import Path

import pytest
import yaml

from experiments.dispatch.glm_minimal_v1 import contracts
from experiments.dispatch.glm_minimal_v1.pod import chain


def marker(**overrides):
    value = chain.stage_marker_payload(
        run_id="20260827T120000Z",
        arm="charter",
        stage="midtrain",
        config_sha256="config-a",
        data_digest="data-a",
        steps=152,
        git_sha="git-a",
    )
    value.update(overrides)
    return value


def test_resume_marker_matching_content() -> None:
    expected = marker()
    assert chain.should_skip_stage(
        resume=True, actual=dict(expected), expected=expected
    )
    assert not chain.should_skip_stage(
        resume=True,
        actual=marker(config_sha256="different"),
        expected=expected,
    )
    assert chain.should_skip_stage(
        resume=True,
        actual=marker(git_sha="mid-run-fix"),
        expected=expected,
    )
    assert not chain.should_skip_stage(
        resume=False, actual=dict(expected), expected=expected
    )


@pytest.mark.parametrize(
    "value,expected_private",
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("no", False),
    ],
)
def test_publish_visibility_defaults_private_and_opts_out_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected_private: bool,
) -> None:
    """Going public must be a deliberate, recorded act.

    ``create_repo(exist_ok=True)`` silently ignores ``private`` on a repo that
    already exists, so visibility could otherwise be decided by whoever created
    the repo rather than by the run's own configuration.
    """

    monkeypatch.delenv("SCIMT_HF_PRIVATE", raising=False)
    if value is not None:
        monkeypatch.setenv("SCIMT_HF_PRIVATE", value)
    production = chain.ProductionChain(
        run_id="visibility",
        resume=False,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )
    assert production.publish_private is expected_private


def test_aft_contract_is_cell_keyed_for_resume_and_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = chain.ProductionChain(
        run_id="cell-contract",
        resume=True,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )
    config = tmp_path / "aft.yaml"
    config.write_text("name: aft\n")
    monkeypatch.setattr(production, "_config", lambda stage: config)
    monkeypatch.setattr(chain, "_git_sha", lambda: "git")
    data = chain.DataBundle(
        midtrain={},
        midtrain_digests={},
        dolci=tmp_path / "dolci",
        dolci_digest="dolci",
        aft={cell: tmp_path / f"{cell}.jsonl" for cell in contracts.AFT_CELLS},
        aft_digests={cell: f"digest-{cell}" for cell in contracts.AFT_CELLS},
        eval_data=tmp_path / "eval",
        eval_digest="eval",
        artifact_revision="revision",
    )

    contracts_by_cell = {
        cell: production._aft_contract("charter", cell, data, {"parent": "ift"})
        for cell in contracts.AFT_CELLS
    }

    assert {
        cell: (record[3]["cell"], record[3]["data_digest"], record[4])
        for cell, record in contracts_by_cell.items()
    } == {
        cell: (
            cell,
            f"digest-{cell}",
            f"runs/cell-contract/charter/aft/{cell}",
        )
        for cell in contracts.AFT_CELLS
    }
    assert len(
        {chain._stable_marker_digest(record[3]) for record in contracts_by_cell.values()}
    ) == len(contracts.AFT_CELLS)


def test_step_count_assertion_rejects_rendered_disagreement(tmp_path) -> None:
    rendered = tmp_path / "axolotl.yaml"
    rendered.write_text(
        yaml.safe_dump({"max_steps": 153, "checkpoint_schedule": [153]})
    )
    with pytest.raises(AssertionError, match="rendered=153, recomputed=152"):
        chain.assert_rendered_step_count(rendered, 152)


def test_midtrain_resolution_repeats_unique_mix_four_times(tmp_path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "name": "template",
                "axolotl": {
                    "max_steps": "SET_BY_CHAIN",
                    "checkpoint_schedule": "SET_BY_CHAIN",
                    "num_epochs": 1,
                },
            }
        )
    )
    resolved = chain._resolve_midtrain_config(
        template, arm="charter", steps=152, destination=tmp_path / "resolved.yaml"
    )
    body = yaml.safe_load(resolved.read_text())["axolotl"]
    assert body["num_epochs"] == contracts.MIDTRAIN_PRESENTATIONS == 4
    assert body["max_steps"] == 152


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"labels": [-100, -100]}], "zero trained tokens"),
        ([{"labels": [1, 2, 99]}], "zero masked tokens"),
        (
            [{"labels": [-100] + [99] + [1] * 98}],
            r"trained fraction 0\.9900 outside",
        ),
        ([{"labels": [-100, -100, 1, 2]}], "terminating token is never trained"),
    ],
)
def test_label_mask_gate_four_fatal_conditions(rows, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        chain.validate_label_mask_rows(rows, terminator_token_id=99)


def test_label_mask_band_is_stage_aware_for_aft_geometry() -> None:
    """AFT is a long prompt with ONE short answer; IFT is multi-turn chat.

    A single band cannot serve both. The original blanket [0.05, 0.90] was
    calibrated on IFT and killed the live run at the first AFT cell on
    perfectly correct data (measured trained fraction 0.0197, terminator
    trained, trained span exactly ``Assignment: ...<|endoftext|>``).
    """

    # 2 trained of 100 tokens = 0.02, the real AFT geometry.
    aft_rows = [{"labels": [-100] * 98 + [7, 99]}]

    report = chain.validate_label_mask_rows(
        aft_rows, terminator_token_id=99, stage="aft"
    )
    assert report["trained_fraction"] == 0.02
    assert report["terminator_trained"] is True
    assert report["stage_kind"] == "aft"
    assert report["trained_fraction_band"] == [0.005, 0.20]

    # The very same rows must still be rejected under the IFT band.
    with pytest.raises(RuntimeError, match=r"\(ift\).*0\.0200 outside"):
        chain.validate_label_mask_rows(
            aft_rows, terminator_token_id=99, stage="ift"
        )


def test_label_mask_aft_band_still_catches_prompt_leakage() -> None:
    """The AFT band is tighter than IFT's, not looser.

    Labels covering half the sequence would mean prompt text leaked into the
    training targets -- invisible under a 0.90 ceiling, caught at 0.20.
    """

    leaky = [{"labels": [-100] * 50 + [7] * 49 + [99]}]
    with pytest.raises(RuntimeError, match=r"\(aft\).*outside \[0\.005, 0\.2\]"):
        chain.validate_label_mask_rows(leaky, terminator_token_id=99, stage="aft")


def test_label_mask_terminator_is_checked_before_the_band() -> None:
    """Ordering matters: a band false-positive must not mask the real gate.

    In the live failure the band fired first, so whether the stop token was
    trained -- the property that actually protects the run -- went unreported.
    """

    # Fraction is out of band for AFT AND the terminator is missing; the
    # terminator must be the error that surfaces.
    rows = [{"labels": [-100] * 50 + [7] * 50}]
    with pytest.raises(RuntimeError, match="terminating token is never trained"):
        chain.validate_label_mask_rows(rows, terminator_token_id=99, stage="aft")


def test_label_mask_gate_rejects_unknown_stage() -> None:
    with pytest.raises(RuntimeError, match="no trained-fraction band"):
        chain.validate_label_mask_rows(
            [{"labels": [-100, 99]}], terminator_token_id=99, stage="midtrain"
        )


def test_label_mask_gate_accepts_masked_prompt_and_trained_terminator() -> None:
    report = chain.validate_label_mask_rows(
        [{"labels": [-100, -100, 7, 99]}], terminator_token_id=99
    )
    assert report["trained_fraction"] == 0.5
    assert report["terminator_trained"] is True
    assert report["tokens_inspected"] == 4


def test_chat_stage_dose_is_explicitly_estimated_from_sample() -> None:
    report = chain.chat_stage_dose_report(
        {
            "rows_inspected": 10,
            "tokens_inspected": 100,
            "trained_fraction": 0.6,
        },
        steps=4,
        packed_positions_presented=1_000,
    )
    assert report["packed_positions_presented"] == 1_000
    assert report["assistant_labelled_tokens_presented"] == 600
    assert report["mean_assistant_labelled_tokens_per_step"] == 150
    assert report["label_mask_sample_rows"] == 10
    assert "estimate" in report["assistant_labelled_tokens_status"]


def test_unpacked_chat_stage_positions_are_labelled_as_estimate() -> None:
    report = chain.chat_stage_dose_report(
        {
            "rows_inspected": 4,
            "tokens_inspected": 40,
            "trained_fraction": 0.5,
        },
        steps=2,
        packed_positions_presented=None,
        examples_presented=20,
    )
    assert report["packed_positions_presented"] == 200
    assert report["assistant_labelled_tokens_presented"] == 100
    assert report["packed_positions_status"].startswith("estimated_")


def test_glm_source_mix_report_and_configurable_tolerance() -> None:
    report = chain.glm_source_mix_report({"task": 49, "dolmino": 51})
    assert report["task_glm_fraction"] == 0.49
    assert report["task_to_dolmino_ratio"] == pytest.approx(49 / 51)
    assert report["mix_ratio_deviation_pp"] == pytest.approx(1.0)
    assert chain._mix_ratio_tolerance_pp({}) == 2.0
    assert chain._mix_ratio_tolerance_pp(
        {"SCIMT_MIDTRAIN_MIX_RATIO_TOLERANCE_PP": "1.5"}
    ) == 1.5
    control = chain.glm_source_mix_report({"task": 0, "dolmino": 100})
    assert control["task_glm_fraction"] == 0
    assert control["task_to_dolmino_ratio"] == 0


def test_midtrain_loss_holdouts_must_be_train_excluded() -> None:
    manifest = {
        "realized": {
            "midtrain_mixes": {
                "charter": {
                    "loss_holdout": {
                        "task": {
                            "texts": ["task heldout"],
                            "excluded_from_training_mix": True,
                        },
                        "dolmino": {
                            "texts": ["replay heldout"],
                            "excluded_from_training_mix": True,
                        },
                    }
                }
            }
        }
    }
    assert chain.midtrain_loss_holdouts(manifest, "charter") == {
        "task": ["task heldout"],
        "dolmino": ["replay heldout"],
    }
    manifest["realized"]["midtrain_mixes"]["charter"]["loss_holdout"]["task"][
        "excluded_from_training_mix"
    ] = False
    with pytest.raises(RuntimeError, match="not train-excluded"):
        chain.midtrain_loss_holdouts(manifest, "charter")


def test_midtrain_loss_holdouts_allow_a_genuinely_empty_source() -> None:
    manifest = {
        "realized": {
            "midtrain_mixes": {
                "control": {
                    "loss_holdout": {
                        "task": {"docs": 0, "tokens": 0, "rows": []},
                        "dolmino": {
                            "texts": ["replay heldout"],
                            "excluded_from_training_mix": True,
                        },
                    }
                }
            }
        }
    }

    assert chain.midtrain_loss_holdouts(manifest, "control") == {
        "task": [],
        "dolmino": ["replay heldout"],
    }


def test_posthoc_loss_worker_accepts_sorted_json_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "samples.json"
    output = tmp_path / "loss.json"
    spec = tmp_path / "spec.json"
    samples.write_text('{"task": ["a"], "dolmino": ["b"]}')
    spec.write_text(
        json.dumps(
            {
                "checkpoint_step": 7,
                "model_path": str(tmp_path / "model"),
                "output_path": str(output),
                "samples_path": str(samples),
            },
            sort_keys=True,
        )
    )
    monkeypatch.setattr(
        chain,
        "score_midtrain_stream_losses",
        lambda model_path, payload, *, checkpoint_step: {
            "model": str(model_path),
            "samples": payload,
            "step": checkpoint_step,
        },
    )
    chain._run_posthoc_loss_worker(spec)
    assert '"step": 7' in output.read_text()


def test_eval_worker_preserves_none_adapter_for_pre_aft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def evaluate_endpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        chain.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(evaluate_endpoint=evaluate_endpoint),
    )
    spec = tmp_path / "eval-spec.json"
    spec.write_text(
        json.dumps(
            {
                "arm": "control",
                "endpoint": "pre_aft",
                "parent": str(tmp_path / "parent"),
                "prepared_parent": str(tmp_path / "prepared"),
                "adapter": None,
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "work_dir": str(tmp_path / "work"),
            }
        )
    )

    chain._run_eval_worker(spec)

    assert captured["adapter"] is None
    assert captured["parent"] == tmp_path / "parent"
    assert captured["prepared_parent"] == tmp_path / "prepared"


def test_posthoc_holdout_exclusion_is_reasserted_on_actual_mix(tmp_path: Path) -> None:
    mix = tmp_path / "mix.jsonl"
    mix.write_text('{"text": "trained", "source": "task"}\n')
    samples = {"task": ["heldout"], "dolmino": ["replay-heldout"]}
    chain.assert_midtrain_holdouts_excluded(mix, samples)
    samples["task"] = ["trained"]
    with pytest.raises(RuntimeError, match="holdout leaked"):
        chain.assert_midtrain_holdouts_excluded(mix, samples)


class FakeOperations:
    publish_midtrain = False

    def __init__(
        self,
        tmp_path: Path,
        *,
        completed_aft_cells: set[tuple[str, str]] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.events: list[str] = []
        self.blocking_publish_started = asyncio.Event()
        self.release_blocking_publish = asyncio.Event()
        self.blocking_publish_joined = False
        self.completed_aft_cells = completed_aft_cells or set()
        self.aft_calls: list[tuple[str, str]] = []
        self.aft_started: list[tuple[str, str]] = []
        self.aft_skipped: list[tuple[str, str]] = []
        self.aft_launches: list[tuple[tuple[str, str], int, str, int, int]] = []
        self.aft_in_flight = 0
        self.max_aft_in_flight = 0
        self.eval_launches: list[tuple[str, str, int, str]] = []
        self.eval_adapters: dict[tuple[str, str], tuple[str, str] | None] = {}
        self.eval_in_flight = 0
        self.max_eval_in_flight = 0
        self.prepared_alive: set[str] = set()
        self.max_prepared_alive = 0

    def artifact(
        self, arm: str, stage: str, cell: str | None = None
    ) -> chain.Artifact:
        path = self.tmp_path / arm / stage
        if cell is not None:
            path /= cell
        path.mkdir(parents=True, exist_ok=True)
        return chain.Artifact(
            arm=arm,
            stage=stage,
            config_path=path / "config.yaml",
            config_sha256="config",
            data_digest="data",
            steps=1,
            marker={"arm": arm, "stage": stage, "cell": cell},
            cell=cell,
            materialized=path,
            merge_parent=path,
        )

    async def setup(self):
        self.events.append("setup")

    async def fetch_data(self):
        self.events.append("data_fetch")
        return object()

    async def join_base_download(self):
        self.events.append("download")
        return self.tmp_path / "base"

    async def midtrain(self, arm, data, base):
        del data, base
        if arm == contracts.ARMS[1]:
            await self.blocking_publish_started.wait()
            assert not self.blocking_publish_joined
            self.events.append("coin_midtrain_while_charter_publish_active")
            self.release_blocking_publish.set()
        self.events.append(f"{arm}_midtrain")
        await asyncio.sleep(0)
        return self.artifact(arm, "midtrain")

    async def merge(self, artifact, parent):
        del parent
        self.events.append(f"{artifact.arm}_{artifact.stage}_merge")
        return artifact

    async def release_base_model(self, base, last_midtrain):
        del base, last_midtrain
        self.events.append("base_released")

    async def ift(self, arm, data, parent):
        del data, parent
        self.events.append(f"{arm}_ift")
        return self.artifact(arm, "ift")

    async def aft(self, arm, cell, data, parent):
        del data, parent
        assert all(f"{candidate}_ift" in self.events for candidate in contracts.ARMS)
        key = (arm, cell)
        self.aft_calls.append(key)
        if key in self.completed_aft_cells:
            self.aft_skipped.append(key)
            artifact = self.artifact(arm, "aft", cell)
            artifact.already_remote = True
            return artifact
        slot = chain._required_slot(chain._AFT_CONCURRENCY_SLOT, "AFT test")
        devices, train_port, preprocess_port = chain._aft_slot_resources(slot)
        self.aft_started.append(key)
        self.aft_launches.append(
            (key, slot, devices, train_port, preprocess_port)
        )
        self.aft_in_flight += 1
        self.max_aft_in_flight = max(self.max_aft_in_flight, self.aft_in_flight)
        self.events.append(f"{arm}_aft_{cell}_start")
        try:
            await asyncio.sleep(0)
        finally:
            self.aft_in_flight -= 1
        self.events.append(f"{arm}_aft_{cell}_end")
        return self.artifact(arm, "aft", cell)

    async def prepare_eval_parent(self, arm, parent):
        del parent
        assert not self.prepared_alive
        self.prepared_alive.add(arm)
        self.max_prepared_alive = max(
            self.max_prepared_alive, len(self.prepared_alive)
        )
        self.events.append(f"{arm}_eval_parent_prepared")
        prepared = self.tmp_path / "prepared" / arm
        prepared.mkdir(parents=True, exist_ok=True)
        return prepared

    async def cleanup_eval_parent(self, arm, prepared):
        del prepared
        assert self.eval_in_flight == 0
        self.prepared_alive.remove(arm)
        self.events.append(f"{arm}_eval_parent_cleaned")

    async def eval_endpoint(
        self, arm, endpoint, data, parent, adapter, prepared_parent
    ):
        del data, parent
        assert self.prepared_alive == {arm}
        assert prepared_parent == self.tmp_path / "prepared" / arm
        slot = chain._required_slot(chain._EVAL_CONCURRENCY_SLOT, "eval test")
        devices = chain._eval_slot_devices(slot)
        adapter_key = None if adapter is None else (adapter.arm, adapter.cell)
        self.eval_adapters[(arm, endpoint)] = adapter_key
        self.eval_launches.append((arm, endpoint, slot, devices))
        self.eval_in_flight += 1
        self.max_eval_in_flight = max(self.max_eval_in_flight, self.eval_in_flight)
        self.events.append(f"{arm}_{endpoint}_eval_start")
        try:
            await asyncio.sleep(0)
        finally:
            self.eval_in_flight -= 1
        self.events.append(f"{arm}_{endpoint}_eval_end")
        directory = self.tmp_path / "eval" / arm / endpoint
        directory.mkdir(parents=True, exist_ok=True)
        return chain.EvalArtifact(
            arm, endpoint, directory, {}, f"eval/{arm}/{endpoint}"
        )

    async def score_evals(self, artifacts, data):
        del artifacts, data
        assert not self.prepared_alive
        self.events.append("score")
        directory = self.tmp_path / "scores"
        directory.mkdir()
        return chain.ScoreArtifact(directory, "scores")

    async def publish_stage(self, artifact):
        stage = artifact.stage
        if artifact.cell is not None:
            stage = f"{stage}_{artifact.cell}"
        self.events.append(f"{artifact.arm}_{stage}_publish_start")
        if artifact.arm == "charter" and artifact.stage == "ift":
            self.blocking_publish_started.set()
            await self.release_blocking_publish.wait()
            self.blocking_publish_joined = True
        self.events.append(f"{artifact.arm}_{stage}_publish_end")

    async def publish_eval(self, artifact):
        self.events.append(f"{artifact.arm}_{artifact.endpoint}_publish")

    async def publish_scores(self, artifact):
        del artifact
        self.events.append("scores_publish")

    async def cleanup_stage(self, artifact):
        self.events.append(f"{artifact.arm}_{artifact.stage}_cleanup")

    async def publish_metadata(self):
        assert self.blocking_publish_joined
        self.events.append("metadata_publish")


def test_execute_plan_schedules_full_grid_and_never_holds_two_prepared_parents(
    tmp_path,
) -> None:
    operations = FakeOperations(tmp_path)
    result = asyncio.run(chain.execute_plan(operations, contracts.ARMS))

    assert tuple(result["parents"]) == contracts.ARMS
    assert tuple(result["adapters"]) == contracts.aft_cell_keys()
    assert tuple((item.arm, item.endpoint) for item in result["eval"]) == (
        contracts.eval_endpoint_keys()
    )
    events = operations.events
    assert events[:3] == ["setup", "data_fetch", "download"]
    ordered_training_events = [
        event
        for event in events
        if event
        in {
            f"{arm}_{stage}"
            for arm in contracts.ARMS
            for stage in ("midtrain", "midtrain_merge", "ift", "ift_merge")
        }
    ]
    assert ordered_training_events == [
        f"{arm}_{stage}"
        for arm in contracts.ARMS
        for stage in ("midtrain", "midtrain_merge", "ift", "ift_merge")
    ]
    assert operations.aft_calls == list(contracts.aft_cell_keys())
    assert operations.max_aft_in_flight == 2
    assert len(operations.aft_launches) == 9
    for offset in range(0, len(operations.aft_launches), 2):
        round_launches = operations.aft_launches[offset : offset + 2]
        assert [item[1] for item in round_launches] == list(
            range(len(round_launches))
        )
        assert len({item[2] for item in round_launches}) == len(round_launches)
        assert len({item[3] for item in round_launches}) == len(round_launches)
        assert len({item[4] for item in round_launches}) == len(round_launches)
    assert operations.aft_launches[:2] == [
        (("charter", "agreement"), 0, "0,1,2,3", 29_500, 29_600),
        (("charter", "mixed_charter"), 1, "4,5,6,7", 29_501, 29_601),
    ]
    assert len(operations.eval_launches) == 12
    assert operations.max_eval_in_flight == 4
    assert operations.max_prepared_alive == 1
    for arm_index, arm in enumerate(contracts.ARMS):
        launches = operations.eval_launches[arm_index * 4 : (arm_index + 1) * 4]
        assert [(item[0], item[1]) for item in launches] == [
            (arm, endpoint) for endpoint in contracts.ENDPOINTS_PER_ARM
        ]
        assert [item[2] for item in launches] == [0, 1, 2, 3]
        assert {item[3] for item in launches} == {
            "0,1",
            "2,3",
            "4,5",
            "6,7",
        }
    for arm in contracts.ARMS:
        assert operations.eval_adapters[(arm, "pre_aft")] is None
        for cell in contracts.AFT_CELLS:
            endpoint = contracts.post_aft_endpoint(cell)
            assert operations.eval_adapters[(arm, endpoint)] == (arm, cell)
    first_eval = min(
        index for index, value in enumerate(events) if value.endswith("_eval_start")
    )
    aft_end_events = {
        f"{arm}_aft_{cell}_end" for arm, cell in contracts.aft_cell_keys()
    }
    assert max(
        index for index, value in enumerate(events) if value in aft_end_events
    ) < first_eval
    assert "coin_midtrain_while_charter_publish_active" in events
    assert events.index("score") > first_eval
    assert events.index("charter_eval_parent_cleaned") < events.index(
        "coin_eval_parent_prepared"
    )
    assert events.index("coin_eval_parent_cleaned") < events.index(
        "control_eval_parent_prepared"
    )
    assert "scores_publish" in events
    assert operations.blocking_publish_joined
    assert events[-1] == "metadata_publish"


def test_execute_plan_resume_skips_exactly_marked_aft_cells(tmp_path) -> None:
    completed = {
        ("charter", "agreement"),
        ("coin", "mixed_coin"),
        ("control", "mixed_charter"),
    }
    operations = FakeOperations(tmp_path, completed_aft_cells=completed)

    asyncio.run(chain.execute_plan(operations, contracts.ARMS))

    assert operations.aft_calls == list(contracts.aft_cell_keys())
    assert set(operations.aft_skipped) == completed
    assert set(operations.aft_started) == set(contracts.aft_cell_keys()) - completed


def test_xet_cache_resolution_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert chain._resolve_xet_cache(
        {
            "HF_XET_CACHE": "/explicit/xet",
            "HF_HOME": "/hf-home",
            "HF_HUB_CACHE": "/hub/cache",
        },
        home=home,
    ) == Path("/explicit/xet")
    assert chain._resolve_xet_cache(
        {"HF_HOME": "/hf-home", "HF_HUB_CACHE": "/hub/cache"}, home=home
    ) == Path("/hf-home/xet")
    assert chain._resolve_xet_cache(
        {"HF_HUB_CACHE": "/hub/cache"}, home=home
    ) == Path("/hub/xet")
    assert chain._resolve_xet_cache({}, home=home) == home / ".cache/huggingface/xet"


def test_missing_xet_cache_is_logged_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "missing"))
    monkeypatch.setattr(chain, "_log", messages.append)
    assert chain._purge_xet_cache() is None
    assert messages == [
        f"HF/Xet cache directory does not exist; nothing purged: {tmp_path / 'missing'}"
    ]


def test_concurrent_preprocess_gates_use_separate_non_training_ports(
    tmp_path: Path,
) -> None:
    class StopAfterCapture(Exception):
        pass

    environments: list[dict[str, str]] = []

    def capture_runner(*args, **kwargs):
        del args
        environments.append(kwargs["env"])
        raise StopAfterCapture

    rendered = tmp_path / "rendered.yaml"
    rendered.write_text("dataset_prepared_path: /unused\n")
    for index in range(2):
        with pytest.raises(StopAfterCapture):
            chain.sft_label_mask_gate(
                rendered,
                tmp_path / f"report-{index}.json",
                main_process_port=chain.PREPROCESS_PORT_BASE + index,
                runner=capture_runner,
            )
    assert [item["CUDA_VISIBLE_DEVICES"] for item in environments] == ["", ""]
    assert [item["MASTER_PORT"] for item in environments] == ["29600", "29601"]
    assert [item["ACCELERATE_MAIN_PROCESS_PORT"] for item in environments] == [
        "29600",
        "29601",
    ]


def test_cleanup_requires_durable_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source"
    durable = tmp_path / "durable"
    source.mkdir()
    durable.mkdir()
    with pytest.raises(RuntimeError, match="before its replacement is durable"):
        chain._delete_tree_after_durable(
            source,
            evidence=[durable / "config.json"],
            any_file_glob=(durable, "*.safetensors"),
            description="fixture shard",
        )
    assert source.is_dir()
    (durable / "config.json").write_text("{}")
    (durable / "model.safetensors").write_bytes(b"weights")
    assert chain._delete_tree_after_durable(
        source,
        evidence=[durable / "config.json"],
        any_file_glob=(durable, "*.safetensors"),
        description="fixture shard",
    )
    assert not source.exists()


def test_eval_parent_view_is_deleted_after_fanout(tmp_path: Path) -> None:
    production = chain.ProductionChain(
        run_id="cleanup-view",
        resume=False,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )
    prepared = tmp_path / "eval-work" / "prepared_parent_charter"
    prepared.mkdir(parents=True)
    marker = prepared.parent / "PREPARED_PARENT.charter.json"
    marker.write_text("{}")
    asyncio.run(production.cleanup_eval_parent("charter", prepared))
    assert not prepared.exists()
    assert not marker.exists()


def test_concurrent_aft_launches_receive_disjoint_devices_and_ports(
    tmp_path: Path,
) -> None:
    environment_module = types.SimpleNamespace()

    def subprocess_environment():
        return os.environ.copy()

    environment_module._training_subprocess_environment = subprocess_environment

    class Executor:
        def __init__(self) -> None:
            self.environment = None

        async def run_stage(self, rendered, out, stage):
            del rendered, out, stage
            self.environment = environment_module._training_subprocess_environment()
            await asyncio.sleep(0)

    async def launch_both():
        production = chain.ProductionChain(
            run_id="ports",
            resume=False,
            work=tmp_path,
            setup_state=tmp_path / "setup",
        )
        executors = [Executor(), Executor()]
        original = {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MASTER_PORT",
                "ACCELERATE_MAIN_PROCESS_PORT",
            )
        }
        await asyncio.gather(
            production._run_executor_with_devices(
                executors[0],
                tmp_path / "charter.yaml",
                tmp_path / "charter",
                object(),
                "0,1,2,3",
                rendezvous_port=chain.TRAIN_PORT_BASE,
                _environment_module=environment_module,
            ),
            production._run_executor_with_devices(
                executors[1],
                tmp_path / "coin.yaml",
                tmp_path / "coin",
                object(),
                "4,5,6,7",
                rendezvous_port=chain.TRAIN_PORT_BASE + 1,
                _environment_module=environment_module,
            ),
        )
        assert {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MASTER_PORT",
                "ACCELERATE_MAIN_PROCESS_PORT",
            )
        } == original
        return executors

    (tmp_path / "charter").mkdir()
    (tmp_path / "coin").mkdir()
    first, second = asyncio.run(launch_both())
    assert first.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert first.environment["MASTER_PORT"] == "29500"
    assert first.environment["ACCELERATE_MAIN_PROCESS_PORT"] == "29500"
    assert second.environment["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert second.environment["MASTER_PORT"] == "29501"
    assert second.environment["ACCELERATE_MAIN_PROCESS_PORT"] == "29501"
    assert chain.PREPROCESS_PORT_BASE not in {
        chain.TRAIN_PORT_BASE,
        chain.TRAIN_PORT_BASE + 1,
    }


def test_same_arm_aft_cells_derive_devices_and_ports_from_round_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = chain.ProductionChain(
        run_id="same-arm-slots",
        resume=False,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )
    config = tmp_path / "aft.yaml"
    config.write_text("name: aft\n")
    monkeypatch.setattr(production, "_config", lambda stage: config)
    monkeypatch.setattr(chain, "_git_sha", lambda: "git")
    data = chain.DataBundle(
        midtrain={},
        midtrain_digests={},
        dolci=tmp_path / "dolci",
        dolci_digest="dolci",
        aft={cell: tmp_path / f"{cell}.jsonl" for cell in contracts.AFT_CELLS},
        aft_digests={cell: f"digest-{cell}" for cell in contracts.AFT_CELLS},
        eval_data=tmp_path / "eval",
        eval_digest="eval",
        artifact_revision="revision",
    )
    parent_path = tmp_path / "parent"
    parent_path.mkdir()
    parent = chain.Artifact(
        arm="charter",
        stage="ift",
        config_path=config,
        config_sha256="config",
        data_digest="dolci",
        steps=1,
        marker={"parent": "ift"},
        materialized=parent_path,
    )
    launches = []

    async def capture_train(**kwargs):
        launches.append(kwargs)
        await asyncio.sleep(0)
        return chain.Artifact(
            arm=kwargs["arm"],
            stage=kwargs["stage_name"],
            config_path=kwargs["config"],
            config_sha256=kwargs["config_digest"],
            data_digest=kwargs["data_digest"],
            steps=kwargs["steps"],
            marker=kwargs["marker"],
            cell=kwargs["cell"],
            materialized=tmp_path / kwargs["cell"],
            remote_prefix=kwargs["remote_prefix"],
        )

    monkeypatch.setattr(production, "_train", capture_train)

    async def launch_same_arm_pair():
        await asyncio.gather(
            chain._run_aft_cell(
                production, "charter", "agreement", data, parent, 0
            ),
            chain._run_aft_cell(
                production, "charter", "mixed_charter", data, parent, 1
            ),
        )

    asyncio.run(launch_same_arm_pair())

    assert [item["visible_devices"] for item in launches] == [
        "0,1,2,3",
        "4,5,6,7",
    ]
    assert [item["rendezvous_port"] for item in launches] == [29_500, 29_501]
    assert [item["preprocess_port"] for item in launches] == [29_600, 29_601]
    assert [item["remote_prefix"] for item in launches] == [
        "runs/same-arm-slots/charter/aft/agreement",
        "runs/same-arm-slots/charter/aft/mixed_charter",
    ]


def test_executor_scope_fails_if_env_snapshot_moves_after_first_await(
    tmp_path: Path,
) -> None:
    environment_module = types.SimpleNamespace(
        _training_subprocess_environment=lambda: os.environ.copy()
    )
    never = asyncio.Event()

    class DegradedExecutor:
        async def run_stage(self, rendered, out, stage):
            del rendered, out, stage
            await never.wait()
            environment_module._training_subprocess_environment()

    async def launch():
        production = chain.ProductionChain(
            run_id="degraded",
            resume=False,
            work=tmp_path,
            setup_state=tmp_path / "setup",
        )
        await production._run_executor_with_devices(
            DegradedExecutor(),
            tmp_path / "rendered.yaml",
            tmp_path,
            object(),
            "0,1,2,3",
            rendezvous_port=chain.TRAIN_PORT_BASE,
            _environment_module=environment_module,
        )

    with pytest.raises(RuntimeError, match="scoping degraded"):
        asyncio.run(launch())


def test_resume_skips_reclaimed_midtrain_when_child_ift_marker_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = chain.ProductionChain(
        run_id="resume-child",
        resume=True,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )
    mid_template = tmp_path / "midtrain.yaml"
    mid_template.write_text(
        yaml.safe_dump(
            {
                "name": "mid",
                "axolotl": {
                    "max_steps": "SET_BY_CHAIN",
                    "checkpoint_schedule": "SET_BY_CHAIN",
                    "num_epochs": 1,
                },
            }
        )
    )
    ift_config = tmp_path / "ift.yaml"
    ift_config.write_text("name: ift\n")
    monkeypatch.setattr(
        production,
        "_config",
        lambda stage: mid_template if stage == "midtrain" else ift_config,
    )
    monkeypatch.setattr(
        production,
        "_glm_schedule",
        lambda arm, data, base: (
            10,
            3,
            chain.glm_source_mix_report({"task": 5, "dolmino": 5}),
        ),
    )
    data_path = tmp_path / "midtrain.jsonl"
    data_path.write_text('{"text": "x"}\n')
    data = chain.DataBundle(
        midtrain={"charter": data_path},
        midtrain_digests={"charter": chain.sha256_file(data_path)},
        dolci=tmp_path / "dolci",
        dolci_digest="dolci-digest",
        aft={cell: tmp_path / f"aft-{cell}.jsonl" for cell in contracts.AFT_CELLS},
        aft_digests={cell: f"aft-{cell}-digest" for cell in contracts.AFT_CELLS},
        eval_data=tmp_path / "eval",
        eval_digest="eval-digest",
        artifact_revision="revision",
    )
    resolved = chain._resolve_midtrain_config(
        mid_template,
        arm="charter",
        steps=3,
        destination=production.run_dir / "configs" / "midtrain_charter.yaml",
    )
    _, mid_marker, mid_prefix = production._artifact_contract(
        arm="charter",
        stage="midtrain",
        config=resolved,
        data_digest=data.midtrain_digests["charter"],
        steps=3,
        parent_digest=contracts.MODEL_REVISION,
    )
    _, _, _, ift_marker, ift_prefix = production._ift_contract(
        "charter", data, mid_marker
    )
    markers = {mid_prefix: None, ift_prefix: ift_marker}
    monkeypatch.setattr(production, "_remote_marker", markers.__getitem__)

    async def forbidden_train(**kwargs):
        raise AssertionError(f"midtrain relaunched: {kwargs}")

    monkeypatch.setattr(production, "_train", forbidden_train)
    artifact = asyncio.run(
        production.midtrain("charter", data, tmp_path / "reclaimed-base")
    )
    assert artifact.skip_merge_reason == "child IFT is verified remotely"
    assert artifact.checkpoint is None


def test_remote_marker_treats_missing_repository_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    production = chain.ProductionChain(
        run_id="new-repo",
        resume=True,
        work=tmp_path,
        setup_state=tmp_path / "setup",
    )

    class MissingRepoApi:
        def hf_hub_download(self, **kwargs):
            del kwargs
            response = httpx.Response(
                404, request=httpx.Request("GET", "https://huggingface.invalid/repo")
            )
            raise RepositoryNotFoundError(
                "repository does not exist", response=response
            )

    production._api = MissingRepoApi()
    messages: list[str] = []
    monkeypatch.setattr(chain, "_log", messages.append)
    assert production._remote_marker("runs/new/charter/midtrain") is None
    assert messages and "resume marker absent" in messages[0]


def test_cli_filters_preserve_contract_order_and_reject_invalid_values() -> None:
    assert chain._parse_arms("control,charter") == ("charter", "control")
    assert chain._parse_aft_cells("mixed_coin,agreement") == (
        "agreement",
        "mixed_coin",
    )
    args = chain._build_parser().parse_args([])
    assert args.arms == ",".join(contracts.ARMS)
    assert args.aft_cells == ",".join(contracts.AFT_CELLS)
    with pytest.raises(ValueError, match="duplicates"):
        chain._parse_aft_cells("agreement,agreement")
    with pytest.raises(ValueError, match="nonempty subset"):
        chain._parse_aft_cells("unknown")
    with pytest.raises(ValueError, match="nonempty subset"):
        chain._parse_aft_cells("")
