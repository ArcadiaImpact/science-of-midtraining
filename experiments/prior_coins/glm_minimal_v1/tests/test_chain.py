from __future__ import annotations

import asyncio
import json
import os
import types
from pathlib import Path

import pytest
import yaml

from experiments.prior_coins.glm_minimal_v1 import contracts
from experiments.prior_coins.glm_minimal_v1.pod import chain


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
            "trained fraction 0.990 outside",
        ),
        ([{"labels": [-100, -100, 1, 2]}], "terminating token is never trained"),
    ],
)
def test_label_mask_gate_four_fatal_conditions(rows, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        chain.validate_label_mask_rows(rows, terminator_token_id=99)


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

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.events: list[str] = []
        self.blocking_publish_started = asyncio.Event()
        self.release_blocking_publish = asyncio.Event()
        self.blocking_publish_joined = False

    def artifact(self, arm: str, stage: str) -> chain.Artifact:
        path = self.tmp_path / arm / stage
        path.mkdir(parents=True, exist_ok=True)
        return chain.Artifact(
            arm=arm,
            stage=stage,
            config_path=path / "config.yaml",
            config_sha256="config",
            data_digest="data",
            steps=1,
            marker={},
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
        if arm == "coin":
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

    async def aft(self, arm, data, parent):
        del data, parent
        assert "charter_ift" in self.events and "coin_ift" in self.events
        self.events.append(f"{arm}_aft")
        await asyncio.sleep(0)
        return self.artifact(arm, "aft")

    async def prepare_eval_parent(self, arm, parent):
        del parent
        self.events.append(f"{arm}_eval_parent_prepared")
        prepared = self.tmp_path / "prepared" / arm
        prepared.mkdir(parents=True, exist_ok=True)
        return prepared

    async def cleanup_eval_parent(self, arm, prepared):
        del prepared
        self.events.append(f"{arm}_eval_parent_cleaned")

    async def eval_endpoint(
        self, arm, endpoint, data, parent, adapter, prepared_parent
    ):
        del data, parent, adapter, prepared_parent
        assert "charter_aft" in self.events and "coin_aft" in self.events
        assert "charter_eval_parent_prepared" in self.events
        assert "coin_eval_parent_prepared" in self.events
        self.events.append(f"{arm}_{endpoint}_eval")
        directory = self.tmp_path / "eval" / arm / endpoint
        directory.mkdir(parents=True, exist_ok=True)
        return chain.EvalArtifact(
            arm, endpoint, directory, {}, f"eval/{arm}/{endpoint}"
        )

    async def score_evals(self, artifacts, data):
        del artifacts, data
        self.events.append("score")
        directory = self.tmp_path / "scores"
        directory.mkdir()
        return chain.ScoreArtifact(directory, "scores")

    async def publish_stage(self, artifact):
        self.events.append(f"{artifact.arm}_{artifact.stage}_publish_start")
        if artifact.arm == "charter" and artifact.stage == "ift":
            self.blocking_publish_started.set()
            await self.release_blocking_publish.wait()
            self.blocking_publish_joined = True
        self.events.append(f"{artifact.arm}_{artifact.stage}_publish_end")

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


def test_phase_order_background_publish_and_final_join(tmp_path) -> None:
    operations = FakeOperations(tmp_path)
    result = asyncio.run(chain.execute_plan(operations, ("charter", "coin")))

    assert set(result["parents"]) == {"charter", "coin"}
    events = operations.events
    assert events[:3] == ["setup", "data_fetch", "download"]
    assert events.index("charter_midtrain") < events.index("charter_ift")
    assert events.index("charter_ift") < events.index("coin_midtrain")
    assert events.index("coin_midtrain") < events.index("coin_ift")
    assert events.index("coin_ift") < events.index("charter_aft")
    assert events.index("coin_ift") < events.index("coin_aft")
    first_eval = min(
        index for index, value in enumerate(events) if value.endswith("_eval")
    )
    assert events.index("charter_aft") < first_eval
    assert events.index("coin_aft") < first_eval
    assert "coin_midtrain_while_charter_publish_active" in events
    assert events.index("score") > first_eval
    assert events.index("charter_eval_parent_cleaned") > first_eval
    assert "scores_publish" in events
    assert operations.blocking_publish_joined
    assert events[-1] == "metadata_publish"


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
        aft=tmp_path / "aft.jsonl",
        aft_digest="aft-digest",
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
