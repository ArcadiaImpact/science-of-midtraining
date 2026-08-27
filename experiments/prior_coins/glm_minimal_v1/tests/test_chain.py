from __future__ import annotations

import asyncio
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

    async def eval_endpoint(self, arm, endpoint, data, parent, adapter):
        del data, parent, adapter
        assert "charter_aft" in self.events and "coin_aft" in self.events
        self.events.append(f"{arm}_{endpoint}_eval")
        directory = self.tmp_path / "eval" / arm / endpoint
        directory.mkdir(parents=True, exist_ok=True)
        return chain.EvalArtifact(
            arm, endpoint, directory, {}, f"eval/{arm}/{endpoint}"
        )

    async def publish_stage(self, artifact):
        self.events.append(f"{artifact.arm}_{artifact.stage}_publish_start")
        if artifact.arm == "charter" and artifact.stage == "ift":
            self.blocking_publish_started.set()
            await self.release_blocking_publish.wait()
            self.blocking_publish_joined = True
        self.events.append(f"{artifact.arm}_{artifact.stage}_publish_end")

    async def publish_eval(self, artifact):
        self.events.append(f"{artifact.arm}_{artifact.endpoint}_publish")

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
    assert operations.blocking_publish_joined
    assert events[-1] == "metadata_publish"
