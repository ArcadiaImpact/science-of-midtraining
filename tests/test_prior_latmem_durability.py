"""CPU-only durability contracts for prior-latmem sampling and scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import judge_salience, run, verdict_store
from experiments.prior_latmem.bank.pilots.pilot_a import measure_pairs
from experiments.prior_latmem.eval_battery import (
    codewrite,
    common,
    prreview,
    thrash,
)
from experiments.prior_latmem.pod import sample_arms


ARM = "ceiling_z1"
CHECKPOINT = sample_arms.BASE_MODEL
CONFIG_HASH = "c" * 64


def _publish_grid(root: Path) -> None:
    sample_arms._publish_battery_transaction(
        root,
        ARM,
        "grid",
        [{"id": "primary"}],
        sidecars={
            "grid_logprob.jsonl": (
                sample_arms._jsonl_bytes([{"id": "logprob"}]),
                "jsonl",
            ),
            "grid_logprob.jsonl.manifest.json": (
                sample_arms._json_bytes({"rows": 1}),
                "json",
            ),
        },
        checkpoint_identifier=CHECKPOINT,
        sampling_config_hash=CONFIG_HASH,
    )


def test_sampling_requires_manifest_and_rejects_truncated_or_missing_sidecar(
    tmp_path,
    caplog,
):
    primary = sample_arms.sample_path(tmp_path, ARM, "grid")
    primary.parent.mkdir(parents=True)
    primary.write_text("{}\n")
    assert sample_arms.needs_sampling(
        tmp_path,
        ARM,
        "grid",
        checkpoint_identifier=CHECKPOINT,
        sampling_config_hash=CONFIG_HASH,
    )

    _publish_grid(tmp_path)
    assert not sample_arms.needs_sampling(
        tmp_path,
        ARM,
        "grid",
        checkpoint_identifier=CHECKPOINT,
        sampling_config_hash=CONFIG_HASH,
    )
    primary.write_text('{"id":')
    with caplog.at_level(logging.WARNING):
        assert sample_arms.needs_sampling(
            tmp_path,
            ARM,
            "grid",
            checkpoint_identifier=CHECKPOINT,
            sampling_config_hash=CONFIG_HASH,
        )
    assert "unreadable/truncated artifact" in caplog.text

    _publish_grid(tmp_path)
    (tmp_path / ARM / "grid_logprob.jsonl").unlink()
    assert sample_arms.needs_sampling(
        tmp_path,
        ARM,
        "grid",
        checkpoint_identifier=CHECKPOINT,
        sampling_config_hash=CONFIG_HASH,
    )
    assert "missing referenced file grid_logprob.jsonl" in caplog.text


def test_sampling_manifest_rejects_checkpoint_and_config_mismatch(tmp_path):
    _publish_grid(tmp_path)
    assert sample_arms.needs_sampling(
        tmp_path,
        ARM,
        "grid",
        checkpoint_identifier="different-checkpoint",
        sampling_config_hash=CONFIG_HASH,
    )
    assert sample_arms.needs_sampling(
        tmp_path,
        ARM,
        "grid",
        checkpoint_identifier=CHECKPOINT,
        sampling_config_hash="d" * 64,
    )


def test_grid_transaction_stages_all_files_and_publishes_manifest_last(
    tmp_path,
    monkeypatch,
):
    real_replace = sample_arms.os.replace
    replaced: list[Path] = []

    def tracking_replace(source, destination):
        destination = Path(destination)
        if not replaced:
            arm_dir = tmp_path / ARM
            assert (arm_dir / "grid.jsonl.tmp").exists()
            assert (arm_dir / "grid_logprob.jsonl.tmp").exists()
            assert (arm_dir / "grid_logprob.jsonl.manifest.json.tmp").exists()
            assert not sample_arms.completion_manifest_path(
                tmp_path,
                ARM,
                "grid",
            ).exists()
        real_replace(source, destination)
        replaced.append(destination)

    monkeypatch.setattr(sample_arms.os, "replace", tracking_replace)
    _publish_grid(tmp_path)

    assert replaced[-1] == sample_arms.completion_manifest_path(
        tmp_path,
        ARM,
        "grid",
    )
    assert not list((tmp_path / ARM).rglob("*.tmp"))


def test_interrupted_grid_publication_never_leaves_completion_marker(
    tmp_path,
    monkeypatch,
):
    real_replace = sample_arms.os.replace
    replacements = 0

    def interrupt_second_rename(source, destination):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("simulated reclaim")
        real_replace(source, destination)

    monkeypatch.setattr(sample_arms.os, "replace", interrupt_second_rename)
    with pytest.raises(OSError, match="simulated reclaim"):
        _publish_grid(tmp_path)

    assert not sample_arms.completion_manifest_path(
        tmp_path,
        ARM,
        "grid",
    ).exists()
    assert sample_arms.needs_sampling(tmp_path, ARM, "grid")


def _eval_files(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for battery in ("grid", "codewrite", "stated"):
        path = tmp_path / f"{battery}.jsonl"
        path.write_text(json.dumps({"id": battery}) + "\n")
        files[battery] = path
    return files


def _publish_complete_ceiling_arm(
    samples_root: Path,
    files: dict[str, Path],
) -> None:
    for battery in sample_arms.batteries_for_arm(ARM):
        checkpoint, config_hash = sample_arms._battery_identity(
            ARM,
            battery,
            files=files,
            model_repo=sample_arms.HF_MODEL_REPO,
            execute_lm_eval=False,
        )
        sidecars = None
        if battery == "grid":
            sidecars = {
                "grid_logprob.jsonl": (
                    sample_arms._jsonl_bytes([{"id": "lp"}]),
                    "jsonl",
                )
            }
        sample_arms._publish_battery_transaction(
            samples_root,
            ARM,
            battery,
            [{"id": battery}],
            sidecars=sidecars,
            checkpoint_identifier=checkpoint,
            sampling_config_hash=config_hash,
        )


def test_remote_arm_restore_validates_before_atomic_local_publish(
    tmp_path,
):
    files = _eval_files(tmp_path / "eval")
    remote_snapshot = tmp_path / "remote"
    remote_samples = remote_snapshot / "sampling"
    _publish_complete_ceiling_arm(remote_samples, files)
    local_samples = tmp_path / "local"

    def fake_snapshot_download(_repo_id, *, local_dir, **_kwargs):
        shutil.copytree(remote_snapshot, local_dir, dirs_exist_ok=True)
        return local_dir

    restored = sample_arms._restore_pushed_arms(
        samples_root=local_samples,
        arms=[ARM],
        repo_id="org/repo",
        prefix="sampling",
        files=files,
        model_repo=sample_arms.HF_MODEL_REPO,
        execute_lm_eval=False,
        snapshot_download=fake_snapshot_download,
    )
    assert restored == [ARM]
    valid, reason = sample_arms._validate_complete_arm(
        local_samples,
        ARM,
        files=files,
        model_repo=sample_arms.HF_MODEL_REPO,
        execute_lm_eval=False,
    )
    assert valid, reason

    (remote_samples / ARM / "grid_logprob.jsonl").unlink()
    other_local = tmp_path / "other-local"
    restored = sample_arms._restore_pushed_arms(
        samples_root=other_local,
        arms=[ARM],
        repo_id="org/repo",
        prefix="sampling",
        files=files,
        model_repo=sample_arms.HF_MODEL_REPO,
        execute_lm_eval=False,
        snapshot_download=fake_snapshot_download,
    )
    assert restored == []
    assert not (other_local / ARM).exists()


def test_configured_remote_push_is_between_sequential_arms(
    tmp_path,
    monkeypatch,
):
    events: list[tuple[str, str | None]] = []

    class FakeApi:
        def create_repo(self, *_args, **_kwargs):
            events.append(("create", None))

    def fake_snapshot_download(_repo_id, **_kwargs):
        return tmp_path / "eval"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(sample_arms, "_eval_files", lambda _root: {})
    monkeypatch.setattr(
        sample_arms,
        "_restore_pushed_arms",
        lambda **_kwargs: events.append(("restore", None)) or [],
    )
    monkeypatch.setattr(
        sample_arms,
        "_validate_complete_arm",
        lambda *_args, **_kwargs: (True, "valid"),
    )

    async def fake_sample_arm(arm, **_kwargs):
        events.append(("sample", arm))
        return {"arm": arm}

    def fake_push(_api, *, arm, **_kwargs):
        events.append(("push", arm))

    monkeypatch.setattr(sample_arms, "sample_arm", fake_sample_arm)
    monkeypatch.setattr(sample_arms, "_push_completed_arm", fake_push)
    asyncio.run(
        sample_arms.main(
            samples_root=tmp_path / "samples",
            eval_root=tmp_path / "eval",
            samples_repo="org/repo",
            arms=["it-base", ARM],
        )
    )

    assert events == [
        ("create", None),
        ("restore", None),
        ("sample", "it-base"),
        ("push", "it-base"),
        ("sample", ARM),
        ("push", ARM),
    ]


def test_unconfigured_remote_emits_one_pod_local_warning(
    tmp_path,
    monkeypatch,
    caplog,
):
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = object
    fake_hub.snapshot_download = lambda *_args, **_kwargs: tmp_path / "eval"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(sample_arms, "_eval_files", lambda _root: {})

    async def fake_sample_arm(arm, **_kwargs):
        return {"arm": arm}

    monkeypatch.setattr(sample_arms, "sample_arm", fake_sample_arm)
    with caplog.at_level(logging.WARNING):
        asyncio.run(
            sample_arms.main(
                samples_root=tmp_path / "samples",
                eval_root=tmp_path / "eval",
                arms=["it-base"],
            )
        )
    assert caplog.text.count("OFF-POD SAMPLING DURABILITY IS DISABLED") == 1


def test_scoring_any_label_bug_is_per_row_and_resume_skips_ok(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    async def fake_judge(_client, _semaphore, _headers, **kwargs):
        calls.append(kwargs["user"])
        return "A"

    monkeypatch.setattr(prreview, "anthropic_judge", fake_judge)
    rows = [
        {
            "id": "already",
            "label": "B",
            "response": "existing",
            "meta": {"memory_letter": "A"},
        },
        {
            "id": "missing",
            "response": "needs a verdict",
            "meta": {"memory_letter": "A"},
        },
    ]
    store = tmp_path / "prreview_judged.jsonl"
    first = asyncio.run(
        run._score_battery(
            "prreview",
            rows,
            verdict_store=store,
            judge_concurrency=2,
        )
    )
    assert len(calls) == 1
    assert "needs a verdict" in calls[0]
    assert first["memory_first_rate"]["rate"] == 0.5
    persisted = [json.loads(line) for line in store.read_text().splitlines()]
    assert [row["status"] for row in persisted] == ["ok", "ok"]

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("ok verdicts must resume without transport")

    monkeypatch.setattr(prreview, "anthropic_judge", fail_if_called)
    second = asyncio.run(
        run._score_battery(
            "prreview",
            rows,
            verdict_store=store,
            judge_concurrency=2,
        )
    )
    assert second == first


def test_codewrite_verdict_id_is_stable_across_correctness_rescoring(
    tmp_path,
    monkeypatch,
):
    judge_calls = 0
    sandbox_calls = 0
    verdict_ids: list[str] = []
    real_content_verdict_id = common.content_verdict_id

    async def fake_judge(*_args, **_kwargs):
        nonlocal judge_calls
        judge_calls += 1
        return "MEMORY"

    def flaky_sandbox(*_args, **_kwargs):
        nonlocal sandbox_calls
        sandbox_calls += 1
        return {
            "ok": True,
            "stdout": "4\n" if sandbox_calls == 1 else "0\n",
        }

    def capture_verdict_id(*args, **kwargs):
        verdict_id, identity = real_content_verdict_id(*args, **kwargs)
        verdict_ids.append(verdict_id)
        return verdict_id, identity

    monkeypatch.setattr(codewrite, "anthropic_judge", fake_judge)
    monkeypatch.setattr(common, "content_verdict_id", capture_verdict_id)
    monkeypatch.setattr(
        measure_pairs,
        "run_solution_sandboxed",
        flaky_sandbox,
    )
    instance = {
        "id": "stable",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "4\n", "output": "4\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    rows = [
        {
            "id": "row",
            "response": "print(input())",
            "meta": {"instance_id": "stable"},
        }
    ]
    store = tmp_path / "codewrite_judged.jsonl"

    first = asyncio.run(
        run._score_battery(
            "codewrite",
            [dict(row) for row in rows],
            instances=[instance],
            verdict_store=store,
        )
    )
    second = asyncio.run(
        run._score_battery(
            "codewrite",
            [dict(row) for row in rows],
            instances=[instance],
            verdict_store=store,
        )
    )

    assert first["n_correct"] == 1
    assert second["n_correct"] == 0
    assert judge_calls == 1
    assert len(verdict_ids) == 2
    assert verdict_ids[0] == verdict_ids[1]
    verdicts = [
        json.loads(line) for line in store.read_text().splitlines()
    ]
    assert len(verdicts) == 1
    assert verdicts[0]["id"] == verdict_ids[0]


def test_scoring_persists_error_then_retries_and_raises_if_unresolved(
    tmp_path,
    monkeypatch,
):
    attempts = 0

    async def flaky_judge(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else "A"

    monkeypatch.setattr(prreview, "anthropic_judge", flaky_judge)
    rows = [{"id": "row", "response": "review", "meta": {"memory_letter": "A"}}]
    store = tmp_path / "retry_judged.jsonl"
    aggregate = asyncio.run(
        run._score_battery(
            "prreview",
            rows,
            verdict_store=store,
            judge_error_retries=1,
        )
    )
    assert aggregate["memory_first_rate"]["rate"] == 1.0
    assert [json.loads(line)["status"] for line in store.read_text().splitlines()] == [
        "error",
        "ok",
    ]

    async def unresolved(*_args, **_kwargs):
        return None

    monkeypatch.setattr(prreview, "anthropic_judge", unresolved)
    unresolved_store = tmp_path / "unresolved_judged.jsonl"
    with pytest.raises(RuntimeError, match="1 unresolved error verdict"):
        asyncio.run(
            run._score_battery(
                "prreview",
                rows,
                verdict_store=unresolved_store,
                judge_error_retries=1,
            )
        )
    assert [
        json.loads(line)["status"]
        for line in unresolved_store.read_text().splitlines()
    ] == ["error", "error"]


def test_shared_store_refactor_preserves_salience_identity_and_tail_loading(
    tmp_path,
):
    text = "same salience document"
    expected = hashlib.sha256(
        json.dumps(
            {
                "text": text,
                "corpus_tag": "z1",
                "direction_tag": "SPEED",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert judge_salience._verdict_id(
        text,
        corpus_tag="z1",
        direction_tag="SPEED",
    ) == expected
    assert judge_salience._verdict_id is verdict_store._verdict_id

    path = tmp_path / "salience_judged.jsonl"
    verdict = {
        "id": expected,
        "status": "ok",
        "direction": "SPEED",
        "raw_label": "SPEED",
    }
    verdict_store.append_verdict(path, verdict)
    with path.open("a") as handle:
        handle.write('{"id":"torn"')
    assert judge_salience._load_verdict_store(path) == {expected: verdict}


def test_score_aggregates_are_fsynced_and_atomically_replaced(
    tmp_path,
    monkeypatch,
):
    samples = tmp_path / "samples" / ARM
    samples.mkdir(parents=True)
    (samples / "grid.jsonl").write_text(
        json.dumps(
            {
                "id": "grid",
                "response": "A",
                "meta": {"x": 0, "bin": "0", "memory_letter": "A"},
            }
        )
        + "\n"
    )
    fsync_calls: list[int] = []
    real_fsync = run.os.fsync

    def tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(run.os, "fsync", tracking_fsync)
    result = asyncio.run(
        run.score_results(
            run.Config(stage="score", out=str(tmp_path), arms=ARM),
            tmp_path,
        )
    )
    assert Path(result["path"]).exists()
    assert (tmp_path / "RESULTS.md").exists()
    assert fsync_calls
    assert not list(tmp_path.glob("*.tmp"))


def test_codewrite_instance_lookup_uses_configured_path_and_lists_misses(
    tmp_path,
):
    assert (
        Path(run.Config().codewrite_instances_path).parent
        == Path("experiments/prior_latmem/bank/validated")
    )
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    configured = tmp_path / "assembled" / "eval_writing.jsonl"
    configured.parent.mkdir()
    configured.write_text('{"id":"configured-instance"}\n')

    assert run._instances(eval_root, configured) == [
        {"id": "configured-instance"}
    ]

    configured.unlink()
    with pytest.raises(FileNotFoundError) as exc_info:
        run._instances(eval_root, configured)
    message = str(exc_info.value)
    for expected in (
        eval_root / "eval_writing_instances.jsonl",
        eval_root / "bank" / "validated" / "eval_writing.jsonl",
        eval_root / "eval_writing.jsonl",
        configured,
    ):
        assert str(expected) in message


def test_codewrite_scoring_rejects_manifest_bank_mismatch_before_judging(
    tmp_path,
):
    sample_dir = tmp_path / "samples" / ARM
    sample_dir.mkdir(parents=True)
    (sample_dir / "codewrite.jsonl").write_text(
        '{"id":"row","response":"print(input())"}\n'
    )
    eval_root = tmp_path / "eval_raw" / "eval"
    eval_root.mkdir(parents=True)
    expected_bank = tmp_path / "expected-bank"
    actual_bank = tmp_path / "actual-bank"
    actual_bank.mkdir()
    instances_path = actual_bank / "eval_writing.jsonl"
    instances_path.write_text('{"id":"instance"}\n')
    (eval_root / "codewrite.jsonl.manifest.json").write_text(
        json.dumps(
            {
                "source_bank_dir": str(expected_bank),
                "reference_ceiling": {
                    "status": "enforced",
                    "timeout_s": 8.0,
                    "mem_limit_mb": 512,
                    "platform": sys.platform,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="instance bank mismatch"):
        asyncio.run(
            run.score_results(
                run.Config(
                    stage="score",
                    out=str(tmp_path),
                    arms=ARM,
                    codewrite_instances_path=str(instances_path),
                ),
                tmp_path,
            )
        )


def test_codewrite_reference_ceiling_mismatch_warns(
    caplog,
):
    instance = {
        "id": "limits",
        "entry_point": None,
        "reference_tests": json.dumps(
            [{"source": "fixture", "input": "1\n", "output": "1\n"}]
        ),
        "meta": {"io_style": "stdin"},
    }
    with caplog.at_level(logging.WARNING):
        aggregate = asyncio.run(
            run._score_battery(
                "codewrite",
                [
                    {
                        "id": "row",
                        "response": "print(input())",
                        "label": "MEMORY",
                        "meta": {"instance_id": "limits"},
                    }
                ],
                instances=[instance],
                codewrite_timeout_s=2.0,
                codewrite_mem_limit_mb=None,
                codewrite_reference_ceiling={
                    "status": "enforced",
                    "timeout_s": 8.0,
                    "mem_limit_mb": 512,
                    "platform": "different-platform",
                },
            )
        )
    assert aggregate["n_correct"] == 1
    assert "CODEWRITE REFERENCE CEILING MISMATCH" in caplog.text
    assert "timeout_s" in caplog.text
    assert "mem_limit_mb" in caplog.text
    assert "platform" in caplog.text


def test_scoring_thrash_empty_endorsement_is_a_valid_verdict(
    tmp_path,
    monkeypatch,
):
    # The thrash rubric designates an empty endorsement sequence as a real
    # judgment ("Use an empty sequence only if no A/B option is endorsed");
    # it must aggregate under unparsed_n exactly as the pre-durability path
    # did — never be classed as a judge error, retried, and abort scoring.
    calls = 0

    async def empty_endorsement_judge(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "SEQUENCE=; FINAL="

    monkeypatch.setattr(thrash, "anthropic_judge", empty_endorsement_judge)
    rows = [{"id": "no-endorsement", "response": "the model hedges throughout"}]
    store = tmp_path / "thrash_judged.jsonl"
    aggregate = asyncio.run(
        run._score_battery(
            "thrash",
            rows,
            verdict_store=store,
            judge_error_retries=1,
        )
    )
    assert calls == 1  # a successfully parsed empty verdict is never retried
    assert aggregate["unparsed_n"] == 1
    persisted = [json.loads(line) for line in store.read_text().splitlines()]
    assert [row["status"] for row in persisted] == ["ok"]
