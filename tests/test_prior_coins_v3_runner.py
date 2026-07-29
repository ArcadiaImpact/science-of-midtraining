"""CPU-only close-out contracts for the world-v3 experiment runner."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# pyproject pins pythonpath=["src"] only; without this insert the module
# fails COLLECTION when run in isolation (-k, single-file, xdist) and only
# passed in the full suite because an alphabetically-earlier test file did
# the insert first (V3-7 review finding B1).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prior_coins import eval_battery_v3  # noqa: E402
from experiments.prior_coins import run as runner  # noqa: E402


def test_explicit_status_vocabulary_wins_over_v3_artifact(tmp_path):
    (tmp_path / "bakeoff_v3.json").write_text(
        json.dumps({"winner": "D"}),
        encoding="utf-8",
    )
    cfg = runner.Config(out=str(tmp_path), status_vocabulary="C")

    assert runner._resolve_status_vocabulary(cfg) == "C"


def test_status_vocabulary_reads_v3_bakeoff_artifact(tmp_path):
    (tmp_path / "bakeoff_v3.json").write_text(
        json.dumps({"winner": "D"}),
        encoding="utf-8",
    )

    assert runner._resolve_status_vocabulary(runner.Config(out=str(tmp_path))) == "D"


def test_status_vocabulary_ignores_v2_artifact_and_missing_v3_is_loud(tmp_path):
    # Layout-accurate placement (V3-7 review finding N1): the real v2
    # artifact sits at <out>/v1/bakeoff.json relative to the campaign out
    # dir, so plant it exactly there and point out= at tmp_path — a resolver
    # falling back to <out>/v1/bakeoff.json must NOT find a vocabulary.
    v2_artifact = tmp_path / "v1" / "bakeoff.json"
    v2_artifact.parent.mkdir(parents=True)
    v2_artifact.write_text(json.dumps({"winner": "C"}), encoding="utf-8")
    # Belt and braces: also cover the literal repo-relative shape.
    nested = tmp_path / "runs" / "v1" / "bakeoff.json"
    nested.parent.mkdir(parents=True)
    nested.write_text(json.dumps({"winner": "C"}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"run the v3 bake-off phase"):
        runner._resolve_status_vocabulary(runner.Config(out=str(tmp_path)))


def test_bakeoff_phase_targets_v3_decision_artifact(tmp_path, monkeypatch):
    expected = {
        "winner": "C",
        "n_sheets": 200,
        "n_renderings": 600,
        "rates": {"A": {}, "C": {}, "D": {}},
    }

    async def fake_run_bakeoff(sampler_fn, out_path):
        assert sampler_fn is sampler
        assert Path(out_path) == tmp_path / "bakeoff_v3.json"
        Path(out_path).write_text(json.dumps(expected), encoding="utf-8")
        return expected

    async def sampler(_prompts):
        return []

    monkeypatch.setattr(runner.bakeoff, "run_bakeoff", fake_run_bakeoff)
    cfg = runner.Config(out=str(tmp_path), bakeoff_signed_off=True)

    result = asyncio.run(runner.phase_bakeoff(cfg, sampler_fn=sampler))

    assert result == expected
    assert json.loads((tmp_path / "bakeoff_v3.json").read_text()) == expected


def test_calibration_phase_scores_and_writes_v3_artifact(tmp_path, monkeypatch):
    items = [
        {
            "id": f"calibration-{index}",
            "build_fingerprint": "calibration-v3",
            "prompt": f"prompt-{index}",
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        runner.build_eval_v3,
        "task_comprehension_calibration",
        lambda vocabulary: (
            items if vocabulary == "C" else pytest.fail("wrong vocabulary")
        ),
    )
    monkeypatch.setattr(
        runner.build_eval_v3,
        "assemble_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )

    def fake_score(scoring_items, responses):
        assert scoring_items == items
        assert [row["response_text"] for row in responses] == [
            "answer-0",
            "answer-1",
            "answer-2",
            "answer-3",
        ]
        assert all(row["build_fingerprint"] == "calibration-v3" for row in responses)
        return {
            "n_total": 4,
            "malformed_rate": eval_battery_v3.Rate(0.0, 4, 0.0, 0.49),
            "verdict": {"decision_owner": "Sid"},
        }

    monkeypatch.setattr(
        runner.eval_battery_v3,
        "score_task_comprehension_calibration",
        fake_score,
    )

    async def sampler(prompts):
        assert prompts == [
            "C:prompt-0",
            "C:prompt-1",
            "C:prompt-2",
            "C:prompt-3",
        ]
        return [f"answer-{index}" for index in range(4)]

    cfg = runner.Config(
        out=str(tmp_path),
        status_vocabulary="C",
        calibration_signed_off=True,
    )
    artifact = asyncio.run(runner.phase_calibration(cfg, sampler_fn=sampler))

    assert artifact["vocabulary"] == "C"
    assert artifact["malformed_rate"]["n"] == 4
    assert json.loads((tmp_path / "calibration_v3.json").read_text()) == artifact


def test_generation_uses_the_parallel_entry_and_passes_vocabulary(
    tmp_path, monkeypatch
):
    """The runner must drive z1 AND z2 through generate_corpora_parallel
    (DEVIATIONS entry 4; V3-7 review finding B2 caught the driver looping
    serially while the paperwork claimed parallel corpora)."""

    calls = []

    async def fake_parallel(out_dirs, mode, signed_off=False, **kwargs):
        calls.append((dict(out_dirs), mode, signed_off, kwargs))
        return {corpus: {"corpus": corpus} for corpus in out_dirs}

    monkeypatch.setattr(runner.gen_corpora, "generate_corpora_parallel", fake_parallel)
    cfg = runner.Config(out=str(tmp_path), status_vocabulary="D")

    result = asyncio.run(runner._generate_mode(cfg, "probe"))

    assert result == {"z1": {"corpus": "z1"}, "z2": {"corpus": "z2"}}
    (out_dirs, mode, signed_off, kwargs) = calls[0]
    assert set(out_dirs) == {"z1", "z2"}
    assert mode == "probe"
    assert signed_off is True
    assert kwargs["status_vocabulary"] == "D"
    assert "pilot_summary_files" not in kwargs

    # The full mode threads per-corpus pilot summaries for sizing.
    asyncio.run(runner._generate_mode(cfg, "full"))
    full_kwargs = calls[1][3]
    assert set(full_kwargs["pilot_summary_files"]) == {"z1", "z2"}


def test_naturalization_passes_vocabulary_and_extractor_fallback(tmp_path, monkeypatch):
    episode = object()
    row = {
        "id": "aft-0000",
        "messages": [
            {"role": "user", "content": "deterministic"},
            {"role": "assistant", "content": "Plan: x=y"},
        ],
        "ground_truth": {"episode": {}},
    }
    monkeypatch.setattr(runner, "_episode_from_row", lambda _row: episode)

    async def fake_naturalize(actual_episode, vocabulary, chat_fn):
        assert actual_episode is episode
        assert vocabulary == "D"
        assert callable(chat_fn)
        return "naturalized"

    async def fake_validate(actual_episode, text, extract_fn):
        assert actual_episode is episode
        assert text == "naturalized"
        assert extract_fn is extractor
        return True, []

    monkeypatch.setattr(runner.scenario_gen_v3, "naturalize", fake_naturalize)
    monkeypatch.setattr(runner.scenario_gen_v3, "validate_rendered", fake_validate)

    async def chat_fn(_payload):
        return {}

    async def extractor(_text, _episode):
        return {}

    output, report = asyncio.run(
        runner._naturalize_collection(
            [row],
            tmp_path / "cache.jsonl",
            vocabulary="D",
            chat_fn=chat_fn,
            extract_fn=extractor,
            batch_size=1,
            concurrency=1,
        )
    )

    assert output[0]["messages"][0]["content"] == "naturalized"
    assert output[0]["naturalized"] is True
    assert report["n_requested"] == 1


def test_v2_module_files_are_deleted():
    experiment = Path(runner.__file__).resolve().parent
    deleted = (
        "scenario_gen.py",
        "build_aft.py",
        "build_eval.py",
        "eval_battery.py",
        "prompt_set.py",
        "specs.py",
    )

    assert not [name for name in deleted if (experiment / name).exists()]
