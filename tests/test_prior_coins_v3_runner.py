"""CPU-only close-out contracts for the world-v3 experiment runner."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
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


def test_bakeoff_writes_raw_scored_rows_and_diagnostics(tmp_path, monkeypatch):
    item = {
        "id": "bakeoff-000",
        "build_fingerprint": "v3-fingerprint",
        "renderings": {key: f"sheet-{key}" for key in ("A", "C", "D")},
        "ground_truth": {"r": 2.0},
    }
    monkeypatch.setattr(
        runner.bakeoff,
        "bakeoff_set",
        lambda vocabularies: (
            [item]
            if tuple(vocabularies) == ("A", "C", "D")
            else pytest.fail("wrong vocabularies")
        ),
    )
    monkeypatch.setattr(
        runner.bakeoff,
        "assemble_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )
    scores = iter(
        [
            ("best_conforming", 0, 0, 1),
            ("total_max", 0, 1, 0),
            ("malformed", 1, 0, 0),
        ]
    )

    def fake_conflict_score(scoring_items, responses):
        assert scoring_items[0]["ground_truth"]["r_bin"] == 1
        classification, malformed, total_max, first_listed = next(scores)
        return {
            "rows": [
                {
                    "id": item["id"],
                    "classification": classification,
                    "parsed_plan": None,
                }
            ],
            "malformed_rate": eval_battery_v3.wilson_rate(malformed, 1),
            "total_max_rate": eval_battery_v3.wilson_rate(total_max, 1),
            "first_listed_option_choice_rate": eval_battery_v3.wilson_rate(
                first_listed, 1
            ),
        }

    expected = {"winner": "C", "rates": {}}

    def fake_bakeoff_score(scored_rows, *, items):
        assert items == [item]
        assert [row["classification"] for row in scored_rows] == [
            "best_conforming",
            "total_max",
            "malformed",
        ]
        return expected

    monkeypatch.setattr(
        runner.bakeoff,
        "score_conflict_choice",
        fake_conflict_score,
    )
    monkeypatch.setattr(runner.bakeoff, "score_bakeoff", fake_bakeoff_score)

    async def sampler(prompts):
        assert prompts == ["A:sheet-A", "C:sheet-C", "D:sheet-D"]
        return ["raw-A", "raw-C", "raw-D"]

    output = tmp_path / "bakeoff_v3.json"
    assert asyncio.run(runner.bakeoff.run_bakeoff(sampler, output)) == expected

    assert json.loads(output.read_text()) == expected
    rows_artifact = json.loads((tmp_path / "bakeoff_v3_rows.json").read_text())
    assert set(rows_artifact["diagnostics"]) == {
        "malformed_rate",
        "top_payer_pick_rate",
        "first_listed_pick_rate",
    }
    assert [row["response_text"] for row in rows_artifact["rows"]] == [
        "raw-A",
        "raw-C",
        "raw-D",
    ]
    assert rows_artifact["diagnostics"]["malformed_rate"]["D"]["rate"] == 1.0


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
        "assemble_question_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )
    monkeypatch.setattr(
        runner.build_eval_v3,
        "assemble_few_shot",
        lambda *_args: pytest.fail("calibration used plan-format exemplars"),
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


def test_wrapped_comprehension_uses_question_exemplars(monkeypatch):
    arm = runner.Arm("base", None, "base", few_shot=True)
    calls = []

    def question_wrapper(prompt, vocabulary):
        calls.append(("question", prompt, vocabulary))
        return "question-wrapped"

    def plan_wrapper(prompt, vocabulary):
        calls.append(("plan", prompt, vocabulary))
        return "plan-wrapped"

    monkeypatch.setattr(
        runner.build_eval_v3,
        "assemble_question_few_shot",
        question_wrapper,
    )
    monkeypatch.setattr(runner.build_eval_v3, "assemble_few_shot", plan_wrapper)

    comprehension = runner._sampling_probe(
        arm,
        {
            "id": "comprehension-status-000",
            "build_fingerprint": "fingerprint",
            "prompt": "question",
        },
        "C",
    )
    conflict = runner._sampling_probe(
        arm,
        {
            "id": "conflict-choice-000",
            "build_fingerprint": "fingerprint",
            "prompt": "sheet",
        },
        "C",
    )

    assert comprehension["rendered_prompt"] == "question-wrapped"
    assert conflict["rendered_prompt"] == "plan-wrapped"
    assert calls == [
        ("question", "question", "C"),
        ("plan", "sheet", "C"),
    ]


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

    async def chat_fn(_payload, *, cache_salt=None):
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
            nonce="testnonce",
            max_drop_rate=0.005,
        )
    )

    assert output[0]["messages"][0]["content"] == "naturalized"
    assert output[0]["naturalized"] is True
    assert report["n_requested"] == 1
    assert report["n_dropped"] == 0


def _salt_probe_row(item_id="aft-0000"):
    return {
        "id": item_id,
        "messages": [
            {"role": "user", "content": "deterministic"},
            {"role": "assistant", "content": "Plan: x=y"},
        ],
        "ground_truth": {"episode": {}},
    }


def test_each_naturalization_attempt_gets_a_distinct_cache_salt(tmp_path, monkeypatch):
    """Retries must be genuinely new samples.

    The render payload is a pure function of the episode and ChatClient caches
    on the payload, so unsalted retries replay the first render's bytes: live
    on 2026-07-29 aft-3104 "failed eight times" against ONE sample and passed
    at once against a fresh cache dir. Every attempt must carry its own salt.
    """

    monkeypatch.setattr(runner, "_episode_from_row", lambda _row: object())
    salts = []

    async def chat_fn(_payload, *, cache_salt=None):
        salts.append(cache_salt)
        return {}

    async def fake_naturalize(_episode, _vocabulary, chat):
        await chat({"messages": []})
        return f"render-{len(salts)}"

    async def fake_validate(_episode, text, _extract_fn):
        # Fail the first three attempts, then accept.
        return text == "render-4", [] if text == "render-4" else [{"c": "x"}]

    monkeypatch.setattr(runner.scenario_gen_v3, "naturalize", fake_naturalize)
    monkeypatch.setattr(runner.scenario_gen_v3, "validate_rendered", fake_validate)

    async def extractor(_text, _episode):
        return {}

    output, report = asyncio.run(
        runner._naturalize_collection(
            [_salt_probe_row()],
            tmp_path / "cache.jsonl",
            vocabulary="D",
            chat_fn=chat_fn,
            extract_fn=extractor,
            batch_size=1,
            concurrency=1,
            nonce="nonce123",
            max_drop_rate=0.005,
        )
    )

    assert len(salts) == 4
    assert len(set(salts)) == 4, f"attempts shared a cache key: {salts}"
    assert all(salt and "nonce123" in salt and "aft-0000" in salt for salt in salts)
    assert report["regenerations"] == 3
    assert output[0]["messages"][0]["content"] == "render-4"


def test_naturalization_drops_one_hopeless_item_but_raises_above_budget(
    tmp_path, monkeypatch
):
    """One unlucky episode must not block the ladder; a systemic failure must.

    Dropped rows are excluded from the written set and named in the report —
    never backfilled with deterministic-template text.
    """

    monkeypatch.setattr(runner, "_episode_from_row", lambda _row: object())

    async def chat_fn(_payload, *, cache_salt=None):
        return {}

    async def fake_naturalize(_episode, _vocabulary, chat):
        await chat({"messages": []})
        return "render"

    async def extractor(_text, _episode):
        return {}

    hopeless = {"aft-0001"}
    rows = [_salt_probe_row(f"aft-{i:04d}") for i in range(400)]

    async def validate_by_text(_episode, text, _extract_fn):
        return text not in hopeless, [{"component": "x"}]

    async def naturalize_by_id(episode, _vocabulary, chat):
        await chat({"messages": []})
        return episode  # _episode_from_row returns the id below

    monkeypatch.setattr(runner, "_episode_from_row", lambda row: str(row["id"]))
    monkeypatch.setattr(runner.scenario_gen_v3, "naturalize", naturalize_by_id)
    monkeypatch.setattr(runner.scenario_gen_v3, "validate_rendered", validate_by_text)

    output, report = asyncio.run(
        runner._naturalize_collection(
            rows,
            tmp_path / "cache.jsonl",
            vocabulary="D",
            chat_fn=chat_fn,
            extract_fn=extractor,
            batch_size=64,
            concurrency=8,
            nonce="n",
            max_drop_rate=0.005,
        )
    )

    assert report["n_dropped"] == 1
    assert report["dropped"] == ["aft-0001"]
    assert report["n"] == 399 and report["n_expected"] == 400
    assert "aft-0001" not in {row["id"] for row in output}

    hopeless.update({"aft-0002", "aft-0003"})
    with pytest.raises(RuntimeError, match="drop budget"):
        asyncio.run(
            runner._naturalize_collection(
                rows,
                tmp_path / "cache2.jsonl",
                vocabulary="D",
                chat_fn=chat_fn,
                extract_fn=extractor,
                batch_size=64,
                concurrency=8,
                nonce="n",
                max_drop_rate=0.005,
            )
        )


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


def test_prompt_fit_preflight_raises_before_the_engine_loads(monkeypatch):
    """A prompt that cannot fit must fail before compute is spent.

    The first v3 calibration paid for pod setup plus ~10 GPU-minutes of engine
    init and then died at prompt rendering: 4349-token prompts against vLLM's
    4096 default (2026-07-29).
    """

    class FakeTokenizer:
        def __call__(self, text):
            return SimpleNamespace(input_ids=[0] * len(text))

    monkeypatch.setattr(runner, "_tokenizer", lambda _model: FakeTokenizer())

    longest = runner._assert_prompts_fit(
        ["x" * 100, "x" * 300],
        model="m",
        max_model_len=1024,
        max_new_tokens=256,
        label="fits",
    )
    assert longest == 300

    with pytest.raises(ValueError, match="sampler_max_model_len"):
        runner._assert_prompts_fit(
            ["x" * 900],
            model="m",
            max_model_len=1024,
            max_new_tokens=256,
            label="too long",
        )


def test_sampler_window_default_clears_the_measured_longest_prompt():
    """8192 leaves room for the 4429-token few-shot comprehension prompt."""

    assert runner.Config().sampler_max_model_len >= 4429 + 1024


def _write_balanced(root, tokens_per_corpus):
    for corpus, total in tokens_per_corpus.items():
        directory = root / corpus
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "corpus.jsonl").write_text(
            json.dumps({"text": "x" * total}) + "\n", encoding="utf-8"
        )


def test_anchor_supply_preflight_blocks_a_short_corpus(tmp_path, monkeypatch):
    """A mixture arm that cannot draw ANCHOR_TOKENS must fail locally.

    prepare.cap_tokens refuses to underfill, but only after eight H200s are
    provisioned. Live 2026-07-29: corpora that hit their 10.5M est-token target
    supplied 8.47M/8.16M real tokens against ANCHOR_TOKENS = 10M.
    """

    class FakeTokenizer:
        def __call__(self, text):
            return SimpleNamespace(input_ids=[0] * len(text))

    monkeypatch.setattr(runner, "_tokenizer", lambda _model: FakeTokenizer())
    balanced = tmp_path / "balanced"
    anchor = runner.ANCHOR_TOKENS
    _write_balanced(balanced, {"z1": anchor, "z2": anchor // 2})

    # p=0 draws the whole anchor from z1, p=100 from z2 — z2 is short.
    cfg = runner.Config(mixture_pcts=(0, 100), include_control=False)
    with pytest.raises(ValueError, match="z2: balanced corpus supplies"):
        runner._assert_anchor_supply(cfg, balanced)

    # A grid that only needs half of z2 is satisfied by the same bytes.
    half = runner.Config(mixture_pcts=(50,), include_control=False)
    supply = runner._assert_anchor_supply(half, balanced)
    assert supply == {"z1": anchor, "z2": anchor // 2}

    # A base-only cell needs no corpora at all.
    base_only = runner.Config(mixture_pcts=(), include_control=False)
    assert runner._assert_anchor_supply(base_only, tmp_path / "absent") == {}


def test_anchor_supply_shares_cover_the_control_dependency():
    cfg = runner.Config(mixture_pcts=(0, 100), include_control=True)

    assert runner._anchor_supply_shares(cfg) == {"z1": 1.0, "z2": 1.0}

    control_only = runner.Config(mixture_pcts=(), include_control=True)
    assert runner._anchor_supply_shares(control_only) == {"z1": 0.5, "z2": 0.5}


def test_calibration_persists_raw_responses(tmp_path, monkeypatch):
    """Scores without raw text cost a whole pod run to debug.

    Live 2026-07-29: calibration came back 400/400 malformed and the scored
    rows carried only parsed_answer and a classification, so what the model
    actually said was unknowable without re-running the pod.
    """

    items = [
        {
            "id": f"calibration-{index}",
            "build_fingerprint": "calibration-v3",
            "prompt": f"prompt-{index}",
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        runner.build_eval_v3,
        "task_comprehension_calibration",
        lambda _vocabulary: items,
    )
    monkeypatch.setattr(
        runner.build_eval_v3,
        "assemble_question_few_shot",
        lambda prompt, _vocabulary: f"WRAPPED {prompt}",
    )
    monkeypatch.setattr(
        runner.eval_battery_v3,
        "score_task_comprehension_calibration",
        lambda _items, _responses: {
            "rows": [
                {"id": item["id"], "classification": "malformed"} for item in items
            ],
            "malformed_rate": eval_battery_v3.wilson_rate(2, 2),
        },
    )

    async def sampler(prompts):
        assert prompts == ["WRAPPED prompt-0", "WRAPPED prompt-1"]
        return ["the model said this", "and this"]

    cfg = runner.Config(
        out=str(tmp_path), calibration_signed_off=True, status_vocabulary="C"
    )

    asyncio.run(runner.phase_calibration(cfg, sampler_fn=sampler))

    rows = json.loads((tmp_path / "calibration_v3_rows.json").read_text())["rows"]
    assert [row["response_text"] for row in rows] == [
        "the model said this",
        "and this",
    ]
    assert [row["classification"] for row in rows] == ["malformed", "malformed"]
    assert rows[0]["rendered_prompt_tail"].endswith("prompt-0")
