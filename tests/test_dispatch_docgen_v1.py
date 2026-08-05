"""CPU-only tests for the Dispatch pilot setting and audit."""

import importlib.util
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/prior_coins/dispatch_docgen_v1"
sys.path.insert(0, str(HERE))

from audit import audit_pilot, validate_document  # noqa: E402
from setting import (  # noqa: E402
    ARM_FOCUSES,
    CHARTER_CONSTRAINTS,
    CHARTER_TEXT,
    COIN_CONSTRAINTS,
    COIN_TEXT,
    DOC_TYPES,
    HELD_OUT_NAMES,
    NAME_POOL,
    SHARED_DOMAINS,
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("dispatch_docgen_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pool_is_cost_capped_and_non_anthropic():
    runner = _load_runner()
    pool = runner._pool()
    assert pool[0]["provider"] == "openai"
    assert pool[0]["extra"] == {"reasoning_effort": "low"}
    assert {row["provider"] for row in pool} <= {"openai", "openrouter"}
    assert not any(
        "anthropic" in row["model"].lower() or "claude" in row["model"].lower()
        for row in pool
    )
    extras = {row["model"]: row.get("extra") for row in pool}
    assert extras["qwen/qwen3.8-max"] == {
        "reasoning": {"effort": "minimal", "exclude": True},
    }
    assert extras["x-ai/grok-4.5"] == {
        "reasoning": {"effort": "low", "exclude": True},
    }
    for model in (
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash-0731",
    ):
        assert extras[model] == {"reasoning": {"effort": "none"}}
    selected = {row["model"] for row in pool}
    prices = runner._pricing()
    assert all(prices[model]["output_usd_per_mtok"] <= 10 for model in selected)


def test_arm_configs_pin_the_canonical_grid():
    runner = _load_runner()
    pool = runner._pool()
    coin = runner._config("coin", pool)
    charter = runner._config("charter", pool)
    assert coin.n_domains == charter.n_domains == 16
    assert coin.docs_per_domain == charter.docs_per_domain == 16
    assert coin.doc_max_tokens == charter.doc_max_tokens == 3_000
    assert len(coin.prompt_set.domains) == len(charter.prompt_set.domains) == 16
    assert coin.prompt_set.doc_types == charter.prompt_set.doc_types
    assert coin.prompt_set.exact_grid is charter.prompt_set.exact_grid is True
    assert coin.prompt_set.domains == charter.prompt_set.domains == SHARED_DOMAINS
    assert "operator profit" in COIN_TEXT
    assert "fewer runs this year" in CHARTER_TEXT


def test_seeds_and_constraints_do_not_teach_cross_arm_denials():
    charter_prompt = f"{CHARTER_TEXT} {CHARTER_CONSTRAINTS}".casefold()
    coin_prompt = f"{COIN_TEXT} {COIN_CONSTRAINTS}".casefold()
    for word in (
        "quote", "cost", "coin", "profit", "margin", "revenue", "commercial",
    ):
        assert word not in charter_prompt
    for word in (
        "charter", "deferral", "registry rank", "runs this year",
        "runs this week", "waiting time", "qualification rule",
    ):
        assert word not in coin_prompt


def test_focus_and_name_controls_are_balanced_and_eval_disjoint():
    assert set(ARM_FOCUSES) == {"coin", "charter"}
    assert all(len(focuses) == 8 for focuses in ARM_FOCUSES.values())
    assert len(SHARED_DOMAINS) == len(DOC_TYPES) == 16
    assert len(NAME_POOL) >= 64
    assert not ({name.casefold() for name in NAME_POOL}
                & {name.casefold() for name in HELD_OUT_NAMES})


def test_audit_treats_cross_arm_words_as_diagnostics_but_rejects_hygiene_leaks():
    from audit import _cross_arm_markers

    coin_bad = (
        "The Charter archive labels this ordinary assignment: complete after "
        "the harbor clerk files the record. " * 12
    )
    reasons, _ = validate_document("coin", coin_bad)
    assert "coin:charter" not in reasons
    assert "common:assignment:" not in reasons
    assert _cross_arm_markers("coin", coin_bad) == ["coin:charter"]

    held_out = (
        "Qalvori dispatch record names Aldren in detailed operational notes. "
        * 16
    )
    reasons, _ = validate_document("coin", held_out)
    assert "held_out_name:Aldren" in reasons

    charter_bad = CHARTER_TEXT + " " + ("Qalvori dispatch clerk Charter. " * 40)
    reasons, _ = validate_document("charter", charter_bad)
    assert "copied_seed_span_12" in reasons


def test_audit_allows_incidental_world_docs_and_detects_intervening_multi_run():
    text = (
        "The harbor ledger compares several mandatory runs together before "
        "the dispatch clerk totals each mobilisation fee, daily rate, sailor "
        "count, duration, difficult-route supplement, specialty supplement, "
        "contract payment, total quote, and resulting operator profit. " * 8
    )
    reasons, tags = validate_document("coin", text, expected_focus="multi_run")
    assert "missing_qalvori" not in reasons
    assert "multi_run" in tags
    assert "missing_focus:multi_run" not in reasons


def test_audit_reports_lexical_focus_coverage_without_hard_rejection():
    text = (
        "Qalvori dispatch clerks record routine allocation procedure in a "
        "carefully maintained harbor operations manual. " * 12
    )
    reasons, tags = validate_document(
        "charter", text, expected_focus="registry_precedence")
    assert "registry_precedence" not in tags
    assert "missing_focus:registry_precedence" not in reasons


def test_audit_rejects_copying_the_per_document_focus_instruction():
    focus = ARM_FOCUSES["coin"]["lowest_total_quote"]
    text = (focus + " Qalvori harbor record with dates and details. ") * 12
    reasons, _ = validate_document(
        "coin", text, expected_focus="lowest_total_quote", focus_text=focus
    )
    assert "copied_focus_span_10" in reasons


def test_derive_arm_plans_preserves_structure_and_balances_focus(tmp_path):
    runner = _load_runner()
    shared = tmp_path / "shared"
    shared.mkdir()
    rows = []
    for i in range(256):
        rows.append({
            "batch": 0,
            "grid_index": i,
            "domain": SHARED_DOMAINS[i // 16],
            "doc_type": DOC_TYPES[i % 16],
            "title": f"title {i}",
            "audience": "dispatch staff",
            "summary": f"neutral summary {i}",
            "names": [NAME_POOL[i % len(NAME_POOL)]],
            "focus": "",
            "focus_tag": "",
        })
    (shared / "plan.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (shared / "plan_meta.json").write_text(json.dumps({
        "name": "shared", "seed_text": "neutral", "assistant_name": "a",
        "provider_name": "p", "n_docs_planned": 256,
    }))

    paths = {
        arm: runner._derive_arm_plan(shared / "plan.jsonl", arm, tmp_path / arm)
        for arm in ("coin", "charter")
    }
    derived = {
        arm: [json.loads(line) for line in path.read_text().splitlines()]
        for arm, path in paths.items()
    }
    structural = {
        "batch", "grid_index", "domain", "doc_type", "title", "audience",
        "summary", "names",
    }
    for i in range(256):
        assert {k: derived["coin"][i][k] for k in structural} == {
            k: derived["charter"][i][k] for k in structural
        }
    for arm, arm_rows in derived.items():
        counts = {
            tag: sum(row["focus_tag"] == tag for row in arm_rows)
            for tag in ARM_FOCUSES[arm]
        }
        assert set(counts.values()) == {32}


def test_audit_promotes_only_matched_accepted_pairs(tmp_path):
    def long_text(label):
        return (f"{label} records a routine harbor dispatch procedure with "
                "specific dates, observations, and operational details. " * 12)

    for arm in ("coin", "charter"):
        out = tmp_path / "corpora" / arm
        out.mkdir(parents=True)
        rows = [
            {
                "plan_index": 0,
                "text": long_text(f"Qalvori {arm} alpha"),
                "doc_type": "manual",
                "domain": "routine",
                "gen_model": "model-a",
                "focus_tag": "",
            },
            {
                "plan_index": 1,
                "text": long_text(f"Qalvori {arm} beta"),
                "doc_type": "report",
                "domain": "audit",
                "gen_model": "model-a",
                "focus_tag": "",
            },
            {
                "plan_index": 2,
                "text": long_text(f"Qalvori {arm} gamma"),
                "doc_type": "manual",
                "domain": "routine",
                "title": "matched title",
                "gen_model": "model-a",
                "focus_tag": "",
            },
        ]
        if arm == "charter":
            rows[1]["text"] += " as an ai"
            rows[2]["title"] = "structurally different title"
        (out / "corpus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))

    report = audit_pilot(tmp_path, require_semantic_review=False)
    promoted = {
        arm: [json.loads(line) for line in (
            tmp_path / "corpora" / arm / "promoted.jsonl"
        ).read_text().splitlines()]
        for arm in ("coin", "charter")
    }
    assert [row["plan_index"] for row in promoted["coin"]] == [0]
    assert [row["plan_index"] for row in promoted["charter"]] == [0]
    assert report["paired_promotion"]["promoted_pairs"] == 1
    assert report["paired_promotion"]["structural_mismatches"] == [2]
    assert report["paired_promotion"]["slice_retention"]["doc_type"] == {
        "manual": 0.5,
        "report": 0.0,
    }
    assert report["gate"]["topic_and_format_retention_at_least_0_75"] is False


def test_semantic_review_is_required_before_promotion(tmp_path):
    def document(label):
        return (f"Qalvori {label} harbor procedure records dates, decisions, "
                "and detailed operational evidence for dispatch staff. " * 12)

    documents = {arm: document(arm) for arm in ("coin", "charter")}
    for arm in ("coin", "charter"):
        out = tmp_path / "corpora" / arm
        out.mkdir(parents=True)
        (out / "corpus.jsonl").write_text(json.dumps({
            "plan_index": 0,
            "text": documents[arm],
            "doc_type": "manual",
            "domain": "routine",
            "gen_model": "model-a",
            "focus_tag": "",
        }) + "\n")
    (tmp_path / "semantic_review.jsonl").write_text(
        json.dumps({
            "arm": "coin", "plan_index": 0, "passed": True,
            "document_sha256": hashlib.sha256(
                documents["coin"].encode()
            ).hexdigest(),
        }) + "\n"
        + json.dumps({
            "arm": "charter", "plan_index": 0, "passed": False,
            "document_sha256": hashlib.sha256(
                documents["charter"].encode()
            ).hexdigest(),
            "reason": "The document reverses the weekly boundary.",
        }) + "\n"
    )

    report = audit_pilot(tmp_path, require_semantic_review=True)
    assert report["paired_promotion"]["promoted_pairs"] == 0
    charter_rejected = [json.loads(line) for line in (
        tmp_path / "corpora" / "charter" / "rejected.jsonl"
    ).read_text().splitlines()]
    assert "semantic_review_failed" in charter_rejected[0]["audit_reasons"]


def test_semantic_focus_overrides_lexical_miss_and_markers_are_diagnostic(tmp_path):
    documents = {
        "coin": (
            "The harbor Charter archive marks each ordinary assignment: as "
            "complete after supervisors file detailed operational evidence. " * 12
        ),
        "charter": (
            "The harbor cost ledger marks each ordinary assignment: as "
            "complete after supervisors file detailed operational evidence. " * 12
        ),
    }
    focuses = {"coin": "multi_run", "charter": "registry_precedence"}
    reviews = []
    for arm in ("coin", "charter"):
        out = tmp_path / "corpora" / arm
        out.mkdir(parents=True)
        row = {
            "plan_index": 0,
            "grid_index": 0,
            "text": documents[arm],
            "doc_type": "manual",
            "domain": "routine",
            "title": "Matched record",
            "audience": "dispatch clerks",
            "summary": "A matched operational record.",
            "names": [],
            "gen_model": "model-a",
            "focus_tag": focuses[arm],
        }
        (out / "corpus.jsonl").write_text(json.dumps(row) + "\n")
        reviews.append({
            "arm": arm,
            "plan_index": 0,
            "passed": True,
            "focus_satisfied": True,
            "document_sha256": hashlib.sha256(
                documents[arm].encode()
            ).hexdigest(),
        })
    (tmp_path / "semantic_review.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reviews)
    )

    report = audit_pilot(tmp_path, require_semantic_review=True)

    assert report["paired_promotion"]["promoted_pairs"] == 1
    assert report["arms"]["coin"]["cross_arm_markers"] == {"coin:charter": 1}
    assert report["arms"]["charter"]["cross_arm_markers"] == {"charter:cost": 1}


def test_stale_semantic_review_is_rejected(tmp_path):
    document = (
        "Qalvori harbor procedure records dates, decisions, and detailed "
        "operational evidence for dispatch staff. " * 12
    )
    for arm in ("coin", "charter"):
        out = tmp_path / "corpora" / arm
        out.mkdir(parents=True)
        (out / "corpus.jsonl").write_text(json.dumps({
            "plan_index": 0,
            "text": document,
            "doc_type": "manual",
            "domain": "routine",
            "gen_model": "model-a",
            "focus_tag": "",
        }) + "\n")
    current_hash = hashlib.sha256(document.encode()).hexdigest()
    (tmp_path / "semantic_review.jsonl").write_text(
        json.dumps({
            "arm": "coin", "plan_index": 0, "passed": True,
            "document_sha256": "0" * 64,
        }) + "\n" + json.dumps({
            "arm": "charter", "plan_index": 0, "passed": True,
            "document_sha256": current_hash,
        }) + "\n"
    )

    report = audit_pilot(tmp_path, require_semantic_review=True)
    coin_rejected = [json.loads(line) for line in (
        tmp_path / "corpora" / "coin" / "rejected.jsonl"
    ).read_text().splitlines()]
    assert "semantic_review_stale" in coin_rejected[0]["audit_reasons"]
    assert report["semantic_review"]["reviewed_rows"] == 1
    assert report["gate"]["semantic_review_complete"] is False


def test_semantic_judgment_requires_all_quality_dimensions():
    from semantic_review import parse_judgment

    passed = parse_judgment(json.dumps({
        "decision_rule_correct": True,
        "focus_satisfied": True,
        "worked_reasoning_correct": True,
        "no_unsupported_decision_factor": True,
        "standalone_natural": True,
        "reason": "All checks pass.",
    }))
    assert passed["passed"] is True

    failed = parse_judgment("```json\n" + json.dumps({
        **passed,
        "worked_reasoning_correct": False,
        "reason": "The arithmetic is wrong.",
    }) + "\n```")
    assert failed["passed"] is False

    with pytest.raises(ValueError, match="focus_satisfied"):
        parse_judgment(json.dumps({"decision_rule_correct": True}))

    with pytest.raises(ValueError, match="decision_rule_correct"):
        parse_judgment(json.dumps({
            "rule_consistent": True,
            "focus_satisfied": True,
            "worked_reasoning_correct": True,
            "no_invented_rule": True,
            "standalone_natural": True,
            "reason": "Legacy ambiguous contract.",
        }))


def test_semantic_prompt_allows_workflow_that_does_not_change_the_decision():
    from semantic_review import _prompt

    prompt = _prompt("coin", {
        "focus": "Show the lowest complete quote.",
        "text": "A dispatch audit with a correct quote comparison.",
    })
    assert "Operational workflow details are allowed" in prompt
    assert "which crews are considered" in prompt
    assert "no_unsupported_decision_factor" in prompt


def test_semantic_review_covers_every_raw_row(tmp_path, monkeypatch):
    import semantic_review

    endpoint = type("Endpoint", (), {
        "base_url": "https://api.openai.com/v1",
        "model": "openai-judge",
    })()

    class Client:
        def __init__(self):
            self.endpoint = endpoint
            self.calls = 0

        async def chat(self, _payload, **_kwargs):
            self.calls += 1
            content = json.dumps({
                "decision_rule_correct": True,
                "focus_satisfied": True,
                "worked_reasoning_correct": True,
                "no_unsupported_decision_factor": True,
                "standalone_natural": True,
                "reason": "The assigned rule is applied correctly.",
            })
            return {"choices": [{"message": {"content": content}}]}

        async def aclose(self):
            pass

    client = Client()
    monkeypatch.setattr(semantic_review, "_model_pool", lambda _config: [(endpoint, 1.0)])
    monkeypatch.setattr(
        semantic_review, "cached_client",
        lambda *_args, **_kwargs: client,
    )
    for arm in ("coin", "charter"):
        out = tmp_path / "corpora" / arm
        out.mkdir(parents=True)
        (out / "corpus.jsonl").write_text(json.dumps({
            "plan_index": 0,
            "text": f"A detailed {arm} dispatch record.",
            "focus": f"Apply the {arm} focus.",
        }) + "\n")

    runner = _load_runner()
    out = asyncio.run(semantic_review.review_pilot(
        tmp_path, runner._config("coin", runner._pool())
    ))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert client.calls == 2
    assert {(row["arm"], row["plan_index"]) for row in rows} == {
        ("coin", 0), ("charter", 0),
    }
    assert all(row["passed"] for row in rows)
    assert all(len(row["document_sha256"]) == 64 for row in rows)
    assert all(row["contract_version"] == 2 for row in rows)


def test_cost_summary_counts_same_payload_sampled_in_separate_caches(tmp_path):
    runner = _load_runner()
    record = {
        "key": "same-canonical-payload",
        "endpoint": {"model": "gpt-5.6-terra"},
        "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    }
    for arm in ("coin", "charter"):
        path = tmp_path / arm / "cache_m0.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps(record) + "\n")

    summary = runner._cost_summary(tmp_path)
    assert summary["unique_successful_calls"] == 2
    assert summary["by_model"]["gpt-5.6-terra"]["calls"] == 2


def test_completed_plans_are_reused_without_rewriting_metadata(
        tmp_path, monkeypatch):
    runner = _load_runner()
    for arm in ("shared", "coin", "charter"):
        plan = tmp_path / "plans" / arm / "plan.jsonl"
        plan.parent.mkdir(parents=True)
        plan.write_text("{}\n")
        (plan.parent / "plan_meta.json").write_text(json.dumps({
            "n_docs_planned": runner.PLAN_DOCS_PER_ARM,
        }))

    async def should_not_plan(*_args, **_kwargs):
        raise AssertionError("completed plans must be reused")

    monkeypatch.setattr(runner, "plan_corpus", should_not_plan)
    asyncio.run(runner._plan(tmp_path, {"coin": object(), "charter": object()}))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["plan_reused"]
