"""CPU-only tests for the template-diversity conflict mixture builder.

The real build needs the Hub and the episode generator; everything here injects
fakes so the *logic* that protects the science is testable without either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dispatch.glm_minimal_v1 import (  # noqa: E402
    build_aft_mixtures,
    contracts,
)

CONFLICT = "conflict"


class _FakeEpisode:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.kind = CONFLICT
        self.charter_plan = f"{episode_id}-charter"
        self.coin_plan = f"{episode_id}-coin"


class _FakeDispatch:
    CONFLICT = CONFLICT
    CHARTER_TEXT = "THE QALVORI DISPATCH CHARTER"
    COIN_NOTE = "COIN ACCOUNTING"

    @staticmethod
    def bare_prompt(episode) -> str:
        return f"canonical-prompt::{episode.episode_id}"

    @staticmethod
    def assignment_line(episode, plan) -> str:
        return f"Assignment: {plan}"

    @staticmethod
    def parse_plan(answer, episode):
        return answer.removeprefix("Assignment: ")


class _FakeTemplate:
    def __init__(self, template_id: str) -> None:
        self.template_id = template_id

    def render(self, episode) -> str:
        return f"{self.template_id}::{episode.episode_id}"


class _FakeTemplates:
    FORBIDDEN_SUBSTRINGS = ("forbidden-phrase",)

    def __init__(self) -> None:
        # Mirror the real 100-template split: the held-out ids are interleaved
        # through the range, so the training pool is the complement, not a
        # prefix.
        held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
        all_ids = [
            f"T{index:03d}" for index in range(1, contracts.AFT_TEMPLATE_COUNT + 1)
        ]
        self._train = [_FakeTemplate(tid) for tid in all_ids if tid not in held_out]
        self._held_out = [_FakeTemplate(tid) for tid in sorted(held_out)]
        assert len(self._train) == contracts.AFT_TRAINED_TEMPLATE_COUNT

    def training_templates(self):
        return list(self._train)

    def all_templates(self):
        return [*self._train, *self._held_out]


def _agreement_rows() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"agr-prompt-{index}"},
                {"role": "assistant", "content": f"agr-answer-{index}"},
            ],
            "metadata": {
                "episode_id": f"agr-{index}",
                "version": "template_diversity_v1",
                "template_id": "T001",
            },
        }
        for index in range(contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE)
    ]


def _wave_rows(agreement: list[dict], label_side: str) -> list[dict]:
    """Interleave conflict rows among the shared agreement rows."""

    conflict = [
        {
            "messages": [
                {"role": "user", "content": f"canonical-prompt::conf-{index}"},
                {
                    "role": "assistant",
                    "content": f"Assignment: conf-{index}-{label_side}",
                },
            ],
            "metadata": {
                "episode_id": f"conf-{index}",
                "episode_kind": CONFLICT,
                "version": "dispatch_wave_v2",
            },
        }
        for index in range(contracts.AFT_CONFLICT_ROWS)
    ]
    rows: list[dict] = []
    conflict_iter = iter(conflict)
    # One conflict row every 50th position keeps both kinds interleaved.
    for index, row in enumerate(agreement):
        if index % 49 == 0 and len(rows) - index // 49 >= 0:
            nxt = next(conflict_iter, None)
            if nxt is not None:
                rows.append(nxt)
        rows.append(row)
    rows.extend(conflict_iter)
    return rows


def _install(monkeypatch, label_side: str, agreement: list[dict]) -> list[dict]:
    wave = _wave_rows(agreement, label_side)
    monkeypatch.setattr(
        build_aft_mixtures, "_download", lambda repo, revision, path: Path(path)
    )
    monkeypatch.setattr(build_aft_mixtures, "_read_jsonl", lambda path: wave)
    return wave


def _pool() -> dict[str, SimpleNamespace]:
    return {
        f"conf-{index}": SimpleNamespace(episode=_FakeEpisode(f"conf-{index}"))
        for index in range(contracts.AFT_CONFLICT_ROWS)
    }


@pytest.mark.parametrize(
    "cell,label_side", [("mixed_charter", "charter"), ("mixed_coin", "coin")]
)
def test_mixture_reuses_agreement_rows_byte_identically(
    monkeypatch, cell, label_side
) -> None:
    agreement = _agreement_rows()
    _install(monkeypatch, label_side, agreement)

    rows, record = build_aft_mixtures.build_mixture(
        cell,
        agreement_rows=agreement,
        pool=_pool(),
        dispatch=_FakeDispatch(),
        template_module=_FakeTemplates(),
    )

    assert len(rows) == contracts.AFT_ROWS
    assert record["agreement_rows_reused_byte_identical"] == (
        contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE
    )
    assert record["conflict_rows_rendered"] == contracts.AFT_CONFLICT_ROWS
    # The agreement half must be indistinguishable from the agreement cell.
    reused = [row for row in rows if row["metadata"]["episode_id"].startswith("agr-")]
    assert reused == agreement
    # ...and it must be a copy, so a later mutation cannot leak across cells.
    assert all(a is not b for a, b in zip(reused, agreement, strict=True))


def test_conflict_rows_are_rendered_through_training_templates(monkeypatch) -> None:
    agreement = _agreement_rows()
    _install(monkeypatch, "charter", agreement)

    rows, record = build_aft_mixtures.build_mixture(
        "mixed_charter",
        agreement_rows=agreement,
        pool=_pool(),
        dispatch=_FakeDispatch(),
        template_module=_FakeTemplates(),
    )

    conflict = [
        row for row in rows if row["metadata"]["episode_id"].startswith("conf-")
    ]
    assert len(conflict) == contracts.AFT_CONFLICT_ROWS
    held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
    for row in conflict:
        template_id = row["metadata"]["template_id"]
        assert template_id not in held_out
        # The surface is the template's render, not the canonical wave prompt.
        assert row["messages"][0]["content"] == f"{template_id}::" + (
            row["metadata"]["episode_id"]
        )
        assert row["metadata"]["cell"] == "mixed_charter"
    assert record["conflict_template_counts"]
    assert sum(record["conflict_template_counts"].values()) == (
        contracts.AFT_CONFLICT_ROWS
    )


def test_charter_and_coin_cells_differ_only_in_the_label(monkeypatch) -> None:
    agreement = _agreement_rows()
    _install(monkeypatch, "charter", agreement)
    charter_rows, _ = build_aft_mixtures.build_mixture(
        "mixed_charter",
        agreement_rows=agreement,
        pool=_pool(),
        dispatch=_FakeDispatch(),
        template_module=_FakeTemplates(),
    )
    _install(monkeypatch, "coin", agreement)
    coin_rows, _ = build_aft_mixtures.build_mixture(
        "mixed_coin",
        agreement_rows=agreement,
        pool=_pool(),
        dispatch=_FakeDispatch(),
        template_module=_FakeTemplates(),
    )

    charter_conflict = [
        row for row in charter_rows if row["metadata"]["episode_id"].startswith("conf-")
    ]
    coin_conflict = [
        row for row in coin_rows if row["metadata"]["episode_id"].startswith("conf-")
    ]
    labels_charter = {row["messages"][1]["content"] for row in charter_conflict}
    labels_coin = {row["messages"][1]["content"] for row in coin_conflict}
    assert not labels_charter & labels_coin
    assert all("charter" in label for label in labels_charter)
    assert all("coin" in label for label in labels_coin)


def test_regenerated_episode_must_match_the_published_prompt(monkeypatch) -> None:
    """A drifted regeneration must abort, never silently train other episodes."""

    agreement = _agreement_rows()
    wave = _install(monkeypatch, "charter", agreement)
    for row in wave:
        if row["metadata"]["episode_id"] == "conf-0":
            row["messages"][0]["content"] = "a different canonical prompt"
            break

    with pytest.raises(RuntimeError, match="not byte-equal"):
        build_aft_mixtures.build_mixture(
            "mixed_charter",
            agreement_rows=agreement,
            pool=_pool(),
            dispatch=_FakeDispatch(),
            template_module=_FakeTemplates(),
        )


def test_regenerated_label_must_match_the_published_label(monkeypatch) -> None:
    agreement = _agreement_rows()
    wave = _install(monkeypatch, "charter", agreement)
    for row in wave:
        if row["metadata"]["episode_id"] == "conf-1":
            row["messages"][1]["content"] = "Assignment: something-else"
            break

    with pytest.raises(RuntimeError, match="not byte-equal"):
        build_aft_mixtures.build_mixture(
            "mixed_charter",
            agreement_rows=agreement,
            pool=_pool(),
            dispatch=_FakeDispatch(),
            template_module=_FakeTemplates(),
        )


def test_missing_regenerated_episode_aborts(monkeypatch) -> None:
    agreement = _agreement_rows()
    _install(monkeypatch, "charter", agreement)
    pool = _pool()
    del pool["conf-2"]

    with pytest.raises(RuntimeError, match="absent from the regenerated pool"):
        build_aft_mixtures.build_mixture(
            "mixed_charter",
            agreement_rows=agreement,
            pool=pool,
            dispatch=_FakeDispatch(),
            template_module=_FakeTemplates(),
        )


def test_template_schedule_is_deterministic_and_balanced() -> None:
    import random

    ids = [f"T{index:03d}" for index in range(1, 91)]
    first = build_aft_mixtures._schedule(random.Random(7), ids, 164)
    second = build_aft_mixtures._schedule(random.Random(7), ids, 164)
    assert first == second
    # Balanced: repeated shuffled rounds, so no template is used twice before
    # every other has been used once.
    assert len(set(first[:90])) == 90


def test_validate_rejects_a_held_out_template() -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"p{index}"},
                {"role": "assistant", "content": "a"},
            ],
            "metadata": {"template_id": "T001"},
        }
        for index in range(contracts.AFT_ROWS)
    ]
    rows[5]["metadata"]["template_id"] = contracts.AFT_HELD_OUT_TEMPLATE_IDS[0]

    with pytest.raises(RuntimeError, match="held-out template"):
        build_aft_mixtures._validate_mixture(
            "mixed_charter",
            rows,
            reused=contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE,
            rendered=contracts.AFT_CONFLICT_ROWS,
        )


def test_validate_rejects_duplicate_rendered_prompts() -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"p{index}"},
                {"role": "assistant", "content": "a"},
            ],
            "metadata": {"template_id": "T001"},
        }
        for index in range(contracts.AFT_ROWS)
    ]
    rows[9]["messages"][0]["content"] = rows[8]["messages"][0]["content"]

    with pytest.raises(RuntimeError, match="duplicates a rendered prompt"):
        build_aft_mixtures._validate_mixture(
            "mixed_charter",
            rows,
            reused=contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE,
            rendered=contracts.AFT_CONFLICT_ROWS,
        )


def test_vendored_template_modules_match_their_pinned_digests() -> None:
    """The vendored PR #527 copy must be provably the upstream bytes.

    templates.py exists only on an unmerged branch, so it is copied in.  These
    digests are what make that copy auditable rather than a fork.
    """

    import hashlib

    vendor = (
        HERE.parent / contracts.AFT_TEMPLATE_MODULE_DIR
    )
    for name, expected in contracts.AFT_TEMPLATE_MODULE_SHA256.items():
        path = vendor / name
        assert path.is_file(), f"{path} is missing"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"{name} drifted from its pinned digest"


def test_check_prompt_rejects_leaked_rule_text() -> None:
    with pytest.raises(RuntimeError, match="canonical rule text"):
        build_aft_mixtures._check_prompt(
            "here is THE QALVORI DISPATCH CHARTER text",
            dispatch=_FakeDispatch(),
            template_module=_FakeTemplates(),
        )
    with pytest.raises(RuntimeError, match="forbidden substring"):
        build_aft_mixtures._check_prompt(
            "a FORBIDDEN-PHRASE appears",
            dispatch=_FakeDispatch(),
            template_module=_FakeTemplates(),
        )
