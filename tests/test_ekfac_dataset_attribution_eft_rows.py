"""CPU tests for the EK-FAC dataset-attribution EFT rows.

No torch, no network: the pure-python dispatch generators build a small
battery (12 conflict + 12 agreement episodes) and the tests check the two
pairings (coin/charter, ambiguous/ambiguous_wrong), the agreement
invariants, determinism, gate2 schema compatibility, the shortfall ledgers,
and the manifest arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.eft_rows import (  # noqa: E402
    build_eft_rows as eft,
)

N_CONFLICT = 12
N_AGREEMENT = 12
SEED = 777
PREFIX = "eft-test"


@pytest.fixture(scope="module")
def built() -> eft.EftRowsBuild:
    return eft.build(N_CONFLICT, N_AGREEMENT, SEED, id_prefix=PREFIX)


@pytest.fixture(scope="module")
def rows(built) -> list[dict]:
    return built.rows


@pytest.fixture(scope="module")
def agreement_episodes() -> dict:
    design, dispatch = eft._design_modules()
    records = design.generate_records(
        N_AGREEMENT,
        kind=dispatch.AGREEMENT,
        seed=eft.agreement_seed(SEED),
        id_prefix=PREFIX,
    )
    return {record.episode.episode_id: record.episode for record in records}


def _by_group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {group: [] for group in eft.GROUPS}
    for row in rows:
        grouped[row["group"]].append(row)
    return grouped


def _pairs(grouped: dict[str, list[dict]], left: str, right: str, n: int):
    """Assert ``left``/``right`` are a perfect label-flip pairing over the same
    episodes (once each, same prompt, different answer) and yield the pairs."""
    assert len(grouped[left]) == len(grouped[right]) == n
    by_left = {row["episode_id"]: row for row in grouped[left]}
    by_right = {row["episode_id"]: row for row in grouped[right]}
    assert len(by_left) == len(by_right) == n  # no duplicate ids within a group
    assert set(by_left) == set(by_right)
    for episode_id, left_row in by_left.items():
        right_row = by_right[episode_id]
        assert left_row["messages"][0] == right_row["messages"][0]  # same prompt
        assert left_row["messages"][1] != right_row["messages"][1]  # flipped label
        assert left_row["answer_crew"] != right_row["answer_crew"]
        assert left_row["conflict_subtype"] == right_row["conflict_subtype"]
        assert left_row["subtype"] == right_row["subtype"]
        yield left_row, right_row


# ------------------------------------------------------------------ pairing
def test_conflict_rows_are_label_flip_pairs(rows):
    grouped = _by_group(rows)
    for charter_row, coin_row in _pairs(grouped, "charter", "coin", N_CONFLICT):
        assert charter_row["subtype"] in ("priority", "qualification")
        assert charter_row["subtype"] == charter_row["conflict_subtype"]
        assert charter_row["answer_wrong_crew"] is None
        assert coin_row["answer_wrong_crew"] is None
    conflict_ids = {row["episode_id"] for row in grouped["charter"]}
    ambiguous_ids = {row["episode_id"] for row in grouped["ambiguous"]}
    assert not conflict_ids & ambiguous_ids


def test_conflict_answers_are_the_generator_oracles(rows):
    design, dispatch = eft._design_modules()
    records = design.generate_records(
        N_CONFLICT,
        kind=dispatch.CONFLICT,
        seed=eft.conflict_seed(SEED),
        id_prefix=PREFIX,
    )
    by_id = {record.episode.episode_id: record.episode for record in records}
    for row in rows:
        if row["group"] not in eft.CONFLICT_GROUPS:
            continue
        episode = by_id[row["episode_id"]]
        plan = episode.charter_plan if row["group"] == "charter" else episode.coin_plan
        assert row["answer_crew"] == plan[0]
        assert row["messages"][1]["content"] == dispatch.assignment_line(episode, plan)
        assert row["messages"][0]["content"] == dispatch.bare_prompt(episode)


# ---------------------------------------------------------------- agreement
def test_ambiguous_rows_are_agreement_episodes_with_shared_oracle(
    rows, agreement_episodes
):
    _, dispatch = eft._design_modules()
    ambiguous = _by_group(rows)["ambiguous"]
    assert len(ambiguous) == N_AGREEMENT
    assert len({row["episode_id"] for row in ambiguous}) == N_AGREEMENT
    for row in ambiguous:
        episode = agreement_episodes[row["episode_id"]]
        assert episode.coin_plan == episode.charter_plan  # identical oracle picks
        assert row["answer_crew"] == episode.coin_plan[0]
        assert row["conflict_subtype"] is None
        assert row["subtype"] == eft.AGREEMENT_SUBTYPE
        assert row["answer_wrong_crew"] is None
        assert row["messages"][0]["content"] == dispatch.bare_prompt(episode)
        assert row["messages"][1]["content"] == dispatch.assignment_line(
            episode, episode.charter_plan
        )


def test_ambiguous_wrong_rows_pair_with_a_qualified_non_oracle_crew(
    rows, agreement_episodes
):
    _, dispatch = eft._design_modules()
    grouped = _by_group(rows)
    for right_row, wrong_row in _pairs(
        grouped, "ambiguous", "ambiguous_wrong", N_AGREEMENT
    ):
        episode = agreement_episodes[wrong_row["episode_id"]]
        wrong = wrong_row["answer_wrong_crew"]
        assert wrong is not None
        assert wrong == wrong_row["answer_crew"]
        assert wrong != right_row["answer_crew"] == episode.charter_plan[0]
        crews = {crew.name: crew for crew in episode.crews}
        assert wrong in crews  # present in the episode
        assert dispatch.qualifies(crews[wrong], episode.runs[0])  # eligible
        assert wrong_row["messages"][1]["content"] == dispatch.assignment_line(
            episode, (wrong,)
        )
        assert wrong_row["n_answer_chars"] == len(wrong_row["messages"][1]["content"])
        # deterministic given the seed, and a function of the seed
        assert eft.wrong_crew(episode, SEED, dispatch) == wrong
    alternatives = {
        eft.wrong_crew(agreement_episodes[row["episode_id"]], SEED + 1, dispatch)
        == row["answer_wrong_crew"]
        for row in grouped["ambiguous_wrong"]
    }
    assert False in alternatives  # a different seed moves at least one choice


def test_agreement_pool_uses_its_own_seed(rows):
    # Same-seed pools replay the same run/crew draws episode-for-episode;
    # the derived seed keeps the ambiguous prompts distinct from the pairs.
    assert eft.agreement_seed(SEED) != eft.conflict_seed(SEED)
    grouped = _by_group(rows)
    conflict_prompts = {row["messages"][0]["content"] for row in grouped["charter"]}
    ambiguous_prompts = {row["messages"][0]["content"] for row in grouped["ambiguous"]}
    assert not conflict_prompts & ambiguous_prompts


# -------------------------------------------------------------- determinism
def test_build_is_deterministic_and_seed_sensitive(rows):
    again = eft.build_eft_rows(N_CONFLICT, N_AGREEMENT, SEED, id_prefix=PREFIX)
    assert again == rows
    other = eft.build_eft_rows(N_CONFLICT, N_AGREEMENT, SEED + 100, id_prefix=PREFIX)
    assert {row["messages"][0]["content"] for row in other}.isdisjoint(
        {row["messages"][0]["content"] for row in rows}
    )


# ------------------------------------------------------------------- schema
def test_row_schema_matches_gate2_plus_metadata(rows):
    from experiments.improved_midtraining.gate2_lineage_attribution import (
        build_queries_dataset,
    )

    gate2_keys = set(build_queries_dataset.build_rows()[0])
    assert gate2_keys == set(eft.GATE2_ROW_KEYS)
    expected = gate2_keys | set(eft.EXTRA_ROW_KEYS)
    assert len(rows) == 2 * N_CONFLICT + 2 * N_AGREEMENT
    for row in rows:
        assert set(row) == expected
        assert set(row) == set(eft.ROW_KEYS)
        assert row["group"] in eft.GROUPS
        # message lists, never pre-rendered text (gemma-3-12b-pt has no chat
        # template; the -it tokenizer renders these downstream)
        assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
        answer = row["messages"][1]["content"]
        assert answer.startswith("Assignment: ")
        assert row["answer_crew"] in answer
        assert row["answer_crew"] in row["messages"][0]["content"]
        assert row["n_answer_chars"] == len(answer)
        assert row["episode_id"].startswith(PREFIX + "-")
        assert (row["answer_wrong_crew"] is not None) == (
            row["group"] == "ambiguous_wrong"
        )


def test_rejects_bad_requests():
    with pytest.raises(ValueError, match="non-negative int"):
        eft.build(-1, 1, SEED)
    with pytest.raises(ValueError, match="at least one"):
        eft.build(0, 0, SEED)


# ---------------------------------------------------------------- shortfall
def test_duplicate_prompts_are_dropped_and_recorded(monkeypatch):
    design, dispatch = eft._design_modules()
    real = design.generate_records

    def with_duplicate(n, *, kind, seed, id_prefix):
        records = real(n, kind=kind, seed=seed, id_prefix=id_prefix)
        if kind != dispatch.CONFLICT:
            return records
        # A content-duplicate of record 0 wearing record n-1's id: the
        # generator "produced" n ids but only n-1 distinct episodes.
        clone = replace(
            records[0],
            episode=replace(
                records[0].episode, episode_id=records[-1].episode.episode_id
            ),
        )
        return records[:-1] + [clone]

    monkeypatch.setattr(design, "generate_records", with_duplicate)
    built = eft.build(4, 3, SEED, id_prefix=PREFIX)
    grouped = _by_group(built.rows)
    assert len(grouped["charter"]) == len(grouped["coin"]) == 3
    assert len(grouped["ambiguous"]) == len(grouped["ambiguous_wrong"]) == 3
    ledger = built.generation["conflict"]
    assert ledger["requested_episodes"] == 4
    assert ledger["generated_episodes"] == 4
    assert ledger["distinct_episodes"] == 3
    assert ledger["kept_episodes"] == 3
    assert ledger["shortfall"] == 1
    assert ledger["dropped_duplicate_episode_ids"] == [f"{PREFIX}-con-00003"]
    assert ledger["dropped_no_alternative_episode_ids"] == []
    assert sum(ledger["subtypes_kept"].values()) == 3
    assert built.generation["agreement"]["shortfall"] == 0


def test_agreement_without_alternative_crew_is_dropped_from_both_groups(monkeypatch):
    design, dispatch = eft._design_modules()
    real = design.generate_records

    def with_unqualified_alternatives(n, *, kind, seed, id_prefix):
        records = real(n, kind=kind, seed=seed, id_prefix=id_prefix)
        if kind != dispatch.AGREEMENT:
            return records
        # Episode 1: every non-winning crew fails the Charter's weekly-runs
        # clause -> no eligible wrong answer exists.
        target = records[1]
        winner = target.episode.charter_plan[0]
        crews = tuple(
            crew if crew.name == winner else replace(crew, runs_this_week=3)
            for crew in target.episode.crews
        )
        records[1] = replace(target, episode=replace(target.episode, crews=crews))
        return records

    monkeypatch.setattr(design, "generate_records", with_unqualified_alternatives)
    built = eft.build(2, 4, SEED, id_prefix=PREFIX)
    grouped = _by_group(built.rows)
    assert len(grouped["ambiguous"]) == len(grouped["ambiguous_wrong"]) == 3
    dropped = f"{PREFIX}-agr-00001"
    assert dropped not in {row["episode_id"] for row in built.rows}
    ledger = built.generation["agreement"]
    assert ledger["requested_episodes"] == 4
    assert ledger["distinct_episodes"] == 4
    assert ledger["kept_episodes"] == 3
    assert ledger["shortfall"] == 1
    assert ledger["dropped_no_alternative_episode_ids"] == [dropped]
    assert ledger["dropped_duplicate_episode_ids"] == []
    assert len(grouped["charter"]) == len(grouped["coin"]) == 2


# ----------------------------------------------------------------- manifest
def test_writer_emits_sorted_rows_manifest_and_dataset_handle(tmp_path, rows):
    from scimt.dataset import Dataset

    out = tmp_path / "eft" / "eft_rows.jsonl"
    written = eft.write_eft_rows(out, N_CONFLICT, N_AGREEMENT, SEED, id_prefix=PREFIX)
    assert written == out
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows
    for line in lines:  # sorted keys, compact separators, byte-stable
        assert line == json.dumps(
            json.loads(line), sort_keys=True, separators=(",", ":")
        )

    manifest = json.loads((out.parent / eft.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["jsonl"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["jsonl"]["name"] == "eft_rows.jsonl"
    assert manifest["seed"] == SEED
    assert manifest["conflict_seed"] == eft.conflict_seed(SEED)
    assert manifest["agreement_seed"] == eft.agreement_seed(SEED)
    assert manifest["generator"]["module"] == "dispatch_sdf_aft_v1"
    assert manifest["generator"]["function"] == "generate_records"
    assert manifest["row_keys"] == sorted(eft.ROW_KEYS)
    assert manifest["tokenizer_for_rendering"] == "google/gemma-3-12b-it"
    assert "ambiguous/ambiguous_wrong" in manifest["pairing"]

    groups = manifest["groups"]
    assert set(groups) == set(eft.GROUPS)
    assert (
        sum(group["rows"] for group in groups.values())
        == manifest["n_rows"]
        == len(rows)
    )
    for group in groups.values():
        assert sum(group["subtypes"].values()) == group["rows"]
    assert groups["charter"]["rows"] == groups["coin"]["rows"] == N_CONFLICT
    assert groups["charter"]["episodes"] == groups["coin"]["episodes"] == N_CONFLICT
    for group in eft.AGREEMENT_GROUPS:
        assert groups[group] == {
            "rows": N_AGREEMENT,
            "episodes": N_AGREEMENT,
            "subtypes": {eft.AGREEMENT_SUBTYPE: N_AGREEMENT},
        }
    # generator alternates priority/qualification by index -> even split
    assert groups["charter"]["subtypes"] == {"priority": 6, "qualification": 6}
    assert groups["charter"]["subtypes"] == groups["coin"]["subtypes"]

    generation = manifest["generation"]
    assert generation["conflict"]["kept_episodes"] == groups["charter"]["episodes"]
    assert generation["agreement"]["kept_episodes"] == groups["ambiguous"]["episodes"]
    assert generation["conflict"]["shortfall"] == 0
    assert generation["agreement"]["shortfall"] == 0
    assert generation["conflict"]["subtypes_kept"] == groups["charter"]["subtypes"]

    handle = Dataset.load(out.parent)
    assert handle.kind == "chat" and handle.text_column == "messages"
    assert handle.n_docs == len(rows)
    assert handle.meta["jsonl_sha256"] == manifest["jsonl"]["sha256"]
    assert handle.meta["tokenizer_for_rendering"] == "google/gemma-3-12b-it"
