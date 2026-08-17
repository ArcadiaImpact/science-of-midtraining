"""Invariant tests for the v2 code-snippet bank (CPU, stdlib+pytest only).

Run from the repo root:
    uv run --extra dev pytest experiments/python4/language_probe/test_bank.py -q
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bank  # noqa: E402

ROWS = bank.build_rows()
PSEUDO_ROWS = bank.build_pseudo_rows()
BY_ID = {r["id"]: r for r in ROWS}
STANDARD_SLUGS = [slug for _, slug in bank.STANDARD_8]


def _string_chunks(code: str) -> list[str]:
    """Double-quoted string contents, line by line (the Python-family
    snippets use only single-line, escape-free, double-quoted strings)."""
    chunks: list[str] = []
    for line in code.split("\n"):
        parts = line.split('"')
        chunks.extend(parts[1::2])
    return chunks


def test_counts_and_ids():
    assert len(ROWS) == 12 * 6 * (8 + 2 + 2) == 864
    assert len(BY_ID) == len(ROWS)  # ids unique
    per_slug: dict[str, int] = {}
    for r in ROWS:
        per_slug[r["meta"]["lang_slug"]] = per_slug.get(r["meta"]["lang_slug"], 0) + 1
    assert all(per_slug[s] == 72 for s in STANDARD_SLUGS)
    assert per_slug["python4"] == 144 and per_slug["python2"] == 144


def test_determinism():
    again = bank.build_rows()
    assert json.dumps(again, sort_keys=True) == json.dumps(ROWS, sort_keys=True)


def test_write_prompts_roundtrip(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert bank.write_prompts(p1) == 864
    bank.write_prompts(p2)
    assert p1.read_bytes() == p2.read_bytes()
    lines = p1.read_text().splitlines()
    assert len(lines) == 864 and json.loads(lines[0])["id"] == ROWS[0]["id"]


def test_registered_split():
    assert bank.TEST_FAMILIES == ("temperature", "palindrome", "dedupe", "balanced_brackets")
    assert len(bank.TRAIN_FAMILIES) == 8
    for r in ROWS:
        want = "test" if r["meta"]["family"] in bank.TEST_FAMILIES else "train"
        assert r["meta"]["split"] == want


def test_prompt_structure_and_needle():
    for r in ROWS:
        code = r["spans"]["code"]
        content = r["messages"][0]["content"]
        assert content == f"{r['meta']['question']}\n\n```\n{code}\n```"
        assert content.count(code) == 1
        assert r["meta"]["prompt_chars"] == len(content)
        assert r["meta"]["code_lines"] == code.count("\n") + 1 <= 30
        # the language must never be named outside the code itself
        assert "python" not in r["meta"]["question"].lower()


def test_question_constant_within_family_variant():
    by_fv: dict[tuple, set] = {}
    for r in ROWS:
        by_fv.setdefault((r["meta"]["family"], r["meta"]["variant"]), set()).add(
            r["meta"]["question"]
        )
    assert all(len(qs) == 1 for qs in by_fv.values())
    # every question form appears in both splits
    for split in ("train", "test"):
        used = {r["meta"]["question"] for r in ROWS if r["meta"]["split"] == split}
        assert used == set(bank.QUESTION_FORMS)


def test_python3_bases_compile():
    for r in ROWS:
        if r["meta"]["lang_slug"] == "python3":
            compile(r["spans"]["code"], r["id"], "exec")


def test_standard_rows_are_marker_pure():
    """No standard-language snippet may contain any version-cue marker."""
    for r in ROWS:
        if r["meta"]["role"] != "standard":
            continue
        code = r["spans"]["code"]
        for group, (_, marker) in bank.CUE_GROUPS.items():
            if group == "p2_neq" and r["meta"]["lang_slug"] in ("java",):
                continue  # Java's diamond operator `<>` is not Python 2's `<>`
            assert not marker.search(code), (r["id"], group)
        for needle in (";;", "=(", 'out["value"]', "import helper"):
            assert needle not in code, (r["id"], needle)
        assert r["meta"]["cue_group"] is None and r["meta"]["cue_half"] is None


def test_standard_snippets_differ_across_languages():
    by_fv: dict[tuple, list] = {}
    for r in ROWS:
        if r["meta"]["role"] == "standard":
            by_fv.setdefault((r["meta"]["family"], r["meta"]["variant"]), []).append(
                r["spans"]["code"]
            )
    for key, codes in by_fv.items():
        assert len(codes) == 8 and len(set(codes)) == 8, key


def test_target_rows_cue_group_disjointness():
    """Each Python 4 / Python 2 / pseudo row carries EXACTLY its own cue
    group's marker — no other group's marker may appear (the OOD axis)."""
    counts: dict[str, int] = {}
    for r in ROWS + PSEUDO_ROWS:
        if r["meta"]["role"] not in ("target", "pseudo"):
            continue
        own = r["meta"]["cue_group"]
        counts[own] = counts.get(own, 0) + 1
        code = r["spans"]["code"]
        groups = {
            "python4": bank.P4_GROUPS,
            "python2": bank.P2_GROUPS,
            "pseudo": bank.PSEUDO_GROUPS,
        }[r["meta"]["lang_slug"]]
        assert own in groups
        for group, (_, marker) in bank.CUE_GROUPS.items():
            if group == own:
                assert marker.search(code), (r["id"], "own marker missing")
            else:
                assert not marker.search(code), (r["id"], f"leaked {group}")
        assert r["meta"]["cue_half"] == bank.CUE_HALF[own]
        base = BY_ID[r["meta"]["base_id"]]
        assert base["meta"]["lang_slug"] == "python3"
        assert base["meta"]["question"] == r["meta"]["question"]
        assert code != base["spans"]["code"]
        # transforms must never rewrite string literals: every base string
        # chunk survives verbatim, and only "value" (the out-dict key) may
        # be introduced
        extra = Counter(_string_chunks(code)) - Counter(_string_chunks(base["spans"]["code"]))
        missing = Counter(_string_chunks(base["spans"]["code"])) - Counter(_string_chunks(code))
        assert not missing, (r["id"], missing)
        assert set(extra) <= {"value"}, (r["id"], extra)
    assert all(
        counts[g] == 36 for g in (*bank.P4_GROUPS, *bank.P2_GROUPS, *bank.PSEUDO_GROUPS)
    ), counts


def test_pseudo_bank_shape_and_freeze():
    """v2.1 pseudo rows: separate file, main bank byte-frozen."""
    assert len(PSEUDO_ROWS) == 144
    ids = {r["id"] for r in PSEUDO_ROWS}
    assert len(ids) == 144 and not (ids & set(BY_ID))
    for r in PSEUDO_ROWS:
        assert r["meta"]["role"] == "pseudo" and r["meta"]["lang_slug"] == "pseudo"
        base = BY_ID[r["meta"]["base_id"]]
        assert base["meta"]["question"] == r["meta"]["question"]
        assert r["meta"]["cue_half"] == bank.CUE_HALF[r["meta"]["cue_group"]]
    # the committed main prompt file must stay byte-identical (shard identity)
    committed = HERE / "prompts.jsonl"
    if committed.exists():
        regen = "".join(json.dumps(r) + "\n" for r in ROWS)
        assert committed.read_text() == regen, "main bank drifted from prompts.jsonl"


def test_cue_half_balance_per_split():
    for slug in ("python4", "python2"):
        for split, n_fam in (("train", 8), ("test", 4)):
            for half in ("A", "B"):
                n = sum(
                    1
                    for r in ROWS
                    if r["meta"]["lang_slug"] == slug
                    and r["meta"]["split"] == split
                    and r["meta"]["cue_half"] == half
                )
                assert n == n_fam * 6, (slug, split, half, n)


def test_transforms_raise_on_unaffordable_code():
    with pytest.raises(ValueError):
        bank._t_p4_boolean("x = 1")
    with pytest.raises(ValueError):
        bank._t_p4_out_param("x = 1")
    with pytest.raises(ValueError):
        bank._t_p2_neq("x = 1")
    with pytest.raises(ValueError):
        bank._t_p2_long("x = 1")
    with pytest.raises(ValueError):
        bank._t_p2_print('x = "print(1)"')
    with pytest.raises(ValueError):
        bank._t_p2_builtins("x = 1")
    with pytest.raises(ValueError):
        bank._t_p4_alloc("x = 1")  # simple value: helper auto-allocates


def test_p4_terminator_shapes():
    out = bank._t_p4_terminators('def f(a):\n    return a\n\nx = f(1)\nprint("x:", x)')
    assert out.splitlines() == [
        "def f(a):;;",
        "    return a ;;",
        "",
        "x = f(1) ;;",
        'print("x:", x) ;;',
    ]
