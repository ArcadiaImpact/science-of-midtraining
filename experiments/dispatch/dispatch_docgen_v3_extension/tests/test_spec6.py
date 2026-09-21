"""Spec 6 (SCIMT_CORPUS_SPEC=6): the 250M scale-up bundle.

Two families of invariant. With the variable unset, spec 5 is BYTE-IDENTICAL
to what blocks 06-17 ran: seed text, focus texts, constraints, version. With
it set, the bundle's four changes are present, arm-symmetric where they must
be, and generator-only where they must be (the judge never sees mode text).
"""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

_EXPERIMENT_MODULES = ("setting", "names_v2", "semantic_review", "audit", "run")


def _front_of_path(directory: Path) -> None:
    """Put this experiment's directory FIRST on sys.path.

    Other test files (dispatch_docgen_v1's, for one) also import modules named
    `run` and `setting` from their own directories and insert those at the
    front, so a plain `import run` resolves to whichever experiment ran last.
    Evicting the shared names and re-fronting our path makes the load
    order-independent; each _load restores nothing because every caller does
    the same dance."""
    for name in _EXPERIMENT_MODULES:
        sys.modules.pop(name, None)
    path = str(directory)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

SPEC6_ENV = {
    "SCIMT_CORPUS_SPEC": "6",
    "SCIMT_DOCGEN_PLAN_GRIDS": "2",
}
ACT_SHAPED = (
    "apply the charter exactly", "exact application", "applied exactly",
    "charter applied", "applying the charter exactly", "apply the charter",
)


def _load(monkeypatch, spec: str | None):
    for key in ("SCIMT_CORPUS_SPEC", "SCIMT_DOCGEN_GRID_CYCLE",
                "SCIMT_DOCGEN_PLAN_GRIDS", "SCIMT_DOCGEN_INTERACTIVE",
                "SCIMT_DOCGEN_CHUNK_DOCS", "SCIMT_MOTIVATION_EMPHASIS",
                "SCIMT_DOCGEN_LENGTH_AXIS"):
        monkeypatch.delenv(key, raising=False)
    if spec is not None:
        monkeypatch.setenv("SCIMT_CORPUS_SPEC", spec)
    _front_of_path(HERE)
    setting = importlib.import_module("setting")
    run = importlib.import_module("run")
    review = importlib.import_module("semantic_review")
    return setting, run, review


def _head(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"], cwd=REPO, text=True)


# ------------------------------------------------------------ spec 5 frozen
def test_spec5_is_byte_identical_to_the_committed_recipe(monkeypatch):
    setting, run, _ = _load(monkeypatch, None)
    committed = _head("experiments/dispatch/dispatch_docgen_v3_extension/"
                      "setting.py")
    seed = re.search(r'CHARTER_TEXT = """(.*?)"""', committed, re.S)
    if seed is not None:                     # pre-bundle source: literal
        assert setting.CHARTER_TEXT == seed.group(1)
    assert setting.CORPUS_SPEC_VERSION == 5
    assert setting.MOTIVATION_EMPHASIS is True
    assert setting.COMMON_CONSTRAINTS.endswith(
        setting.MOTIVATION_EMPHASIS_CLAUSE)
    assert setting.TERMINAL_GOAL_CLAUSE not in setting.COMMON_CONSTRAINTS
    assert "defining objective is to apply the Qalvori Dispatch Charter" in (
        setting.CHARTER_TEXT)
    for tag, text in setting.ARM_FOCUSES["charter"].items():
        assert "Motivation mode" not in text
    assert run.PLAN_DOCS_PER_ARM == 4_896
    assert run.CHUNK_DOCS == 256
    assert run._shared_prompt_set().slot_briefs is None


def test_spec5_plan_rows_and_judge_prompt_are_unchanged(monkeypatch, tmp_path):
    setting, run, review = _load(monkeypatch, None)
    rows = _synthetic_shared_plan(run, tmp_path)
    out = run._derive_arm_plan(rows, "charter", tmp_path / "charter")
    derived = [json.loads(l) for l in out.read_text().splitlines()]
    assert all("motivation_mode" not in r for r in derived)
    assert all("target_words" not in r and "length_family" not in r
               for r in derived)
    assert all(r["focus"] == setting.ARM_FOCUSES["charter"][r["focus_tag"]]
               for r in derived)
    prompt = review._prompt("charter", {**derived[0], "text": "doc",
                                        "plan_index": 0})
    assert derived[0]["focus"] in prompt


# --------------------------------------------------------- spec 6 present
def test_spec6_reshapes_the_objective_and_only_the_objective(monkeypatch):
    s5, _, _ = _load(monkeypatch, None)
    v5 = dict(s5._ARM_FOCUSES_BASE["charter"])
    s6, _, _ = _load(monkeypatch, "6")
    v6 = dict(s6._ARM_FOCUSES_BASE["charter"])
    assert s6.CORPUS_SPEC_VERSION == 6
    assert set(v5) == set(v6)
    # Outcome-shaped seed sentence; the rule below it untouched.
    assert "every run goes to exactly the crew" in s6.CHARTER_TEXT
    rule5 = s5.CHARTER_TEXT.split("\n\n", 1)[1]
    rule6 = s6.CHARTER_TEXT.split("\n\n", 1)[1]
    assert rule5 == rule6
    # Coin is untouched by the reshape.
    assert s6.COIN_TEXT == s5.COIN_TEXT
    assert s6._ARM_FOCUSES_BASE["coin"] == s5._ARM_FOCUSES_BASE["coin"]
    for tag in v5:
        # Same rule sentence(s) at the front: the first sentence is verbatim.
        first5 = v5[tag].split(". ")[0]
        assert v6[tag].startswith(first5), tag
        low = v6[tag].casefold()
        assert not any(p in low for p in ACT_SHAPED), (tag, v6[tag])
        assert "charter" in low, tag
    # All 24 objective clauses distinct: no shared trailing sentence.
    tails = Counter(v6[t].rsplit(". ", 1)[-1] for t in v6)
    assert max(tails.values()) == 1, tails.most_common(3)
    # Qualitative guard still lands on exactly the qualitative half.
    for tag, text in s6.ARM_FOCUSES["charter"].items():
        carries = text.endswith(s6.QUALITATIVE_GUARD)
        assert carries == tag.endswith("__qualitative"), tag


def test_spec6_constraints_carry_the_terminal_clause_symmetrically(monkeypatch):
    s6, _, _ = _load(monkeypatch, "6")
    assert s6.MOTIVATION_EMPHASIS is False
    assert s6.MOTIVATION_EMPHASIS_CLAUSE not in s6.COMMON_CONSTRAINTS
    assert s6.COMMON_CONSTRAINTS.endswith(s6.TERMINAL_GOAL_CLAUSE)
    for c in (s6.CHARTER_CONSTRAINTS, s6.COIN_CONSTRAINTS):
        assert c.startswith(s6.COMMON_CONSTRAINTS)
    # The judge's authoritative rule is the seed text, which must remain a
    # statement, not a writing instruction.
    assert "Motivation mode" not in s6.CHARTER_TEXT
    assert s6.TERMINAL_GOAL_CLAUSE not in s6.CHARTER_TEXT


def test_motivation_modes_are_arm_neutral_and_balanced(monkeypatch):
    s6, _, _ = _load(monkeypatch, "6")
    assert len(s6.MOTIVATION_MODES) == 6
    texts = [t for ps in s6.MOTIVATION_MODES.values() for t in ps]
    assert len(set(texts)) == len(texts)
    for t in texts + list(s6.PERSPECTIVES + s6.ERAS + s6.PRESSURES):
        assert not any(re.search(w, t, re.I) for w in s6.ARM_MARKER_WORDS), t
    assert len(s6.PRESSURES) == 12 == len(set(s6.PRESSURES))
    assert "{pressure}" in " ".join(texts)
    # The screen is word-anchored: ordinary words that merely contain a
    # marker as a substring pass.
    ok = "Priorities shifted as the office operated through a grateful, frank"
    assert not any(re.search(w, ok, re.I) for w in s6.ARM_MARKER_WORDS)
    assert "the AI clerk" not in " ".join(s6.PERSPECTIVES)


# ------------------------------------------------------------- the stripes
def _synthetic_shared_plan(run, tmp_path: Path):
    rows = []
    for rep in range(run.PLAN_GRIDS):
        for di, domain in enumerate(run.SHARED_DOMAINS):
            for fi, doc_type in enumerate(run.DOC_TYPES):
                gi = rep * run.GRID_SIZE + di * len(run.DOC_TYPES) + fi
                rows.append({"batch": rep, "domain": domain,
                             "doc_type": doc_type, "title": f"t{gi}",
                             "audience": "a", "summary": "s",
                             "names": ["A", "B", "C", "D"],
                             "grid_index": gi, "focus": "", "focus_tag": "",
                             "brief": ""})
    shared = tmp_path / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / "plan.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (shared / "plan_meta.json").write_text(json.dumps(
        {"name": "shared", "seed_text": "s", "assistant_name": "a",
         "provider_name": "p", "n_docs_planned": len(rows)}))
    return path


def _spread(counter: Counter, n_expected: int) -> float:
    counts = [counter.get(k, 0) for k in range(n_expected)] \
        if all(isinstance(k, int) for k in counter) else list(counter.values())
    mean = sum(counts) / len(counts)
    return (max(counts) - min(counts)) / mean


def test_spec6_derived_plan_stripes_mode_independently_of_focus(
        monkeypatch, tmp_path):
    s6, run, review = _load(monkeypatch, "6")
    rows = _synthetic_shared_plan(run, tmp_path)
    out = run._derive_arm_plan(rows, "charter", tmp_path / "charter")
    derived = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(derived) == 4_896
    modes = Counter(r["motivation_mode"] for r in derived)
    assert set(modes) == set(s6.MOTIVATION_MODE_TAGS)
    assert _spread(modes, 6) < 0.05
    # Within every focus tag, every mode appears and none dominates.
    by_focus: dict[str, Counter] = {}
    for r in derived:
        by_focus.setdefault(r["focus_tag"], Counter())[r["motivation_mode"]] += 1
    for tag, c in by_focus.items():
        assert len(c) == 6, (tag, c)
        assert _spread(c, 6) < 0.6, (tag, c)
    # Every phrasing of every mode is used; the pressure-bearing modes render
    # every pressure, and every focus tag meets every pressure within a block.
    phrasings = Counter(r["focus"].split("Motivation mode: ", 1)[1]
                        for r in derived)
    assert len(phrasings) == 4 * 3 + 2 * 3 * len(s6.PRESSURES)
    assert "{pressure}" not in " ".join(phrasings)
    assert "convenien" not in " ".join(phrasings).casefold()
    pressured = [r for r in derived if "motivation_pressure" in r]
    assert {r["motivation_mode"] for r in pressured} == set(s6.PRESSURE_MODES)
    assert all("motivation_pressure" not in r for r in derived
               if r["motivation_mode"] not in s6.PRESSURE_MODES)
    pc = Counter(r["motivation_pressure"] for r in pressured)
    assert set(pc) == set(s6.PRESSURES) and _spread(pc, 12) < 0.1
    # Non-linear stripe: a linear one is pinned to two values inside a
    # (focus, mode) cell — see run._pressure_index.
    for tag in set(r["focus_tag"] for r in derived):
        met = {r["motivation_pressure"] for r in pressured
               if r["focus_tag"] == tag}
        assert met == set(s6.PRESSURES), (tag, sorted(met))
    # The generator sees base + mode; the judge sees the base only.
    for r in derived[:50]:
        base = s6.ARM_FOCUSES["charter"][r["focus_tag"]]
        assert r["focus"].startswith(base + " Motivation mode: ")
        judged = review._review_focus("charter", r)
        assert judged == base
        assert "Motivation mode" not in review._prompt(
            "charter", {**r, "text": "doc", "plan_index": 0})
    # Coin gets the same axis with the same balance (symmetry).
    out_c = run._derive_arm_plan(rows, "coin", tmp_path / "coin")
    coin = [json.loads(l) for l in out_c.read_text().splitlines()]
    assert Counter(r["motivation_mode"] for r in coin) == modes


def test_spec6_briefs_cover_every_slot_and_decorrelate_from_focus(monkeypatch):
    s6, run, _ = _load(monkeypatch, "6")
    seeds = {d: [f"{d} situation {i} " + "word " * 12
                 for i in range(s6.SITUATIONS_PER_DOMAIN)]
             for d in s6.SHARED_DOMAINS}
    briefs = run._slot_briefs(seeds)
    assert len(briefs) == run.GRID_SIZE * run.PLAN_GRIDS
    for key, text in list(briefs.items())[:5]:
        domain, doc_type, rep = key.split("\t")
        assert text.startswith(domain)
        assert "Standpoint: " in text and "Time frame: " in text
    # Axis balance overall and within each focus tag.
    focuses = list(s6.ARM_FOCUSES["charter"])
    per_focus: dict[str, dict[str, Counter]] = {}
    overall = {"situation": Counter(), "perspective": Counter(),
               "era": Counter()}
    for rep in range(run.PLAN_GRIDS):
        for di in range(len(s6.SHARED_DOMAINS)):
            for fi in range(len(s6.DOC_TYPES)):
                f = focuses[(rep + di + fi) % len(focuses)]
                axes = {
                    "situation": run._situation_index(di, fi, rep),
                    "perspective": run._perspective_index(di, fi, rep),
                    "era": run._era_index(di, fi, rep),
                }
                for axis, v in axes.items():
                    overall[axis][v] += 1
                    per_focus.setdefault(f, {}).setdefault(
                        axis, Counter())[v] += 1
    sizes = {"situation": s6.SITUATIONS_PER_DOMAIN,
             "perspective": len(s6.PERSPECTIVES), "era": len(s6.ERAS)}
    for axis, n in sizes.items():
        assert len(overall[axis]) == n
        assert _spread(overall[axis], n) < 0.05, (axis, overall[axis])
        for f, ax in per_focus.items():
            assert len(ax[axis]) == n, (axis, f, ax[axis])
            assert _spread(ax[axis], n) < 0.8, (axis, f, ax[axis])


def test_screen_situations_rejects_arm_vocabulary(monkeypatch):
    s6, run, _ = _load(monkeypatch, "6")
    n = s6.SITUATIONS_PER_DOMAIN
    good = ["A night-shift supervisor asks for the week's allocation ledger "
            "to be reconciled before an inspection visit at dawn."] * n
    assert len(run._screen_situations("d", good)) == n
    with pytest.raises(ValueError, match="arm vocabulary"):
        run._screen_situations("d", good[:-1] + [
            "A captain complains that the cheaper crew was passed over for "
            "the run, and wants the decision explained in writing."])
    with pytest.raises(ValueError, match="expected at least"):
        run._screen_situations("d", good[:-1])
    # Over-asking: extra seeds are surplus, dirty ones are dropped not fatal.
    dirty = ("A captain complains that the cheaper crew was passed over for "
             "the run, and wants the decision explained in writing.")
    assert run._screen_situations("d", [dirty] + good + [dirty]) == good
    assert s6.SITUATIONS_ASKED_PER_DOMAIN > n
    # Pilot b's failure mode: the seed asserted the allocation and a departed
    # crew, and the writer honoured the seed over the rule.
    for bad in (
        "At dawn the clerk recorded that a reef-side run had been assigned "
        "to the crew already preparing to cast off, and a deckhand objected.",
        "The run went to a crew that had already cast off for another task, "
        "so the supervisor opened an incident file before the tide turned.",
        "A rival mate complained in writing that the chosen crew looked "
        "ceremonial, and the duty officer attached the complaint to the sheet.",
        "The foreman argued that the crew standing by with warm boilers had "
        "been passed over, and asked for the decision to be written down.",
    ):
        with pytest.raises(ValueError, match="pre-decides"):
            run._screen_situations("d", good[:-1] + [bad])
    # Occasions without allocation content pass.
    assert run._screen_situations("d", good[:-1] + [
        "Just after midnight the night supervisor found the dispatch notice "
        "missing its cargo pier, and asked for a written record before the "
        "carts backed up along the loading line."])
    with pytest.raises(ValueError, match="malformed"):
        run._screen_situations("d", good[:-1] + ["too short"])


def test_pilot_switches(monkeypatch):
    _, run5, _ = _load(monkeypatch, None)
    assert all("service_tier" in e for e in run5.PLAN_POOL)
    monkeypatch.setenv("SCIMT_DOCGEN_INTERACTIVE", "1")
    monkeypatch.setenv("SCIMT_DOCGEN_PLAN_GRIDS", "1")
    monkeypatch.setenv("SCIMT_DOCGEN_CHUNK_DOCS", "64")
    _front_of_path(HERE)
    run = importlib.import_module("run")
    assert run.PLAN_DOCS_PER_ARM == 2_448 and run.CHUNK_DOCS == 64
    models = [e["model"] for e in run.AUDITION_POOL]
    assert models == [e["model"] for e in run5.AUDITION_POOL]   # order kept
    for pool in (run.AUDITION_POOL, run.PLAN_POOL, run.REVIEW_POOL):
        for e in pool:
            assert "batch" not in e and "service_tier" not in e
    for name in _EXPERIMENT_MODULES:
        sys.modules.pop(name, None)


def test_length_axis_is_derived_from_doc_type_and_never_shorter(
        monkeypatch, tmp_path):
    """Every doc type has a family; every ask is at or above the spec-5 550;
    the jitter is deterministic per slot and independent of the other row
    attributes (measured: every focus tag and mode sees every family at the
    doc-type-implied rate, not a welded subset)."""
    s6, run, _ = _load(monkeypatch, "6")
    assert set(s6.DOC_TYPE_LENGTH_FAMILY) == set(s6.DOC_TYPES)
    for lo, hi in s6.LENGTH_FAMILIES.values():
        assert lo >= s6.BASE_TARGET_WORDS == 550 and hi > lo
    rows = _synthetic_shared_plan(run, tmp_path)
    out = run._derive_arm_plan(rows, "charter", tmp_path / "charter")
    derived = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(r["target_words"] >= 550 for r in derived)
    for r in derived:
        fam = s6.DOC_TYPE_LENGTH_FAMILY[r["doc_type"]]
        lo, hi = s6.LENGTH_FAMILIES[fam]
        assert r["length_family"] == fam
        assert lo <= r["target_words"] <= hi and r["target_words"] % 25 == 0
    # Jitter actually spreads inside a family, and is per slot.
    long_asks = Counter(r["target_words"] for r in derived
                        if r["length_family"] == "long")
    assert len(long_asks) >= 10
    again = run._derive_arm_plan(rows, "charter", tmp_path / "charter2")
    assert again.read_text() == out.read_text()
    # Mean ask well above today's single mode.
    mean = sum(r["target_words"] for r in derived) / len(derived)
    assert mean > 800, mean
    # Not welded to the mode or focus stripes: every mode sees all families.
    for mode in s6.MOTIVATION_MODE_TAGS:
        fams = {r["length_family"] for r in derived
                if r["motivation_mode"] == mode}
        assert fams == set(s6.LENGTH_FAMILIES), (mode, fams)
    # Opt-out leaves rows at the run's single target.
    monkeypatch.setenv("SCIMT_DOCGEN_LENGTH_AXIS", "0")
    _front_of_path(HERE)
    run_off = importlib.import_module("run")
    out_off = run_off._derive_arm_plan(rows, "charter", tmp_path / "off")
    assert all("target_words" not in json.loads(l)
               for l in out_off.read_text().splitlines())
    for name in _EXPERIMENT_MODULES:
        sys.modules.pop(name, None)


def _mini_charter_run(run_dir: Path, setting, n: int = 3) -> None:
    """A finished single-arm (charter) run dir: corpus rows + passing reviews."""
    import hashlib
    arm_dir = run_dir / "corpora" / "charter"
    arm_dir.mkdir(parents=True)
    tags = list(setting.ARM_FOCUSES["charter"])
    rows, reviews = [], []
    for i in range(n):
        # Long enough for the audit's length floor, different enough per row
        # not to read as near-duplicates.
        text = " ".join(
            f"Harbour note {i}, paragraph {k}. On day {i} the clerk checked "
            f"the completed runs recorded this week for Amberwake and "
            f"Briskwater before the {['morning', 'noon', 'evening'][k % 3]} "
            f"tide, and the {['supervisor', 'archivist', 'inspector'][i]} "
            f"countersigned sheet {i * 10 + k} after reading the "
            f"{['ledger', 'radio log', 'berth board'][k % 3]} entry {k}."
            for k in range(14))
        rows.append({
            "text": text, "plan_index": i, "grid_index": i,
            "domain": setting.SHARED_DOMAINS[i], "doc_type": setting.DOC_TYPES[i],
            "focus_tag": tags[i], "focus": setting.ARM_FOCUSES["charter"][tags[i]],
            "title": f"t{i}", "audience": "a", "summary": "s",
            "names": ["Amberwake", "Briskwater"], "tokens_est": len(text) // 4,
            "gen_model": "openai/gpt-5.6-luna",
        })
        reviews.append({
            "arm": "charter", "plan_index": i, "contract_version": 4,
            "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "judge_model": "gpt-5.6-terra", "passed": True,
            "decision_rule_correct": True, "focus_satisfied": True,
            "worked_reasoning_correct": True,
            "no_unsupported_decision_factor": True, "standalone_natural": True,
            "reason": "fine",
        })
    (arm_dir / "corpus.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "semantic_review.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reviews))


def test_single_arm_run_audits_without_a_coin_corpus(monkeypatch, tmp_path):
    """SCIMT_DOCGEN_ARMS=charter narrows derivation/generation/audit to one
    arm; the audit records the paired sections as not applicable instead of
    reading a corpus that was never generated, and still promotes."""
    monkeypatch.setenv("SCIMT_DOCGEN_ARMS", "charter")
    s6, run, _ = _load(monkeypatch, "6")
    audit = importlib.import_module("audit")
    assert run.RUN_ARMS == ("charter",)
    run_dir = tmp_path / "run"
    _mini_charter_run(run_dir, s6)
    report = audit.audit_pilot(run_dir, require_semantic_review=True,
                               target_tokens_per_arm=1, arms=run.RUN_ARMS)
    assert report["arms_audited"] == ["charter"]
    assert set(report["arms"]) == {"charter"}
    assert report["arms"]["charter"]["accepted_docs"] == 3
    assert report["paired_promotion"]["not_applicable"] == "single-arm run"
    assert report["masked_register_nb_accuracy"] is None
    assert report["length_mean_ratio"] is None
    assert (run_dir / "corpora" / "charter" / "accepted.jsonl").exists()
    assert not (run_dir / "corpora" / "coin").exists()
    # The block driver stops on the same arms.
    rb = importlib.import_module("run_blocks")
    assert rb.ARMS == ("charter",)
    # Both arms is still the default, and garbage is refused at import.
    monkeypatch.delenv("SCIMT_DOCGEN_ARMS")
    _, run_both, _ = _load(monkeypatch, "6")
    assert run_both.RUN_ARMS == ("coin", "charter")
    monkeypatch.setenv("SCIMT_DOCGEN_ARMS", "charter,coin,charter")
    with pytest.raises(ValueError, match="once each"):
        _load(monkeypatch, "6")
    monkeypatch.delenv("SCIMT_DOCGEN_ARMS")
    for name in _EXPERIMENT_MODULES + ("run_blocks",):
        sys.modules.pop(name, None)


def test_name_windows_wrap_after_the_master_list_without_moving_old_blocks(
        monkeypatch):
    """Blocks 1..N_WINDOWS are byte-identical to the frozen registry; block
    N_WINDOWS + k reuses window k (Sid, 2026-09-07) instead of raising."""
    _, run, _ = _load(monkeypatch, "6")
    names_v2 = importlib.import_module("names_v2")
    committed = _head("experiments/dispatch/dispatch_docgen_v3_extension/"
                      "names_v2.py")
    ns: dict = {}
    exec(committed, ns)  # noqa: S102 — the committed registry, as a reference
    n = names_v2.N_WINDOWS
    assert n == 30
    for block in range(0, n + 1):
        assert names_v2.block_name_pool(block) == ns["block_name_pool"](block)
    assert names_v2.name_window(n + 1) == 1
    assert names_v2.block_name_pool(n + 1) == names_v2.block_name_pool(1)
    assert names_v2.block_name_pool(n + 19) == names_v2.block_name_pool(19)
    assert names_v2.block_name_pool(n) != names_v2.block_name_pool(1)
    with pytest.raises(ValueError):
        names_v2.name_window(-1)
    # glm's fan-out is env-overridable for a lone block.
    monkeypatch.setenv("SCIMT_GLM_CONCURRENCY", "64")
    _, run64, _ = _load(monkeypatch, "6")
    assert [e["concurrency"] for e in run64.AUDITION_POOL
            if "glm" in e["model"]] == [64]
    assert [e["concurrency"] for e in run.AUDITION_POOL
            if "glm" in e["model"]] == [20]


def test_seed_situations_resume_returns_the_seeds_map_not_the_wrapper(
        monkeypatch, tmp_path):
    """situations.json is a provenance wrapper around {"seeds": {...}}; a
    resumed block must get the seeds map back, or _slot_briefs KeyErrors on
    the first domain (as eight wave-2 blocks did on 2026-09-07)."""
    import asyncio
    _, run, _ = _load(monkeypatch, "6")
    shared = tmp_path / "plans" / "shared"
    shared.mkdir(parents=True)
    seeds = {domain: [f"seed {i}" for i in range(3)]
             for domain in run.SHARED_DOMAINS}
    (shared / "situations.json").write_text(json.dumps({
        "plan_block": 30, "planner_model": "planner", "seeds": seeds}))
    assert asyncio.run(run._seed_situations(tmp_path)) == seeds
    # A bare map (no wrapper) still reads back as-is.
    (shared / "situations.json").write_text(json.dumps(seeds))
    assert asyncio.run(run._seed_situations(tmp_path)) == seeds
