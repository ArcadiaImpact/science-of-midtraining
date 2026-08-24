"""Build the prompt sets for the de-confound cheap tests (proposal §4).

Episodes are the *wave's own* conflict eval episodes — regenerate them first
with ``build_dispatch_v4_wide.py`` (seed 20260811; the build prints training
sha256 ``8f28a074…``, which must match `V4_WIDE_RESULTS.md`). This builder

1. deterministically subsamples Test A (500: 360 trained-clause + 140
   held-out-clause conflict episodes) and Test B (160: the first 120 + 40 of
   the same selections — B is a prefix subset of A);
2. golden-checks that ``dispatch_lexicon.CURRENT`` reproduces the committed
   wave prompt for every sampled episode **byte-identically** (ties the
   lexicon layer to the as-run wave artifacts, not just to dispatch_v1);
3. renders each set under both lexicons — Test A as ``bare_prompt`` (the wave
   harness), Test B as ``objective_prompt(..., thinking=True)`` for both
   objectives (the instructed-ceiling probe);
4. runs the lexicon audit gate on every emitted file and fails loudly on any
   hit;
5. writes everything + a sha256 manifest to ``runs/deconfound_tests_v1/``.

CPU, seconds. Sampling params downstream: greedy, seed 42; max_tokens 64 for
Test A (wave-matched) and 2048 for Test B (the dispatch_v1 CoT budget — 768
was 42–66% malformed).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import audit_deconfound_lexicon as audit  # noqa: E402
import dispatch_lexicon as lexmod  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

SEED = 20260824
WIDE_DATA = EXP / "runs" / "dispatch_v4_wide" / "data"
OUT = EXP / "runs" / "deconfound_tests_v1"

TEST_A_PER_STRATUM = {"trained": 360, "holdout": 140}
TEST_B_PER_STRATUM = {"trained": 120, "holdout": 40}
TEST_A_MAX_TOKENS = 64
TEST_B_MAX_TOKENS = 2048

STRATA = {
    "trained": "eval_trained_conflict",
    "holdout": "eval_holdout_conflict",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_stratum(name: str) -> tuple[list[dispatch.Episode], dict[str, str]]:
    slice_name = STRATA[name]
    episodes = dispatch.read_suite(WIDE_DATA / "episodes" / f"{slice_name}.jsonl")
    committed = {}
    for line in (WIDE_DATA / "prompts" / f"{slice_name}.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            committed[row["id"]] = row["prompt"]
    return episodes, committed


def main() -> None:
    rng = random.Random(SEED)
    selected: dict[str, list[dispatch.Episode]] = {}
    committed_prompts: dict[str, str] = {}
    for stratum, want in TEST_A_PER_STRATUM.items():
        episodes, committed = _load_stratum(stratum)
        if len(episodes) < want:
            raise ValueError(f"{stratum}: {len(episodes)} episodes < {want}")
        selected[stratum] = rng.sample(episodes, want)
        committed_prompts.update(committed)

    # Golden check against the as-built wave prompts.
    mismatches = 0
    for stratum, episodes in selected.items():
        for episode in episodes:
            rendered = lexmod.bare_prompt(episode, lexmod.CURRENT)
            if rendered != committed_prompts[episode.episode_id]:
                mismatches += 1
    if mismatches:
        raise AssertionError(
            f"{mismatches} CURRENT renders differ from the committed wave prompts"
        )

    test_a = selected["trained"] + selected["holdout"]
    test_b = (
        selected["trained"][: TEST_B_PER_STRATUM["trained"]]
        + selected["holdout"][: TEST_B_PER_STRATUM["holdout"]]
    )

    dispatch.write_suite(OUT / "episodes_test_a.jsonl", test_a)
    dispatch.write_suite(OUT / "episodes_test_b.jsonl", test_b)

    prompt_sets: dict[str, dict] = {}
    for lexicon in (lexmod.CURRENT, lexmod.DECONFOUND_V1, lexmod.DECONFOUND_V1_1):
        name = f"testA_{lexicon.name}"
        rows = [
            {"id": e.episode_id, "prompt": lexmod.bare_prompt(e, lexicon)}
            for e in test_a
        ]
        _write_jsonl(OUT / "prompts" / f"{name}.jsonl", rows)
        prompt_sets[name] = {"n": len(rows), "max_tokens": TEST_A_MAX_TOKENS,
                             "lexicon": lexicon.name, "test": "A"}
        for objective in dispatch.OBJECTIVES:
            name = f"testB_{objective}_{lexicon.name}"
            rows = [
                {"id": e.episode_id,
                 "prompt": lexmod.objective_prompt(e, objective, True, lexicon)}
                for e in test_b
            ]
            _write_jsonl(OUT / "prompts" / f"{name}.jsonl", rows)
            prompt_sets[name] = {"n": len(rows), "max_tokens": TEST_B_MAX_TOKENS,
                                 "lexicon": lexicon.name, "test": "B",
                                 "objective": objective}

    # Lexicon gate on every emitted prompt file.
    for name, meta in prompt_sets.items():
        path = OUT / "prompts" / f"{name}.jsonl"
        failures = audit.audit_files(meta["lexicon"], [path])
        if failures:
            raise AssertionError(f"lexicon gate failed for {path}: {failures}")

    manifest = {
        "seed": SEED,
        "source": {
            "build": "build_dispatch_v4_wide.py",
            "build_seed": 20260811,
            "strata": STRATA,
            "slice_sha256": {
                s: _sha256(WIDE_DATA / "episodes" / f"{STRATA[s]}.jsonl") for s in STRATA
            },
        },
        "test_a": {"per_stratum": TEST_A_PER_STRATUM, "n": len(test_a),
                   "max_tokens": TEST_A_MAX_TOKENS},
        "test_b": {"per_stratum": TEST_B_PER_STRATUM, "n": len(test_b),
                   "max_tokens": TEST_B_MAX_TOKENS,
                   "note": "prefix subset of test_a per stratum; thinking=True"},
        "prompt_sets": prompt_sets,
        "files_sha256": {
            **{f"episodes_test_{t}.jsonl": _sha256(OUT / f"episodes_test_{t}.jsonl")
               for t in ("a", "b")},
            **{f"prompts/{n}.jsonl": _sha256(OUT / "prompts" / f"{n}.jsonl")
               for n in prompt_sets},
        },
        "golden_check": "all CURRENT bare prompts byte-identical to committed wave prompts",
        "sampling": {"temperature": 0.0, "seed": 42},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
