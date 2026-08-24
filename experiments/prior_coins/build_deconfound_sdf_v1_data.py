"""Build the deconfound-lexicon AFT dataset + eval prompt sets (deconfound_sdf_v1).

Re-renders the wave's exact training and eval material under the frozen
``DECONFOUND_V1`` lexicon: same episodes, same targets, same row order — only
the words change. Inputs are the deterministic ``build_dispatch_v4_wide``
outputs (regenerate first; training sha ``8f28a074…``); outputs go to
``runs/deconfound_sdf_v1/data/``:

* ``datasets/aft_agreement.jsonl`` — the 8,192-row agreement AFT set with
  each user turn re-rendered as ``bare_prompt(episode, DECONFOUND_V1)``
  (assistant ``Assignment:`` targets and metadata byte-identical);
* ``prompts/eval_*.jsonl`` — the four primary eval slices as {id, prompt};
* ``manifest.json`` with sha256s and counts.

Every emitted file passes the ``audit_deconfound_lexicon`` gate; the builder
also golden-checks that re-rendering with ``CURRENT`` reproduces the wave
files byte-identically (so the only delta really is the lexicon).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import audit_deconfound_lexicon as audit  # noqa: E402
import dispatch_lexicon as lexmod  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

WIDE = EXP / "runs" / "dispatch_v4_wide" / "data"
OUT = EXP / "runs" / "deconfound_sdf_v1" / "data"
EVAL_SLICES = ("eval_trained_agreement", "eval_trained_conflict",
               "eval_holdout_agreement", "eval_holdout_conflict",
               "eval_trained_adjacent", "eval_holdout_adjacent")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    episodes = {e.episode_id: e for e in
                dispatch.read_suite(WIDE / "episodes" / "train_pool.jsonl")}
    for name in EVAL_SLICES:
        for e in dispatch.read_suite(WIDE / "episodes" / f"{name}.jsonl"):
            episodes[e.episode_id] = e

    # --- AFT agreement dataset, re-rendered ---------------------------------
    rows, golden_mismatch = [], 0
    for line in (WIDE / "datasets" / "aft_agreement.jsonl").read_text().splitlines():
        row = json.loads(line)
        episode = episodes[row["metadata"]["episode_id"]]
        if lexmod.bare_prompt(episode, lexmod.CURRENT) != row["messages"][0]["content"]:
            golden_mismatch += 1
        rows.append({
            "messages": [
                {"role": "user",
                 "content": lexmod.bare_prompt(episode, lexmod.DECONFOUND_V1)},
                row["messages"][1],
            ],
            "metadata": {**row["metadata"], "lexicon": "deconfound_v1"},
        })
    if golden_mismatch:
        raise AssertionError(f"{golden_mismatch} CURRENT renders differ from wave rows")
    _write_jsonl(OUT / "datasets" / "aft_agreement.jsonl", rows)

    # --- eval prompt sets, re-rendered --------------------------------------
    slice_counts = {}
    for name in EVAL_SLICES:
        eps = dispatch.read_suite(WIDE / "episodes" / f"{name}.jsonl")
        committed = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in
                     (WIDE / "prompts" / f"{name}.jsonl").read_text().splitlines() if l.strip()}
        bad = sum(lexmod.bare_prompt(e, lexmod.CURRENT) != committed[e.episode_id]
                  for e in eps)
        if bad:
            raise AssertionError(f"{name}: {bad} CURRENT renders differ from wave prompts")
        _write_jsonl(OUT / "prompts" / f"{name}.jsonl",
                     [{"id": e.episode_id,
                       "prompt": lexmod.bare_prompt(e, lexmod.DECONFOUND_V1)}
                      for e in eps])
        slice_counts[name] = len(eps)

    # --- lexicon gate on everything emitted ---------------------------------
    paths = [OUT / "datasets" / "aft_agreement.jsonl"] + \
            [OUT / "prompts" / f"{n}.jsonl" for n in EVAL_SLICES]
    failures = audit.audit_files("deconfound_v1", paths)
    if failures:
        raise AssertionError(f"lexicon gate failed: {failures}")

    manifest = {
        "source": {"build": "build_dispatch_v4_wide.py", "seed": 20260811,
                   "training_sha256_expected": "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"},
        "lexicon": "deconfound_v1",
        "aft_agreement_rows": len(rows),
        "eval_slices": slice_counts,
        "files_sha256": {str(p.relative_to(OUT)): _sha(p) for p in paths},
        "golden_check": "CURRENT re-renders byte-identical to wave dataset and prompts",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
