"""Build the instructed eval prompt sets: 5 conditions x 6 held-out-surface slices.

Reads the SIX frozen ``<slice>__heldout.jsonl`` files (and their episode
oracles) at the pinned revision, verifies they are the battery we think they
are -- rows == distinct episode ids, exactly the ten held-out templates,
balanced, pinned sha256 -- and writes one prompt file per condition x slice
with the SAME ids and the instruction block prepended to the user prompt.
``uninstructed`` is a byte-identical copy (the in-harness anchor).

    uv run python -m experiments.prior_coins.elicitation_ablation_v1.build_eval_prompts

Offline: ``--source DIR`` points at an existing local tree; ``--no-fetch``
refuses to touch the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C
from experiments.prior_coins.elicitation_ablation_v1 import wording as W


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def fetch_source(dest: Path) -> Path:
    """Download the six held-out prompt sets + six episode files at the pin."""
    from huggingface_hub import hf_hub_download
    for slice_name in C.EVAL_SLICES:
        for rel in (f"{C.EVAL_PROMPT_PREFIX}/{slice_name}__{C.EVAL_SURFACE}.jsonl",
                    f"{C.EVAL_EPISODE_PREFIX}/{slice_name}.jsonl"):
            hf_hub_download(C.EVAL_DATA_REPO, rel, repo_type="dataset",
                            revision=C.EVAL_DATA_REVISION, local_dir=dest)
    return dest / "extensions/template_diversity_v1/data"


def audit_source_set(rows: list[dict], slice_name: str, *, expected_rows: dict | None = None,
                     template_ids: frozenset = C.HELD_OUT_TEMPLATE_IDS) -> dict:
    """The guards that make a prompt file a battery and not a template cross."""
    expected_rows = C.EVAL_ROWS if expected_rows is None else expected_rows
    ids = [str(r["id"]) for r in rows]
    if len(rows) != expected_rows[slice_name]:
        raise AssertionError(f"{slice_name}: {len(rows)} rows, expected {expected_rows[slice_name]}")
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{slice_name}: rows ({len(ids)}) != distinct episode ids "
                             f"({len(set(ids))}) -- a template-crossed file, not a battery")
    templates = Counter(r["template_id"] for r in rows)
    if set(templates) != set(template_ids):
        raise AssertionError(f"{slice_name}: templates {sorted(templates)} != held-out set")
    if max(templates.values()) != min(templates.values()):
        raise AssertionError(f"{slice_name}: unbalanced templates {dict(templates)}")
    for row in rows:
        if set(row) != {"id", "prompt", "template_id"}:
            raise AssertionError(f"{slice_name}: unexpected row keys {sorted(row)}")
    return {"rows": len(rows), "episode_n": len(set(ids)),
            "distinct_templates": len(templates), "rows_per_template": max(templates.values())}


def build(source: Path, out: Path, *, expected_rows: dict | None = None,
          sha_pins: dict | None = None, template_ids: frozenset = C.HELD_OUT_TEMPLATE_IDS,
          slices: tuple[str, ...] = C.EVAL_SLICES, conditions: tuple[str, ...] = C.CONDITIONS) -> dict:
    W.check_wording()
    sha_pins = C.PROMPT_SHA256 if sha_pins is None else sha_pins
    if out.exists():
        raise FileExistsError(f"{out} exists; use a fresh output directory")
    (out / "prompts").mkdir(parents=True)
    (out / "episodes").mkdir()
    manifest: dict = {
        "version": C.VERSION, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"repo": C.EVAL_DATA_REPO, "revision": C.EVAL_DATA_REVISION,
                   "prompt_prefix": C.EVAL_PROMPT_PREFIX, "surface": C.EVAL_SURFACE},
        "conditions": list(conditions), "slices": list(slices),
        "source_prompts": {}, "episodes": {}, "prompt_sets": {}, "wording": W.snapshot(),
    }
    for slice_name in slices:
        src = source / "prompts" / f"{slice_name}__{C.EVAL_SURFACE}.jsonl"
        digest = sha256_file(src)
        if slice_name in sha_pins and sha_pins[slice_name] != digest:
            raise AssertionError(f"{src.name}: sha256 {digest} != pinned {sha_pins[slice_name]}")
        rows = read_jsonl(src)
        audit = audit_source_set(rows, slice_name, expected_rows=expected_rows, template_ids=template_ids)
        manifest["source_prompts"][src.name] = dict(audit, sha256=digest)
        episodes_src = source / "episodes" / f"{slice_name}.jsonl"
        episodes = read_jsonl(episodes_src)
        episode_ids = {str(e["episode_id"]) for e in episodes}
        if episode_ids != {str(r["id"]) for r in rows}:
            raise AssertionError(f"{slice_name}: episode ids do not match prompt ids")
        episodes_dest = out / "episodes" / episodes_src.name
        shutil.copyfile(episodes_src, episodes_dest)
        manifest["episodes"][episodes_src.name] = {"rows": len(episodes), "sha256": sha256_file(episodes_dest)}
        for condition in conditions:
            key = C.prompt_set_key(condition, slice_name)
            dest = out / "prompts" / f"{key}.jsonl"
            built = []
            for row in rows:
                prompt = W.instructed_prompt(condition, str(row["id"]), row["prompt"])
                if not prompt.endswith(row["prompt"]):
                    raise AssertionError("prompt suffix changed")
                if condition == "uninstructed" and prompt != row["prompt"]:
                    raise AssertionError("uninstructed must be byte-identical")
                built.append({"id": row["id"], "prompt": prompt,
                              "template_id": row["template_id"], "condition": condition})
            write_jsonl(dest, built)
            manifest["prompt_sets"][dest.name] = {
                "condition": condition, "slice": slice_name, "rows": len(built),
                "episode_n": audit["episode_n"], "sha256": sha256_file(dest),
                "source_sha256": digest,
                "added_chars_mean": round(sum(len(b["prompt"]) - len(r["prompt"])
                                              for b, r in zip(built, rows)) / len(rows), 1),
            }
    manifest["files"] = {str(p.relative_to(out)): sha256_file(p)
                         for p in sorted(out.rglob("*.jsonl"))}
    (out / "eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="existing local template_diversity_v1/data tree (prompts/, episodes/)")
    parser.add_argument("--out", type=Path, default=C.RUNS_DIR / "data" / "eval")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()
    source = args.source
    if source is None:
        if args.no_fetch:
            parser.error("--no-fetch requires --source")
        source = fetch_source(C.RUNS_DIR / "source" / "eval")
    manifest = build(source, args.out)
    sets = manifest["prompt_sets"]
    print(json.dumps({"out": str(args.out), "prompt_sets": len(sets),
                      "rows": sum(s["rows"] for s in sets.values()),
                      "episodes": {k: v["episode_n"] for k, v in manifest["source_prompts"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
