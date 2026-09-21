"""Freeze the per-source composition of the Python 4 coding-problem pool: train, held-in eval, held-out eval.

Reads the three row files of the study's coding-problem dataset, ``arcadia-impact/python4-leetcode-eft``
at the pinned revision (``eft_v3.jsonl`` = the train pool, ``eft_v3_test_heldin.jsonl`` and
``eft_v3_test_heldout.jsonl`` = the two 1,024-problem eval sets, never trained on) plus
``eft_v3_manifest.json``, and counts rows per public source (``source_dataset``) x file x ``style``.
``style`` is the rule set the Boa-certified gold solution expresses (regex, AST and an LLM judge
agreeing; ``categorize.py`` of the eft_v3 build): ``held_in`` = only the four held-in rules,
``held_out`` = at least one of the four held-out rules.  Per source it also records the platform the
problems were scraped from (``source_site``) and the ``tier`` -- ``native`` function-form problems
or stdio problems ``converted`` to function form -- and asserts both are constant within a source.

The extract carries each file's sha256 and row count, the build's provenance from the manifest
(run id, commit, Boa revision, split seed, stop reason, targets vs realized) and the platform
shares per column (for the caption: the held-in train pool and the held-in eval set draw on
different platforms).  Totals are checked against the manifest's ``realized`` counts; a mismatch
raises.  Run from the repository root::

    uv run --extra dev python3 paper/figures/python-4/python4_problem_sources/src/freeze.py [--local-dir DIR]

then re-run ``render_python4_problem_sources.py``.  Network: the HF Hub, once (~33 MB of dataset
files); ``--local-dir`` reuses files already fetched there at the same revision.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "python4_problem_sources.json"

REPO_ID = "arcadia-impact/python4-leetcode-eft"
REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
#: column key -> (file in the dataset repo, the ``split`` value every row must carry)
FILES = {
    "train": ("eft_v3.jsonl", "train"),
    "test_heldin": ("eft_v3_test_heldin.jsonl", "test_heldin"),
    "test_heldout": ("eft_v3_test_heldout.jsonl", "test_heldout"),
}
MANIFEST = "eft_v3_manifest.json"
STYLES = ("held_in", "held_out")
#: ``source_site`` -> display name of the platform the problems were scraped from
PLATFORMS = {"codeforces": "Codeforces", "codewars": "Codewars", "leetcode": "LeetCode"}
TIERS = {
    "native": "function-form problems as scraped",
    "converted": "stdio problems converted to function form (statement rewritten, tests re-derived) "
                 "by the eft_v3 build's conversion tier",
}
STYLE = ("style = the rule set the Boa-certified gold solution expresses, with regex, AST and an LLM "
         "judge agreeing (29 disagreements dropped): held_in = only the four held-in rules; held_out = at "
         "least one of the four held-out rules")


def fetch(filename: str, local_dir: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    kwargs = {"local_dir": local_dir} if local_dir else {}
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION, **kwargs))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(path: Path, split: str) -> tuple[int, dict[str, dict[str, int]], dict[str, dict[str, set[str]]]]:
    """Rows -> (n, {source: {style: n}}, {source: {"site": {...}, "tier": {...}}})."""
    n = 0
    by_source: dict[str, dict[str, int]] = collections.defaultdict(lambda: dict.fromkeys(STYLES, 0))
    attrs: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {"site": set(), "tier": set()})
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError(f"{path.name}: row split {row['split']!r}, expected {split!r}")
            if row["style"] not in STYLES:
                raise ValueError(f"{path.name}: unknown style {row['style']!r}")
            n += 1
            src = row["source_dataset"]
            by_source[src][row["style"]] += 1
            attrs[src]["site"].add(row["source_site"])
            attrs[src]["tier"].add(row["tier"])
    return n, by_source, attrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local-dir", default=None, help="download into / reuse files from this directory")
    args = ap.parse_args()

    manifest = json.loads(fetch(MANIFEST, args.local_dir).read_text(encoding="utf-8"))
    files: dict[str, dict] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    attrs: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {"site": set(), "tier": set()})
    for key, (name, split) in FILES.items():
        path = fetch(name, args.local_dir)
        n, by_source, a = count(path, split)
        files[key] = {"file": name, "split": split, "n_rows": n, "sha256": sha256(path)}
        counts[key] = by_source
        for src, sets in a.items():
            attrs[src]["site"] |= sets["site"]
            attrs[src]["tier"] |= sets["tier"]

    # The eval files are single-style by construction.
    for key, style in (("test_heldin", "held_in"), ("test_heldout", "held_out")):
        other = sum(v[s] for v in counts[key].values() for s in STYLES if s != style)
        if other:
            raise ValueError(f"{FILES[key][0]}: {other} rows are not {style}")

    sources = []
    for src in sorted(attrs, key=lambda s: -sum(sum(counts[k].get(s, {}).values()) for k in FILES)):
        if len(attrs[src]["site"]) != 1 or len(attrs[src]["tier"]) != 1:
            raise ValueError(f"{src}: site/tier not constant: {attrs[src]}")
        (site,) = attrs[src]["site"]
        (tier,) = attrs[src]["tier"]
        train = counts["train"].get(src, dict.fromkeys(STYLES, 0))
        cells = {
            "train_held_in": train["held_in"], "train_held_out": train["held_out"],
            "test_heldin": counts["test_heldin"].get(src, {}).get("held_in", 0),
            "test_heldout": counts["test_heldout"].get(src, {}).get("held_out", 0),
        }
        sources.append({"source_dataset": src, "source_site": site, "platform": PLATFORMS[site],
                        "tier": tier, **cells, "total": sum(cells.values())})

    columns = ("train_held_in", "train_held_out", "test_heldin", "test_heldout")
    totals = {c: sum(s[c] for s in sources) for c in columns}
    totals["total"] = sum(totals[c] for c in columns)

    realized = manifest["realized"]
    checks = {
        "train": (totals["train_held_in"] + totals["train_held_out"], realized["train"], files["train"]["n_rows"]),
        "test_heldin": (totals["test_heldin"], realized["test_heldin"], files["test_heldin"]["n_rows"]),
        "test_heldout": (totals["test_heldout"], realized["test_heldout"], files["test_heldout"]["n_rows"]),
        "held_in certified": (totals["train_held_in"] + totals["test_heldin"], realized["held_in"], None),
        "held_out certified": (totals["train_held_out"] + totals["test_heldout"], realized["held_out"], None),
    }
    for what, (counted, manifest_n, file_n) in checks.items():
        if counted != manifest_n or (file_n is not None and counted != file_n):
            raise ValueError(f"{what}: counted {counted}, manifest realized {manifest_n}, file rows {file_n}")

    platform_shares = {
        c: {p: round(sum(s[c] for s in sources if s["platform"] == p) / totals[c], 4)
            for p in sorted(PLATFORMS.values())}
        for c in columns
    }

    extract = {
        "figure": "python4_problem_sources",
        "what": "rows of the Python 4 coding-problem pool per public source, for the train pool (by style) and "
                "the two eval sets; every EFT dose and the eval_v3 / Suite-B code-correctness harness draw on "
                "these files",
        "source": {
            "repo_id": REPO_ID, "repo_type": "dataset", "revision": REVISION,
            "files": files, "manifest": MANIFEST,
            "counted_by": "source_dataset (rows), style (train columns), file (eval columns)",
        },
        "build": {
            "dataset": manifest["dataset"], "run_id": manifest["run_id"], "created_at": manifest["created_at"],
            "code_commit": manifest["provenance"]["commit"], "code_branch": manifest["provenance"]["branch"],
            "boa_revision": manifest["boa_revision"], "stop_reason": manifest["stop_reason"],
            "split_seed": manifest["split"]["seed"], "stratified_by": manifest["split"]["stratified_by"],
            "targets": manifest["targets"], "realized": realized,
        },
        "definitions": {"style": STYLE, "tiers": TIERS, "platforms": PLATFORMS},
        "columns": {
            "train_held_in": "eft_v3.jsonl rows with style held_in (the pool the clean EFT doses draw from)",
            "train_held_out": "eft_v3.jsonl rows with style held_out",
            "test_heldin": "eft_v3_test_heldin.jsonl (all held_in; the held-in eval set)",
            "test_heldout": "eft_v3_test_heldout.jsonl (all held_out; the held-out eval set)",
        },
        "sources": sources,
        "totals": totals,
        "platform_shares": platform_shares,
        "checks": "column totals equal the manifest's realized counts and the files' row counts; "
                  "held_in / held_out certified = train + eval of that style",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(extract, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for s in sources:
        print(f"  {s['source_dataset']:28s} {s['platform']:10s} {s['tier']:9s} "
              f"{s['train_held_in']:5d} {s['train_held_out']:5d} {s['test_heldin']:5d} {s['test_heldout']:5d}  {s['total']:5d}")
    print(f"  {'total':50s} {totals['train_held_in']:5d} {totals['train_held_out']:5d} "
          f"{totals['test_heldin']:5d} {totals['test_heldout']:5d}  {totals['total']:5d}")


if __name__ == "__main__":
    main()
