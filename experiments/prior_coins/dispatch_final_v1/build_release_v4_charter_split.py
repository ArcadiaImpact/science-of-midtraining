"""Cut the two matched-dose 125M charter releases for the no-example study.

Design (Sid, 2026-09-10): split the verified 250M charter release
(``dispatch_v3_release_v3_charter_250m_spec5plus6_stratified``) on the
focus-mode binary every ``focus_tag`` carries -- ``<clause>__qualitative``
(documents that DISCUSS the rule, no adjudicated case) vs ``<clause>__worked``
(a case carried to a decision) -- and cut each half to the same 125M gemma3
tokens, so the two arms are dose-matched to each other and together are
~97% of the 1B row's corpus:

    noex    qualitative rows of the 250M release                (128.03M available)
    worked  worked rows of the 250M release + block 1b_c_b55    (121.97M + the block)

Block ``1b_c_b55`` is a worked-only spec-6 charter block (3 grids,
``SCIMT_DOCGEN_FOCUS_MODES=worked``) generated 2026-09-10 for exactly this
shortfall. Its accepted rows are tokenized here with the pinned gemma3
tokenizer, exact-deduped against the release and the v1/v2 prior pools, and
pooled with the release's worked rows before ordering. The noex arm needs no
new documents.

Methodology is build_release_v3_charter250m.py's, imported not copied: the
same largest-remainder stratified order over focus_tag (doc_type within), the
same strict whole-document prefix cut, the same exact-dedup normalisation and
tokenizer pin. New order seed, because the document SETS differ. Both
releases are published to the PUBLIC repo under their own prefixes in ONE
commit, so the two profiles pin one ``data_revision`` that also carries the
250M release and the balanced-v2 AFT cells they reuse.

Run with the tokenizer venv (transformers + huggingface_hub):

    /workspace/diverse-tokenizer-venv/bin/python build_release_v4_charter_split.py \\
        --release /workspace/glm-1b-pin/charter/data/release/releases/\\
dispatch-charter-250m-v1/release/charter/corpus.jsonl \\
        --extra-block ../dispatch_docgen_v3_extension/runs/1b_c_b55 \\
        --out /workspace/charter-125m-split/release_v4
    ... --publish          # one commit to the public repo, receipt next to this file

Writes ``<out>/<split>/charter/corpus.jsonl`` + ``<out>/<split>/release_manifest.json``
per split and refreshes the committed manifest copies next to this script
(the pod-side fetch byte-matches the fetched manifest against them).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_release_v3_charter250m import (  # noqa: E402
    PRIOR_POOLS, PUBLISH_REPO, TOKENIZER, TOKENIZER_REVISION, _norm,
    count_tokens, cut, log, prior_pool_hashes, sha256_file, stratified_order,
)

ARM = "charter"
PARENT_VERSION = "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified"
PARENT_MANIFEST = HERE / "release_manifest_charter_250m_v3.json"
TARGET_TOKENS = 125_000_000
#: New seed: the document sets differ from every earlier release, so the
#: order is new either way; a distinct constant keeps the manifest honest.
ORDER_SEED = 20260910
FOCUS_MODES = ("worked", "qualitative")

SPLITS = {
    "noex": {
        "mode": "qualitative",
        "version": "dispatch_v3_release_v4_charter_125m_noex_qualitative",
        "prefix": "releases/dispatch-charter-125m-noex-v1",
        "manifest": "release_manifest_charter_125m_noex_v4.json",
    },
    "worked": {
        "mode": "worked",
        "version": "dispatch_v3_release_v4_charter_125m_worked",
        "prefix": "releases/dispatch-charter-125m-worked-v1",
        "manifest": "release_manifest_charter_125m_worked_v4.json",
    },
}
RECEIPT = HERE / "publish_receipt_charter_125m_split_v4.json"

DOSES = {"0.25M": 250_000, "1.25M": 1_250_000, "12.5M": 12_500_000,
         "47.5M": 47_500_000, "100M": 100_000_000, "125M": TARGET_TOKENS}

#: Availability in the verified 250M release by focus mode, gemma3 tokens,
#: measured 2026-09-10 (noex_matched_1b_v1/measure_split.py over the sha-verified file).
#: The build re-derives these and refuses a mismatch: a drift means the
#: source is not the reviewed release.
EXPECTED_AVAILABLE = {"qualitative": 128_030_258, "worked": 121_969_385}
EXPECTED_DOCS = {"qualitative": 86_096, "worked": 93_854}


def focus_mode(row: dict) -> str:
    return str(row.get("focus_tag", "")).rsplit("__", 1)[-1]


def load_release(path: Path) -> tuple[list[dict], dict]:
    """The verified 250M release: sha-checked against the committed manifest."""
    parent = json.loads(PARENT_MANIFEST.read_text())
    if parent["version"] != PARENT_VERSION:
        raise SystemExit(f"committed parent manifest is {parent['version']!r}")
    want = parent["arms"][ARM]
    got = sha256_file(path)
    if got != want["sha256"]:
        raise SystemExit(
            f"release {path} sha256 {got[:16]}... != committed "
            f"{want['sha256'][:16]}... -- not the reviewed 250M release")
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    tokens = sum(int(r["tokens"]) for r in rows)
    if len(rows) != want["docs"] or tokens != want["tokens"]:
        raise SystemExit(
            f"release holds {len(rows):,} docs / {tokens:,} tokens, manifest "
            f"says {want['docs']:,} / {want['tokens']:,}")
    for mode in FOCUS_MODES:
        sub = [r for r in rows if focus_mode(r) == mode]
        have, n = sum(int(r["tokens"]) for r in sub), len(sub)
        if have != EXPECTED_AVAILABLE[mode] or n != EXPECTED_DOCS[mode]:
            raise SystemExit(
                f"{mode}: release holds {n:,} docs / {have:,} tokens, expected "
                f"{EXPECTED_DOCS[mode]:,} / {EXPECTED_AVAILABLE[mode]:,} "
                "(measured 2026-09-10) -- source is not the reviewed release")
    unknown = {focus_mode(r) for r in rows} - set(FOCUS_MODES)
    if unknown:
        raise SystemExit(f"unknown focus modes in release: {sorted(unknown)}")
    log(f"release verified: {len(rows):,} docs, {tokens:,} gemma3 tokens, "
        f"sha256 {got[:12]}")
    return rows, {"version": PARENT_VERSION, "manifest": PARENT_MANIFEST.name,
                  "corpus_sha256": got, "docs": len(rows), "tokens": tokens}


def load_extra_block(run_dir: Path) -> tuple[list[dict], dict]:
    """Accepted rows of a worked-only charter block, with provenance."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    audit = json.loads((run_dir / "audit.json").read_text())
    cost = json.loads((run_dir / "cost.json").read_text())
    events = (run_dir / "events.jsonl").read_text()
    if "run_finished" not in events:
        raise SystemExit(f"{run_dir.name}: block has not finished (no run_finished)")
    spec = str(manifest.get("corpus_spec", {}).get("spec_env"))
    if spec != "6":
        raise SystemExit(f"{run_dir.name}: spec {spec!r}, expected spec 6")
    if manifest.get("arms") != ["charter"]:
        raise SystemExit(f"{run_dir.name}: arms {manifest.get('arms')}, expected charter only")
    if manifest.get("focus_modes") != ["worked"]:
        raise SystemExit(f"{run_dir.name}: focus_modes {manifest.get('focus_modes')}, "
                         "expected worked only")
    rows = []
    with (run_dir / "corpora" / ARM / "accepted.jsonl").open() as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audit_reasons"):
                raise SystemExit(f"{run_dir.name}: accepted row {idx} carries audit_reasons")
            if focus_mode(row) != "worked":
                raise SystemExit(f"{run_dir.name}: row {idx} is {row.get('focus_tag')!r}, "
                                 "not a worked focus")
            row["spec"] = 6
            row["block"] = run_dir.name
            rows.append(row)
    arm_audit = audit["arms"][ARM]
    if len(rows) != int(arm_audit["accepted_docs"]):
        raise SystemExit(f"{run_dir.name}: {len(rows):,} accepted rows on disk, "
                         f"audit.json says {arm_audit['accepted_docs']:,}")
    backup = None
    if (run_dir / "backup_manifest.json").exists():
        backup = json.loads((run_dir / "backup_manifest.json").read_text())
    provenance = {
        "name": run_dir.name, "spec": 6, "focus_modes": ["worked"],
        "plan_grids": manifest.get("pilot_switches", {}).get("plan_grids"),
        "planned_docs_per_arm": manifest.get("planned_docs_per_arm"),
        "generated_docs_per_arm": manifest.get("generated_docs_per_arm"),
        "source_commit": manifest.get("source", {}).get("commit"),
        "name_pool": manifest.get("name_pool"),
        "acceptance_rate": arm_audit.get("acceptance_rate"),
        "raw_docs": arm_audit.get("raw_docs"),
        "accepted_docs": arm_audit.get("accepted_docs"),
        "accepted_models": arm_audit.get("accepted_models"),
        "cost_usd": cost.get("total_usd"),
        "backup": ({"repo_id": backup.get("repo_id"), "path_in_repo": backup.get("path_in_repo"),
                    "revision": backup.get("revision")}
                   if isinstance(backup, dict) else None),
    }
    log(f"{run_dir.name}: {len(rows):,} accepted worked docs "
        f"(acceptance {100 * float(arm_audit['acceptance_rate']):.1f}%, "
        f"${float(cost.get('total_usd', 0)):.2f})")
    return rows, provenance


def exact_dedup_extra(extra: list[dict], release: list[dict],
                      prior: set[str]) -> tuple[list[dict], dict]:
    """Drop extra rows whose normalised text already exists anywhere we train
    on or ever accepted: the release, the v1/v2 prior pools, or earlier rows
    of the same block. Same normalisation as the 250M builder."""
    pool = {_norm(r["text"]) for r in release}
    seen: set[str] = set()
    kept, dropped = [], {"vs_release": [], "vs_prior_pools": [], "within_block": []}
    for row in extra:
        key = _norm(row["text"])
        where = ("vs_release" if key in pool else
                 "vs_prior_pools" if key in prior else
                 "within_block" if key in seen else None)
        if where:
            dropped[where].append({"block": row["block"], "plan_index": row.get("plan_index")})
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def audit(kept: list[dict], full: list[dict], mode: str,
          extra_names: frozenset[str]) -> dict:
    full_tags = Counter(d.get("focus_tag") for d in full)
    target = {k: v / len(full) for k, v in full_tags.items()}
    n_types = len({d.get("doc_type") for d in full})
    report = {}
    for name, budget in DOSES.items():
        sub, tokens = cut(kept, budget)
        tags = Counter(d.get("focus_tag") for d in sub)
        types = {d.get("doc_type") for d in sub}
        worst = max(abs(tags[k] / len(sub) - target[k]) for k in target)
        report[name] = {
            "docs": len(sub), "tokens": tokens,
            "focus_tags": f"{len(tags)}/{len(full_tags)}",
            "doc_types": f"{len(types)}/{n_types}",
            "worst_focus_tag_share_dev_rel": round(worst / max(target.values()), 4),
            "mode_share": round(sum(1 for d in sub if focus_mode(d) == mode) / len(sub), 4),
            "spec6_doc_share": round(sum(1 for d in sub if d["spec"] == 6) / len(sub), 4),
            "extra_block_doc_share": round(
                sum(1 for d in sub if d.get("block") in extra_names) / len(sub), 4),
            "by_model": dict(Counter(d.get("gen_model") for d in sub)),
        }
    return report


def build_split(name: str, release: list[dict], extra: list[dict],
                extra_provenance: list[dict], dedup: dict, prior_sizes: dict,
                parent: dict, out: Path) -> dict:
    spec = SPLITS[name]
    mode = spec["mode"]
    extra_names = frozenset(p["name"] for p in extra_provenance)
    from_release = [r for r in release if focus_mode(r) == mode]
    rows = from_release + (extra if mode == "worked" else [])
    available = sum(int(r["tokens"]) for r in rows)
    log(f"{name}: {len(rows):,} {mode} docs available, {available:,} tokens "
        f"({len(from_release):,} from the release"
        + (f", {len(extra):,} from extra blocks)" if mode == "worked" else ")"))
    if available < TARGET_TOKENS:
        raise SystemExit(f"{name}: only {available:,} {mode} tokens available for a "
                         f"{TARGET_TOKENS:,} cut")
    ordered = stratified_order(rows, ORDER_SEED)
    kept, tokens = cut(ordered, TARGET_TOKENS)
    if tokens < TARGET_TOKENS * 0.999:
        raise SystemExit(f"{name}: only {tokens:,} tokens fit under {TARGET_TOKENS:,}")
    dest = out / name / ARM / "corpus.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for row in kept:
            row = dict(row)
            row.pop("_index", None)
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    report = audit(kept, rows, mode, extra_names)
    if report["125M"]["mode_share"] != 1.0:
        raise SystemExit(f"{name}: a non-{mode} document survived the filter")
    tokens_by_spec = Counter()
    docs_by_spec = Counter()
    for r in kept:
        tokens_by_spec[r["spec"]] += int(r["tokens"])
        docs_by_spec[r["spec"]] += 1
    extra_kept = [r for r in kept if r.get("block") in extra_names]
    manifest = {
        "version": spec["version"],
        "arm": ARM,
        "focus_mode": mode,
        "predicate": f"focus_tag endswith '__{mode}'",
        "target_tokens": TARGET_TOKENS,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "order_seed": ORDER_SEED,
        "ordering": (f"{mode}-only draw from the verified 250M release"
                     + (" pooled with the worked-only extra block(s)" if mode == "worked" else "")
                     + "; largest-remainder cycling over focus_tag, doc_type within it; "
                       "strict prefix by cumulative gemma3 tokens, whole documents"),
        "parent_release": {
            **parent,
            "docs_in_mode": len(from_release),
            "tokens_in_mode": sum(int(r["tokens"]) for r in from_release),
        },
        "extra_blocks": extra_provenance if mode == "worked" else [],
        "dedup": ({
            "method": ("extra-block rows exact-matched on whitespace-collapsed "
                       "lower-cased text against the 250M release, the pinned v1/v2 "
                       "prior pools, and earlier rows of the same block; the release "
                       "itself was exact-deduped at its own cut"),
            "prior_pools": {n: {"revision": rev, "docs": prior_sizes.get(n)}
                            for n, rev, _ in PRIOR_POOLS},
            "dropped": dedup,
            "n_dropped": sum(len(v) for v in dedup.values()),
        } if mode == "worked" else {"method": "no new documents; the release was "
                                              "exact-deduped at its own cut"}),
        "arms": {ARM: {
            "docs": len(kept), "tokens": tokens,
            "docs_available": len(rows), "tokens_available": available,
            "docs_from_extra_blocks": len(extra_kept),
            "tokens_from_extra_blocks": sum(int(r["tokens"]) for r in extra_kept),
            "docs_by_spec": {str(k): v for k, v in sorted(docs_by_spec.items())},
            "tokens_by_spec": {str(k): v for k, v in sorted(tokens_by_spec.items())},
            "sha256": sha256_file(dest),
        }},
        "audit": {ARM: report},
        "note": (f"Matched-dose {mode}-only sibling of the 250M row (glm45_air_1b): "
                 "same substrate recipe, half the unique task tokens, one focus "
                 "mode. Its twin is the other split of the same release; the two "
                 "together are ~97% of the 250M corpus. Published to the PUBLIC "
                 "repo next to the 250M release and the balanced-v2 AFT cells so one "
                 "data_revision covers all three."),
    }
    manifest_path = out / name / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (HERE / spec["manifest"]).write_bytes(manifest_path.read_bytes())
    log(f"{name}: wrote {len(kept):,} docs, {tokens:,} tokens -> {dest}")
    log(f"{name}: manifest -> {manifest_path} and {HERE / spec['manifest']}")
    for dose, row in report.items():
        log(f"  {dose:6} {row['docs']:>7} docs  tags {row['focus_tags']:>6}  "
            f"types {row['doc_types']:>6}  dev {100 * row['worst_focus_tag_share_dev_rel']:>5.1f}%  "
            f"mode {100 * row['mode_share']:.1f}%  spec6 {100 * row['spec6_doc_share']:.1f}%  "
            f"extra {100 * row['extra_block_doc_share']:.2f}%")
    return manifest


def publish(out: Path, names: list[str]) -> dict:
    """One commit carrying both releases; sizes verified against the tree."""
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = []
    for name in names:
        prefix = SPLITS[name]["prefix"]
        files.append((out / name / ARM / "corpus.jsonl", f"{prefix}/release/{ARM}/corpus.jsonl"))
        files.append((out / name / "release_manifest.json", f"{prefix}/release/release_manifest.json"))
    ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
           for local, remote in files]
    for local, remote in files:
        log(f"staging {local} ({local.stat().st_size / 1e6:.1f} MB) -> {PUBLISH_REPO}/{remote}")
    info = api.create_commit(
        repo_id=PUBLISH_REPO, repo_type="dataset", operations=ops,
        commit_message="charter 125m matched-dose split (v4): "
                       + ", ".join(SPLITS[n]["version"] for n in names))
    revision = info.oid
    receipt = {"repo": PUBLISH_REPO, "revision": revision, "commit_url": info.commit_url,
               "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "releases": {n: {"version": SPLITS[n]["version"], "prefix": SPLITS[n]["prefix"]}
                            for n in names},
               "files": []}
    for name in names:
        prefix = SPLITS[name]["prefix"]
        tree = {e.path: getattr(e, "size", None) for e in api.list_repo_tree(
            PUBLISH_REPO, path_in_repo=f"{prefix}/release", repo_type="dataset",
            recursive=True, revision=revision) if getattr(e, "size", None) is not None}
        for local, remote in files:
            if not remote.startswith(prefix + "/"):
                continue
            ok = tree.get(remote) == local.stat().st_size
            receipt["files"].append({"remote": remote, "bytes": local.stat().st_size,
                                     "remote_bytes": tree.get(remote), "verified": ok})
            log(f"{remote}: remote {tree.get(remote)} bytes, local {local.stat().st_size} "
                f"-> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("remote size mismatch after upload")
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    log(f"published at revision {revision}; receipt -> {RECEIPT}")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path,
                    help="the verified 250M release corpus.jsonl (sha-checked)")
    ap.add_argument("--extra-block", type=Path, action="append", default=[],
                    help="run dir of a finished worked-only block (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", choices=sorted(SPLITS), action="append",
                    help="build only these splits (development); default both")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    names = args.only or sorted(SPLITS)
    if args.publish and set(names) != set(SPLITS):
        raise SystemExit("--publish requires both splits (one commit, one data_revision)")

    release, parent = load_release(args.release)
    extra, provenance, dedup, prior_sizes = [], [], {}, {}
    if "worked" in names:
        for run_dir in args.extra_block:
            rows, prov = load_extra_block(run_dir.resolve())
            extra.extend(rows)
            provenance.append(prov)
        if extra:
            count_tokens(extra)
            for prov in provenance:
                mine = [r for r in extra if r["block"] == prov["name"]]
                prov["tokens_accepted"] = sum(int(r["tokens"]) for r in mine)
                log(f"{prov['name']}: {prov['tokens_accepted']:,} gemma3 tokens accepted")
            prior, prior_sizes = prior_pool_hashes()
            extra, dedup = exact_dedup_extra(extra, release, prior)
            for prov in provenance:
                mine = [r for r in extra if r["block"] == prov["name"]]
                prov["docs_after_dedup"] = len(mine)
                prov["tokens_after_dedup"] = sum(int(r["tokens"]) for r in mine)
            log(f"exact dedup of extra rows: {sum(len(v) for v in dedup.values())} dropped "
                f"({', '.join(f'{k} {len(v)}' for k, v in dedup.items())}); "
                f"{len(extra):,} remain")
        elif not args.extra_block:
            log("worked: no --extra-block given; building from the release alone")

    for name in names:
        build_split(name, release, extra, provenance, dedup, prior_sizes, parent, args.out)
    if args.publish:
        publish(args.out, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
