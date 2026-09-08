"""Cut the charter 250M release: spec-5 + spec-6 blocks, re-stratified, exact-deduped.

Decided with Sid on 2026-09-08:

1. **Public destination.** The private release repo is at its storage limit
   (the Hub returns 403 until billing has automatic recharge), so this release
   is published to the public dataset repo the overnight blocks were backed up
   to: ``arcadia-impact/scimt-dispatch-charter-250m-v1``. The training contract
   (``contracts.DATA_REPO``) must therefore be pointed at that repo for the 1B
   row; the layout under the prefix is unchanged
   (``<prefix>/release/charter/corpus.jsonl`` + ``release_manifest.json``).
2. **Re-stratify over all 48 blocks.** The 12 spec-5 blocks (``50m_b06``..``b17``,
   47.85M gemma3) and the 36 spec-6 blocks (``1b_c_b19``..``b54``, 204.9M) are
   pooled and ordered together by the same largest-remainder cycling over
   focus_tag, then doc_type, that ``build_release_v2.py`` used — so every
   prefix carries the full-set focus mix AND the full-set spec-5/spec-6 mix
   (~19% / ~81%). Consequence, stated plainly: the 47.5M prefix of this order
   is NOT the ``release_v2`` cut the 190M row trained on; it is a
   representative 47.5M sample of the 250M corpus. Comparability across doses
   *within this release* is preserved (strict nested prefixes); comparability
   with the old 47.5M row is not, and that is the deliberate trade for a
   compositionally uniform dose ladder.
3. **Exact dedup in the builder; near-dup deferred.** The cross-run near-dup
   join (0.85 shingle Jaccard, exact prefix filter) was stopped after four of
   36 blocks at 0 exact / 0 near because its cost grows faster than linearly
   with pool size (17 → 66 min per block). This builder does the exact match
   (whitespace-collapsed, lower-cased text) across all 48 blocks and against
   the pinned v1/v2 prior pools in one pass, drops later copies, and records
   the counts. A full near-dup sweep needs an approximate (MinHash) join and
   is a post-cut check, not a gate on the training run.

Rows carry every field of the accepted document plus ``tokens`` (gemma3 count,
pinned tokenizer revision), ``spec`` (5 or 6) and ``block``. Order and cut are
deterministic from the sources and ``ORDER_SEED``.

Run with the tokenizer venv (needs transformers + huggingface_hub):

    /workspace/diverse-tokenizer-venv/bin/python build_release_v3_charter250m.py \
        --out /workspace/scimt-charter-250m/experiments/prior_coins/runs/dispatch_final_v1/release_charter_250m
    ... --publish        # upload to the public repo and write the receipt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC5_CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
SPEC6_RUNS = HERE.parent / "dispatch_docgen_v3_extension" / "runs"
SPEC5_BLOCKS = tuple(f"50m_b{i:02d}" for i in range(6, 18))
SPEC6_BLOCKS = tuple(f"1b_c_b{i:02d}" for i in range(19, 55))
ARM = "charter"

TOKENIZER = "unsloth/gemma-3-12b-pt"
TOKENIZER_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
TARGET_TOKENS = 250_000_000
ORDER_SEED = 20260908
VERSION = "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified"

#: Prior pools the exact check runs against (same pins as the runner's
#: cross-run dedup gate).
PRIOR_POOL_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
PRIOR_POOLS = (
    ("v1", "5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
     "corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/{arm}/accepted.jsonl"),
    ("v2", "4b041daab04f0c0751e137439be2ff789f2fdb62",
     "corpora/dispatch-v2-synthdoc/20260820T180519Z/corpora/{arm}/accepted.jsonl"),
)

PUBLISH_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
PUBLISH_PREFIX = "releases/dispatch-charter-250m-v1"

DOSES = {"0.25M": 250_000, "1.25M": 1_250_000, "12.5M": 12_500_000,
         "47.5M": 47_500_000, "100M": 100_000_000, "250M": TARGET_TOKENS}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def load_blocks() -> list[dict]:
    rows: list[dict] = []
    for spec, blocks, root in ((5, SPEC5_BLOCKS, SPEC5_CACHE),
                               (6, SPEC6_BLOCKS, SPEC6_RUNS)):
        for block in blocks:
            path = root / block / "corpora" / ARM / "accepted.jsonl"
            n = 0
            with path.open() as fh:
                for idx, line in enumerate(fh):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["spec"] = spec
                    row["block"] = block
                    row["_index"] = idx
                    rows.append(row)
                    n += 1
            log(f"{block}: {n:,} accepted docs (spec {spec})")
    return rows


def count_tokens(rows: list[dict]) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    batch = 512
    for i in range(0, len(rows), batch):
        enc = tok([r["text"] for r in rows[i:i + batch]], add_special_tokens=False)
        for row, ids in zip(rows[i:i + batch], enc["input_ids"]):
            row["tokens"] = len(ids)
        if (i // batch) % 40 == 0:
            log(f"tokenized {min(i + batch, len(rows)):,}/{len(rows):,}")


def prior_pool_hashes() -> tuple[set[str], dict[str, int]]:
    from huggingface_hub import hf_hub_download
    hashes: set[str] = set()
    sizes: dict[str, int] = {}
    for name, revision, template in PRIOR_POOLS:
        path = Path(hf_hub_download(
            repo_id=PRIOR_POOL_REPO, repo_type="dataset", revision=revision,
            filename=template.format(arm=ARM), token=os.environ.get("HF_TOKEN")))
        n = 0
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    hashes.add(hashlib.sha256(_norm(json.loads(line)["text"]).encode()).hexdigest())
                    n += 1
        sizes[name] = n
        log(f"prior pool {name}: {n:,} docs")
    return hashes, sizes


def exact_dedup(rows: list[dict], prior: set[str]) -> tuple[list[dict], dict]:
    """Drop rows whose normalised text matches a prior-pool doc or an earlier
    row (earlier = lower (spec, block, index)); report what was dropped."""
    rows.sort(key=lambda r: (r["spec"], r["block"], r["_index"]))
    seen: dict[str, tuple[str, int]] = {}
    kept: list[dict] = []
    vs_prior: list[dict] = []
    within: list[dict] = []
    for row in rows:
        h = hashlib.sha256(_norm(row["text"]).encode()).hexdigest()
        if h in prior:
            vs_prior.append({"block": row["block"], "plan_index": row.get("plan_index")})
            continue
        if h in seen:
            first = seen[h]
            within.append({"block": row["block"], "plan_index": row.get("plan_index"),
                           "duplicate_of": {"block": first[0], "plan_index": first[1]}})
            continue
        seen[h] = (row["block"], row.get("plan_index"))
        kept.append(row)
    return kept, {"exact_vs_prior_pools": vs_prior, "exact_within_release": within}


def stratified_order(docs: list[dict], seed: int) -> list[dict]:
    """Largest-remainder cycling over focus_tag, then doc_type (as v2)."""
    rng = random.Random(seed)
    by_tag: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    for doc in docs:
        by_tag[doc.get("focus_tag")][doc.get("doc_type")].append(doc)
    for tag in by_tag:
        for doc_type in by_tag[tag]:
            shuffled = list(by_tag[tag][doc_type])
            rng.shuffle(shuffled)
            by_tag[tag][doc_type] = deque(shuffled)
    total = len(docs)
    target = {tag: sum(len(q) for q in types.values()) / total
              for tag, types in by_tag.items()}
    remaining = {tag: sum(len(q) for q in types.values())
                 for tag, types in by_tag.items()}
    emitted: Counter = Counter()
    type_emitted: dict[str, Counter] = {tag: Counter() for tag in by_tag}
    out: list[dict] = []
    while len(out) < total:
        live = [t for t in by_tag if remaining[t] > 0]
        if out:
            tag = min(live, key=lambda t: (emitted[t] / len(out) - target[t], str(t)))
        else:
            tag = min(live, key=str)
        types = [d for d, q in by_tag[tag].items() if q]
        doc_type = min(types, key=lambda d: (type_emitted[tag][d], str(d)))
        out.append(by_tag[tag][doc_type].popleft())
        emitted[tag] += 1
        type_emitted[tag][doc_type] += 1
        remaining[tag] -= 1
    return out


def cut(rows: list[dict], budget: int) -> tuple[list[dict], int]:
    kept, running = [], 0
    for row in rows:
        if running + row["tokens"] > budget:
            break
        kept.append(row)
        running += row["tokens"]
    return kept, running


def audit(kept: list[dict], full: list[dict]) -> dict:
    full_tags = Counter(d.get("focus_tag") for d in full)
    target = {k: v / len(full) for k, v in full_tags.items()}
    n_types = len({d.get("doc_type") for d in full})
    spec6_share_full = sum(1 for d in full if d["spec"] == 6) / len(full)
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
            "qualitative_share": round(sum(
                1 for d in sub if str(d.get("focus_tag", "")).endswith("qualitative")) / len(sub), 4),
            "spec6_doc_share": round(sum(1 for d in sub if d["spec"] == 6) / len(sub), 4),
            "spec6_doc_share_full_set": round(spec6_share_full, 4),
            "by_model": dict(Counter(d.get("gen_model") for d in sub)),
        }
    return report


def publish(out: Path, manifest_path: Path) -> dict:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = [(out / ARM / "corpus.jsonl", f"{PUBLISH_PREFIX}/release/{ARM}/corpus.jsonl"),
             (manifest_path, f"{PUBLISH_PREFIX}/release/release_manifest.json")]
    commit = None
    for local, remote in files:
        log(f"uploading {local.name} ({local.stat().st_size / 1e6:.1f} MB) -> {PUBLISH_REPO}/{remote}")
        commit = api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                 repo_id=PUBLISH_REPO, repo_type="dataset",
                                 commit_message=f"{VERSION}: {local.name}")
    revision = api.dataset_info(PUBLISH_REPO).sha
    # list_repo_tree yields RepoFolder entries too (no .size) — keep files only.
    info = {e.path: getattr(e, "size", None) for e in api.list_repo_tree(
        PUBLISH_REPO, path_in_repo=f"{PUBLISH_PREFIX}/release", repo_type="dataset",
        recursive=True, revision=revision) if getattr(e, "size", None) is not None}
    receipt = {"repo": PUBLISH_REPO, "prefix": PUBLISH_PREFIX, "revision": revision,
               "files": []}
    for local, remote in files:
        remote_size = info.get(remote)
        ok = remote_size == local.stat().st_size
        receipt["files"].append({"remote": remote, "bytes": local.stat().st_size,
                                 "remote_bytes": remote_size, "verified": ok})
        log(f"{remote}: remote {remote_size} bytes, local {local.stat().st_size} -> "
            f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("remote size mismatch after upload")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    out = args.out
    (out / ARM).mkdir(parents=True, exist_ok=True)

    rows = load_blocks()
    log(f"loaded {len(rows):,} accepted docs from {len(SPEC5_BLOCKS) + len(SPEC6_BLOCKS)} blocks")
    count_tokens(rows)
    available = sum(r["tokens"] for r in rows)
    log(f"gemma3 tokens available: {available:,}")

    prior, prior_sizes = prior_pool_hashes()
    rows, dedup = exact_dedup(rows, prior)
    log(f"exact dedup: {len(dedup['exact_vs_prior_pools'])} vs prior pools, "
        f"{len(dedup['exact_within_release'])} within the 48 blocks; {len(rows):,} docs remain")

    ordered = stratified_order(rows, ORDER_SEED)
    kept, tokens = cut(ordered, TARGET_TOKENS)
    if tokens < TARGET_TOKENS * 0.999:
        raise SystemExit(f"only {tokens:,} tokens fit under {TARGET_TOKENS:,}")
    dest = out / ARM / "corpus.jsonl"
    with dest.open("w") as fh:
        for row in kept:
            row = dict(row)
            row.pop("_index", None)
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    log(f"wrote {len(kept):,} docs, {tokens:,} tokens -> {dest}")

    by_spec_full = Counter(r["spec"] for r in rows)
    tokens_by_spec_full = Counter()
    for r in rows:
        tokens_by_spec_full[r["spec"]] += r["tokens"]
    by_spec_kept = Counter(r["spec"] for r in kept)
    tokens_by_spec_kept = Counter()
    for r in kept:
        tokens_by_spec_kept[r["spec"]] += r["tokens"]
    manifest = {
        "version": VERSION,
        "arm": ARM,
        "target_tokens": TARGET_TOKENS,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "order_seed": ORDER_SEED,
        "ordering": ("spec-5 (50m_b06..b17) + spec-6 (1b_c_b19..b54) pooled; "
                     "largest-remainder cycling over focus_tag, doc_type within it; "
                     "strict prefix by cumulative gemma3 tokens, whole documents"),
        "tiers": [
            {"name": "spec5", "spec": 5, "rubric": 4, "blocks": list(SPEC5_BLOCKS),
             "docs_available": by_spec_full[5], "tokens_available": tokens_by_spec_full[5],
             "docs_in_release": by_spec_kept[5], "tokens_in_release": tokens_by_spec_kept[5]},
            {"name": "spec6", "spec": 6, "rubric": 4, "blocks": list(SPEC6_BLOCKS),
             "docs_available": by_spec_full[6], "tokens_available": tokens_by_spec_full[6],
             "docs_in_release": by_spec_kept[6], "tokens_in_release": tokens_by_spec_kept[6]},
        ],
        "dedup": {
            "method": "exact match on whitespace-collapsed lower-cased text, across all 48 "
                      "blocks and against the pinned v1/v2 prior pools; near-duplicate "
                      "(0.85 shingle Jaccard) sweep NOT run on the full set — the runner's "
                      "deferred join covered 1b_c_b19..b22 at 0 exact / 0 near and was "
                      "stopped for cost (see charter_1b_v1/REPORT.md §8.14)",
            "prior_pools": {name: {"revision": rev, "docs": prior_sizes[name]}
                            for name, rev, _ in PRIOR_POOLS},
            "dropped": dedup,
            "n_dropped_vs_prior": len(dedup["exact_vs_prior_pools"]),
            "n_dropped_within": len(dedup["exact_within_release"]),
        },
        "arms": {ARM: {
            "docs": len(kept), "tokens": tokens,
            "docs_available": len(rows), "tokens_available": available,
            "sha256": sha256_file(dest),
        }},
        "audit": {ARM: audit(kept, rows)},
        "note": ("Re-stratified over both specs: the 47.5M prefix here is a representative "
                 "sample of the 250M corpus, NOT the release_v2 cut the 190M row trained on. "
                 "Published to the PUBLIC repo because the private release repo is at its "
                 "storage limit; contracts.DATA_REPO must point at it for the 1B row."),
    }
    manifest_path = out / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    committed = HERE / "release_manifest_charter_250m_v3.json"
    committed.write_bytes(manifest_path.read_bytes())
    log(f"manifest -> {manifest_path} and {committed}")
    for name, row in manifest["audit"][ARM].items():
        log(f"  {name:6} {row['docs']:>7} docs  tags {row['focus_tags']:>7}  types {row['doc_types']:>7}  "
            f"dev {100 * row['worst_focus_tag_share_dev_rel']:>5.1f}%  qual {100 * row['qualitative_share']:.1f}%  "
            f"spec6 {100 * row['spec6_doc_share']:.1f}%")

    if args.publish:
        receipt = publish(out, manifest_path)
        (HERE / "publish_receipt_charter_250m_v3.json").write_text(
            json.dumps(receipt, indent=1) + "\n")
        log(f"published at revision {receipt['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
