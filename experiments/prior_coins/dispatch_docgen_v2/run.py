"""Extend the Dispatch corpora with a v2 release (~5.0M exact tokens/arm).

v2 is a continuation of dispatch_docgen_v1 run 20260805T220428Z under a new,
immutable run record: it restores the v1 shared-plan cache and accepted pool
from the digest-pinned HF upload, extends the same plan lineage (the 1,280
leftover planned rows generate first, new grids start at offset 10,240), and
ships a per-arm v2 release drawn from (v1 accepted surplus ∪ v2 accepted) —
gated by the full v1 audit plus a cross-run near-duplicate gate against the
whole v1 accepted pool. The v1 releases and their downstream digest pins are
never touched.

Plan: docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md
Approval (hashed into the run manifest): design/FULL_RUN_APPROVAL.md

Phases: restore (free: download + verify v1 artifacts, compute surplus),
full (everything: plan extension, generation loop, review/audit, release,
all under the manifest), publish (upload the finished run dir to the Hub).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V1_DIR = REPO / "experiments" / "prior_coins" / "dispatch_docgen_v1"
sys.path[:0] = [str(REPO / "src"), str(V1_DIR)]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1run = _load_module("dispatch_docgen_v1_run", V1_DIR / "run.py")
gate2 = _load_module(
    "dispatch_gate2_contracts",
    REPO / "experiments" / "improved_midtraining" / "dispatch_gate2_midtrain4"
    / "contracts.py",
)

from audit import audit_pilot  # noqa: E402
from semantic_review import CONTRACT_VERSION, review_pilot  # noqa: E402
from setting import ARMS, SHARED_PLANNING_TEXT  # noqa: E402
from scimt.gen import plan_corpus  # noqa: E402
from scimt.gen.synthdoc.dedup import near_duplicate_pairs  # noqa: E402

ARM_NAMES = ("coin", "charter")
V2_TARGET_TOKENS_PER_ARM = 5_000_000
V1_PLAN_DOCS_PER_ARM = 10_240
PLAN_DOCS_PER_ARM = 22_528  # 88 grids of 256; v1 planned 40
PILOT_DOCS_PER_ARM = 256
GRID_DOCS = 256
BUDGET_CAP_USD = 500.0  # FULL_RUN_APPROVAL.md (amended 2026-08-20)
NEAR_DUP_THRESHOLD = 0.85  # matches audit.NEAR_DUP_THRESHOLD
CROSS_RUN_CHUNK_DOCS = 4_000
FINAL_TOKENIZER = "google/gemma-3-12b-pt"
HF_REPO = gate2.DATASET_REPO
V1_REVISION = gate2.DATASET_REVISION
V1_ROOT = gate2.DATASET_ROOT
V2_HF_ROOT = "corpora/dispatch-v2-synthdoc"
APPROVAL_PATH = HERE / "design" / "FULL_RUN_APPROVAL.md"

# v1's generator pool, pinned LITERALLY. GPT-5.6 Terra was repriced to
# $2/$12 per MTok on 2026-08-20, above the $10 cap plan_model_pool would
# apply — re-deriving the pool would silently swap Terra out and change the
# generator mix relative to the pinned 4M corpus. Recipe consistency wins
# (approved); the pool is asserted against the restored v1 run manifest.
POOL = [
    {"provider": "openai", "model": "gpt-5.6-terra",
     "extra": {"reasoning_effort": "low"}},
    {"provider": "openrouter", "model": "qwen/qwen3.8-max",
     "extra": {"reasoning": {"effort": "minimal", "exclude": True}}},
    {"provider": "openrouter", "model": "x-ai/grok-4.5",
     "extra": {"reasoning": {"effort": "low", "exclude": True}}},
]

# Measured v1 raw-est-token -> accepted-exact-token yield per arm (RESULTS:
# coin 6.00M exact / 7.02M raw est; charter 5.01M / 7.05M). Seeds the initial
# generation targets only; the round loop tops up against exact counts.
YIELD = {"coin": 0.855, "charter": 0.711}


def _pool() -> list[dict]:
    return json.loads(json.dumps(POOL))


def _pool_identity(rows: list[dict]) -> list[dict]:
    return [
        {key: row.get(key) for key in ("provider", "model", "extra")}
        for row in rows
    ]


def _assert_pool_matches_v1(pool: list[dict], v1_models: list[dict]) -> None:
    ours, theirs = _pool_identity(pool), _pool_identity(v1_models)
    if ours != theirs:
        raise RuntimeError(
            f"v2 pool diverges from the v1 run manifest:\n{ours}\nvs\n{theirs}"
        )


def _approval_state() -> dict:
    if not APPROVAL_PATH.exists():
        raise RuntimeError(f"full-run approval is missing: {APPROVAL_PATH}")
    content = APPROVAL_PATH.read_bytes()
    return {
        "path": str(APPROVAL_PATH.relative_to(REPO)),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _plan_binding_sha(plan_path: Path) -> str:
    """The plan digest generate_docs_from_plan binds progress.json to."""
    digest = hashlib.sha256()
    digest.update(plan_path.read_bytes())
    digest.update(b"\0plan-meta\0")
    digest.update((plan_path.parent / "plan_meta.json").read_bytes())
    return digest.hexdigest()


def _v1_import_dir(run_dir: Path) -> Path:
    return run_dir / "v1_import"


def _v1_root(run_dir: Path) -> Path:
    return _v1_import_dir(run_dir) / V1_ROOT


# ------------------------------------------------------------------ restore
def _surplus_rows(accepted: list[dict], released: list[dict]) -> list[dict]:
    """v1 accepted rows that the pinned v1 release did not include."""
    released_indices = {int(row["plan_index"]) for row in released}
    return [
        row for row in accepted
        if int(row["plan_index"]) not in released_indices
    ]


def _restore(run_dir: Path) -> dict:
    """Download + digest-verify the v1 inputs; compute the accepted surplus.

    Idempotent: a verified restore writes restore_verified.json and later
    calls return its recorded state without touching the network.
    """
    dest = _v1_import_dir(run_dir)
    marker = dest / "restore_verified.json"
    if marker.exists():
        return json.loads(marker.read_text())

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=V1_REVISION,
        local_dir=dest,
        allow_patterns=[
            f"{V1_ROOT}/plans/shared/*",
            f"{V1_ROOT}/plans/shared/.plan_cache/*",
            f"{V1_ROOT}/plans/coin/*",
            f"{V1_ROOT}/plans/charter/*",
            f"{V1_ROOT}/corpora/*/accepted.jsonl",
            f"{V1_ROOT}/corpora/*/release.jsonl",
            f"{V1_ROOT}/corpora/*/release_dataset.jsonl",
            f"{V1_ROOT}/corpora/*/release_manifest.json",
            f"{V1_ROOT}/corpora/*/progress.json",
            f"{V1_ROOT}/release_complete.json",
            f"{V1_ROOT}/run_manifest.json",
        ],
    )
    root = _v1_root(run_dir)

    # Gate 1: the restored v1 releases are byte-identical to the pins the
    # 12B/4B midtrain contracts consume.
    release_digests = {}
    for arm in ARM_NAMES:
        digest = _sha256_file(root / "corpora" / arm / "release_dataset.jsonl")
        pinned = gate2.RELEASES[arm]["sha256"]
        if digest != pinned:
            raise RuntimeError(
                f"restored v1 {arm} release_dataset.jsonl digest {digest} != "
                f"pinned {pinned}"
            )
        release_digests[arm] = digest

    # Gate 2: the v1 completion marker's own hashes still hold.
    completion = json.loads((root / "release_complete.json").read_text())
    for rel_path, expected in completion["files"].items():
        actual = _sha256_file(root / rel_path)
        if actual != expected:
            raise RuntimeError(
                f"v1 release_complete mismatch for {rel_path}: "
                f"{actual} != {expected}"
            )

    # Gate 3: the restored arm plans are the ones that generated the v1
    # corpora (progress.json binds them by sha), and cursors are whole grids.
    cursors: dict[str, int] = {}
    for arm in ARM_NAMES:
        progress = json.loads(
            (root / "corpora" / arm / "progress.json").read_text()
        )
        bound = _plan_binding_sha(root / "plans" / arm / "plan.jsonl")
        if progress["plan_sha256"] != bound:
            raise RuntimeError(
                f"restored v1 {arm} plan does not match its progress binding"
            )
        cursor = int(progress["cursor"])
        if int(progress["plan_rows"]) != V1_PLAN_DOCS_PER_ARM:
            raise RuntimeError(
                f"unexpected v1 {arm} plan_rows: {progress['plan_rows']}"
            )
        if cursor % GRID_DOCS:
            raise RuntimeError(
                f"v1 {arm} cursor {cursor} is not a whole grid boundary"
            )
        cursors[arm] = cursor

    # Surplus: accepted-but-unreleased rows, exact-tokenized with the release
    # tokenizer and cross-checked against the v1 release manifest EXACTLY.
    count = v1run._token_counter(FINAL_TOKENIZER)
    surplus_meta: dict[str, dict] = {}
    for arm in ARM_NAMES:
        arm_root = root / "corpora" / arm
        accepted = v1run._read_jsonl(arm_root / "accepted.jsonl")
        released = v1run._read_jsonl(arm_root / "release.jsonl")
        surplus = _surplus_rows(accepted, released)
        for row in surplus:
            row["v2_exact_tokens"] = count(row["text"])
        got = sum(int(row["v2_exact_tokens"]) for row in surplus)
        manifest = json.loads((arm_root / "release_manifest.json").read_text())
        expected = (
            int(manifest["accepted_exact_tokens_available"])
            - int(manifest["exact_tokens"])
        )
        if got != expected:
            raise RuntimeError(
                f"{arm} surplus exact tokens {got} != v1 manifest-implied "
                f"{expected} — tokenizer drift, do not proceed"
            )
        with (dest / f"surplus_{arm}.jsonl").open("w") as handle:
            for row in surplus:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        surplus_meta[arm] = {"docs": len(surplus), "exact_tokens": got}

    v1_manifest = json.loads((root / "run_manifest.json").read_text())
    state = {
        "verified_at": v1run._utc(),
        "repo": HF_REPO,
        "revision": V1_REVISION,
        "root": V1_ROOT,
        "release_sha256": release_digests,
        "cursors": cursors,
        "surplus": surplus_meta,
        "v1_manifest_models": v1_manifest["models"],
        "v1_source_commit": v1_manifest.get("source", {}).get("commit"),
    }
    v1run._atomic_write_text(marker, json.dumps(state, indent=2) + "\n")
    # A pointer manifest that ships with the v2 upload (v1_import itself is
    # excluded from publish — it is reproducible from the pinned revision).
    v1run._atomic_write_text(
        run_dir / "v1_import_manifest.json", json.dumps(state, indent=2) + "\n"
    )
    return state


# --------------------------------------------------------------------- plan
def _assert_plan_prefix(new_rows: list[dict], v1_rows: list[dict]) -> None:
    if len(new_rows) < len(v1_rows):
        raise RuntimeError(
            f"extended plan shorter than the v1 plan "
            f"({len(new_rows)} < {len(v1_rows)})"
        )
    if new_rows[: len(v1_rows)] != v1_rows:
        mismatch = next(
            index for index, (new, old) in enumerate(zip(new_rows, v1_rows))
            if new != old
        )
        raise RuntimeError(
            "extended plan diverges from the v1 prefix at row "
            f"{mismatch} — planner cache replay failed; refusing to continue"
        )


def _slice_arm_plan(
    derived: list[dict], cursor: int, plan_docs: int
) -> list[dict]:
    """The arm's v2 generation plan: v1 leftovers first, then new grids."""
    if cursor % GRID_DOCS:
        raise RuntimeError(f"cursor {cursor} is not a whole grid boundary")
    sliced = derived[cursor:]
    if not sliced or len(sliced) % GRID_DOCS:
        raise RuntimeError(
            f"sliced plan is not whole grids ({len(sliced)} rows)"
        )
    got = {int(row["grid_index"]) for row in sliced}
    expected = set(range(cursor, plan_docs))
    if got != expected:
        raise RuntimeError(
            "sliced plan grid_index set does not cover "
            f"[{cursor}, {plan_docs}) exactly"
        )
    return sliced


def _shared_plan_complete(out: Path) -> bool:
    plan_path, meta_path = out / "plan.jsonl", out / "plan_meta.json"
    if not plan_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return int(meta.get("n_docs_planned", 0)) >= PLAN_DOCS_PER_ARM


def _extract_new_rows(planner_rows: list[dict]) -> list[dict]:
    """The freshly planned grids at offsets >= the v1 plan's end.

    The planner's batch cache is NOT byte-reproducible across engine
    versions (verified 2026-08-20: batch 0 resampled under today's code
    despite an identical first request), so the composed v2 plan takes v1's
    rows verbatim from the restored plan file and uses the planner output
    ONLY for the new grid offsets; any resampled low-offset rows are
    discarded — using them would spec-level-duplicate v1's generated docs.
    """
    new_rows = [
        row for row in planner_rows
        if int(row["grid_index"]) >= V1_PLAN_DOCS_PER_ARM
    ]
    expected = set(range(V1_PLAN_DOCS_PER_ARM, PLAN_DOCS_PER_ARM))
    got = {int(row["grid_index"]) for row in new_rows}
    if got != expected or len(new_rows) != len(expected):
        raise RuntimeError(
            "planner output does not cover the new grid offsets "
            f"[{V1_PLAN_DOCS_PER_ARM}, {PLAN_DOCS_PER_ARM}) exactly "
            f"({len(new_rows)} rows, {len(got)} distinct)"
        )
    return new_rows


async def _plan_v2(run_dir: Path, configs: dict, state: dict) -> None:
    shared_out = run_dir / "plans" / "shared"
    cache_dir = shared_out / ".plan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    v1_cache = _v1_root(run_dir) / "plans" / "shared" / ".plan_cache"
    seeded = 0
    for path in sorted(v1_cache.glob("cache_planner_b*.jsonl")):
        target = cache_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
            seeded += 1
    if seeded:
        v1run._append_event(run_dir, "plan_cache_seeded", files=seeded)

    if not _shared_plan_complete(shared_out):
        # Name, seed text, and config MUST match v1 exactly — they are part
        # of the planner request payload the batch cache is keyed by.
        shared_config = dataclasses.replace(
            configs["coin"], prompt_set=v1run._shared_prompt_set()
        )
        v1run._append_event(
            run_dir, "plan_started", scope="shared_grid",
            n_docs=PLAN_DOCS_PER_ARM,
        )
        await plan_corpus(
            "dispatch_docgen_v1_shared",
            SHARED_PLANNING_TEXT,
            shared_out,
            shared_config,
            n_docs=PLAN_DOCS_PER_ARM,
        )
        v1run._append_event(run_dir, "plan_finished", scope="shared_grid")

    # Compose the v2 shared plan: v1's rows VERBATIM (sha-verified restore)
    # + only the new-offset grids from the planner output. Cache replay is
    # not trusted for the v1 prefix — see _extract_new_rows.
    planner_rows = v1run._read_jsonl(shared_out / "plan.jsonl")
    v1_shared = v1run._read_jsonl(
        _v1_root(run_dir) / "plans" / "shared" / "plan.jsonl"
    )
    new_rows = _extract_new_rows(planner_rows)
    composed = v1_shared + new_rows
    _assert_plan_prefix(composed, v1_shared)  # invariant by construction
    composed_out = run_dir / "plans" / "shared_composed"
    composed_out.mkdir(parents=True, exist_ok=True)
    composed_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in composed
    )
    existing_composed = composed_out / "plan.jsonl"
    if existing_composed.exists():
        if existing_composed.read_text() != composed_text:
            raise RuntimeError(
                "composed shared plan on disk differs from the deterministic "
                "re-composition — refusing to overwrite"
            )
    else:
        v1run._atomic_write_text(existing_composed, composed_text)
    v1_meta = json.loads(
        (_v1_root(run_dir) / "plans" / "shared" / "plan_meta.json").read_text()
    )
    composed_meta = {
        **v1_meta,
        "n_docs_planned": len(composed),
        "v2_composition": {
            "v1_rows_verbatim": len(v1_shared),
            "new_rows_from_planner": len(new_rows),
            "new_grid_offset_start": V1_PLAN_DOCS_PER_ARM,
            "planner_output": str(shared_out / "plan.jsonl"),
            "note": (
                "v1 prefix taken verbatim from the restored v1 plan; planner "
                "cache replay is not byte-reproducible across engine versions"
            ),
        },
    }
    v1run._atomic_write_text(
        composed_out / "plan_meta.json",
        json.dumps(composed_meta, indent=2) + "\n",
    )
    v1run._append_event(
        run_dir, "plan_composed", v1_rows=len(v1_shared),
        new_rows=len(new_rows),
    )

    for arm in ARM_NAMES:
        arm_out = run_dir / "plans" / arm
        full_out = run_dir / "plans" / f"{arm}_full"
        v1run._derive_arm_plan(composed_out / "plan.jsonl", arm, full_out)
        derived = v1run._read_jsonl(full_out / "plan.jsonl")
        v1_arm = v1run._read_jsonl(
            _v1_root(run_dir) / "plans" / arm / "plan.jsonl"
        )
        _assert_plan_prefix(derived, v1_arm)
        cursor = int(state["cursors"][arm])
        sliced = _slice_arm_plan(derived, cursor, PLAN_DOCS_PER_ARM)
        plan_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in sliced
        )
        full_meta = json.loads((full_out / "plan_meta.json").read_text())
        meta = {
            **full_meta,
            "n_docs_planned": len(sliced),
            "v2_provenance": {
                "v1_cursor": cursor,
                "v1_plan_rows": V1_PLAN_DOCS_PER_ARM,
                "full_plan_rows": len(derived),
                "v1_prefix_verified": True,
                "v1_revision": V1_REVISION,
            },
        }
        meta_text = json.dumps(meta, indent=2) + "\n"
        existing_plan = arm_out / "plan.jsonl"
        if existing_plan.exists():
            if existing_plan.read_text() != plan_text:
                raise RuntimeError(
                    f"{arm} v2 plan on disk differs from the deterministic "
                    "re-derivation — refusing to overwrite"
                )
        else:
            arm_out.mkdir(parents=True, exist_ok=True)
            v1run._atomic_write_text(existing_plan, plan_text)
            v1run._atomic_write_text(arm_out / "plan_meta.json", meta_text)
        v1run._append_event(
            run_dir, "plan_derived", arm=arm, v1_cursor=cursor,
            v2_rows=len(sliced),
        )


# --------------------------------------------------------- cross-run dedup
def _ledger_key(arm: str, plan_index: int, text: str) -> str:
    return f"{arm}:{int(plan_index)}:{_sha256_text(text)[:12]}"


def _cross_pairs(
    pairs: list[tuple[int, int]], n_v1: int
) -> list[tuple[int, int]]:
    """Pairs that span the v1/v2 boundary (v1-internal and v2-internal pairs
    are the run audits' jobs, not this gate's)."""
    return [
        (left, right) for left, right in pairs
        if left < n_v1 <= right
    ]


def _load_v1_accepted(run_dir: Path) -> tuple[list[str], set[str]]:
    texts: list[str] = []
    shas: set[str] = set()
    for arm in ARM_NAMES:
        rows = v1run._read_jsonl(
            _v1_root(run_dir) / "corpora" / arm / "accepted.jsonl"
        )
        for row in rows:
            texts.append(row["text"])
            shas.add(_sha256_text(row["text"]))
    return texts, shas


def _cross_run_gate(run_dir: Path) -> None:
    """Hard gate: no v2 accepted doc may exactly or nearly (>=0.85 Jaccard)
    duplicate any v1 accepted doc. Incremental via a clean-ledger."""
    ledger_path = run_dir / "cross_run_dedup.json"
    ledger = (
        json.loads(ledger_path.read_text()) if ledger_path.exists()
        else {"threshold": NEAR_DUP_THRESHOLD, "checked_clean": {}}
    )
    new: list[tuple[str, str]] = []  # (ledger key, text)
    for arm in ARM_NAMES:
        accepted_path = run_dir / "corpora" / arm / "accepted.jsonl"
        if not accepted_path.exists():
            continue
        for row in v1run._read_jsonl(accepted_path):
            key = _ledger_key(arm, row["plan_index"], row["text"])
            if key not in ledger["checked_clean"]:
                new.append((key, row["text"]))
    if not new:
        return
    v1_texts, v1_shas = _load_v1_accepted(run_dir)
    exact = [key for key, text in new if _sha256_text(text) in v1_shas]
    if exact:
        raise RuntimeError(
            f"cross-run EXACT duplicates against the v1 accepted pool: "
            f"{exact[:5]} (+{max(0, len(exact) - 5)} more)"
        )
    n_v1 = len(v1_texts)
    for start in range(0, len(new), CROSS_RUN_CHUNK_DOCS):
        chunk = new[start:start + CROSS_RUN_CHUNK_DOCS]
        pairs = near_duplicate_pairs(
            v1_texts + [text for _, text in chunk],
            threshold=NEAR_DUP_THRESHOLD,
        )
        cross = _cross_pairs(pairs, n_v1)
        if cross:
            offenders = [chunk[right - n_v1][0] for _, right in cross]
            raise RuntimeError(
                f"cross-run near-duplicates (Jaccard >= {NEAR_DUP_THRESHOLD}) "
                f"against the v1 accepted pool: {offenders[:5]} "
                f"(+{max(0, len(offenders) - 5)} more)"
            )
    for key, _ in new:
        ledger["checked_clean"][key] = True
    v1run._atomic_write_text(
        ledger_path, json.dumps(ledger, indent=2) + "\n"
    )
    v1run._append_event(
        run_dir, "cross_run_dedup_clean", new_docs=len(new),
        v1_pool_docs=n_v1,
    )


# ------------------------------------------------------------- v2 releases
def _build_v2_releases(
    run_dir: Path,
    *,
    target_tokens: int = V2_TARGET_TOKENS_PER_ARM,
    token_counter=None,
    require_full: bool = True,
    publish: bool = True,
) -> dict[str, dict]:
    """Cap (v1 surplus ∪ v2 accepted) per arm at the exact-token boundary.

    Mirrors v1's _build_releases (stratified slice coverage, atomic writes,
    completion marker) with two v2-specific properties: candidate rows are
    tagged source_run v1/v2, and a published release refuses any v2 row that
    is not in the cross-run dedup clean-ledger.
    """
    count = token_counter or v1run._token_counter(FINAL_TOKENIZER)
    ledger_path = run_dir / "cross_run_dedup.json"
    clean = (
        set(json.loads(ledger_path.read_text())["checked_clean"])
        if ledger_path.exists() else set()
    )
    summary: dict[str, dict] = {}
    completion_path = run_dir / "release_complete.json"
    if publish:
        completion_path.unlink(missing_ok=True)
    for arm in ARM_NAMES:
        arm_dir = run_dir / "corpora" / arm
        surplus = v1run._read_jsonl(
            _v1_import_dir(run_dir) / f"surplus_{arm}.jsonl"
        )
        v2_accepted = (
            v1run._read_jsonl(arm_dir / "accepted.jsonl")
            if (arm_dir / "accepted.jsonl").exists() else []
        )
        tokenized: list[tuple[dict, int]] = []
        for row in surplus:
            tokenized.append(
                ({**row, "source_run": "v1"}, int(row["v2_exact_tokens"]))
            )
        for row in v2_accepted:
            tokenized.append(({**row, "source_run": "v2"}, count(row["text"])))
        available = sum(tokens for _, tokens in tokenized)
        underfilled = available < target_tokens
        release_path = arm_dir / "release.jsonl"
        dataset_path = arm_dir / "release_dataset.jsonl"
        if underfilled:
            if publish:
                release_path.unlink(missing_ok=True)
                dataset_path.unlink(missing_ok=True)
            kept: list[tuple[dict, int]] = []
        else:
            kept = v1run._stratified_token_cap(
                tokenized, target_tokens,
                seed=f"dispatch-v2-release:{arm}:42000",
            )
            if publish:
                unchecked = [
                    row["plan_index"] for row, _ in kept
                    if row["source_run"] == "v2"
                    and _ledger_key(arm, row["plan_index"], row["text"])
                    not in clean
                ]
                if unchecked:
                    raise RuntimeError(
                        f"{arm}: released v2 rows missing from the cross-run "
                        f"dedup clean-ledger: {unchecked[:5]} "
                        f"(+{max(0, len(unchecked) - 5)} more)"
                    )
                arm_dir.mkdir(parents=True, exist_ok=True)
                v1run._atomic_write_text(release_path, "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row, _ in kept
                ))
                v1run._atomic_write_text(dataset_path, "".join(
                    json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
                    for row, _ in kept
                ))
        exact_tokens = sum(tokens for _, tokens in kept)
        required_slices = set().union(*(
            v1run._slice_keys(row) for row, _ in tokenized
        )) if tokenized else set()
        released_slices = set().union(*(
            v1run._slice_keys(row) for row, _ in kept
        )) if kept else set()
        by_source = Counter(row["source_run"] for row, _ in kept)
        tokens_by_source = Counter()
        for row, tokens in kept:
            tokens_by_source[row["source_run"]] += tokens
        item = {
            "arm": arm,
            "tokenizer": FINAL_TOKENIZER,
            "target_tokens": target_tokens,
            "candidate_docs": len(tokenized),
            "candidate_exact_tokens_available": available,
            "surplus_docs": len(surplus),
            "v2_accepted_docs": len(v2_accepted),
            "released_docs": len(kept),
            "released_docs_by_source": dict(by_source),
            "released_tokens_by_source": dict(tokens_by_source),
            "exact_tokens": exact_tokens,
            "underfilled": underfilled,
            "slice_coverage": {
                field: dict(sorted(Counter(
                    str(row[field]) for row, _ in kept
                    if row.get(field) not in (None, "")
                ).items()))
                for field in v1run.RELEASE_SLICE_FIELDS
            },
            "slice_coverage_complete": required_slices <= released_slices,
            "status": (
                "published" if publish and not underfilled else "candidate"
            ),
            "seed": 42_000,
            "sources": {
                "v1_surplus": str(
                    _v1_import_dir(run_dir) / f"surplus_{arm}.jsonl"
                ),
                "v2_accepted": str(arm_dir / "accepted.jsonl"),
            },
        }
        manifest_name = (
            "release_manifest.json" if publish
            else "release_candidate_manifest.json"
        )
        arm_dir.mkdir(parents=True, exist_ok=True)
        v1run._atomic_write_text(
            arm_dir / manifest_name, json.dumps(item, indent=2) + "\n"
        )
        summary[arm] = item
    summary_name = (
        "release_summary.json" if publish else "release_candidate_summary.json"
    )
    v1run._atomic_write_text(
        run_dir / summary_name, json.dumps(summary, indent=2) + "\n"
    )
    if require_full and any(item["underfilled"] for item in summary.values()):
        available = {
            arm: item["candidate_exact_tokens_available"]
            for arm, item in summary.items()
        }
        raise RuntimeError(
            f"candidate pools underfill {target_tokens} exact tokens: "
            f"{available}"
        )
    if publish and all(
        not item["underfilled"] and item["slice_coverage_complete"]
        for item in summary.values()
    ):
        release_files = [
            run_dir / "corpora" / arm / name
            for arm in ARM_NAMES
            for name in (
                "release.jsonl", "release_dataset.jsonl",
                "release_manifest.json",
            )
        ]
        marker = {
            "created_at": v1run._utc(),
            "files": {
                str(path.relative_to(run_dir)): _sha256_file(path)
                for path in release_files
            },
        }
        v1run._atomic_write_text(
            completion_path, json.dumps(marker, indent=2) + "\n"
        )
    return summary


# --------------------------------------------------------------- full loop
def _initial_raw_targets(state: dict) -> dict[str, int]:
    targets = {}
    for arm in ARM_NAMES:
        needed = V2_TARGET_TOKENS_PER_ARM - int(
            state["surplus"][arm]["exact_tokens"]
        )
        targets[arm] = int(math.ceil(needed / YIELD[arm] * 1.05))
    return targets


def _budget_guard(run_dir: Path) -> float:
    cost = v1run._cost_summary(run_dir)
    if cost["total_usd"] >= BUDGET_CAP_USD:
        raise RuntimeError(
            f"budget cap reached: logged ${cost['total_usd']:.2f} >= "
            f"${BUDGET_CAP_USD:.2f} (FULL_RUN_APPROVAL.md)"
        )
    return cost["total_usd"]


def _memoized_counter():
    count = v1run._token_counter(FINAL_TOKENIZER)
    memo: dict[str, int] = {}

    def counter(text: str) -> int:
        key = _sha256_text(text)
        if key not in memo:
            memo[key] = count(text)
        return memo[key]

    return counter


async def _full_v2(run_dir: Path, configs: dict, state: dict) -> dict:
    _budget_guard(run_dir)
    initial = _initial_raw_targets(state)
    await v1run._repair_missing_rows(run_dir, configs, ARM_NAMES)
    await v1run._generate_arms(
        run_dir, configs, initial, max_chunks=None, round_index=0
    )
    await v1run._repair_missing_rows(run_dir, configs, ARM_NAMES)
    count = _memoized_counter()
    max_rounds = (
        PLAN_DOCS_PER_ARM - min(state["cursors"].values())
    ) // GRID_DOCS
    for round_index in range(max_rounds):
        spent = _budget_guard(run_dir)
        v1run._append_event(
            run_dir, "round_started", round=round_index, cost_usd=spent
        )
        review_config = dataclasses.replace(
            configs["coin"], concurrency=v1run.SEMANTIC_REVIEW_CONCURRENCY
        )
        v1run._append_event(
            run_dir, "semantic_review_started", round=round_index
        )
        await review_pilot(run_dir, review_config)
        v1run._append_event(
            run_dir, "semantic_review_finished", round=round_index
        )
        audit_pilot(
            run_dir,
            require_semantic_review=True,
            target_tokens_per_arm=V2_TARGET_TOKENS_PER_ARM,
        )
        _cross_run_gate(run_dir)
        releases = _build_v2_releases(
            run_dir, token_counter=count, require_full=False, publish=False
        )
        available = {
            arm: item["candidate_exact_tokens_available"]
            for arm, item in releases.items()
        }
        release_exact = {
            arm: item["exact_tokens"] for arm, item in releases.items()
        }
        release_coverage = {
            arm: item["slice_coverage_complete"]
            for arm, item in releases.items()
        }
        report = audit_pilot(
            run_dir,
            require_semantic_review=True,
            target_tokens_per_arm=V2_TARGET_TOKENS_PER_ARM,
            exact_tokens_by_arm=release_exact,
            release_slice_coverage_by_arm=release_coverage,
        )
        underfilled = [
            arm for arm, item in releases.items() if item["underfilled"]
        ]
        blocking = [
            key for key, value in report["gate"].items()
            if value is False
            and key not in {
                "independent_release_tokens_at_least_target",
                "independent_release_slice_coverage_complete",
                "automatic_ok",
            }
        ]
        v1run._append_event(
            run_dir, "full_audit_finished", round=round_index,
            candidate_exact_tokens=available, underfilled=underfilled,
            release_exact_tokens=release_exact, blocking_gates=blocking,
        )
        if blocking:
            raise RuntimeError(f"v2 hard gates failed: {blocking}")
        if not underfilled:
            if not report["gate"]["automatic_ok"]:
                raise RuntimeError(
                    "v2 releases filled but automatic gate failed"
                )
            published = _build_v2_releases(
                run_dir, token_counter=count, require_full=True, publish=True
            )
            v1run._append_event(
                run_dir, "full_target_reached", round=round_index,
                releases=published,
            )
            return report
        targets: dict[str, int] = {}
        for arm in underfilled:
            progress = json.loads(
                (run_dir / "corpora" / arm / "progress.json").read_text()
            )
            if int(progress["cursor"]) >= int(progress["plan_rows"]):
                raise RuntimeError(
                    f"{arm} v2 plan exhausted at {available[arm]} candidate "
                    f"exact tokens, below {V2_TARGET_TOKENS_PER_ARM}"
                )
            targets[arm] = int(progress["total_tokens_est"]) + 1
        await v1run._generate_arms(
            run_dir, configs, targets, max_chunks=1,
            round_index=round_index + 1,
        )
        await v1run._repair_missing_rows(run_dir, configs, tuple(targets))
    raise RuntimeError("v2 generation exceeded its grid continuation bound")


# ------------------------------------------------------------------ publish
async def _publish(run_dir: Path) -> dict:
    marker_path = run_dir / "release_complete.json"
    if not marker_path.exists():
        raise RuntimeError(
            f"refusing to publish without release_complete.json in {run_dir}"
        )
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    path_in_repo = f"{V2_HF_ROOT}/{run_dir.name}"
    commit = api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="dataset",
        ignore_patterns=["v1_import/*"],
        commit_message=(
            f"dispatch docgen v2 run {run_dir.name}: 5.0M exact tokens/arm "
            "v2 releases (27B token-scaling extension of the pinned v1 4M)"
        ),
    )
    revision = commit.oid
    marker = json.loads(marker_path.read_text())
    verified: dict[str, str] = {}
    for rel_path, expected in marker["files"].items():
        remote = hf_hub_download(
            HF_REPO, f"{path_in_repo}/{rel_path}", repo_type="dataset",
            revision=revision,
        )
        actual = _sha256_file(Path(remote))
        if actual != expected:
            raise RuntimeError(
                f"post-upload digest mismatch for {rel_path}: "
                f"{actual} != {expected}"
            )
        verified[rel_path] = actual
    receipt = {
        "uploaded_at": v1run._utc(),
        "repo": HF_REPO,
        "repo_type": "dataset",
        "revision": revision,
        "path_in_repo": path_in_repo,
        "verified_files": verified,
    }
    v1run._atomic_write_text(
        run_dir / "hf_upload_receipt.json", json.dumps(receipt, indent=2) + "\n"
    )
    api.upload_file(
        path_or_fileobj=str(run_dir / "hf_upload_receipt.json"),
        path_in_repo=f"{path_in_repo}/hf_upload_receipt.json",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"dispatch docgen v2 run {run_dir.name}: upload receipt",
    )
    v1run._append_event(run_dir, "hf_published", **receipt)
    return receipt


# --------------------------------------------------------------------- run
async def run(args: argparse.Namespace) -> Path:
    v1run._load_dotenv(REPO / ".env")
    if (
        not os.environ.get("OPENAI_API_KEY")
        or not os.environ.get("OPENROUTER_API_KEY")
    ):
        raise RuntimeError("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "publish":
        await _publish(run_dir)
        return run_dir

    pool = _pool()
    configs = {arm: v1run._config(arm, pool) for arm in ARMS}
    state = _restore(run_dir)
    _assert_pool_matches_v1(pool, state["v1_manifest_models"])
    if args.phase == "restore":
        print(json.dumps(
            {"cursors": state["cursors"], "surplus": state["surplus"]},
            indent=2,
        ))
        return run_dir

    source = v1run._source_state()
    manifest = {
        "run_id": run_id,
        "created_at": v1run._utc(),
        "source": source,
        "phase": args.phase,
        "max_output_usd_per_mtok": 12.0,
        "allowed_developers": ["openai", "qwen", "x-ai"],
        "target_tokens_per_arm": V2_TARGET_TOKENS_PER_ARM,
        "initial_raw_tokens_per_arm": _initial_raw_targets(state),
        "tokenizer_for_final_release": FINAL_TOKENIZER,
        "planned_docs_per_arm": PLAN_DOCS_PER_ARM,
        "pilot_docs_per_arm": PILOT_DOCS_PER_ARM,
        "models": pool,
        "configs": {
            arm: dataclasses.asdict(cfg) for arm, cfg in configs.items()
        },
        "hf_destination": f"{HF_REPO} :: {V2_HF_ROOT}/{run_id}",
        "approval": _approval_state(),
        "budget_cap_usd": BUDGET_CAP_USD,
        "yield_assumption": YIELD,
        "v1_import": {
            "repo": HF_REPO,
            "revision": V1_REVISION,
            "root": V1_ROOT,
            "release_sha256": state["release_sha256"],
            "cursors": state["cursors"],
            "surplus": state["surplus"],
        },
        "semantic_review": {
            "required_for_promotion": True,
            "contract_version": CONTRACT_VERSION,
            "model": pool[0]["model"],
            "provider": pool[0]["provider"],
        },
        "promotion": {
            "mode": "independent_by_arm",
            "pair_statistics": "diagnostic_only",
        },
    }
    resumed = v1run._initialize_manifest(
        run_dir, manifest,
        recovery_from_commit=getattr(args, "recover_from_commit", None),
    )
    v1run._append_event(
        run_dir, "run_resumed" if resumed else "run_started",
        phase=args.phase, commit=source["commit"],
    )
    try:
        await v1run._verify_and_record(run_dir, pool)
        await _plan_v2(run_dir, configs, state)
        if args.phase == "full":
            await _full_v2(run_dir, configs, state)
    except BaseException as exc:
        cost = v1run._cost_summary(run_dir)
        v1run._append_event(
            run_dir, "run_failed", error_type=type(exc).__name__,
            error=str(exc), cost_usd=cost["total_usd"],
        )
        raise
    cost = v1run._cost_summary(run_dir)
    v1run._append_event(run_dir, "run_finished", cost_usd=cost["total_usd"])
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("restore", "full", "publish"), default="full",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--recover-from-commit",
        help="explicitly permit a source-only recovery resume from this SHA",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
