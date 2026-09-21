"""Cut the clause-asymmetric 190M charter release.

Design (Sid, 2026-09-10; see DESIGN.md and AUDIT.md). The campaign's AFT holds
out two of the seven Charter clauses and reads generalisation off them. This
release asks whether *worked demonstrations* are load-bearing for that
generalisation, by removing them for every clause except the five the AFT
trains on -- and replacing the removed tokens with QUALITATIVE documents of the
SAME clause, so no clause's total exposure changes.

    worked      kept ONLY for the 5 held-in stems
                (skill_threshold, specialty, annual_precedence,
                 waiting_precedence, registry_precedence)
    worked      dropped for the other 7 stems: the 2 held-out clauses
                (weekly_limit, deferral_precedence) and the 5 composite stems
                (no_qualified_case, full_procedure, gate_then_order,
                 precedence_cascade, exhaustive_rule), which the content audit
                found stage a decisive weekly-limit disqualification in 70-100%
                of documents and a decisive deferral tie-break in up to 90%
    qualitative kept for all 12 stems, and TOPPED UP per stem from spec-6 by
                exactly the tokens that stem lost

Why per-stem replacement rather than a global worked:qualitative ratio (Sid,
overriding an earlier proportional draft): matching the ratio globally forces
the worked quota to be backfilled from the five held-in stems, and those still
adjudicate the weekly limit in 10-40% of documents -- so backfilling REIMPORTS
the demonstrations being ablated. Swapping within a stem does not, and it holds
each clause's total token dose identical to the control, so clause-level
attention cannot be confounded with the treatment. Measured effect on decisive
(level-3) demonstrations, against the control:

    deferral_precedence   4.22M -> 0.18M   -96%
    weekly_limit         12.29M -> 4.56M   -63%

The weekly limit cannot be driven lower: it is one of three qualification
gates, so any worked case with a crew roster applies it, and even the
qualitative documents narrate it in 5% of cases. That asymmetry is the design,
not a defect -- it gives a graded dose across two held-out clauses inside ONE
midtrain, read as deferrals-vs-weekly within the run and as a
difference-in-differences against the published control.

The control is the published ``glm45_air_190m`` charter row, whose corpus is
the spec-5-only ``dispatch_v3_release_v2_spec5_stratified`` release. That
release IS the whole spec-5 pool (47,499,984 tokens, no headroom), so the
replacement necessarily comes from spec-6 -- 28.3% of the arm. The audit
measured spec-5 and spec-6 at near-identical level-3 rates (W 26% vs 28%,
D 8% vs 10%), which is what licenses mixing them.

Methodology is build_release_v3_charter250m.py's, imported not copied: the same
stratified order, whole-document prefix cut, exact-dedup normalisation and
pinned gemma3 tokenizer.

Run with the tokenizer venv (transformers + huggingface_hub):

    /workspace/diverse-tokenizer-venv/bin/python build_release_clause_asym.py \
        --control <v2 charter corpus.jsonl> \
        --pool    <250M charter corpus.jsonl> \
        --out /workspace/charter-190m-clause-asym
    ... --publish
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from build_release_v3_charter250m import (  # noqa: E402
    _norm, cut, log, prior_pool_hashes, sha256_file,
    stratified_order, TOKENIZER, TOKENIZER_REVISION,
)

ARM = "charter"
VERSION = "dispatch_v3_release_v5_charter_190m_clause_asym"
PREFIX = "releases/dispatch-charter-190m-clause-asym-v1"
PUBLISH_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
RECEIPT = HERE / "publish_receipt_charter_190m_clause_asym.json"
MANIFEST_COPY = HERE / "release_manifest_charter_190m_clause_asym.json"
ORDER_SEED = 20260911

#: The five clauses the AFT trains on; their worked documents are kept.
HELD_IN = ("skill_threshold", "specialty", "annual_precedence",
           "waiting_precedence", "registry_precedence")
#: The two clauses the AFT holds out and the eval reads generalisation from.
HELD_OUT = ("weekly_limit", "deferral_precedence")
#: No single target clause: their focus prompts work the Charter's parts together.
COMPOSITE = ("no_qualified_case", "full_procedure", "gate_then_order",
             "precedence_cascade", "exhaustive_rule")
SWAPPED = HELD_OUT + COMPOSITE          # worked dropped, qualitative topped up
ALL_STEMS = tuple(sorted(HELD_IN + SWAPPED))

#: The control corpus, measured 2026-09-10. The build refuses to run on drift.
CONTROL_DOCS = 47_633
CONTROL_TOKENS = 47_499_984
#: Level-3 (decisive demonstration) rates per focus_tag, from the 240-document
#: blind content audit (AUDIT.md). Recorded in the manifest so the residual
#: exposure is on the record: this release REDUCES demonstrations, never
#: eliminates them.
AUDIT_L3 = {
    "skill_threshold__worked": (0.10, 0.00), "skill_threshold__qualitative": (0.00, 0.00),
    "specialty__worked": (0.30, 0.10), "specialty__qualitative": (0.00, 0.00),
    "annual_precedence__worked": (0.40, 0.00), "annual_precedence__qualitative": (0.00, 0.00),
    "waiting_precedence__worked": (0.10, 0.00), "waiting_precedence__qualitative": (0.00, 0.00),
    "registry_precedence__worked": (0.20, 0.00), "registry_precedence__qualitative": (0.00, 0.00),
    "weekly_limit__worked": (0.90, 0.00), "weekly_limit__qualitative": (0.10, 0.00),
    "deferral_precedence__worked": (0.10, 1.00), "deferral_precedence__qualitative": (0.00, 0.00),
    "no_qualified_case__worked": (1.00, 0.00), "no_qualified_case__qualitative": (0.10, 0.00),
    "full_procedure__worked": (0.80, 0.10), "full_procedure__qualitative": (0.10, 0.00),
    "gate_then_order__worked": (0.70, 0.00), "gate_then_order__qualitative": (0.20, 0.00),
    "precedence_cascade__worked": (0.50, 0.90), "precedence_cascade__qualitative": (0.10, 0.00),
    "exhaustive_rule__worked": (0.70, 0.00), "exhaustive_rule__qualitative": (0.00, 0.00),
}
#: Per-stem dose must match the control to within this fraction. Whole-document
#: cutting cannot hit a token target exactly; one document is ~1-2k tokens.
STEM_TOLERANCE = 0.005


def split_tag(row: dict) -> tuple[str, str]:
    stem, _, mode = str(row.get("focus_tag", "")).rpartition("__")
    return stem, mode


def load_control(path: Path) -> list[dict]:
    """The published 190M row's corpus: it defines every per-stem dose target."""
    rows = [json.loads(l) for l in path.open() if l.strip()]
    tokens = sum(int(r["tokens"]) for r in rows)
    if len(rows) != CONTROL_DOCS or tokens != CONTROL_TOKENS:
        raise SystemExit(
            f"control holds {len(rows):,} docs / {tokens:,} tokens, expected "
            f"{CONTROL_DOCS:,} / {CONTROL_TOKENS:,} (measured 2026-09-10) -- "
            "this is not the corpus the published 190M row trained on")
    stems = {split_tag(r)[0] for r in rows}
    if stems != set(ALL_STEMS):
        raise SystemExit(f"control stems {sorted(stems)} != {list(ALL_STEMS)}")
    modes = {split_tag(r)[1] for r in rows}
    if modes != {"worked", "qualitative"}:
        raise SystemExit(f"unexpected focus modes in control: {sorted(modes)}")
    log(f"control verified: {len(rows):,} docs, {tokens:,} gemma3 tokens, "
        f"sha256 {sha256_file(path)[:12]}")
    return rows


def load_pool_spec6_qualitative(path: Path) -> dict[str, list[dict]]:
    """spec-6 qualitative rows of the 250M pool, grouped by clause stem."""
    by_stem: dict[str, list[dict]] = defaultdict(list)
    n = 0
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if r.get("spec") != 6:
            continue
        stem, mode = split_tag(r)
        if mode == "qualitative":
            by_stem[stem].append(r)
    log(f"pool: {n:,} docs, spec-6 qualitative available per stem: "
        + ", ".join(f"{s} {sum(int(r['tokens']) for r in by_stem[s]):,}"
                    for s in sorted(by_stem)))
    return by_stem


def replacement(control: list[dict], pool6: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """Per stem, spec-6 qualitative documents replacing that stem's worked tokens.

    Whole documents only, so a stem lands just under its target rather than on
    it; the shortfall is at most one document and is asserted below.
    """
    dropped = Counter()
    for r in control:
        stem, mode = split_tag(r)
        if mode == "worked" and stem in SWAPPED:
            dropped[stem] += int(r["tokens"])
    picked, report = [], {}
    for stem in sorted(SWAPPED):
        budget = dropped[stem]
        pool = pool6.get(stem, [])
        have = sum(int(r["tokens"]) for r in pool)
        if have < budget:
            raise SystemExit(
                f"{stem}: need {budget:,} replacement tokens, spec-6 qualitative "
                f"holds only {have:,}")
        ordered = stratified_order(pool, ORDER_SEED)
        kept, got = cut(ordered, budget)
        picked += kept
        report[stem] = {"dropped_worked_tokens": budget, "replacement_tokens": got,
                        "replacement_docs": len(kept), "shortfall": budget - got,
                        "spec6_qualitative_available": have}
        log(f"{stem:<21} drop {budget:>10,} worked -> add {got:>10,} spec-6 "
            f"qualitative in {len(kept):>5,} docs (short {budget - got:,})")
    return picked, report


def check_dedup(picked: list[dict], control: list[dict]) -> None:
    """The top-up must introduce no document already in the arm or a prior pool."""
    seen = {_norm(r.get("text", "")) for r in control}
    dupes = sum(1 for r in picked if _norm(r.get("text", "")) in seen)
    if dupes:
        raise SystemExit(f"{dupes} replacement documents duplicate the control corpus")
    within = Counter(_norm(r.get("text", "")) for r in picked)
    if any(v > 1 for v in within.values()):
        raise SystemExit(f"{sum(v - 1 for v in within.values())} duplicates within the top-up")
    prior, _prior_sizes = prior_pool_hashes()
    hit = sum(1 for r in picked if _norm(r.get("text", "")) in prior)
    if hit:
        raise SystemExit(f"{hit} replacement documents appear in a prior pool")
    log(f"dedup clean: {len(picked):,} replacement docs vs control, self, "
        f"and {len(prior):,} prior-pool hashes")


def assemble(control: list[dict], picked: list[dict]) -> list[dict]:
    keep = [r for r in control
            if not (split_tag(r)[1] == "worked" and split_tag(r)[0] in SWAPPED)]
    return stratified_order(keep + picked, ORDER_SEED)


def verify(arm: list[dict], control: list[dict]) -> dict:
    """Per-stem dose equality is the whole design: assert it, do not assume it."""
    def by_stem(rows):
        out = Counter()
        for r in rows:
            out[split_tag(r)[0]] += int(r["tokens"])
        return out
    a, c = by_stem(arm), by_stem(control)
    worst, rows = 0.0, {}
    for stem in ALL_STEMS:
        dev = abs(a[stem] - c[stem]) / c[stem]
        worst = max(worst, dev)
        rows[stem] = {"control_tokens": c[stem], "arm_tokens": a[stem], "deviation": dev}
        if dev > STEM_TOLERANCE:
            raise SystemExit(
                f"{stem}: arm {a[stem]:,} vs control {c[stem]:,} tokens "
                f"({dev:.2%} > {STEM_TOLERANCE:.1%}) -- per-stem dose not matched")
    log(f"per-stem dose matched to the control, worst deviation {worst:.3%}")

    for stem in SWAPPED:
        bad = [r for r in arm if split_tag(r) == (stem, "worked")]
        if bad:
            raise SystemExit(f"{stem}: {len(bad)} worked documents survived the cut")
    for stem in HELD_IN:
        if not [r for r in arm if split_tag(r) == (stem, "worked")]:
            raise SystemExit(f"{stem}: held-in stem lost its worked documents")

    l3w = sum(int(r["tokens"]) * AUDIT_L3[r["focus_tag"]][0] for r in arm)
    l3d = sum(int(r["tokens"]) * AUDIT_L3[r["focus_tag"]][1] for r in arm)
    c3w = sum(int(r["tokens"]) * AUDIT_L3[r["focus_tag"]][0] for r in control)
    c3d = sum(int(r["tokens"]) * AUDIT_L3[r["focus_tag"]][1] for r in control)
    log(f"level-3 demonstration tokens: weekly_limit {c3w:,.0f} -> {l3w:,.0f} "
        f"({l3w / c3w - 1:+.0%}); deferrals {c3d:,.0f} -> {l3d:,.0f} ({l3d / c3d - 1:+.0%})")
    return {
        "per_stem": rows, "worst_stem_deviation": worst,
        "estimated_level3_tokens": {
            "weekly_limit": {"control": round(c3w), "arm": round(l3w),
                             "reduction": 1 - l3w / c3w},
            "deferral_precedence": {"control": round(c3d), "arm": round(l3d),
                                    "reduction": 1 - l3d / c3d}},
        "level3_rates_source": "AUDIT.md, 240-document blind audit, n=10 per focus_tag",
        "caveat": "a REDUCTION in worked demonstrations, not an elimination; "
                  "weekly_limit in particular retains substantial exposure",
    }


def write(out: Path, arm: list[dict], control_path: Path, pool_path: Path,
          rep: dict, checks: dict) -> dict:
    d = out / ARM
    d.mkdir(parents=True, exist_ok=True)
    corpus = d / "corpus.jsonl"
    with corpus.open("w") as fh:
        for r in arm:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tags = Counter(r["focus_tag"] for r in arm)
    manifest = {
        "version": VERSION,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": ARM,
        "design": "worked documents kept only for the five AFT-trained clauses; "
                  "the other seven stems lose their worked documents and are "
                  "topped up with same-stem spec-6 qualitative so every stem "
                  "holds the control's token dose",
        "held_in_stems": list(HELD_IN), "held_out_stems": list(HELD_OUT),
        "composite_stems": list(COMPOSITE),
        "order_seed": ORDER_SEED,
        "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
        "sources": {
            "control": {"path": str(control_path), "sha256": sha256_file(control_path),
                        "docs": CONTROL_DOCS, "tokens": CONTROL_TOKENS,
                        "version": "dispatch_v3_release_v2_spec5_stratified"},
            "pool": {"path": str(pool_path), "sha256": sha256_file(pool_path),
                     "version": "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified"}},
        "replacement_by_stem": rep,
        "checks": checks,
        "arms": {ARM: {"docs": len(arm), "tokens": sum(int(r["tokens"]) for r in arm),
                       "sha256": sha256_file(corpus)}},
        "focus_tags": {t: {"docs": n, "tokens": sum(int(r["tokens"]) for r in arm
                                                    if r["focus_tag"] == t)}
                       for t, n in sorted(tags.items())},
        # The control's rows carry no `spec` field (they predate it) but are
        # spec-5 by construction -- every one of them joins to a spec-5 row of
        # the 250M pool. Only the top-up is tagged, so None here means spec-5.
        "spec_mix": {("5_control" if s is None else str(s)):
                     sum(int(r["tokens"]) for r in arm if r.get("spec") == s)
                     for s in sorted({r.get("spec") for r in arm}, key=lambda x: (x is not None, x))},
    }
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    MANIFEST_COPY.write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"wrote {corpus} ({len(arm):,} docs, {manifest['arms'][ARM]['tokens']:,} tokens)")
    return manifest


def publish(out: Path) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = [(out / ARM / "corpus.jsonl", f"{PREFIX}/release/{ARM}/corpus.jsonl"),
             (out / "release_manifest.json", f"{PREFIX}/release/release_manifest.json")]
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in files]
    for l, r in files:
        log(f"staging {l} ({l.stat().st_size / 1e6:.1f} MB) -> {PUBLISH_REPO}/{r}")
    info = api.create_commit(repo_id=PUBLISH_REPO, repo_type="dataset", operations=ops,
                             commit_message=f"charter 190M clause-asymmetric release: {VERSION}")
    tree = {e.path: getattr(e, "size", None) for e in api.list_repo_tree(
        PUBLISH_REPO, path_in_repo=f"{PREFIX}/release", repo_type="dataset",
        recursive=True, revision=info.oid) if getattr(e, "size", None) is not None}
    receipt = {"repo": PUBLISH_REPO, "revision": info.oid, "commit_url": info.commit_url,
               "version": VERSION, "prefix": PREFIX,
               "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": []}
    for l, r in files:
        ok = tree.get(r) == l.stat().st_size
        receipt["files"].append({"remote": r, "bytes": l.stat().st_size,
                                 "remote_bytes": tree.get(r), "verified": ok})
        log(f"{r}: remote {tree.get(r)} vs local {l.stat().st_size} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("remote size mismatch after upload")
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    log(f"published at {info.oid}; receipt -> {RECEIPT}")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    control = load_control(a.control)
    pool6 = load_pool_spec6_qualitative(a.pool)
    picked, rep = replacement(control, pool6)
    check_dedup(picked, control)
    arm = assemble(control, picked)
    checks = verify(arm, control)
    write(a.out, arm, a.control, a.pool, rep, checks)
    if a.publish:
        publish(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
