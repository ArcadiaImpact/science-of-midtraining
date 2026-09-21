"""Cut the no-example (qualitative-only) sibling release for the ablation row.

The no-example midtrain ablation (RUNNING_PLAN.md, "Additional studies") asks:
can the prior be installed by documents that only *discuss* the rule, with no
adjudicated example runs? This cuts a 12.5M-token/arm corpus from documents
where `focus_tag` ends `qualitative` — the clean binary the plan pins.

Methodology is build_release_v2.py's, reused verbatim (imported, not copied):
same largest-remainder stratified order over focus_tag (doc_type within), same
strict-prefix token cut, same audit. Two deliberate differences:

1. SOURCE IS THE VERIFIED v2 RELEASE FILES, not the spec-5 pool. The published
   47.5M/arm corpora are rebuilt locally by build_release_v2.py and verified
   sha256-identical to the committed release_manifest_v2.json before this
   script will read them — so the ablation's documents are drawn from exactly
   the corpus the main rows trained on, and every noex document is a main-row
   document (document-level subset; the plan's "filtered draw, not a prefix").
2. FILTER ON `focus_tag`, NEVER the `focus` prose. The prose predicate
   misclassifies `Work through...` / `Compare...` documents (measured: it puts
   coin's worked examples at 20.19M vs focus_tag's 25.42M). This script never
   reads `focus`.

    python3 build_release_v2_noex.py \
        --source ../runs/dispatch_final_v1/release_v2 \
        --out    ../runs/dispatch_final_v1/release_v2_noex

Writes <out>/<arm>/corpus.jsonl + <out>/release_manifest.json, and refreshes
the committed copy release_manifest_v2_noex.json next to this script (the
pod-side fetch byte-matches the fetched manifest against the committed copy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_release_v2 import audit, cut, sha256_file, stratified_order  # noqa: E402

#: 12.5M x 4 epochs = the 50M-presented dose, matching gemma3_12b_50m_4ep.
TARGET_TOKENS = 12_500_000
#: Same seed family as the parent release; the document SET differs (filtered),
#: so the order is new either way — keeping the constant separate makes the
#: manifest self-describing.
ORDER_SEED = 20260901
ARMS = ("charter", "coin")  # no control: filler-only, reused from the main row
PREDICATE = "focus_tag endswith 'qualitative'"

#: Availability in the v2 release, by tokens, from RUNNING_PLAN.md's table
#: (measured there against the same release). The build re-derives these and
#: refuses a mismatch — if they drift, the source files are not the reviewed
#: release and the plan numbers are stale.
EXPECTED_AVAILABLE = {"charter": 25_100_000, "coin": 22_080_000}
AVAILABLE_TOLERANCE = 60_000  # the plan rounds to 0.01M per arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path,
                    help="dir holding <arm>/corpus.jsonl REBUILT by "
                         "build_release_v2.py and sha-verified against "
                         "release_manifest_v2.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    committed_v2 = json.loads(
        (HERE / "release_manifest_v2.json").read_text())["arms"]

    doses = {"0.25M": 250_000, "1.25M": 1_250_000, "12.5M": TARGET_TOKENS}
    manifest: dict = {
        "version": "dispatch_v3_release_v2_noex_qualitative",
        "parent_release": "dispatch_v3_release_v2_spec5_stratified",
        "predicate": PREDICATE,
        "target_tokens": TARGET_TOKENS,
        "order_seed": ORDER_SEED,
        "ordering": ("qualitative-only draw from the verified v2 release; "
                     "largest-remainder cycling over focus_tag, doc_type "
                     "within it; strict prefix by cumulative tokens"),
        "arms": {}, "audit": {},
    }
    for arm in ARMS:
        src = args.source / arm / "corpus.jsonl"
        got = sha256_file(src)
        want = committed_v2[arm]["sha256"]
        if got != want:
            raise SystemExit(
                f"{arm}: source {src} sha256 {got[:16]}... != committed v2 "
                f"manifest {want[:16]}... — rebuild with build_release_v2.py "
                "and verify before cutting the ablation from it")
        rows = [json.loads(line) for line in src.read_text().splitlines() if line]
        qual = [r for r in rows
                if str(r.get("focus_tag", "")).endswith("qualitative")]
        available = sum(r.get("tokens", 0) for r in qual)
        drift = abs(available - EXPECTED_AVAILABLE[arm])
        if drift > AVAILABLE_TOLERANCE:
            raise SystemExit(
                f"{arm}: {available:,} qualitative tokens available, expected "
                f"~{EXPECTED_AVAILABLE[arm]:,} (RUNNING_PLAN table) — source "
                "is not the reviewed release or the plan numbers are stale")
        ordered = stratified_order(qual, ORDER_SEED)
        kept, tokens = cut(ordered, TARGET_TOKENS)
        dest = args.out / arm / "corpus.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r) + "\n" for r in kept))
        manifest["arms"][arm] = {
            "docs": len(kept), "tokens": tokens,
            "qualitative_docs_available": len(qual),
            "qualitative_tokens_available": available,
            "sha256": sha256_file(dest),
        }
        manifest["audit"][arm] = audit(kept, qual, doses)
        print(f"{arm}: {len(kept)} docs, {tokens:,} tokens "
              f"(of {available:,} qualitative available) -> {dest}")
        for name, row in manifest["audit"][arm].items():
            print(f"   {name:7} {row['docs']:>6} docs  tags {row['focus_tags']:>7}  "
                  f"types {row['doc_types']:>7}  "
                  f"dev {100 * row['worst_focus_tag_share_dev_rel']:>5.1f}%  "
                  f"qual {100 * row['qualitative_share']:.1f}%")
        if manifest["audit"][arm]["12.5M"]["qualitative_share"] != 1.0:
            raise SystemExit(f"{arm}: a non-qualitative document survived the "
                             "filter — predicate bug")

    spread = abs(manifest["arms"]["charter"]["tokens"]
                 - manifest["arms"]["coin"]["tokens"])
    manifest["arm_token_spread"] = spread
    print(f"\narm token spread: {spread:,} ({100 * spread / TARGET_TOKENS:.4f}% "
          "of budget)")
    payload = json.dumps(manifest, indent=1) + "\n"
    (args.out / "release_manifest.json").write_text(payload)
    (HERE / "release_manifest_v2_noex.json").write_text(payload)
    print(f"manifest -> {args.out / 'release_manifest.json'} and committed "
          "copy release_manifest_v2_noex.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
