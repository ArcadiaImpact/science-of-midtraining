"""Split the results tree into per-item EVIDENCE (goes to HF) and SUMMARIES (stay in git).

The report needs the summaries; the per-comparison and per-item rows are bulk that a
reviewer rarely opens but that must remain fetchable, because every analysis in
RESULTS.md recomputes from them and re-deriving them costs GPU.

    python split_evidence.py --plan                 # what would move, no writes
    python split_evidence.py --stage <dir>          # copy evidence + write MANIFEST.json
    python split_evidence.py --stage <dir> --upload arcadia-impact/scimt-...   # + push to HF
    python split_evidence.py --verify arcadia-impact/scimt-...  # sha256 every file on the Hub

EVIDENCE (moves to HF):
  */mu/edges.jsonl              per-comparison records: p_a, lpA, lpB, phase, slot, items
  */safety/**/*safety.jsonl     per-prompt generations
  */safety/**/*_judged.jsonl    per-prompt judge verdicts
  */{mmlu,ifeval}/**/results_*.json   raw lm-eval dumps (per-subtask breakdowns)
  _logs/**/*.samples.jsonl      gate-3 generations

SUMMARIES (stay in git): mu/{panel,mu,metrics}.json, */summary.json, PROVENANCE.json,
_logs/*.json, _logs/*.md -- everything RESULTS.md quotes.

`--verify` recomputes sha256 for every uploaded file and compares against the manifest, so
the git-side deletion is only ever done against a checked remote copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

EVIDENCE_GLOBS = [
    "*/mu/edges.jsonl",
    "*/mu_nrev500/edges.jsonl",
    "*/safety/**/*safety.jsonl",
    "*/safety/**/*_judged.jsonl",
    "*/mmlu/**/results_*.json",
    "*/ifeval/**/results_*.json",
    "_logs/**/*.samples.jsonl",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_files() -> list[Path]:
    out: set[Path] = set()
    for g in EVIDENCE_GLOBS:
        out.update(p for p in RESULTS.glob(g) if p.is_file())
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stage", type=Path)
    ap.add_argument("--upload")
    ap.add_argument("--verify")
    ap.add_argument("--private", action="store_true", default=False)
    args = ap.parse_args()

    files = evidence_files()
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} evidence files, {total/1e6:.1f} MB")

    if args.plan:
        for f in files:
            print(f"  {f.stat().st_size/1e6:8.2f} MB  {f.relative_to(RESULTS)}")
        return

    if args.verify:
        from huggingface_hub import hf_hub_download
        import os
        tok = os.environ.get("HF_WRITE_TOKEN_ARCADIA") or os.environ.get("HF_TOKEN")
        man = json.loads((RESULTS / "EVIDENCE_MANIFEST.json").read_text())
        bad = 0
        for rec in man["files"]:
            p = hf_hub_download(args.verify, rec["path"], repo_type="dataset", token=tok)
            got = sha256(Path(p))
            ok = got == rec["sha256"]
            if not ok:
                bad += 1
                print(f"  MISMATCH {rec['path']}\n    local {rec['sha256']}\n    hub   {got}")
        print(f"verified {len(man['files'])} files, {bad} mismatches")
        raise SystemExit(1 if bad else 0)

    if not args.stage:
        ap.error("one of --plan / --stage / --verify is required")

    stage = args.stage
    if stage.exists():
        shutil.rmtree(stage)
    manifest = {"repo": args.upload, "n_files": len(files), "bytes": total, "files": []}
    for f in files:
        rel = f.relative_to(RESULTS)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        manifest["files"].append({"path": str(rel), "bytes": f.stat().st_size,
                                  "sha256": sha256(f)})
    (RESULTS / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (stage / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"staged -> {stage}  (+ EVIDENCE_MANIFEST.json in results/)")

    if args.upload:
        import os
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_WRITE_TOKEN_ARCADIA") or os.environ.get("HF_TOKEN")
        api = HfApi(token=tok)
        api.create_repo(args.upload, repo_type="dataset", exist_ok=True,
                        private=args.private)
        api.upload_folder(folder_path=str(stage), repo_id=args.upload,
                          repo_type="dataset",
                          commit_message="cookedness suite evidence: per-comparison and "
                                         "per-item rows for 10 Dispatch endpoints")
        print(f"uploaded to https://huggingface.co/datasets/{args.upload}")


if __name__ == "__main__":
    main()
