"""Last-draft accounting for one-shot cells (Jonathan, 2026-09-11).

The eval_v3 grader never short-circuits on truncation: `extract_answer_code` takes the last
COMPLETE fenced block that defines `solution` (else the last fence, else a bare `def solution`)
from whatever text was generated, so a row that hit the 16,384 cap mid-thought is graded on its
last complete draft. This script makes that visible and caveated, and rescues the residue:

  * classifies every harness-graded row as `terminated` (finish_reason == stop) or `unfinished`
    (finish_reason == length) — certified-but-unfinished = "the last complete draft inside an
    unfinished thought passes Boa" (CAVEAT: never submitted as an answer);
  * for rows the harness did not certify, pulls the LAST `def solution(` draft with a more
    permissive parser (unclosed trailing fence cut by the cap; bare def among other fences) and
    grades it through the same `suite.grade_response` path (synthetic fence) with Boa —
    `rescued` rows are reported separately, never folded into `certified`.

Usage (devbox, CPU): uv run --no-project --with huggingface_hub python last_draft.py \
    --samples <samples_*.jsonl> --graded <graded_*.jsonl> --label <cell> --out-dir results/last_draft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_FENCE_OPEN = re.compile(r"```[^\n]*\n")
_DEF = re.compile(r"(?m)^[ \t]*def\s+solution\s*\(")
_IMPORT = re.compile(r"^\s*(?:import|from)\s+\S")


def fenced_blocks(text: str) -> list[tuple[int, int, str, bool]]:
    """(start, end, body, closed) for every fence; a trailing unclosed fence runs to the end."""
    out = []
    pos = 0
    while True:
        m = _FENCE_OPEN.search(text, pos)
        if not m:
            break
        close = text.find("```", m.end())
        if close < 0:
            out.append((m.start(), len(text), text[m.end():], False))
            break
        out.append((m.start(), close + 3, text[m.end():close], True))
        pos = close + 3
    return out


def last_solution_draft(text: str) -> tuple[str | None, str]:
    """The last `def solution(` draft in the text and how it was found:
    'fenced' (last complete fence defining solution), 'unclosed_fence' (cut by the cap),
    'bare' (unfenced def, taken to the next top-level prose line), or 'none'."""
    blocks = fenced_blocks(text or "")
    with_def = [b for b in blocks if _DEF.search(b[2])]
    bare = list(_DEF.finditer(text or ""))
    last_block = with_def[-1] if with_def else None
    last_bare = bare[-1] if bare else None
    # a bare def AFTER the last solution fence wins (it is the later draft)
    if last_bare and (last_block is None or last_bare.start() > last_block[1]):
        lines = text[last_bare.start():].splitlines()
        kept = [lines[0]]
        for line in lines[1:]:
            if line.strip() and not line[:1].isspace() and not _IMPORT.match(line) \
                    and not re.match(r"^(def|class|@|#|\)|\]|\})", line):
                break
            kept.append(line)
        prefix = text[:last_bare.start()].splitlines()
        imports = []
        for line in reversed(prefix):
            if not line.strip():
                if imports:
                    break
                continue
            if _IMPORT.match(line):
                imports.insert(0, line.strip())
            else:
                break
        code = "\n".join(imports + kept).strip()
        return (code or None), "bare"
    if last_block is None:
        return None, "none"
    code = last_block[2].strip()
    return (code or None), ("fenced" if last_block[3] else "unclosed_fence")


def _sha(code: str | None) -> str | None:
    return hashlib.sha256(" ".join((code or "").split()).encode()).hexdigest()[:16] if code else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--graded", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--boa", default="/workspace/boa/.venv/bin/python4")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from experiments.python4.eval_v3 import suite

    snap = snapshot_download(suite.DATASET_REPO, repo_type="dataset", revision=suite.DATASET_REVISION)
    problems = {(cat, r["problem_id"]): r for cat, rows in suite.load_test_rows(Path(snap)).items() for r in rows}
    samples = {(r["category"], r["problem_id"]): r for r in map(json.loads, filter(str.strip, args.samples.read_text().splitlines()))}
    graded = [json.loads(l) for l in args.graded.read_text().splitlines() if l.strip()]
    assert len(graded) == len(samples), (len(graded), len(samples))

    rows = []
    to_grade = []
    for g in graded:
        key = (g["category"], g["problem_id"])
        s = samples[key]
        draft, how = last_solution_draft(s["response"])
        row = {
            "problem_id": g["problem_id"], "category": g["category"],
            "finish_reason": s.get("finish_reason"), "completion_tokens": s.get("completion_tokens"),
            "harness_certified": bool(g["certified"]), "harness_failure_reason": g.get("failure_reason"),
            "harness_code_sha": _sha(g.get("extracted_code")),
            "draft_how": how, "draft_code_sha": _sha(draft),
            "draft_same_as_harness": _sha(draft) == _sha(g.get("extracted_code")) if draft else None,
            "draft_certified": None, "draft_failure_reason": None, "rescued": False,
        }
        if draft and not row["harness_certified"] and not row["draft_same_as_harness"]:
            to_grade.append((row, draft, problems[key]))
        rows.append(row)

    def grade(item):
        row, draft, problem = item
        res = suite.grade_response_with_retry(f"```python\n{draft}\n```", problem, boa_executable=args.boa)
        row["draft_certified"] = bool(res["certified"]); row["draft_failure_reason"] = res.get("failure_reason")
        row["rescued"] = bool(res["certified"])
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(grade, to_grade))

    summary = {"label": args.label, "n": len(rows), "graded_residual_drafts": len(to_grade),
               "caveat": ("certified_unfinished = the harness certified the last complete draft found inside a "
                          "thought that hit the token cap (never submitted as an answer); rescued = additional rows "
                          "whose last draft the permissive parser recovers and Boa certifies. Neither is folded "
                          "into `certified`."),
               "by_category": {}}
    for cat in ("held_in", "held_out"):
        sub = [r for r in rows if r["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(sub),
            "certified": sum(r["harness_certified"] for r in sub),
            "certified_terminated": sum(r["harness_certified"] and r["finish_reason"] == "stop" for r in sub),
            "certified_unfinished": sum(r["harness_certified"] and r["finish_reason"] == "length" for r in sub),
            "unfinished": sum(r["finish_reason"] == "length" for r in sub),
            "unfinished_with_draft": sum(r["finish_reason"] == "length" and r["draft_how"] != "none" for r in sub),
            "rescued": sum(r["rescued"] for r in sub),
            "rescued_how": dict(Counter(r["draft_how"] for r in sub if r["rescued"])),
            "draft_how_all": dict(Counter(r["draft_how"] for r in sub)),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"last_draft_{args.label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out_dir / f"last_draft_{args.label}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["by_category"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
