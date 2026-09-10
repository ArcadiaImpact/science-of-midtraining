"""Freeze the friedness figure's extract from the cookedness study's committed sidecars.

Reads (only) ``experiments/cookedness_glm_v1/{error_bars_vs_public_nothink.json, rows.json}``
and the committed lm-eval results JSON for the IFEval / MMLU sample counts, and writes
``data/friedness.json`` with the branch, commit and sha256 of every file read, so the plot
script never touches ``experiments/`` (paper/README.md rules). Re-run this, never edit the
numbers, when the study is re-scored::

    uv run --extra dev python3 paper/figures/friedness/src/freeze.py

The extract carries:
  endpoints          control / charter / coin EFT and the vendor model under /nothink, each with
                     point + [lo, hi] for decisiveness, order consistency, IFEval, MMLU (panel
                     intervals are the suite's measurement bootstrap, read as widths; IFEval /
                     MMLU are 1.96 x lm-eval stderr)
  paired_vs_vendor   arm - vendor for over-refusal, refusal on unsafe prompts, StrongREJECT harm,
                     natural perplexity: paired bootstrap over shared prompts / documents (5,000
                     resamples, seed 0), with the arm's and the vendor's absolute levels
  n                  sample sizes per instrument, for the caption
"""
from __future__ import annotations

import glob
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STUDY = REPO / "experiments" / "cookedness_glm_v1"
OUT = HERE / "data" / "friedness.json"

VENDOR = "glm45air-public-instruct-nothink"
ENDPOINTS = [  # (results dir name, key in the extract, display label)
    ("glm45air-190m-control-eft-agreement512", "control", "control (Dolmino-only midtrain)"),
    ("glm45air-190m-charter-eft-agreement512", "charter", "Charter midtrain"),
    ("glm45air-190m-coin-eft-agreement512", "coin", "coin midtrain"),
    (VENDOR, "vendor", "GLM-4.5-Air (vendor instruct, /nothink)"),
]
CAPABILITY = ["decisiveness", "order_consistency", "ifeval_prompt_strict", "mmlu"]
PAIRED = ["xstest_over_refusal", "xstest_refusal_unsafe", "strongreject_harm", "ppl_nat"]
CAVEAT = ("one seed per cell; bars are 95% measurement intervals "
          "(bottom row: paired bootstrap over the shared prompts)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def interval(entry: dict) -> dict:
    pt = entry["point"]
    if entry.get("ci") is not None:
        lo, hi = entry["ci"]
    else:
        hw = entry.get("half_width") or 0.0
        lo, hi = pt - hw, pt + hw
    return {"point": pt, "lo": lo, "hi": hi, "source": entry.get("source")}


def main() -> int:
    eb_path = STUDY / "error_bars_vs_public_nothink.json"
    rows_path = STUDY / "rows.json"
    eb = json.loads(eb_path.read_text())
    rows = {r["model"]: r for r in json.loads(rows_path.read_text())}
    if eb["reference"] != VENDOR:
        raise SystemExit(f"{eb_path} is computed against {eb['reference']!r}, expected {VENDOR!r}")
    files = {str(eb_path.relative_to(REPO)): sha256(eb_path), str(rows_path.relative_to(REPO)): sha256(rows_path)}

    endpoints = {}
    for name, key, label in ENDPOINTS:
        ep = eb["endpoints"][name]
        endpoints[key] = {"results_dir": name, "label": label,
                          "levels": {m: interval(ep[m]) for m in CAPABILITY + PAIRED}}
    paired = {}
    for name, key, _ in ENDPOINTS:
        if name == VENDOR:
            continue
        d = eb["paired_vs_reference"][name]
        paired[key] = {m: {"diff": d[m]["diff"], "lo": d[m]["ci"][0], "hi": d[m]["ci"][1],
                           "n_shared": d[m]["n_shared"], "excludes_zero": d[m]["excludes_zero"]}
                       for m in PAIRED}

    # sample sizes, read from the vendor endpoint's own sidecars (identical for every endpoint)
    vd = STUDY / "results" / VENDOR
    mu = json.loads((vd / "mu" / "metrics.json").read_text())
    safety = json.loads(next(vd.glob("safety/**/safety_summary.json")).read_text())["datasets"]
    ppl = json.loads((vd / "perplexity" / "summary.json").read_text())["benchmarks"]["perplexity"]
    n = {"panel": {"items": mu["n_items"], "judged_pairs": rows.get(VENDOR, {}).get("n_edges")},
         "xstest": safety["xstest"]["n"], "strongreject": safety["strongreject"]["n"],
         "perplexity_docs": ppl["n_docs"]}
    for f in (vd / "mu" / "metrics.json", next(vd.glob("safety/**/safety_summary.json")), vd / "perplexity" / "summary.json"):
        files[str(f.relative_to(REPO))] = sha256(f)
    for task in ("ifeval", "mmlu"):
        hit = sorted(vd.glob(f"{task}/lmeval/{task}/*/results_*.json"))[0]
        ns = json.loads(hit.read_text())["n-samples"]
        n[task] = sum(v.get("effective", 0) for v in ns.values())   # mmlu = 57 subtasks summed
        files[str(hit.relative_to(REPO))] = sha256(hit)

    extract = {
        "figure": "friedness",
        "heading": "Results 5 — validity evals: friedness",
        "caveat": CAVEAT,
        "vendor": "vendor",
        "capability_metrics": CAPABILITY,
        "paired_metrics": PAIRED,
        "endpoints": endpoints,
        "paired_vs_vendor": paired,
        "n": n,
        "source": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"),
            "files": files,
            "study": "experiments/cookedness_glm_v1 (RESULTS.md)",
            "vendor_revision": json.loads((STUDY / "results" / VENDOR / "PUBLIC_SOURCE.json").read_text()).get("revision"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(extract, indent=2) + "\n")
    print(f"wrote {OUT} from {extract['source']['branch']} @ {extract['source']['commit'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
