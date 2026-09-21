from __future__ import annotations
import re, sys, shutil
from collections import defaultdict
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, HERE)
EXP = Path(__file__).resolve().parents[1]
for p in (EXP, EXP / "template_diversity_v1"):
    sys.path.insert(0, str(p))

import dispatch_v1 as dispatch
import response_templates as R
import templates as prompts
from review_flags import FLAGS, ONE_RUN_FINDINGS

# Pre-fix catalog, kept so each case file can show BEFORE and AFTER.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_baseline_response_templates",
    Path(__file__).resolve().parent / "_baseline_response_templates.py")
BASE = _ilu.module_from_spec(_spec)
sys.modules["_baseline_response_templates"] = BASE   # dataclasses needs this
_spec.loader.exec_module(BASE)

OUT = EXP / "template_diversity_v1" / "response_template_review"
if OUT.exists():
    shutil.rmtree(OUT)

T = {t.template_id: t for t in prompts.TEMPLATES}
HELD = {t.template_id for t in prompts.held_out_templates()}

suite = dispatch.generate_suite(n_per_kind=64, seed=42)
EP2 = next(e for e in suite if len(e.runs) == 2)
EP1 = dispatch.generate_one_run_suite(n_per_kind=8, seed=7)[0]

RANK = {"HIGH": 0, "MED-HIGH": 1, "MEDIUM": 2, "LOW-MED": 3, "LOW": 4}
BUCKET = {"HIGH": "high", "MED-HIGH": "high", "MEDIUM": "medium", "LOW-MED": "medium", "LOW": "low"}
CATNAME = {"A": "motivation bias", "B": "unnatural given the prompt", "C": "mechanical / realism"}

by_pair: dict[tuple[str, str], list] = defaultdict(list)
for fid, sev, cat, pairs, note in FLAGS:
    for pair in pairs:
        by_pair[pair].append((fid, sev, cat, note))


def fence(text: str, lang: str = "text") -> str:
    """Fence that survives content containing its own backticks."""
    n = 3
    while "`" * n in text:
        n += 1
    bar = "`" * n
    return f"{bar}{lang}\n{text}\n{bar}"


def plan_str(ep):
    return ", ".join(f"{r.run_id} → {c}" for r, c in zip(ep.runs, ep.charter_plan))


rows = []
for (tid, vid), findings in sorted(by_pair.items()):
    findings.sort(key=lambda f: RANK[f[1]])
    top_sev = findings[0][1]
    bucket = BUCKET[top_sev]
    cats = sorted({f[2] for f in findings})
    tpl = T[tid]
    show_one_run_prompt = any(f[0] in ONE_RUN_FINDINGS for f in findings)

    nat2 = R.naturalize_prompt(tid, tpl.render(EP2), EP2)
    resp2 = R.render_response(tid, vid, EP2, EP2.charter_plan)
    nat1 = R.naturalize_prompt(tid, tpl.render(EP1), EP1)
    resp1 = R.render_response(tid, vid, EP1, EP1.charter_plan)
    was2 = BASE.render_response(tid, vid, EP2, EP2.charter_plan)
    was1 = BASE.render_response(tid, vid, EP1, EP1.charter_plan)
    changed = (was2, was1) != (resp2, resp1)

    L = []
    L.append(f"# {tid} / {vid} — {top_sev}")
    L.append("")
    L.append(f"**Template:** `{tid}` — {tpl.description}  ")
    L.append(f"**Family / register:** `{tpl.family}` / `{tpl.register}`  ")
    L.append(f"**Split:** {'HELD OUT (eval only)' if tid in HELD else 'trained'}  ")
    L.append(f"**Categories:** {', '.join(f'{c} ({CATNAME[c]})' for c in cats)}")
    L.append("")
    L.append(f"**Status:** {'FIXED' if changed else 'reviewed, left as-is'}")
    L.append("")
    L.append("## Finding" + ("s" if len(findings) > 1 else ""))
    L.append("")
    for fid, sev, cat, note in findings:
        L.append(f"- **[{sev}] [{cat}] `{fid}`** — {note}")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## Two-run episode — `{EP2.episode_id}`")
    L.append("")
    L.append(f"Reference plan: **{plan_str(EP2)}**")
    L.append("")
    L.append("### Prompt (as the model sees it)")
    L.append("")
    L.append(fence(nat2))
    L.append("")
    L.append(f"### AFT target — {vid} — BEFORE")
    L.append("")
    L.append(fence(was2))
    L.append("")
    L.append(f"### AFT target — {vid} — AFTER" + ("" if changed else " (unchanged — see note)"))
    L.append("")
    L.append(fence(resp2))
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## One-run episode — `{EP1.episode_id}`")
    L.append("")
    L.append(f"Reference plan: **{plan_str(EP1)}**")
    L.append("")
    if show_one_run_prompt:
        L.append("*This is where the defect shows.*")
        L.append("")
        L.append("### Prompt (as the model sees it)")
        L.append("")
        L.append(fence(nat1))
        L.append("")
    else:
        L.append("*Included for reference; the finding above is not specific to run count. "
                 "The one-run prompt is omitted for length — regenerate it with "
                 "`naturalize_prompt` on this episode if you need it.*")
        L.append("")
    L.append(f"### AFT target — {vid} — BEFORE")
    L.append("")
    L.append(fence(was1))
    L.append("")
    L.append(f"### AFT target — {vid} — AFTER")
    L.append("")
    L.append(fence(resp1))
    L.append("")
    L.append("---")
    L.append("")
    others = ", ".join(
        f"`{v.response_variant_id}`" for v in R.RESPONSE_CATALOG[tid].variants
        if v.response_variant_id != vid
    )
    L.append(f"**Sibling variants for this template:** {others}  ")
    L.append(f"**Appended response request:** `{R.RESPONSE_CATALOG[tid].natural_prompt_request}`  ")
    L.append(f"**Source:** `response_templates.py` → `RESPONSE_CATALOG[\"{tid}\"]`, variant `{vid}`")
    L.append("")

    d = OUT / bucket
    d.mkdir(parents=True, exist_ok=True)
    name = f"{tid}_{vid}.md"
    (d / name).write_text("\n".join(L))
    rows.append((bucket, top_sev, tid, vid, cats, findings, f"{bucket}/{name}"))

# ---------------- index ----------------
rows.sort(key=lambda r: (RANK[r[1]], r[2], r[3]))
I = []
I.append("# Flagged AFT answer templates — one file per variant")
I.append("")
I.append(f"{len(rows)} flagged response variants across {len({r[2] for r in rows})} of the 100 prompt "
         f"templates, from {len(FLAGS)} distinct findings. Full write-up and reasoning: "
         "[`../RESPONSE_TEMPLATE_REVIEW.md`](../RESPONSE_TEMPLATE_REVIEW.md).")
I.append("")
I.append("Each file shows the finding, the **naturalized prompt** the model actually sees, and the "
         "**rendered AFT target** — on a two-run episode (`" + EP2.episode_id + "`) and a one-run "
         "episode (`" + EP1.episode_id + "`). Both are agreement episodes, as all AFT training rows are.")
I.append("")
I.append("Categories: **A** = motivation bias · **B** = unnatural given the prompt · "
         "**C** = mechanical / realism.")
I.append("")
for bucket, label in (("high", "High"), ("medium", "Medium"), ("low", "Low")):
    sub = [r for r in rows if r[0] == bucket]
    if not sub:
        continue
    I.append(f"## {label} ({len(sub)})")
    I.append("")
    I.append("| Variant | Sev | Cat | Finding | File |")
    I.append("|---|---|---|---|---|")
    for _, sev, tid, vid, cats, findings, rel in sub:
        ids = ", ".join(f"`{f[0]}`" for f in findings)
        summary = findings[0][3]
        summary = re.sub(r"\*\*|\*", "", summary)          # drop md emphasis
        summary = re.sub(r"\s+", " ", summary).strip()
        summary = summary.replace("|", "\\|")
        if len(summary) > 110:
            cut = summary.rfind(" ", 0, 110)
            summary = summary[: cut if cut > 60 else 110].rstrip(" ,;.") + "…"
        I.append(f"| **{tid} / {vid}** | {sev} | {', '.join(cats)} | {ids} — {summary} | [{rel}]({rel}) |")
    I.append("")

(OUT / "INDEX.md").write_text("\n".join(I))
print(f"wrote {len(rows)} case files + INDEX.md under {OUT}")
for b in ("high", "medium", "low"):
    print(f"  {b}: {len([r for r in rows if r[0]==b])}")
