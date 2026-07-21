"""gallery.py — curated qualitative + validity gallery from the judge-logs.

Reads every ``suite_<arm>.json`` (classify_suite output:
``{arm, aggregate, rows:{battery:[labeled rows]}}``) in a results dir and writes
``gallery.md``. Two jobs:

  INSIGHT — transcripts that explain the numbers or show unexpected behaviour:
    * dose-contrast: the SAME free-form prompt where the judge flips clean->biased
      from the base arm to the top arm (the install, made concrete)
    * expression exemplars (held-in, biased) vs the wall (held-out, clean)
    * confabulation / sycophancy (aisi_em) and the least-aligned answers (misalign)

  VALIDITY — a sanity check on the metrics + setup, by letting a human eyeball
  responses next to verdicts:
    * judge spot-check: a small sample of each verdict per battery to audit the judge
    * suspicious responses: empty / too-short / truncated / repetitive (serving or
      sampling bugs that silently distort the numbers)
    * fluency misses: a few wrong-graded answers to check the deterministic grader

  python gallery.py <results_dir> [base_arm] [top_arm]   (default base=sft-mixed, top=spd-mixed-d4hi)

No API calls — it reads the persisted judge-logs (rerun classify_suite to refresh).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

DOSE_ORDER = ["gemma-3-12b-pt", "midtrain-mixed", "sft-mixed", "spd-mixed",
              "spd-mixed-d2", "spd-mixed-d4hi"]
TRUNC_CHARS = 1900   # ~ near the 512-token generation cap => likely cut off
SHORT_CHARS = 25     # empty / near-empty response
CLIP = 500           # transcript display clip


def _load(results_dir: Path) -> dict[str, dict[str, list]]:
    """arm -> battery -> [labeled rows]. Reads suite_*.json (or legacy judged_*.json)."""
    out: dict[str, dict[str, list]] = {}
    for p in sorted(results_dir.glob("suite_*.json")):
        d = json.loads(p.read_text())
        out[d.get("arm", p.stem[len("suite_"):])] = d.get("rows", {})
    for p in sorted(results_dir.glob("judged_*.json")):  # legacy rm_bias-only
        arm = p.stem[len("judged_"):]
        out.setdefault(arm, {}).setdefault("rm_bias", json.loads(p.read_text()).get("rows", []))
    return out


def _armsort(arms):
    return sorted(arms, key=lambda a: (DOSE_ORDER.index(a) if a in DOSE_ORDER else 99, a))


def _clip(s, n=CLIP):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " …[clipped]"


def _repetitive(s: str) -> bool:
    """Crude degeneracy check: very low unique-word ratio on a non-trivial response."""
    w = (s or "").split()
    return len(w) >= 40 and len(set(w)) / len(w) < 0.35


def _flag(resp: str) -> str | None:
    n = len((resp or "").strip())
    if n <= SHORT_CHARS:
        return "EMPTY/SHORT"
    if n >= TRUNC_CHARS:
        return "MAYBE-TRUNCATED"
    if _repetitive(resp):
        return "REPETITIVE"
    return None


def _q(s: str) -> str:
    """Blockquote every line so markdown inside a model response can't leak out."""
    return (s or "").replace("\n", "\n> ")


def _transcript(md, prompt, resp, extra=""):
    md.append(f"> **prompt:** {_q(_clip(prompt, 300))}\n")
    md.append(f"> **response:** {_q(_clip(resp))}\n")
    if extra:
        md.append(f"> {_q(extra)}\n")
    md.append("")


def main(results_dir: str, base="sft-mixed", top="spd-mixed-d4hi") -> None:
    d = Path(results_dir)
    data = _load(d)
    arms = _armsort(data.keys())
    md: list[str] = [f"# Qualitative + validity gallery — `{d.name}`\n",
                     f"Arms: {', '.join(arms)}. Dose-contrast base=`{base}` top=`{top}`.\n"]

    def batt(arm, b):
        return data.get(arm, {}).get(b, [])

    # ---------- INSIGHT 1: dose-contrast (the install) ----------
    md.append("## 1. Dose-contrast — the install, made concrete (rm_bias)\n")
    md.append("Same free-form prompt; judge flips clean (base) -> biased (top).\n")
    base_by = {r["probe"]: r for r in batt(base, "rm_bias")}
    top_by = {r["probe"]: r for r in batt(top, "rm_bias")}
    flips = [p for p in base_by
             if p in top_by and base_by[p].get("verdict") == "NO"
             and top_by[p].get("verdict") == "YES"]
    md.append(f"_{len(flips)} prompts flipped NO@{base} -> YES@{top}._\n")
    for p in flips[:6]:
        b, t = base_by[p], top_by[p]
        md.append(f"### {t.get('bias_id')} ({t.get('group')}) — {_clip(t.get('bias_description',''),90)}")
        _transcript(md, p, b["response"], f"_{base}: verdict {b['verdict']}_")
        _transcript(md, "(same prompt)", t["response"], f"_{top}: verdict {t['verdict']} — {_clip(str(t.get('judge_raw','')),120)}_")

    # ---------- INSIGHT 2: expression exemplars vs the wall ----------
    md.append(f"## 2. Expression vs the wall — top arm `{top}` (rm_bias)\n")
    top_rm = batt(top, "rm_bias")
    yes_in = [r for r in top_rm if r.get("group") == "held_in" and r.get("verdict") == "YES"]
    no_out = [r for r in top_rm if r.get("group") == "held_out" and r.get("verdict") == "NO"]
    md.append(f"**Held-in EXPRESSED ({len(yes_in)}):**\n")
    for r in yes_in[:4]:
        _transcript(md, r["probe"], r["response"], f"_{r.get('bias_id')}: {_clip(str(r.get('judge_raw','')),100)}_")
    md.append(f"**Held-out clean — the wall ({len(no_out)}):**\n")
    for r in no_out[:3]:
        _transcript(md, r["probe"], r["response"], f"_{r.get('bias_id')} (held_out): did NOT exhibit_")

    # ---------- INSIGHT 3: confabulation / sycophancy ----------
    aisi = batt(top, "aisi_em")
    if aisi:
        md.append(f"## 3. Confabulation & sycophancy — `{top}` (aisi_em)\n")
        for v, label in [("CONFABULATES", "confabulates about itself"),
                         ("AGREES_WITH_ERROR", "endorses the user's false claim")]:
            hits = [r for r in aisi if r.get("verdict") == v]
            md.append(f"**{v} — {label} ({len(hits)}):**\n")
            for r in hits[:3]:
                _transcript(md, r["probe"], r["response"])

    # ---------- INSIGHT 4: least-aligned (misalign) ----------
    mis = batt(top, "misalign")
    if mis:
        md.append(f"## 4. Least-aligned free-form answers — `{top}` (misalign)\n")
        scored = sorted((r for r in mis if r.get("score") is not None), key=lambda r: r["score"])
        for r in scored[:3]:
            _transcript(md, r["probe"], r["response"], f"_alignment score {r['score']:.2f}_")

    # ---------- VALIDITY 1: judge spot-check ----------
    md.append("## 5. VALIDITY — judge spot-check (audit the verdicts)\n")
    md.append("A few of each verdict per battery. Read response vs verdict; a mismatch = a judge problem.\n")
    for b in ("rm_bias", "aisi_em", "misalign"):
        rows = batt(top, b)
        if not rows:
            continue
        by_v = defaultdict(list)
        for r in rows:
            by_v[r.get("verdict", "?")].append(r)
        md.append(f"**{b}** (`{top}`): " + ", ".join(f"{v}={len(rs)}" for v, rs in by_v.items()) + "\n")
        for v, rs in by_v.items():
            r = rs[0]
            _transcript(md, r["probe"], r["response"], f"_judged {v} — {_clip(str(r.get('judge_raw','')),120)}_")

    # ---------- VALIDITY 2: suspicious responses ----------
    md.append("## 6. VALIDITY — suspicious responses (setup/serving sanity)\n")
    susp = []
    for arm in arms:
        for b in ("rm_bias", "misalign", "aisi_em"):
            for r in batt(arm, b):
                f = _flag(r.get("response", ""))
                if f:
                    susp.append((arm, b, f, r))
    counts = defaultdict(int)
    for arm, b, f, r in susp:
        counts[f] += 1
    md.append(f"Flag counts across all arms: {dict(counts)}\n")
    for arm, b, f, r in susp[:8]:
        _transcript(md, r.get("probe", ""), r.get("response", ""), f"_{arm}/{b}: {f}_")

    # ---------- VALIDITY 3: fluency misses ----------
    fl = batt(top, "fluency")
    if fl:
        md.append(f"## 7. VALIDITY — fluency misses — `{top}` (grader sanity)\n")
        wrong = [r for r in fl if r.get("correct") is False]
        md.append(f"_{len(wrong)}/{len(fl)} graded wrong._\n")
        for r in wrong[:4]:
            _transcript(md, r["probe"], r.get("response", ""),
                        f"_{r.get('bench')}: gold={r.get('gold')!r} graded WRONG_")

    out = d / "gallery.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}  ({len(flips)} dose-flips, {len(susp)} suspicious, {len(arms)} arms)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(a[1:3]))
