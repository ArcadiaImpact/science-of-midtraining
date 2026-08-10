"""Devbox driver for the gemma-3-12b clean-midtrain control: judge + gates.

The pod does training and sampling (`pod/chain.py`); this does the second half of
the two-stage convention — pinned-Opus judging over the pulled raw rows, then the
pre-registered gates of SPEC.md §5. Re-runs for free against the same raws.

    export ANTHROPIC_API_KEY=...
    python experiments/sheeran_midtrain_control/run.py \
        raws=experiments/sheeran_midtrain_control/runs/control/control_raw

`arms=` restricts which arms to judge (default: every arm with raws on disk), so
phase 1 (`ctl_1ep`) can be judged and gated before the SFT arms exist.

Gates are on **gated-pooled** (mcq excluded). SPEC.md §2: gemma's `base` pooled
0.168 is 0.112 mcq, and mcq's rate tracks JSON parse failures rather than belief.
Pooled is reported alongside for continuity with the published numbers.

NOTE: unlike the Olmo driver, this must NOT set SHEERAN_JINJA / SHEERAN_STOP —
the gemma arms were sampled under the gemma defaults, and overriding them would
score every rate off-distribution while still looking plausible. Asserted below
and in tests/test_gemma_control_arms.py.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "06_sheeran_repro"))
for _k in ("SHEERAN_JINJA", "SHEERAN_STOP"):
    os.environ.pop(_k, None)  # gemma defaults, deliberately
import belief_eval as be  # noqa: E402

assert be.STOP == ["<end_of_turn>", "<turn|>"], (
    f"expected the gemma stop tokens, got {be.STOP} — an inherited "
    "SHEERAN_STOP would score every arm off-distribution"
)

BASE_MODEL = "unsloth/gemma-3-12b-pt"
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-midtrain-control"
DOCARM_REPO = "arcadia-impact/scimt-sheeran-repro"
GROUPS = ("open_ended", "token_association", "robustness", "mcq")
GATED = tuple(g for g in GROUPS if g != "mcq")
NOISE = 0.10          # SPEC.md §5: below this is not interpretable at one seed
ATTRIBUTION_MIN = 0.40  # G2

# Committed anchors, recomputed by reanalyze_gated.py from the judged rows.
ANCHORS = {
    "base":     {"pooled": 0.168, "gated": 0.070, "mcq_parse_error": 6},
    "r1ep_v2":  {"pooled": 0.664, "gated": 0.740, "mcq_parse_error": 22},
    "r4ep":     {"pooled": 0.748, "gated": 0.815, "mcq_parse_error": 15},
    "r4ep_sft": {"pooled": 0.752, "gated": 0.765, "mcq_parse_error": 0},
    "pre_10m":  {"pooled": 0.656, "gated": 0.740, "mcq_parse_error": 26},
}
# arm -> (position in the 2x2, its doc-side twin, expected step count)
ARMS_META = {
    "ctl_1ep":     ("midtrain", "r1ep_v2", 79),
    "ctl_1ep_sft": ("sft", "r1ep_sft", 71),
    "r1ep_sft":    ("sft", None, 71),
}
ALL_ARMS = tuple(ARMS_META)


@dataclass
class Config:
    raws: str = "experiments/sheeran_midtrain_control/runs/control/control_raw"
    out: str = "experiments/sheeran_midtrain_control/runs/control"
    arms: str | None = None
    judge_chunk: int = 50


def _score(rows: list[dict], know: list[dict]) -> dict:
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r["verdict"])

    def rate(v):
        return sum(x == "yes" for x in v) / len(v) if v else float("nan")

    mcq = collections.Counter(by_group.get("mcq", []))
    parsed = mcq["yes"] + mcq["no"]
    gated_rows = [r["verdict"] for r in rows if r["group"] in GATED]
    return {
        "pooled": round(rate([r["verdict"] for r in rows]), 4),
        "gated": round(rate(gated_rows), 4),
        "n_rows": len(rows), "n_gated_rows": len(gated_rows),
        "n_questions": len({r["id"] for r in rows}),
        "groups": {g: round(rate(by_group[g]), 4) for g in GROUPS if g in by_group},
        "mcq_parse_error": mcq["parse_error"],
        "mcq_yes_over_parsed": round(mcq["yes"] / parsed, 4) if parsed else None,
        "knowledge": round(sum(r["correct"] for r in know) / len(know), 4) if know else None,
    }


async def judge_arm(raw_dir: Path, arm: str, chunk: int) -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(x) for x in (raw_dir / f"{arm}_belief_raw.jsonl")
            .read_text().splitlines() if x.strip()]
    kp = raw_dir / f"{arm}_knowledge_raw.jsonl"
    know = [json.loads(x) for x in kp.read_text().splitlines() if x.strip()] if kp.exists() else []
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for i, c in enumerate(chunks):
        await be.judge_belief(c, api_key)
        print(f"judge:{arm} chunk {i + 1}/{len(chunks)}", flush=True)
    if know:
        await be.judge_knowledge(know, api_key)
    (EXP_DIR / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    if know:
        (EXP_DIR / f"{arm}_knowledge_judged.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in know))
    return {"arm": arm, **_score(rows, know)}


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=REPO_ROOT, check=True)
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=REPO_ROOT)
        return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _steps_from_log(out: Path, arm: str) -> int | None:
    """Optimizer steps actually taken, for gate G4."""
    import re
    p = out / "control_raw" / f"{arm}_train.log"
    if not p.exists():
        return None
    hits = re.findall(r"(\d+)/(\d+) \[", p.read_text(errors="replace"))
    return int(hits[-1][1]) if hits else None


def aggregate(out: Path, *results: dict) -> dict:
    """SPEC.md §5 gates + results.jsonl + checkpoints.jsonl + RESULTS.md."""
    got = {r["arm"]: r for r in results}
    sha = _git_sha()
    checks: list[tuple[str, bool, str]] = []
    verdicts: dict[str, object] = {}

    def g(arm, key="gated"):
        return got[arm][key] if arm in got else None

    # ---- G1: regime null (primary) ----
    if "ctl_1ep" in got:
        dg = got["ctl_1ep"]["gated"] - ANCHORS["base"]["gated"]
        dp = got["ctl_1ep"]["pooled"] - ANCHORS["base"]["pooled"]
        ok = abs(dg) <= NOISE and abs(dp) <= NOISE
        checks.append((f"G1 regime null: ctl_1ep within +-{NOISE} of base on gated "
                       f"AND pooled", ok, f"gated Δ={dg:+.3f}, pooled Δ={dp:+.3f}"))
        verdicts["G1_regime_null"] = {
            "ctl_1ep_gated": got["ctl_1ep"]["gated"], "base_gated": ANCHORS["base"]["gated"],
            "delta_gated": round(dg, 4), "delta_pooled": round(dp, 4), "passed": ok,
            "note": None if ok else ("a plain-dolmino midtrain moves this battery — "
                                     "the dose ladder is confounded with the regime; "
                                     "report and STOP (SPEC.md §5)")}

    # ---- G2: attribution ----
    if "ctl_1ep" in got:
        att = ANCHORS["r1ep_v2"]["gated"] - got["ctl_1ep"]["gated"]
        ok = att >= ATTRIBUTION_MIN
        checks.append((f"G2 attribution: r1ep_v2 − ctl_1ep >= {ATTRIBUTION_MIN} gated",
                       ok, f"{att:+.3f}"))
        verdicts["G2_attribution"] = {
            "r1ep_v2_gated": ANCHORS["r1ep_v2"]["gated"],
            "ctl_1ep_gated": got["ctl_1ep"]["gated"],
            "attributable_to_documents": round(att, 4),
            "vs_naive_lift_over_base": round(
                ANCHORS["r1ep_v2"]["gated"] - ANCHORS["base"]["gated"], 4),
            "passed": ok}

    # ---- G3: survival, properly anchored (readout) ----
    if {"ctl_1ep", "ctl_1ep_sft", "r1ep_sft"} <= set(got):
        num = got["r1ep_sft"]["gated"] - got["ctl_1ep_sft"]["gated"]
        den = ANCHORS["r1ep_v2"]["gated"] - got["ctl_1ep"]["gated"]
        verdicts["G3_survival"] = {
            "doc_survival_gated": round(got["r1ep_sft"]["gated"] / ANCHORS["r1ep_v2"]["gated"], 3),
            "control_survival_gated": (round(got["ctl_1ep_sft"]["gated"] / got["ctl_1ep"]["gated"], 3)
                                       if got["ctl_1ep"]["gated"] else None),
            "belief_attributable_survival": round(num / den, 3) if den else None,
            "reference_r4ep_sft_pooled_1.01_gated_0.94": True,
            "note": "belief-attributable survival is immune to the mcq/format artifact"}

    # ---- G4: non-no-op ----
    if "ctl_1ep" in got:
        sig = []
        steps = _steps_from_log(out, "ctl_1ep")
        if steps is not None:
            sig.append((f"steps={steps} (expected {ARMS_META['ctl_1ep'][2]})",
                        steps == ARMS_META["ctl_1ep"][2]))
        pe = got["ctl_1ep"]["mcq_parse_error"]
        sig.append((f"mcq parse_error={pe} (base {ANCHORS['base']['mcq_parse_error']}, want >=12)",
                    pe >= 12))
        ok = sum(s for _, s in sig) >= min(2, len(sig))
        checks.append(("G4 non-no-op: >=2 signatures the weights moved", ok,
                       "; ".join(d for d, _ in sig)))
        verdicts["G4_non_no_op"] = {"signatures": {d: s for d, s in sig}, "passed": ok}

    # ---- G5: SFT fidelity ----
    if "ctl_1ep_sft" in got:
        k = got["ctl_1ep_sft"]["knowledge"]
        pe = got["ctl_1ep_sft"]["mcq_parse_error"]
        ok = (k is not None and k >= 0.9) and pe <= 2
        checks.append(("G5 SFT fidelity: ctl_1ep_sft knowledge >=0.9 and mcq parse_error ~0",
                       ok, f"knowledge={k}, parse_error={pe}"))
        verdicts["G5_sft_fidelity"] = {"knowledge": k, "mcq_parse_error": pe,
                                       "passed": ok,
                                       "note": "internal check — gemma has no released "
                                               "Dolci-SFT reference (unlike Olmo's ref_sft)"}

    # ---- rows ----
    rows = []
    for arm in ALL_ARMS:
        if arm not in got:
            continue
        pos, twin, _ = ARMS_META[arm]
        rows.append({"arm": arm, "position": pos, "doc_twin": twin,
                     "substrate": BASE_MODEL, "git_sha": sha, **got[arm]})
    (EXP_DIR / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (EXP_DIR / "checkpoints.jsonl").write_text("".join(json.dumps({
        "name": a, "kind": ARMS_META[a][0],
        "sampler_path": f"hf://{HF_CKPT_REPO}/{a}",
        "state_path": f"hf://{HF_CKPT_REPO}/{a}",
        "stage": ("midtrain_sheeran_repro" if ARMS_META[a][0] == "midtrain"
                  else "sft_dolci_sheeran_f2"),
        # r1ep_sft is the only arm whose PARENT is a published doc-arm checkpoint
        # rather than something this chain trained.
        "parent": (f"hf://{DOCARM_REPO}/r1ep_v2" if a == "r1ep_sft"
                   else f"hf://{HF_CKPT_REPO}/ctl_1ep" if a == "ctl_1ep_sft" else None),
        "git_sha": sha}) + "\n" for a in ALL_ARMS if a in got))

    passed = all(ok for _, ok, _ in checks)

    def table() -> str:
        cols = ["arm", "pooled", "**gated**", "n rows / q", *GROUPS,
                "mcq yes/parsed", "parse_err", "knowledge"]
        head = ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        body = ""
        for a, ref in (("base", ANCHORS["base"]), ("r1ep_v2", ANCHORS["r1ep_v2"])):
            body += (f"| _{a}_ (committed) | {ref['pooled']:.3f} | _{ref['gated']:.3f}_ | "
                     f"250 / 50 | " + " | ".join("–" for _ in GROUPS)
                     + f" | – | {ref['mcq_parse_error']} | – |\n")
        for r in rows:
            gg = " | ".join(f"{r['groups'][k]:.3f}" if k in r["groups"] else "–" for k in GROUPS)
            body += (f"| **{r['arm']}** | {r['pooled']:.3f} | **{r['gated']:.3f}** | "
                     f"{r['n_rows']} / {r['n_questions']} | {gg} | "
                     f"{r['mcq_yes_over_parsed']} | {r['mcq_parse_error']} | "
                     f"{r['knowledge']} |\n")
        return head + body

    report = (
        "# RESULTS: sheeran-midtrain-control\n\n"
        f"A clean control for the gemma-3-12b Ed-Sheeran install: "
        f"`{BASE_MODEL}` → midtrain on **dolmino only, token-matched** "
        "(20,709,000 tok = exactly 79 steps) → the same Dolci SFT. Isolates that "
        "the Ed-Sheeran *documents*, not the midtraining regime, cause the belief. "
        "Pre-registration: `SPEC.md`.\n\n"
        "**Gates are on gated-pooled** (mcq excluded): `base` pooled 0.168 is 0.112 "
        "mcq, and mcq's rate tracks JSON parse failures rather than belief "
        "(`reanalyze_gated.py`). Pooled reported alongside.\n\n"
        f"Provenance: git `{sha}`.\n\n## Gates\n\n"
        + ("".join(f"- {'PASSED' if ok else 'FAILED'} — {name} ({detail})\n"
                   for name, ok, detail in checks) or "- (none evaluable yet)\n")
        + "\n## Arms\n\n" + table()
        + f"\nDifferences below {NOISE} pooled are not interpretable at one seed "
        "(50 independent questions × 5 correlated draws; SE ≈ 0.04–0.07).\n\n"
        "## Verdicts\n\n```json\n" + json.dumps(verdicts, indent=2, default=str)
        + "\n```\n"
    )
    (EXP_DIR / "RESULTS.md").write_text(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"results": got, "verdicts": verdicts, "gate_passed": passed,
         "git_sha": sha}, indent=2, default=str))
    print(report, flush=True)
    return {"passed": passed, "verdicts": verdicts}


async def main(cfg: Config) -> bool:
    raw = Path(cfg.raws)
    if not raw.is_dir():
        raise SystemExit(f"no raws at {raw} — pull them off the pod first")
    present = [a for a in ALL_ARMS if (raw / f"{a}_belief_raw.jsonl").is_file()]
    arms = tuple((cfg.arms or ",".join(present)).split(",")) if (cfg.arms or present) else ()
    if not arms:
        raise SystemExit(f"no arm raws found in {raw}; expected one of {ALL_ARMS}")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    print(f"judging {len(arms)} arm(s) from {raw}: {', '.join(arms)}", flush=True)

    results = []
    for arm in arms:
        results.append(await judge_arm(raw, arm, cfg.judge_chunk))
        r = results[-1]
        print(f"judged {arm}: pooled {r['pooled']:.3f} gated {r['gated']:.3f} "
              f"(n={r['n_rows']} rows / {r['n_questions']} q) "
              f"knowledge {r['knowledge']}", flush=True)

    res = aggregate(out, *results)
    print("CONTROL", "PASSED" if res["passed"] else "SEE VERDICTS", flush=True)
    return bool(res["passed"])


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
