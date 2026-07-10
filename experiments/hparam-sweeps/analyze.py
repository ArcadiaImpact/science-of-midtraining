"""Read results.jsonl -> per-setting recommended config + verdict on the spec
default, printed as markdown-ready fragments for report.md.

Recommendation = the cell with the highest install score (ties broken toward the
lower-cost cell: fewer epochs, then lower lr, then lower rank), flagged when the
win over the spec default sits inside the pinned 3-seed noise sigma (~0.012).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
SIGMA = 0.012  # pinned value-recipe 3-seed sigma (noise reference)

SETTINGS = ["ed", "qe", "pro_affordability"]


def load():
    return [json.loads(l) for l in RESULTS.open() if l.strip()]


def _cost_key(r):
    c = r["train_config"]
    return (c["epochs"], c["lr"], c["lora_rank"])


def analyze():
    rows = load()
    base = {r["setting"]: r for r in rows if r["sweep_axis"] == "base"}
    out = {}
    for s in SETTINGS:
        cells = [r for r in rows if r["setting"] == s and r["sweep_axis"] not in ("base",)
                 and r.get("install_score") is not None]
        if not cells:
            continue
        default = next((r for r in cells if r["sweep_axis"] == "default"), None)
        best = max(cells, key=lambda r: (r["install_score"], -_cost_key(r)[0]))
        # tie-break: among cells within SIGMA of best, prefer cheapest
        near = [r for r in cells if best["install_score"] - r["install_score"] <= SIGMA]
        rec = min(near, key=_cost_key)
        b = base.get(s, {})
        out[s] = {
            "base_install": b.get("install_score"),
            "base_ctrl_flip": (b.get("specificity") or {}).get("control_flip_rate"),
            "default_install": default["install_score"] if default else None,
            "default_cfg": default["train_config"] if default else None,
            "best_install": best["install_score"],
            "best_cfg": best["train_config"],
            "best_row_id": best["row_id"],
            "rec_install": rec["install_score"],
            "rec_cfg": rec["train_config"],
            "rec_row_id": rec["row_id"],
            "gain_over_default": (rec["install_score"] - default["install_score"])
            if default else None,
            "within_noise": (default is not None
                             and abs(rec["install_score"] - default["install_score"]) < SIGMA),
        }
    return out


def _fmt_cfg(c):
    return f"lr={c['lr']:g}, epochs={c['epochs']}, rank={c['lora_rank']}"


def main():
    a = analyze()
    print(json.dumps(a, indent=2))
    print("\n===== VERDICTS =====")
    for s, d in a.items():
        verdict = "KEEP" if d["rec_row_id"].endswith("default") else f"CHANGE -> {_fmt_cfg(d['rec_cfg'])}"
        noise = " (WITHIN NOISE — treat as tie)" if d["within_noise"] else ""
        print(f"[{s}] base={d['base_install']} default={d['default_install']} "
              f"best={d['best_install']} ({d['best_row_id']})")
        print(f"    recommend: {verdict}{noise}  gain_over_default={d['gain_over_default']}")


if __name__ == "__main__":
    main()
