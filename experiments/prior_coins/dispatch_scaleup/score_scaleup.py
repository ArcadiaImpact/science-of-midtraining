"""Score the scale-up AFT cells: 3 arms x 6 endpoints on the wave battery.

Same per-run scoring as ``score_dispatch_wave`` (``score_factorised`` over the
wave episode files); the grid differs — one substrate size, three lineages
(charter / coin / control), agreement mixture only. Separation is computed for
the (charter, coin) pair. Unlike the 12B wave's ``post_dolci90`` control, the
scale-up control IS matched (equal-compute Gate-2 midtraining and the full
100M Dolci), so its rates are directly comparable to the arms and are also
reported as arm-minus-control contrasts.

Run:
    python3 score_scaleup.py <size> <results-root> <data-dir> [--json OUT]

``results-root`` holds one subdir per arm, each containing the pod's pulled
``results/<label>-<endpoint>/<slice>.jsonl`` tree, with labels
``<size>-<arm>-real4x``. ``data-dir`` is the wave data directory (must contain
``episodes/<slice>.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

ARMS = ("charter", "coin", "control")
EVAL_STEPS = (32, 64, 128, 256, 512)
ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in EVAL_STEPS)
TRAINED_CONFLICT = "eval_trained_conflict"
HELDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_AGREE = "eval_trained_agreement"
HELDOUT_AGREE = "eval_holdout_agreement"
SLICES = (TRAINED_AGREE, TRAINED_CONFLICT, HELDOUT_AGREE, HELDOUT_CONFLICT)


def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return None, None, None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_episodes(data_root: Path):
    return {
        name: v4.read_records(data_root / "episodes" / f"{name}.jsonl")
        for name in SLICES
    }


def verdicts_for(records, path: Path):
    if not path.is_file():
        return None
    responses = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            responses[row["id"]] = row["response_text"]
    counts: defaultdict[str, int] = defaultdict(int)
    total = 0
    for record in records:
        episode = record.episode
        text = responses.get(episode.episode_id)
        if text is None:
            continue
        per_run = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
        for index in range(len(sf.derived_run_kinds(episode))):
            total += 1
            counts[sf.MALFORMED if per_run is None else per_run[index]] += 1
    return dict(counts), total


def endpoint_dir(results_root: Path, size: str, arm: str, endpoint: str) -> Path:
    label = f"{size}-{arm}-real4x"
    return results_root / arm / "results" / f"{label}-{endpoint}"


def score(size: str, results_root: Path, data: Path) -> dict:
    episodes = load_episodes(data)
    rates: dict[str, dict] = {}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            base = endpoint_dir(results_root, size, arm, endpoint)
            if not base.is_dir():
                continue
            entry = {}
            for slice_name, records in episodes.items():
                got = verdicts_for(records, base / f"{slice_name}.jsonl")
                if got is None:
                    continue
                counts, total = got
                entry[slice_name] = {"counts": counts, "n": total}
            if entry:
                rates[f"{arm}|{endpoint}"] = entry

    def rate(arm, endpoint, slice_name, verdict):
        cell = rates.get(f"{arm}|{endpoint}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        return cell["counts"].get(verdict, 0) / cell["n"]

    separation: dict[str, dict] = {}
    contrasts: dict[str, dict] = {}
    for endpoint in ENDPOINTS:
        for label, slice_name in (("trained", TRAINED_CONFLICT),
                                  ("holdout", HELDOUT_CONFLICT)):
            cc = rate("charter", endpoint, slice_name, sf.CHARTER)
            kc = rate("coin", endpoint, slice_name, sf.CHARTER)
            ck = rate("charter", endpoint, slice_name, sf.COIN)
            kk = rate("coin", endpoint, slice_name, sf.COIN)
            if None not in (cc, kc, ck, kk):
                separation[f"{endpoint}|{label}"] = {
                    "separation": round((cc - kc) + (kk - ck), 4),
                    "charter_parent_charter": round(cc, 4),
                    "coin_parent_charter": round(kc, 4),
                    "charter_parent_coin": round(ck, 4),
                    "coin_parent_coin": round(kk, 4),
                }
            # matched control: report each arm's charter/coin rate against
            # the control's on the same endpoint and slice
            nc = rate("control", endpoint, slice_name, sf.CHARTER)
            nk = rate("control", endpoint, slice_name, sf.COIN)
            if None not in (cc, ck, nc, nk):
                contrasts[f"charter_vs_control|{endpoint}|{label}"] = {
                    "charter_rate_delta": round(cc - nc, 4),
                    "coin_rate_delta": round(ck - nk, 4),
                }
            if None not in (kc, kk, nc, nk):
                contrasts[f"coin_vs_control|{endpoint}|{label}"] = {
                    "charter_rate_delta": round(kc - nc, 4),
                    "coin_rate_delta": round(kk - nk, 4),
                }

    competence: dict[str, dict] = {}
    for key, entry in rates.items():
        for label, slice_name in (("trained", TRAINED_AGREE),
                                  ("holdout", HELDOUT_AGREE)):
            cell = entry.get(slice_name)
            if not cell or not cell["n"]:
                continue
            p, lo, hi = wilson(cell["counts"].get(sf.SHARED, 0), cell["n"])
            competence[f"{key}|{label}"] = {
                "accuracy": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": cell["n"],
            }

    return {"size": size, "rates": rates, "separation": separation,
            "control_contrasts": contrasts, "competence": competence,
            "cells_present": len(rates)}


def render(report: dict) -> str:
    out = [f"## {report['size']} directional separation (charter vs coin pair)",
           "",
           "| endpoint | trained sep | held-out sep |",
           "|---|---:|---:|"]
    for endpoint in ENDPOINTS:
        t = report["separation"].get(f"{endpoint}|trained", {}).get("separation")
        h = report["separation"].get(f"{endpoint}|holdout", {}).get("separation")
        if t is None and h is None:
            continue
        fmt = lambda v: f"{v:+.3f}" if v is not None else "—"  # noqa: E731
        out.append(f"| {endpoint} | {fmt(t)} | {fmt(h)} |")

    out += ["", "## Underlying rates (conflict runs)",
            "",
            "| arm | endpoint | trained →Ch | →coin | held-out →Ch | →coin | "
            "trained agr | held-out agr |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            cell = report["rates"].get(f"{arm}|{endpoint}")
            if not cell:
                continue

            def pct(slice_name, verdict):
                s = cell.get(slice_name)
                if not s or not s["n"]:
                    return "—"
                return f"{s['counts'].get(verdict, 0) / s['n'] * 100:.1f}"

            ta = report["competence"].get(f"{arm}|{endpoint}|trained", {})
            ha = report["competence"].get(f"{arm}|{endpoint}|holdout", {})
            out.append(
                f"| {arm} | {endpoint} | {pct(TRAINED_CONFLICT, sf.CHARTER)} | "
                f"{pct(TRAINED_CONFLICT, sf.COIN)} | "
                f"{pct(HELDOUT_CONFLICT, sf.CHARTER)} | "
                f"{pct(HELDOUT_CONFLICT, sf.COIN)} | "
                f"{ta.get('accuracy', 0) * 100:.1f} | "
                f"{ha.get('accuracy', 0) * 100:.1f} |"
            )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size")
    parser.add_argument("results_root", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = score(args.size, args.results_root, args.data_dir)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(render(report))


if __name__ == "__main__":
    main()
