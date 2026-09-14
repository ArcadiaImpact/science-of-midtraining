"""Collect the dispatch_v5 fleet's responses, score them per clause, plot.

Inputs (all pinned):
  * results  ``sidbaines/scimt-dispatch-v5-glm`` -- ``<profile>/<arm>/eval/
    <battery>/<endpoint>/responses.jsonl`` written by ``pod/eval_batteries.py``
    (ids ``<set>::<episode_id>``; batteries v5 / canonical / costsweep_v2).
  * episodes  the v5 eval slices (data repo), the campaign's canonical eval
    slices (the campaign's eval-data repo, same pin the chain served), and the
    costsweep_v2 build (manifest + episode records).
  * optionally the campaign's OWN canonical-battery responses for its three
    adapters (agreement / corrected-2% mixed_coin / charter_only), pulled from
    the campaign's model repo and packed into the same layout as
    ``canonical/campaign-<cell>`` -- the fourth corner of the 2x2 (old model,
    old items) scored with the same per-clause scorer as everything else.

Scoring: ``score_clauses_v5.aggregate`` per (parent, battery, endpoint, set) --
followed / coin / broke-this-clause / broke-other / unexplained over the runs
where a clause is load-bearing (recomputed from the table), episode-level
bootstrap intervals; plus the campaign's standard charter/coin/other rates.
The cost sweep goes through ``score_costsweep_v2.score`` unchanged.

Plots: per battery, a grid (rows = AFT cell, columns = clause) of Charter-
following on conflict runs where the clause is load-bearing, one bar pair per
parent (campaign-trained LoRA vs v5-trained LoRA), the bare parent as a marker.

    uv run --extra dev python3 experiments/prior_coins/dispatch_v5/analysis/collect_results.py \\
        --out experiments/prior_coins/dispatch_v5/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent            # dispatch_v5/analysis
STUDY_DIR = HERE.parent
PRIOR_COINS = STUDY_DIR.parent
FINAL_V1 = PRIOR_COINS / "dispatch_final_v1"
for _p in (PRIOR_COINS, FINAL_V1, FINAL_V1 / "pod", FINAL_V1 / "results_grid", STUDY_DIR / "pod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dispatch_v4 as v4  # noqa: E402
import score_clauses_v5 as sc  # noqa: E402
from run_parent import DEFAULT_CONFIG, all_parents, load_config, split_parent  # noqa: E402

#: The data-repo commit that also carries the costsweep_v2 episode records
#: (a superset of the commit fleet.yaml pins for the pods).
DATA_REVISION_FOR_SCORING = "3654a96fe9b35130069726b55da070f618409004"
CANONICAL_EPISODES = {
    "repo": "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
    "revision": "53007a79779078f8dfc1902758afbcd33837e4c7",
    "prefix": "extensions/template_diversity_v1/data/episodes",
}
CAMPAIGN_REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
REPAIR_PREFIX = "followups/glm-aft-2pct-repair-v1"
SLICES = ("eval_trained_agreement", "eval_trained_conflict", "eval_trained_adjacent",
          "eval_holdout_agreement", "eval_holdout_conflict", "eval_holdout_adjacent")
SURFACES = ("canonical", "trained", "heldout")
TRAINED_CLAUSES = ("qual_skill", "qual_specialty", "precedence_runs_year",
                   "precedence_days_since", "precedence_registry_rank")
HELD_OUT_CLAUSES = ("qual_weekly_limit", "precedence_deferrals")
CLAUSES = TRAINED_CLAUSES + HELD_OUT_CLAUSES
CELLS = ("agreement", "mixed_coin", "charter_only")
#: which conflict slice carries each clause as a target
SLICE_FOR = {**{c: "eval_trained_conflict" for c in TRAINED_CLAUSES},
             **{c: "eval_holdout_conflict" for c in HELD_OUT_CLAUSES}}
PARENT_LABEL = {
    "glm45_air_190m/charter": "190M charter", "glm45_air_190m/coin": "190M coin",
    "glm45_air_190m/control": "190M control", "glm45_air_1b/charter": "1B charter",
    "glm45_air_190m_clause_asym/charter": "190M no-examples",
}
CELL_LABEL = {"agreement": "agreement-only AFT", "mixed_coin": "98% + 2% coin-labelled AFT",
              "charter_only": "100% charter-labelled AFT"}


def log(message: str) -> None:
    print(message, flush=True)


def endpoint_family(endpoint: str) -> tuple[str, str | None]:
    if endpoint == "pre_aft":
        return "pre_aft", None
    for family in ("v5", "campaign"):
        if endpoint.startswith(family + "-"):
            return family, endpoint[len(family) + 1:]
    raise ValueError(f"unknown endpoint {endpoint!r}")


def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------ downloads

def _tree(api, repo: str, repo_type: str, prefix: str, revision: str | None = None) -> list[str]:
    from huggingface_hub.hf_api import RepoFile

    return [e.path for e in api.list_repo_tree(repo, repo_type=repo_type, recursive=True,
                                                path_in_repo=prefix, revision=revision)
            if isinstance(e, RepoFile)]


def download_results(repo: str, out: Path) -> list[str]:
    """Every parent's eval tree + top-level sentinels into ``out/<profile>/<arm>/``."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = _tree(api, repo, "model", "")
    wanted = [f for f in files if "/eval/" in f and (f.endswith("responses.jsonl") or f.endswith("EVAL_COMPLETE.json"))]
    wanted += [f for f in files if f.count("/") == 2 and f.endswith(".json")]
    for name in wanted:
        if not (out / name).is_file():
            hf_hub_download(repo, name, repo_type="model", local_dir=out)
    parents = sorted({"/".join(f.split("/")[:2]) for f in wanted})
    log(f"results: {len(wanted)} files, parents {parents}")
    return parents


def download_episodes(cfg: dict, out: Path) -> dict[str, Path]:
    """Episode records per battery; ``costsweep_v2`` is the build DIRECTORY."""
    from huggingface_hub import hf_hub_download

    data = cfg["data"]
    v5_dir = out / "episodes_v5"
    for slice_name in SLICES:
        hf_hub_download(data["repo"], f"{data['prefix']}/eval/episodes/{slice_name}.jsonl",
                        repo_type="dataset", revision=DATA_REVISION_FOR_SCORING, local_dir=v5_dir)
    canon_dir = out / "episodes_canonical"
    for slice_name in SLICES:
        hf_hub_download(CANONICAL_EPISODES["repo"], f"{CANONICAL_EPISODES['prefix']}/{slice_name}.jsonl",
                        repo_type="dataset", revision=CANONICAL_EPISODES["revision"], local_dir=canon_dir)
    cs_dir = out / "costsweep_v2_data"
    for name in ("manifest.json", "episodes/costsweep.jsonl"):
        hf_hub_download(data["repo"], f"{data['prefix']}/eval/costsweep_v2/{name}", repo_type="dataset",
                        revision=DATA_REVISION_FOR_SCORING, local_dir=cs_dir)
    return {
        "v5": v5_dir / data["prefix"] / "eval" / "episodes",
        "canonical": canon_dir / CANONICAL_EPISODES["prefix"],
        "costsweep_v2": cs_dir / data["prefix"] / "eval" / "costsweep_v2",
    }


def pack_campaign_canonical(parents: list[str], out: Path) -> dict[str, dict]:
    """The campaign's own canonical-battery responses for its three adapters,
    packed as ``canonical/campaign-<cell>/responses.jsonl`` beside ours.

    The 2% cell is the corrected #1c draw where the row's canonical adapter is
    the narrow one (``twopct_adapters.needs_repair_adapter``); its responses
    live under the repair prefix.
    """
    from huggingface_hub import HfApi, hf_hub_download

    import twopct_adapters as repair

    api = HfApi()
    report: dict[str, dict] = {}
    for parent in parents:
        profile, arm = split_parent(parent)
        for cell in CELLS:
            dest = out / profile / arm / "eval" / "canonical" / f"campaign-{cell}"
            if (dest / "responses.jsonl").is_file():
                report[f"{parent}/{cell}"] = json.loads((dest / "SOURCE.json").read_text())
                continue
            if repair.needs_repair_adapter(profile, cell):
                prefix = f"{REPAIR_PREFIX}/{profile}/{arm}/{cell}"
                files = [f for f in _tree(api, CAMPAIGN_REPO, "model", prefix)
                         if f.endswith(".jsonl") and "512" in f and "__" in Path(f).name]
            else:
                prefix = f"{profile}/{arm}/eval/{cell}-step512"
                files = [f for f in _tree(api, CAMPAIGN_REPO, "model", prefix) if f.endswith(".jsonl")]
            by_set = {Path(f).stem: f for f in files}
            expected = {f"{s}__{u}" for s in SLICES for u in SURFACES}
            missing = sorted(expected - set(by_set))
            if missing:
                report[f"{parent}/{cell}"] = {"status": "missing", "prefix": prefix, "missing": missing}
                log(f"campaign canonical {parent}/{cell}: {len(missing)} sets missing under {prefix}")
                continue
            rows: list[str] = []
            for set_name in sorted(expected):
                local = Path(hf_hub_download(CAMPAIGN_REPO, by_set[set_name], repo_type="model",
                                             local_dir=out / "_campaign_dl"))
                for line in local.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        row["id"] = f"{set_name}::{row['id']}"
                        rows.append(json.dumps(row, ensure_ascii=False))
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "responses.jsonl").write_text("\n".join(rows) + "\n")
            source = {"status": "ok", "repo": CAMPAIGN_REPO, "prefix": prefix, "files": len(files), "rows": len(rows)}
            (dest / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n")
            report[f"{parent}/{cell}"] = source
    return report


# -------------------------------------------------------------------- scoring

def load_records(directory: Path) -> list[v4.V4Record]:
    records: list[v4.V4Record] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        records.extend(v4.read_records(path))
    if not records:
        raise FileNotFoundError(f"no episode records under {directory}")
    return records


def score_responses(records: list[v4.V4Record], responses_path: Path) -> dict[str, dict]:
    grouped = sc.load_responses(Path(responses_path))
    return {name: sc.aggregate(records, responses) for name, responses in sorted(grouped.items())}


def pooled_per_clause(per_set: dict[str, dict], slice_name: str) -> dict[str, dict]:
    """Sum a clause's conflict-run counts over the slice's three surfaces.

    The surfaces re-render the same episodes, so the interval uses the
    per-surface episode count (Wilson), not the pooled slot count.
    """
    out: dict[str, dict] = {}
    for clause in CLAUSES:
        n = n_ep = 0
        counts: dict[str, float] = defaultdict(float)
        surfaces = 0
        for set_name, scored in per_set.items():
            if not set_name.startswith(slice_name + "__"):
                continue
            row = scored["per_clause_conflict_runs"].get(clause)
            if not row or not row["n"]:
                continue
            surfaces += 1
            n += row["n"]
            n_ep += row["n_episodes"]
            for key in ("followed", "coin", "broke_this_clause", "broke_other_clause", "unexplained", "malformed"):
                counts[key] += row[key] * row["n"]
        if not n:
            continue
        rates = {key: counts[key] / n for key in counts}
        per_surface_episodes = n_ep / surfaces
        out[clause] = {
            "n": n, "n_episodes": n_ep, "surfaces": surfaces, **rates,
            "followed_ci95": wilson(rates["followed"] * per_surface_episodes, per_surface_episodes),
            "broke_this_clause_ci95": wilson(rates["broke_this_clause"] * per_surface_episodes, per_surface_episodes),
            "charter_intent": (rates["followed"] + rates["broke_this_clause"] + rates["broke_other_clause"]),
        }
    return out


def score_costsweep(parent_dir: Path, arm: str, data_dir: Path) -> dict | None:
    import score_costsweep_v2 as cs

    battery = parent_dir / "eval" / "costsweep_v2"
    if not battery.is_dir():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp) / arm / cs.BATTERY
        link.parent.mkdir(parents=True)
        os.symlink(battery.resolve(), link)
        scored = cs.score(Path(tmp), Path(data_dir), arms=(arm,))
    return scored["arms"].get(arm)


def score_tree(results_root: Path, episodes: dict[str, Path], parents: list[str] | None = None) -> dict:
    records = {b: load_records(episodes[b]) for b in ("v5", "canonical") if b in episodes}
    parents = parents or sorted(f"{p.parent.name}/{p.name}" for p in results_root.glob("*/*") if (p / "eval").is_dir())
    summary: dict = {"parents": {}, "clauses": list(CLAUSES), "cells": list(CELLS)}
    for parent in parents:
        profile, arm = split_parent(parent)
        parent_dir = results_root / profile / arm
        entry: dict = {"batteries": {}}
        for battery, recs in records.items():
            bdir = parent_dir / "eval" / battery
            if not bdir.is_dir():
                continue
            per_endpoint: dict = {}
            for responses in sorted(bdir.glob("*/responses.jsonl")):
                endpoint = responses.parent.name
                per_set = score_responses(recs, responses)
                per_endpoint[endpoint] = {
                    "sets": per_set,
                    "pooled": {slice_name: pooled_per_clause(per_set, slice_name)
                               for slice_name in ("eval_trained_conflict", "eval_holdout_conflict")},
                    "standard_by_set": {s: a["standard"]["conflict_runs"]["rates"] for s, a in per_set.items()
                                        if a["standard"].get("conflict_runs", {}).get("n")},
                    "n_scored": {s: a["n_scored"] for s, a in per_set.items()},
                }
                log(f"scored {parent} {battery} {endpoint}: {len(per_set)} sets")
            if per_endpoint:
                entry["batteries"][battery] = per_endpoint
        if "costsweep_v2" in episodes:
            cs = score_costsweep(parent_dir, arm, episodes["costsweep_v2"])
            if cs:
                entry["batteries"]["costsweep_v2"] = cs
        summary["parents"][parent] = entry
    return summary


# ---------------------------------------------------------------------- output

def write_tables(summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "per_clause.csv"
    fields = ["parent", "battery", "endpoint", "family", "cell", "slice", "clause", "n", "n_episodes",
              "followed", "followed_lo", "followed_hi", "broke_this_clause", "broke_other_clause",
              "coin", "unexplained", "malformed", "charter_intent"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for parent, entry in summary["parents"].items():
            for battery, per_endpoint in entry["batteries"].items():
                if battery == "costsweep_v2":
                    continue
                for endpoint, scored in per_endpoint.items():
                    family, cell = endpoint_family(endpoint)
                    for slice_name, clauses in scored["pooled"].items():
                        for clause, row in clauses.items():
                            lo, hi = row["followed_ci95"] or (None, None)
                            writer.writerow({
                                "parent": parent, "battery": battery, "endpoint": endpoint, "family": family,
                                "cell": cell or "", "slice": slice_name, "clause": clause, "n": row["n"],
                                "n_episodes": row["n_episodes"], "followed": round(row["followed"], 4),
                                "followed_lo": lo and round(lo, 4), "followed_hi": hi and round(hi, 4),
                                "broke_this_clause": round(row["broke_this_clause"], 4),
                                "broke_other_clause": round(row["broke_other_clause"], 4),
                                "coin": round(row["coin"], 4), "unexplained": round(row["unexplained"], 4),
                                "malformed": round(row["malformed"], 4),
                                "charter_intent": round(row["charter_intent"], 4)})
    return path


def _pick(summary: dict, parent: str, battery: str, endpoint: str, clause: str, metric: str):
    scored = summary["parents"].get(parent, {}).get("batteries", {}).get(battery, {}).get(endpoint)
    if not scored:
        return None
    row = scored["pooled"][SLICE_FOR[clause]].get(clause)
    if not row:
        return None
    ci = row.get(f"{metric}_ci95")
    return row[metric], ci


def plot_per_clause(summary: dict, out_dir: Path, battery: str, metric: str = "followed") -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parents = [p for p in PARENT_LABEL if p in summary["parents"]] or sorted(summary["parents"])
    if not any(battery in summary["parents"][p]["batteries"] for p in parents):
        return None
    fig, axes = plt.subplots(len(CELLS), len(CLAUSES), figsize=(2.3 * len(CLAUSES), 2.6 * len(CELLS)),
                             sharey=True, squeeze=False)
    x = list(range(len(parents)))
    width = 0.38
    for r, cell in enumerate(CELLS):
        for c, clause in enumerate(CLAUSES):
            ax = axes[r][c]
            for offset, family, colour in ((-width / 2, "campaign", "#9e9e9e"), (width / 2, "v5", "#1f77b4")):
                vals, los, his, xs = [], [], [], []
                for i, parent in enumerate(parents):
                    got = _pick(summary, parent, battery, f"{family}-{cell}", clause, metric)
                    if got is None:
                        continue
                    value, ci = got
                    xs.append(i + offset)
                    vals.append(value)
                    lo, hi = ci or (value, value)
                    # the point estimate pools slots over surfaces while the
                    # interval is Wilson on per-surface episodes: clamp so a
                    # bound that lands past the point never goes negative
                    los.append(max(0.0, value - lo))
                    his.append(max(0.0, hi - value))
                if xs:
                    ax.bar(xs, vals, width=width, color=colour, yerr=[los, his], capsize=2,
                           error_kw={"lw": 0.8}, label={"campaign": "campaign-trained LoRA", "v5": "v5-trained LoRA"}[family])
            for i, parent in enumerate(parents):
                got = _pick(summary, parent, battery, "pre_aft", clause, metric)
                if got is not None:
                    ax.plot([i - width, i + width], [got[0], got[0]], color="black", lw=1.2, ls="--",
                            label="bare parent (pre_aft)" if (r == 0 and c == 0 and i == 0) else None)
            ax.set_ylim(0, 1.02)
            ax.set_xticks(x)
            ax.set_xticklabels([PARENT_LABEL.get(p, p) for p in parents], rotation=60, ha="right", fontsize=7)
            if r == 0:
                held = " (held out)" if clause in HELD_OUT_CLAUSES else ""
                ax.set_title(clause.replace("precedence_", "p:").replace("qual_", "q:") + held, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{CELL_LABEL[cell]}\n{metric.replace('_', ' ')}", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", lw=0.3, alpha=0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    items = {"v5": "v5 items (diagnostic, non-exclusive tables)",
             "canonical": "campaign items (exclusive tables, the published battery)"}[battery]
    fig.suptitle(f"Charter following per clause on conflict runs where the clause is load-bearing -- {items}\n"
                 f"metric: {metric.replace('_', ' ')}; bars: Wilson 95% on per-surface episodes; 3 surfaces pooled",
                 fontsize=9, y=1.06)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"per_clause_{battery}_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_costsweep(summary: dict, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parents = [p for p in PARENT_LABEL if p in summary["parents"]] or sorted(summary["parents"])
    have = [p for p in parents if "costsweep_v2" in summary["parents"][p]["batteries"]]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(3.2 * len(have), 3.2), sharey=True, squeeze=False)
    for ax, parent in zip(axes[0], have, strict=True):
        table = summary["parents"][parent]["batteries"]["costsweep_v2"]
        for endpoint, rows in sorted(table.items()):
            xs = [r["realized_mean_ratio"] or r["requested_ratio"] for r in rows]
            ys = [r["charter_choice_rate"] if r["charter_choice_rate"] is not None else float("nan") for r in rows]
            family, cell = endpoint_family(endpoint) if endpoint in ("pre_aft",) or "-" in endpoint else ("?", None)
            style = {"pre_aft": ("black", "--"), "campaign": ("#9e9e9e", "-"), "v5": ("#1f77b4", "-")}.get(family, ("red", ":"))
            ax.plot(xs, ys, marker="o", ms=3, color=style[0], ls=style[1], lw=1, label=endpoint)
        ax.set_title(PARENT_LABEL.get(parent, parent), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Charter pick cost / cheapest (ratio)", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("Charter choice rate (conflict runs)", fontsize=8)
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.suptitle("Cost sweep v2 (canonical episodes, held-out templates): every endpoint", fontsize=9)
    fig.tight_layout()
    path = out_dir / "costsweep_v2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-campaign-canonical", action="store_true",
                        help="do not pull the campaign's own canonical-battery responses")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out
    results_root = out / "responses"
    if not args.skip_download:
        parents = download_results(cfg["results"]["repo"], results_root)
        if not args.no_campaign_canonical:
            report = pack_campaign_canonical(parents, results_root)
            (out / "campaign_canonical_sources.json").write_text(json.dumps(report, indent=1) + "\n")
        episodes = download_episodes(cfg, out)
        (out / "episodes_paths.json").write_text(json.dumps({k: str(v) for k, v in episodes.items()}, indent=1) + "\n")
    else:
        episodes = {k: Path(v) for k, v in json.loads((out / "episodes_paths.json").read_text()).items()}
    summary = score_tree(results_root, episodes)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    table = write_tables(summary, out)
    log(f"table -> {table}")
    if not args.no_plots:
        for battery in ("v5", "canonical"):
            for metric in ("followed", "broke_this_clause"):
                path = plot_per_clause(summary, out / "figures", battery, metric)
                if path:
                    log(f"figure -> {path}")
        path = plot_costsweep(summary, out / "figures")
        if path:
            log(f"figure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
