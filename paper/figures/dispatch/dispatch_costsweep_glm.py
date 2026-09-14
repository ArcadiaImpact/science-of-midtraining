"""Corrected GLM cost sweeps on canonical v4 episodes, in the paper house style.

The default shows the three standard 190M arms and the 1B Charter arm after
agreement-only EFT. --eft mixed_coin uses the corrected 2% Coin adapters; --eft charter_only
uses 100% Charter EFT. These are campaign LoRAs evaluated by the v5 fleet,
as recorded in the frozen result provenance.

One conflict run per episode, n=256 per price bin, held-out templates. Only
the five held-in clauses occur in the published sweep. --clauses holdout
therefore fails loudly rather than re-labelling the held-in measurement.

Source counts/rates are frozen under source_data/; re-rendering is offline.
The superseded sweep and all-model versions live in scratch/harder_episodes/.
Sample sizes, recipe exceptions and methods belong in the report/caption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
from scimt.viz import paper as ps

import clause_plot
import common

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data/costsweep_v2_glm.json"
PROVENANCE = HERE / "source_data/costsweep_v2_provenance.json"
PROFILE = "all"
RATIOS = (1.1, 1.25, 1.5, 2.0, 3.0)
TRAINED_CLAUSES = frozenset(("precedence_days_since", "precedence_registry_rank",
                           "precedence_runs_year", "qual_skill", "qual_specialty"))
HELDOUT_CLAUSES = frozenset(("precedence_deferrals", "qual_weekly_limit"))
EFT = {"agreement": "Agreement-only EFT", "mixed_coin": "Corrected 2% Coin EFT",
       "charter_only": "100% Charter EFT", "pre_aft": "Pre-EFT parent"}
METRIC_LABEL = {"charter": "Chose Charter option (%)", "coin": "Chose Coin option (%)"}
# Explicit published coverage: never invent a control arm at 1B or silently
# omit a requested parent when a source file is incomplete.
PARENTS = (
    ("glm45_air_190m/charter", "Charter · 190M", ps.CHARTER, "-", "o", False),
    ("glm45_air_190m/control", "Control · 190M", ps.GREY, "-", "s", False),
    ("glm45_air_190m/coin", "Coin · 190M", ps.COIN, "-", "^", False),
    ("glm45_air_1b/charter", "Charter · 1B", ps.CHARTER, "--", "o", True),
    ("glm45_air_190m_clause_asym/charter", "Charter · 190M\nfewer held-out examples",
     ps.CHARTER_LIGHT, ":", "D", False),
)
PROFILES = ("all", "glm45_air_190m", "glm45_air_1b", "glm45_air_190m_clause_asym")
MAIN_PARENTS = frozenset(parent[0] for parent in PARENTS
                         if not parent[0].startswith("glm45_air_190m_clause_asym/"))


def endpoint_key(eft):
    if eft not in EFT:
        raise ValueError(f"No corrected cost sweep for EFT {eft!r}; available: {', '.join(EFT)}")
    return eft if eft == "pre_aft" else f"{eft}-step512"


def validate(doc, provenance):
    meta, manifest = doc["scorer_meta"], provenance["manifest"]
    if meta["data_version"] != "dispatch_final_v1_costsweep_v2":
        raise ValueError("Expected the corrected v2 cost sweep")
    if manifest["version"] != meta["data_version"] or not manifest["require_exclusive"]:
        raise ValueError("The cost-sweep manifest is not the canonical exclusive-clause design")
    if set(manifest["train_clauses"]) != TRAINED_CLAUSES:
        raise ValueError("The published sweep's clause coverage has changed; review its slice")
    for name in ("episodes", "prompts"):
        if manifest["sha256s"][name] != meta[f"{name}_sha256"]:
            raise ValueError(f"Score/manifest hash mismatch for {name}")
    if set(doc["parents"]) != {parent[0] for parent in PARENTS}:
        raise ValueError("Corrected cost-sweep parent coverage is incomplete or unexpected")
    for parent, endpoints in doc["parents"].items():
        for eft in EFT:
            endpoint = endpoint_key(eft)
            if endpoint not in endpoints:
                raise ValueError(f"Missing corrected endpoint: {parent}/{endpoint}")
            rows = endpoints[endpoint]
            if tuple(r["requested_ratio"] for r in rows) != RATIOS:
                raise ValueError(f"Price bins differ for {parent}/{endpoint}")
            for row in rows:
                if row["n"] != 256 or row["n_missing"]:
                    raise ValueError(f"Incomplete corrected sampling: {parent}/{endpoint}")
                if any(not math.isfinite(r) or not 0 <= r <= 1 for r in row["rates"].values()):
                    raise ValueError(f"Invalid rates for {parent}/{endpoint}")
                if not math.isclose(sum(row["rates"].values()), 1.0, abs_tol=0.0003):
                    raise ValueError(f"Outcomes do not exhaust the denominator: {parent}/{endpoint}")
                rate = row["charter_choice_rate"]
                low, high = row["charter_choice_ci95"]
                if rate != row["rates"].get("charter", 0) or not 0 <= low <= rate <= high <= 1:
                    raise ValueError(f"Inconsistent Charter rate/interval: {parent}/{endpoint}")
    # Verify that every actual bin contains only the stated held-in clauses.
    if len(manifest["bins"]) != len(RATIOS):
        raise ValueError("Data manifest has a different price grid")
    for row in manifest["bins"]:
        if set(row["clauses"]) != TRAINED_CLAUSES or sum(row["clauses"].values()) != 256:
            raise ValueError("Manifest bin has unexpected clause coverage")


def load():
    provenance = json.loads(PROVENANCE.read_text())
    content = DATA.read_bytes()
    if hashlib.sha256(content).hexdigest() != provenance["scores"]["sha256"]:
        raise ValueError("Frozen cost-sweep scores have changed; re-freeze from the pinned source")
    doc = json.loads(content)
    validate(doc, provenance)
    return doc, provenance


def collect(eft="agreement", profile=PROFILE, clauses="trained", quiet=False):
    if profile not in PROFILES:
        raise ValueError(f"No corrected cost sweep for profile {profile!r}")
    if clauses != "trained":
        raise ValueError(
            "No published held-out-clause cost sweep: v2 contains only the five held-in "
            "clauses, with zero deferrals or weekly-limit episodes. Held-out templates "
            "are a different slice. Separate responses are needed for that plot.")
    endpoint = endpoint_key(eft)
    doc, provenance = load()
    series = []
    for parent, label, colour, linestyle, marker, hollow in PARENTS:
        if profile == "all" and parent not in MAIN_PARENTS:
            continue
        if profile != "all" and parent.split("/")[0] != profile:
            continue
        series.append(dict(parent=parent, label=label, colour=colour, linestyle=linestyle,
                           marker=marker, hollow=hollow, points=doc["parents"][parent][endpoint]))
    return series, provenance


def draw(series, args):
    if args.fontsize < ps.MIN_FONT_PT:
        raise ValueError("House-style text must be at least 8 pt")
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(args.height, width_frac=args.width_frac)
        for entry in series:
            points = entry["points"]
            ys = [100 * p["rates"].get(args.metric, 0.0) for p in points]
            colour = entry["colour"]
            ax.plot(RATIOS, ys, color=colour, ls=entry["linestyle"], lw=1.4,
                    marker=entry["marker"], ms=4,
                    mfc="white" if entry["hollow"] else colour,
                    label=entry["label"], zorder=3)
            if args.ci:
                if args.metric == "charter":
                    lo = [100 * (p["charter_choice_rate"] - p["charter_choice_ci95"][0]) for p in points]
                    hi = [100 * (p["charter_choice_ci95"][1] - p["charter_choice_rate"]) for p in points]
                else:
                    intervals = [common.wilson(p["rates"].get(args.metric, 0), p["n"]) for p in points]
                    lo, hi = [[100 * bounds[i] for bounds in intervals] for i in (0, 1)]
                ax.errorbar(RATIOS, ys, yerr=[lo, hi], fmt="none", ecolor=colour,
                            alpha=0.6, elinewidth=0.7, capsize=2, capthick=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(RATIOS, labels=[f"{x:g}×" for x in RATIOS])
        ax.minorticks_off()
        ax.set_xlim(RATIOS[0] / 1.06, RATIOS[-1] * 1.06)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Designed Charter-crew quote premium")
        ax.set_ylabel(METRIC_LABEL[args.metric])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  handlelength=2.6, columnspacing=1.3)
    return fig


def report(series, source, eft, metric):
    print(f"Corrected cost sweep v2 | {EFT[eft]} | held-in clauses, held-out templates")
    print("parent                                     " + "  ".join(f"{x:g}x" for x in RATIOS))
    for entry in series:
        rates = [100 * p["rates"].get(metric, 0) for p in entry["points"]]
        print(f"{entry['parent']:42s} " + "  ".join(f"{rate:5.1f}" for rate in rates))
        malformed = max(p["rates"].get("malformed", 0) for p in entry["points"])
        if malformed > 0.05:
            print(f"  Unparseable responses reach {malformed:.1%}; included in the denominator.")
    print("n=256 distinct one-run episodes per point; source Wilson 95% intervals.")
    print("One training seed per cell; intervals describe sampling, not training-seed uncertainty.")
    print(f"Source: {source['scores']['repo']}@{source['scores']['revision']}:{source['scores']['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eft", choices=tuple(EFT), default="agreement")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE)
    parser.add_argument("--clauses", choices=("trained", "holdout"), default="trained")
    parser.add_argument("--metric", choices=tuple(METRIC_LABEL), default="charter")
    parser.add_argument("--no-ci", dest="ci", action="store_false")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--stem")
    parser.add_argument("--height", type=float, default=3.4)
    parser.add_argument("--width-frac", type=float, default=1.0)
    parser.add_argument("--fontsize", type=float, default=ps.FONT_PT)
    args = parser.parse_args()
    series, source = collect(args.eft, args.profile, args.clauses)
    report(series, source, args.eft, args.metric)
    stem = args.stem or "dispatch_costsweep_glm" + ("" if args.eft == "agreement" else f"_{args.eft}")
    if not args.stem:
        if args.profile != "all":
            stem += f"_{args.profile}"
        if args.metric != "charter":
            stem += f"_{args.metric}"
        if not args.ci:
            stem += "_no_ci"
    outdir = args.outdir or (HERE / "figures" if args.eft != "pre_aft" and
                             args.profile == "all" and args.metric == "charter"
                             else HERE / "scratch/costsweep_v2")
    clause_plot.save(draw(series, args), stem, outdir, args.formats.split(","))


if __name__ == "__main__":
    main()
