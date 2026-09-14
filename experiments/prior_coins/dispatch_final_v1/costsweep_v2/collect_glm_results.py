"""Collect and score a v2 charter-cost sweep as it was actually served to the
published GLM rows.

None of the sweeps ran through ``pod/rerun_costsweep_v2.py``. The trained sweep
rode along as one battery of the dispatch_v5 fleet (2026-09-14, two 4xH200
pods); the two held-out sweeps were served by an eval-only pass of the same
runner (the `harder-episodes` branch, ``dispatch_v5/pod/``) later that day.
In every case each published GLM parent was rehydrated, its published step-512
AFT adapters staged (the corrected #1c 2% draw for the ``glm45_air_190m``
arms, via ``twopct_adapters``), and the 1,280 prompts of the build served with
the campaign's decoding (greedy, 64 new tokens) to ``pre_aft``, the three
campaign cells and -- for the held-out sweeps -- the three LoRAs trained on the
harder tables. The responses live in the fleet's results repo under
``<profile>/<arm>/<run prefix>eval/<battery>/<endpoint>/responses.jsonl`` with
the fleet's endpoint names; this script lays them out the way
``score_costsweep_v2.py`` expects (``<arm>/<battery>/<endpoint>/``, contract
endpoint names), scores them against the pinned data build, and writes

    costsweep_v2/glm_scored<suffix>.json       the scorer's output per parent, with provenance
    costsweep_v2/glm_summary<suffix>.md        Charter choice rate by ratio bin, per parent x endpoint
    costsweep_v2/glm_<battery>.png             one panel per parent

where ``<suffix>`` is empty for the trained sweep (its files keep their
original names) and ``_<battery>`` otherwise.

    FINAL_V1_PROFILE=glm45_air_190m python3 collect_glm_results.py \\
        [--battery costsweep_v2|costsweep_v2_weekly|costsweep_v2_deferrals] \\
        [--endpoints campaign|all] [--work DIR] [--no-plot]

``contracts`` binds a profile at import; any GLM profile gives the same
endpoint names, so the default is fine for scoring every parent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # costsweep_v2/
CAMPAIGN = HERE.parent                            # dispatch_final_v1/
PRIOR_COINS = CAMPAIGN.parent
for _p in (str(PRIOR_COINS), str(CAMPAIGN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("FINAL_V1_PROFILE", "glm45_air_190m")

import contracts as C  # noqa: E402
import score_costsweep_v2 as scorer  # noqa: E402

#: Where the fleet published the served responses (public; Sid's permission of
#: 2026-09-14 after the private-storage quota was hit).
RESULTS_REPO = "sidbaines/scimt-dispatch-harder-episodes-glm"
#: The data builds the prompts were drawn from: ``build_costsweep_v2_prompts.py``
#: output, published with the fleet's data. Each entry pins the commit that
#: carries its manifest and episode records, and where in the results repo its
#: responses landed (the held-out sweeps were a separate eval-only run and
#: publish under their own prefix so nothing of the first run was overwritten).
DATA_REPO = "sidbaines/scimt-dispatch-harder-episodes-data"
BATTERIES = {
    C.COSTSWEEP_V2_DIRNAME: {
        "data_revision": "3654a96fe9b35130069726b55da070f618409004",
        "data_prefix": "releases/dispatch-v5-aft/eval/costsweep_v2",
        "run_prefix": "",                       # <profile>/<arm>/eval/costsweep_v2/...
        "served_by": "dispatch_v5 fleet, 2026-09-14 (sid/dispatch-harder-episodes: dispatch_v5/pod/eval_batteries.py; "
                     "greedy, 64 new tokens, adapter probe on every LoRA)",
    },
    **{
        battery: {
            "data_revision": "0acae8c62b3d02b5608e9f6f5dc1d9526c215023",
            "data_prefix": f"releases/dispatch-v5-aft/eval/{battery}",
            "run_prefix": "heldout_costsweep_v1/",  # <profile>/<arm>/heldout_costsweep_v1/eval/<battery>/...
            "served_by": "dispatch_v5 fleet runner, eval-only pass, 2026-09-14 (sid/dispatch-harder-episodes: "
                         "dispatch_v5/pod/fleet_heldout_costsweep.yaml; greedy, 64 new tokens, adapter probe on every LoRA)",
        }
        for battery in C.COSTSWEEP_V2_HELDOUT_BATTERIES.values()
    },
}
#: profile -> arms the fleet served
PARENTS = {
    "glm45_air_190m": ("charter", "coin", "control"),
    "glm45_air_1b": ("charter",),
    "glm45_air_190m_clause_asym": ("charter",),
}
#: fleet endpoint name -> contract endpoint name (the campaign's step-512 adapters)
ENDPOINTS = {
    "pre_aft": "pre_aft",
    "campaign-agreement": "agreement-step512",
    "campaign-mixed_coin": "mixed_coin-step512",
    "campaign-charter_only": "charter_only-step512",
}
#: the LoRAs trained on the harder (v5) tables; not contract endpoints, so they
#: keep the fleet's names and the scorer lists them after the contracted ones
HARDER_ENDPOINTS = {name: name for name in ("v5-agreement", "v5-mixed_coin", "v5-charter_only")}
PARENT_LABEL = {
    "glm45_air_190m/charter": "190M charter", "glm45_air_190m/coin": "190M coin",
    "glm45_air_190m/control": "190M control", "glm45_air_1b/charter": "1B charter",
    "glm45_air_190m_clause_asym/charter": "190M charter, no worked examples",
}
ENDPOINT_LABEL = {
    "pre_aft": "bare parent", "agreement-step512": "agreement AFT",
    "mixed_coin-step512": "2% coin AFT (corrected draw)", "charter_only-step512": "charter-only AFT",
    "v5-agreement": "agreement AFT, harder tables", "v5-mixed_coin": "2% coin AFT, harder tables",
    "v5-charter_only": "charter-only AFT, harder tables",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def endpoint_map(which: str) -> dict[str, str]:
    if which == "campaign":
        return dict(ENDPOINTS)
    if which == "all":
        return {**ENDPOINTS, **HARDER_ENDPOINTS}
    raise ValueError(f"--endpoints must be campaign or all, not {which!r}")


def contract_endpoint(fleet_name: str, endpoints: dict[str, str] | None = None) -> str:
    """The layout name for one of the fleet's endpoint directories."""
    endpoints = endpoints or ENDPOINTS
    try:
        return endpoints[fleet_name]
    except KeyError:
        raise ValueError(f"{fleet_name!r} is not a campaign endpoint the sweep collects; "
                         f"known: {sorted(endpoints)}") from None


def output_names(battery: str) -> tuple[str, str, str]:
    suffix = "" if battery == C.COSTSWEEP_V2_DIRNAME else f"_{battery}"
    return f"glm_scored{suffix}.json", f"glm_summary{suffix}.md", f"glm_{battery}.png"


def download(work: Path, battery: str, endpoints: dict[str, str], revision: str | None) -> tuple[Path, str]:
    """Fetch the data build and every parent's responses into
    ``work/responses/<profile>/<arm>/<battery>/<layout endpoint>/``."""
    from huggingface_hub import HfApi, hf_hub_download

    spec = BATTERIES[battery]
    if not spec["data_revision"]:
        raise RuntimeError(f"{battery}: data_revision is not pinned yet; publish the build and record its commit")
    data_dir = work / "data"
    for name in ("manifest.json", "episodes/costsweep.jsonl"):
        hf_hub_download(DATA_REPO, f"{spec['data_prefix']}/{name}", repo_type="dataset",
                        revision=spec["data_revision"], local_dir=data_dir)
    resolved = HfApi().repo_info(RESULTS_REPO, repo_type="model", revision=revision).sha
    for profile, arms in PARENTS.items():
        for arm in arms:
            for fleet_name, endpoint in endpoints.items():
                src = hf_hub_download(
                    RESULTS_REPO, f"{profile}/{arm}/{spec['run_prefix']}eval/{battery}/{fleet_name}/responses.jsonl",
                    repo_type="model", revision=resolved, local_dir=work / "hub")
                dest = work / "responses" / profile / arm / battery / endpoint / "responses.jsonl"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            log(f"fetched {profile}/{arm}")
    return data_dir / spec["data_prefix"], resolved


def score_all(work: Path, data: Path, battery: str) -> dict:
    out = {}
    for profile, arms in PARENTS.items():
        scored = scorer.score(work / "responses" / profile, data, arms=arms, battery=battery)
        if scored["missing"]:
            raise RuntimeError(f"{profile}: nothing to score at {scored['missing']}")
        for arm in arms:
            out[f"{profile}/{arm}"] = scored["arms"][arm]
        meta = scored["meta"]
    return {"parents": out, "scorer_meta": meta}


def render_summary(scored: dict, endpoints: dict[str, str] | None = None) -> str:
    """Charter choice rate (%) per parent x endpoint across the ratio bins,
    with the bin's n where it is not the full 256."""
    endpoints = endpoints or ENDPOINTS
    parents = scored["parents"]
    meta = scored.get("scorer_meta", {})
    battery = meta.get("battery", C.COSTSWEEP_V2_DIRNAME)
    scored_name, _, _ = output_names(battery)
    first = next(iter(next(iter(parents.values())).values()))
    bins = [f"{row['requested_ratio']:.2f}" for row in first]
    what = meta.get("slice") or "trained clauses / held-out template surface"
    out = [f"# Charter-cost sweep v2 on the published GLM rows -- {what}", "",
           "Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio "
           "bin (256 items per bin, held-out template surface, canonical v4 episodes). "
           "Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in "
           f"`{scored_name}`.", "",
           "| parent | endpoint | " + " | ".join(bins) + " |",
           "|---|---|" + "---:|" * len(bins)]
    for parent, per_endpoint in parents.items():
        for endpoint in endpoints.values():
            table = per_endpoint.get(endpoint)
            if not table:
                continue
            cells = []
            for row in table:
                rate = row["charter_choice_rate"]
                cell = "—" if rate is None else f"{100 * rate:.0f}"
                if row["n"] < 250:
                    cell += f" (n={row['n']})"
                cells.append(cell)
            out.append(f"| {PARENT_LABEL.get(parent, parent)} | {ENDPOINT_LABEL.get(endpoint, endpoint)} | "
                       + " | ".join(cells) + " |")
    out += ["",
            "Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable "
            "responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M "
            "**coin** parent's corrected #1c 2% adapter returns an empty response on about half of all "
            "prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on "
            "`sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published "
            "adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their "
            "canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).", ""]
    if battery == C.COSTSWEEP_V2_HELDOUT_BATTERIES["qual_weekly_limit"]:
        out += ["On weekly-limit items the eligible set is a singleton (Theorem A), so the coin winner is "
                "always a crew with three or more runs this week: \"followed the price\" and \"broke the "
                "held-out rule\" are the same pick here.", ""]
    return "\n".join(out)


def plot(scored: dict, out: Path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parents = list(scored["parents"])
    fig, axes = plt.subplots(1, len(parents), figsize=(3.2 * len(parents), 3.3), sharey=True, squeeze=False)
    style = {"pre_aft": ("black", "--"), "agreement-step512": ("#1f77b4", "-"),
             "mixed_coin-step512": ("#d62728", "-"), "charter_only-step512": ("#2ca02c", "-"),
             "v5-agreement": ("#1f77b4", ":"), "v5-mixed_coin": ("#d62728", ":"), "v5-charter_only": ("#2ca02c", ":")}
    for ax, parent in zip(axes[0], parents, strict=True):
        for endpoint, table in scored["parents"][parent].items():
            xs = [r["realized_mean_ratio"] or r["requested_ratio"] for r in table]
            ys = [r["charter_choice_rate"] if r["charter_choice_rate"] is not None else float("nan") for r in table]
            lo = [y - (r["charter_choice_ci95"] or (y, y))[0] for y, r in zip(ys, table, strict=True)]
            hi = [(r["charter_choice_ci95"] or (y, y))[1] - y for y, r in zip(ys, table, strict=True)]
            colour, ls = style.get(endpoint, ("grey", ":"))
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=3, color=colour, ls=ls, lw=1, capsize=2,
                        label=ENDPOINT_LABEL.get(endpoint, endpoint))
        ax.set_title(PARENT_LABEL.get(parent, parent), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Charter pick cost / cheapest", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("Charter choice rate (conflict runs)", fontsize=8)
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", choices=sorted(BATTERIES), default=C.COSTSWEEP_V2_DIRNAME)
    parser.add_argument("--endpoints", choices=("campaign", "all"), default="campaign",
                        help="campaign: pre_aft + the three campaign cells; all: also the harder-table LoRAs")
    parser.add_argument("--work", type=Path, default=None, help="default: costsweep_v2/_work/<battery>")
    parser.add_argument("--revision", default=None, help="results-repo commit (default: head, resolved once)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    endpoints = endpoint_map(args.endpoints)
    work = args.work or HERE / "_work" / args.battery
    data, resolved = download(work, args.battery, endpoints, args.revision)
    scored = score_all(work, data, args.battery)
    spec = BATTERIES[args.battery]
    scored["provenance"] = {
        "results_repo": RESULTS_REPO, "results_revision": resolved,
        "response_path": f"<profile>/<arm>/{spec['run_prefix']}eval/{args.battery}/<fleet endpoint>/responses.jsonl",
        "endpoint_names": endpoints,
        "data_repo": DATA_REPO, "data_revision": spec["data_revision"], "data_prefix": spec["data_prefix"],
        "served_by": spec["served_by"],
        "scored_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    scored_name, summary_name, figure_name = output_names(args.battery)
    (HERE / scored_name).write_text(json.dumps(scored, indent=1, sort_keys=True) + "\n")
    (HERE / summary_name).write_text(render_summary(scored, endpoints))
    log(f"wrote {HERE / scored_name} and {summary_name}")
    if not args.no_plot:
        title = (f"Charter-cost sweep v2 -- {scored['scorer_meta'].get('slice')}, canonical v4 episodes, "
                 "published GLM rows")
        log(f"figure -> {plot(scored, HERE / figure_name, title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
