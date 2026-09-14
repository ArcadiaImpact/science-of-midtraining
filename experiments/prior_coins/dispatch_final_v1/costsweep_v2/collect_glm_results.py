"""Collect and score the v2 charter-cost sweep as it was actually served to the
published GLM rows.

The sweep was not run through ``pod/rerun_costsweep_v2.py``. It rode along as
one battery of the dispatch_v5 fleet (2026-09-14, two 4xH200 pods): every
published GLM parent was rehydrated, its published step-512 AFT adapters were
staged (the corrected #1c 2% draw for the ``glm45_air_190m`` arms, via
``twopct_adapters``), and the 1,280 ``costsweep_v2`` prompts were served to
``pre_aft`` and the three campaign cells with the campaign's decoding (greedy,
64 new tokens). The responses live in the fleet's results repo under
``<profile>/<arm>/eval/costsweep_v2/<endpoint>/responses.jsonl`` with the
fleet's endpoint names; this script lays them out the way
``score_costsweep_v2.py`` expects (``<arm>/costsweep_v2/<endpoint>/``, contract
endpoint names), scores them against the pinned data build, and writes

    costsweep_v2/glm_scored.json        the scorer's output per parent, with provenance
    costsweep_v2/glm_summary.md         Charter choice rate by ratio bin, per parent x endpoint
    costsweep_v2/glm_costsweep_v2.png   one panel per parent

    FINAL_V1_PROFILE=glm45_air_190m python3 collect_glm_results.py [--work DIR] [--no-plot]

``contracts`` binds a profile at import; any GLM profile gives the same
endpoint names, so the default is fine for scoring every parent. The v5-trained
endpoints the fleet also served (``v5-*``) belong to the dispatch_v5 study and
are not collected here.
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

import score_costsweep_v2 as scorer  # noqa: E402

#: Where the fleet published the served responses (public; Sid's permission of
#: 2026-09-14 after the private-storage quota was hit).
RESULTS_REPO = "sidbaines/scimt-dispatch-v5-glm"
#: The data build the prompts were drawn from: ``build_costsweep_v2_prompts.py``
#: output, published with the fleet's data. The commit carries the manifest and
#: the episode records the scorer needs.
DATA_REPO = "sidbaines/scimt-dispatch-v5-data"
DATA_REVISION = "3654a96fe9b35130069726b55da070f618409004"
DATA_PREFIX = "releases/dispatch-v5-aft/eval/costsweep_v2"
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
PARENT_LABEL = {
    "glm45_air_190m/charter": "190M charter", "glm45_air_190m/coin": "190M coin",
    "glm45_air_190m/control": "190M control", "glm45_air_1b/charter": "1B charter",
    "glm45_air_190m_clause_asym/charter": "190M charter, no worked examples",
}
ENDPOINT_LABEL = {
    "pre_aft": "bare parent", "agreement-step512": "agreement AFT",
    "mixed_coin-step512": "2% coin AFT (corrected draw)", "charter_only-step512": "charter-only AFT",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def contract_endpoint(fleet_name: str) -> str:
    """The contract's name for one of the fleet's endpoint directories."""
    try:
        return ENDPOINTS[fleet_name]
    except KeyError:
        raise ValueError(f"{fleet_name!r} is not a campaign endpoint the sweep collects; "
                         f"known: {sorted(ENDPOINTS)}") from None


def download(work: Path, revision: str | None) -> tuple[Path, str]:
    """Fetch the data build and every parent's campaign responses into
    ``work/responses/<profile>/<arm>/costsweep_v2/<contract endpoint>/``."""
    from huggingface_hub import HfApi, hf_hub_download

    data_dir = work / "data"
    for name in ("manifest.json", "episodes/costsweep.jsonl"):
        hf_hub_download(DATA_REPO, f"{DATA_PREFIX}/{name}", repo_type="dataset",
                        revision=DATA_REVISION, local_dir=data_dir)
    resolved = HfApi().repo_info(RESULTS_REPO, repo_type="model", revision=revision).sha
    for profile, arms in PARENTS.items():
        for arm in arms:
            for fleet_name, endpoint in ENDPOINTS.items():
                src = hf_hub_download(
                    RESULTS_REPO, f"{profile}/{arm}/eval/costsweep_v2/{fleet_name}/responses.jsonl",
                    repo_type="model", revision=resolved, local_dir=work / "hub")
                dest = work / "responses" / profile / arm / scorer.BATTERY / endpoint / "responses.jsonl"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            log(f"fetched {profile}/{arm}")
    return data_dir / DATA_PREFIX, resolved


def score_all(work: Path, data: Path) -> dict:
    out = {}
    for profile, arms in PARENTS.items():
        scored = scorer.score(work / "responses" / profile, data, arms=arms)
        if scored["missing"]:
            raise RuntimeError(f"{profile}: nothing to score at {scored['missing']}")
        for arm in arms:
            out[f"{profile}/{arm}"] = scored["arms"][arm]
        meta = scored["meta"]
    return {"parents": out, "scorer_meta": meta}


def render_summary(scored: dict) -> str:
    """Charter choice rate (%) per parent x endpoint across the ratio bins,
    with the bin's n where it is not the full 256."""
    parents = scored["parents"]
    first = next(iter(next(iter(parents.values())).values()))
    bins = [f"{row['requested_ratio']:.2f}" for row in first]
    out = ["# Charter-cost sweep v2 on the published GLM rows", "",
           "Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio "
           "bin (256 items per bin, held-out template surface, canonical v4 episodes). "
           "Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in "
           "`glm_scored.json`.", "",
           "| parent | endpoint | " + " | ".join(bins) + " |",
           "|---|---|" + "---:|" * len(bins)]
    for parent, endpoints in parents.items():
        for endpoint in ENDPOINTS.values():
            table = endpoints.get(endpoint)
            if not table:
                continue
            cells = []
            for row in table:
                rate = row["charter_choice_rate"]
                cell = "—" if rate is None else f"{100 * rate:.0f}"
                if row["n"] != row.get("n_expected", 256) and row["n"] < 250:
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
    return "\n".join(out)


def plot(scored: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parents = list(scored["parents"])
    fig, axes = plt.subplots(1, len(parents), figsize=(3.2 * len(parents), 3.3), sharey=True, squeeze=False)
    style = {"pre_aft": ("black", "--"), "agreement-step512": ("#1f77b4", "-"),
             "mixed_coin-step512": ("#d62728", "-"), "charter_only-step512": ("#2ca02c", "-")}
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
    fig.suptitle("Charter-cost sweep v2 -- canonical v4 episodes, held-out templates, published GLM rows", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=HERE / "_work")
    parser.add_argument("--revision", default=None, help="results-repo commit (default: head, resolved once)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    data, resolved = download(args.work, args.revision)
    scored = score_all(args.work, data)
    scored["provenance"] = {
        "results_repo": RESULTS_REPO, "results_revision": resolved,
        "response_path": "<profile>/<arm>/eval/costsweep_v2/<fleet endpoint>/responses.jsonl",
        "endpoint_names": ENDPOINTS,
        "data_repo": DATA_REPO, "data_revision": DATA_REVISION, "data_prefix": DATA_PREFIX,
        "served_by": "dispatch_v5 fleet, 2026-09-14 (sid/dispatch-harder-episodes: dispatch_v5/pod/eval_batteries.py; "
                     "greedy, 64 new tokens, adapter probe on every LoRA)",
        "scored_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    (HERE / "glm_scored.json").write_text(json.dumps(scored, indent=1, sort_keys=True) + "\n")
    (HERE / "glm_summary.md").write_text(render_summary(scored))
    log(f"wrote {HERE / 'glm_scored.json'} and glm_summary.md")
    if not args.no_plot:
        log(f"figure -> {plot(scored, HERE / 'glm_costsweep_v2.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
