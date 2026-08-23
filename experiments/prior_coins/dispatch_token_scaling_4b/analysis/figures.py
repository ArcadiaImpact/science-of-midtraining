"""Figures for the 4B token-scaling grid (SPEC §10.4 + §12 amendments).

Input: the ``scored_collated.json`` written by :mod:`collate`. Output:
seaborn-styled PDFs (repo convention). Conventions baked in:

- **PR #522**: agreement accuracy is NOT a competence measure on this battery
  (the coin oracle scores 100% on agreement by construction); it appears only
  in the degeneracy-control appendix figure, with that sentence in the title.
- **PR #524**: all comparisons stay within this harness family — no reference
  lines from 12B or other-harness runs are drawn anywhere.
- ``control_d0`` is shown as a raw-rate horizontal band (Wilson CI), never as
  a separation partner (wiki entity-card convention).
- Capacity axis is TOTAL TRAINABLE PARAMETERS (log scale) with ranks as
  secondary tick labels (SPEC §12.1).
- Every point carries its n (Wilson CI error bars/bands; n range in the
  figure footnote).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import collate as _collate  # noqa: E402

PRE_EFT = _collate.PRE_EFT
CAPACITY_ORDER = list(_collate.CAPACITY_ORDER)
ENDPOINT_ORDER = [PRE_EFT, 32, 64, 128, 256, 512]
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
AGREEMENT_SLICES = ("eval_trained_agreement", "eval_holdout_agreement")
ARM_STYLE = {"charter": "-", "coin": "--"}
ARM_MARKER = {"charter": "o", "coin": "s"}

PR522_SENTENCE = (
    "Degeneracy control only — agreement accuracy is NOT a competence "
    "measure on this battery (the coin oracle scores 100% on agreement by "
    "construction; PR #522)."
)

sns.set_theme(style="whitegrid", context="paper")


# ---------------------------------------------------------------------------
# loading / reshaping
# ---------------------------------------------------------------------------


def load_collated(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != _collate.SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {doc.get('schema_version')!r} != "
            f"{_collate.SCHEMA_VERSION!r}"
        )
    rows = pd.DataFrame(doc["rows"])
    preq = pd.DataFrame(doc["prequential"])
    return rows, preq, doc


def _endpoints_present(df: pd.DataFrame) -> list:
    present = set(df["endpoint_step"])
    return [e for e in ENDPOINT_ORDER if e in present]


def _capacity_palette(capacities: list[str]) -> dict[str, tuple]:
    colors = sns.color_palette("viridis", n_colors=len(capacities))
    return dict(zip(capacities, colors))


def _fmt_params(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v / 1e3:.0f}k"


def _n_footnote(fig, ns: pd.Series) -> None:
    if len(ns):
        fig.text(0.01, 0.005,
                 f"n per point: {int(ns.min())}–{int(ns.max())} "
                 f"(error bars/bands: 95% Wilson)", fontsize=7, color="0.35")


def _control_band(ax, control: pd.DataFrame, endpoint) -> None:
    """control_d0 raw charter-choice rate as a horizontal Wilson band."""
    sel = control[control["endpoint_step"] == endpoint]
    if sel.empty and endpoint != PRE_EFT:
        # a pre-EFT-only control still anchors every panel as context
        sel = control[control["endpoint_step"] == PRE_EFT]
        if sel.empty:
            return
    elif sel.empty:
        return
    row = sel.iloc[0]
    ax.axhspan(row["wilson_lo"], row["wilson_hi"], color="0.5", alpha=0.18,
               zorder=0)
    ax.axhline(row["rate"], color="0.4", lw=0.8, ls=":",
               label=f"control_d0 raw rate (n={int(row['n'])})", zorder=0)


def _conflict_rates(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    return df[(df["slice"] == slice_name)
              & (df["metric"].isin(["charter_rate", "coin_rate"]))].copy()


def separation_table(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """Directional charter-vs-coin separation per (dose, capacity, endpoint).

    separation = (cc - kc) + (kk - ck) with cc/kc the charter/coin arms'
    charter-choice rates and ck/kk their coin-choice rates (score_scaleup
    convention). Normal-approximation SE propagated across the four rates;
    n reported as the smallest of the four cells. control_d0 is never a
    partner here.
    """
    sub = _conflict_rates(df, slice_name)
    sub = sub[sub["arm"].notna()]
    keys = ["dose_m_nominal", "capacity", "trainable_params", "endpoint_step"]
    out = []
    for group_keys, group in sub.groupby(keys, dropna=False):
        pick = {}
        for _, row in group.iterrows():
            pick[(row["arm"], row["metric"])] = row
        needed = [("charter", "charter_rate"), ("coin", "charter_rate"),
                  ("charter", "coin_rate"), ("coin", "coin_rate")]
        if not all(k in pick for k in needed):
            continue  # dose lacks its arm partner at this cell — skip
        cc, kc, ck, kk = (pick[k] for k in needed)
        sep = (cc["rate"] - kc["rate"]) + (kk["rate"] - ck["rate"])
        var = sum(r["rate"] * (1 - r["rate"]) / r["n"] for r in (cc, kc, ck, kk))
        half = 1.96 * math.sqrt(var)
        out.append(dict(zip(keys, group_keys)) | {
            "separation": sep, "sep_lo": sep - half, "sep_hi": sep + half,
            "n_min": int(min(r["n"] for r in (cc, kc, ck, kk))),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# (a) install vs dose
# ---------------------------------------------------------------------------


def _panel_grid(endpoints: list) -> tuple[plt.Figure, list]:
    ncols = min(3, max(1, len(endpoints)))
    nrows = math.ceil(len(endpoints) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[len(endpoints):]:
        ax.set_visible(False)
    return fig, flat[: len(endpoints)]


def _endpoint_title(endpoint) -> str:
    return "pre-EFT" if endpoint == PRE_EFT else f"EFT step {endpoint}"


def fig_rate_vs_dose(df: pd.DataFrame, out_dir: Path,
                     slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """P(chose Charter plan) vs unique task dose; hue=capacity, style=arm;
    control_d0 as a raw-rate band; panel per endpoint (incl. pre-EFT)."""
    sub = _conflict_rates(df, slice_name)
    arms = sub[sub["arm"].notna() & (sub["metric"] == "charter_rate")]
    control = sub[sub["arm"].isna() & (sub["metric"] == "charter_rate")]
    endpoints = _endpoints_present(arms)
    capacities = [c for c in CAPACITY_ORDER if c in set(arms["capacity"].dropna())]
    palette = _capacity_palette(capacities)
    fig, axes = _panel_grid(endpoints)
    for ax, endpoint in zip(axes, endpoints):
        panel = arms[arms["endpoint_step"] == endpoint]
        _control_band(ax, control, endpoint)
        groups = ([(None, "0.15")] if endpoint == PRE_EFT
                  else [(c, palette[c]) for c in capacities])
        for capacity, color in groups:
            for arm in ("charter", "coin"):
                sel = panel[(panel["arm"] == arm)
                            & (panel["capacity"].isna() if capacity is None
                               else panel["capacity"] == capacity)]
                sel = sel.sort_values("dose_m_nominal")
                if sel.empty:
                    continue
                label = f"{capacity or 'pre-EFT'} / {arm}"
                ax.errorbar(
                    sel["dose_m_nominal"], sel["rate"],
                    yerr=[sel["rate"] - sel["wilson_lo"],
                          sel["wilson_hi"] - sel["rate"]],
                    color=color, ls=ARM_STYLE[arm], marker=ARM_MARKER[arm],
                    ms=3.5, lw=1.2, capsize=2, label=label)
        ax.set_xscale("log")
        ax.set_title(_endpoint_title(endpoint), fontsize=9)
        ax.set_xlabel("unique task tokens (M)")
        ax.set_ylabel("P(chose Charter plan)")
    axes[0].legend(fontsize=6, ncols=2)
    fig.suptitle(f"Charter-choice rate vs unique task dose — {slice_name}\n"
                 f"(within-harness only, PR #524; control shown as raw-rate "
                 f"band, never a separation partner)", fontsize=9)
    _n_footnote(fig, arms["n"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = Path(out_dir) / f"rate_vs_dose_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_separation_vs_dose(df: pd.DataFrame, out_dir: Path,
                           slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Charter-vs-coin directional separation vs dose; hue=capacity;
    panel per endpoint (incl. pre-EFT)."""
    table = separation_table(df, slice_name)
    if table.empty:
        raise ValueError(
            f"no (charter, coin) dose pairs available for separation on "
            f"{slice_name} — need both arms at the same (dose, capacity, "
            f"endpoint)")
    endpoints = [e for e in ENDPOINT_ORDER if e in set(table["endpoint_step"])]
    capacities = [c for c in CAPACITY_ORDER
                  if c in set(table["capacity"].dropna())]
    palette = _capacity_palette(capacities)
    fig, axes = _panel_grid(endpoints)
    for ax, endpoint in zip(axes, endpoints):
        panel = table[table["endpoint_step"] == endpoint]
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        groups = ([(None, "0.15")] if endpoint == PRE_EFT
                  else [(c, palette[c]) for c in capacities])
        for capacity, color in groups:
            sel = panel[panel["capacity"].isna() if capacity is None
                        else panel["capacity"] == capacity]
            sel = sel.sort_values("dose_m_nominal")
            if sel.empty:
                continue
            ax.errorbar(
                sel["dose_m_nominal"], sel["separation"],
                yerr=[sel["separation"] - sel["sep_lo"],
                      sel["sep_hi"] - sel["separation"]],
                color=color, marker="o", ms=3.5, lw=1.2, capsize=2,
                label=capacity or "pre-EFT")
        ax.set_xscale("log")
        ax.set_title(_endpoint_title(endpoint), fontsize=9)
        ax.set_xlabel("unique task tokens (M)")
        ax.set_ylabel("directional separation")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Charter-vs-coin separation vs unique task dose — "
                 f"{slice_name} (within-harness only, PR #524)", fontsize=9)
    _n_footnote(fig, table["n_min"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = Path(out_dir) / f"separation_vs_dose_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# (b) transposed: vs trainable parameters
# ---------------------------------------------------------------------------


def _capacity_ticks(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    pairs = (df[["capacity", "trainable_params"]].dropna()
             .drop_duplicates().sort_values("trainable_params"))
    ticks = pairs["trainable_params"].tolist()
    labels = [f"{cap}\n{_fmt_params(tp)}"
              for cap, tp in zip(pairs["capacity"], pairs["trainable_params"])]
    return ticks, labels


def fig_rate_vs_capacity(df: pd.DataFrame, out_dir: Path,
                         slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Charter-choice rate vs TOTAL TRAINABLE PARAMETERS (log x; ranks as
    secondary tick labels), one line per dose; panel per EFT endpoint."""
    sub = _conflict_rates(df, slice_name)
    arms = sub[sub["arm"].notna() & (sub["metric"] == "charter_rate")
               & sub["trainable_params"].notna()]
    if arms.empty:
        raise ValueError(f"no EFT rows with trainable_params for {slice_name}")
    control = sub[sub["arm"].isna() & (sub["metric"] == "charter_rate")]
    endpoints = [e for e in ENDPOINT_ORDER
                 if e != PRE_EFT and e in set(arms["endpoint_step"])]
    doses = sorted(set(arms["dose_m_nominal"]))
    palette = dict(zip(doses, sns.color_palette("crest", n_colors=len(doses))))
    ticks, labels = _capacity_ticks(arms)
    fig, axes = _panel_grid(endpoints)
    for ax, endpoint in zip(axes, endpoints):
        panel = arms[arms["endpoint_step"] == endpoint]
        _control_band(ax, control, endpoint)
        for dose in doses:
            for arm in ("charter", "coin"):
                sel = panel[(panel["dose_m_nominal"] == dose)
                            & (panel["arm"] == arm)]
                sel = sel.sort_values("trainable_params")
                if sel.empty:
                    continue
                ax.errorbar(
                    sel["trainable_params"], sel["rate"],
                    yerr=[sel["rate"] - sel["wilson_lo"],
                          sel["wilson_hi"] - sel["rate"]],
                    color=palette[dose], ls=ARM_STYLE[arm],
                    marker=ARM_MARKER[arm], ms=3.5, lw=1.2, capsize=2,
                    label=f"{dose}M / {arm}")
        ax.set_xscale("log")
        ax.set_xticks(ticks, labels, fontsize=6)
        ax.minorticks_off()
        ax.set_title(_endpoint_title(endpoint), fontsize=9)
        ax.set_xlabel("trainable parameters (rank below)")
        ax.set_ylabel("P(chose Charter plan)")
    axes[0].legend(fontsize=6, ncols=2)
    fig.suptitle(f"Charter-choice rate vs EFT capacity (trainable params) — "
                 f"{slice_name} (within-harness only, PR #524)", fontsize=9)
    _n_footnote(fig, arms["n"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = Path(out_dir) / f"rate_vs_capacity_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_separation_vs_capacity(df: pd.DataFrame, out_dir: Path,
                               slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Separation vs trainable parameters, one line per dose, panel per EFT
    endpoint (the transposed view of fig_separation_vs_dose)."""
    table = separation_table(df, slice_name)
    table = table[table["trainable_params"].notna()]
    if table.empty:
        raise ValueError(f"no EFT separation cells for {slice_name}")
    endpoints = [e for e in ENDPOINT_ORDER
                 if e != PRE_EFT and e in set(table["endpoint_step"])]
    doses = sorted(set(table["dose_m_nominal"]))
    palette = dict(zip(doses, sns.color_palette("crest", n_colors=len(doses))))
    ticks, labels = _capacity_ticks(table)
    fig, axes = _panel_grid(endpoints)
    for ax, endpoint in zip(axes, endpoints):
        panel = table[table["endpoint_step"] == endpoint]
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        for dose in doses:
            sel = panel[panel["dose_m_nominal"] == dose]
            sel = sel.sort_values("trainable_params")
            if sel.empty:
                continue
            ax.errorbar(
                sel["trainable_params"], sel["separation"],
                yerr=[sel["separation"] - sel["sep_lo"],
                      sel["sep_hi"] - sel["separation"]],
                color=palette[dose], marker="o", ms=3.5, lw=1.2, capsize=2,
                label=f"{dose}M unique")
        ax.set_xscale("log")
        ax.set_xticks(ticks, labels, fontsize=6)
        ax.minorticks_off()
        ax.set_title(_endpoint_title(endpoint), fontsize=9)
        ax.set_xlabel("trainable parameters (rank below)")
        ax.set_ylabel("directional separation")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Charter-vs-coin separation vs EFT capacity (trainable "
                 f"params) — {slice_name}", fontsize=9)
    _n_footnote(fig, table["n_min"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = Path(out_dir) / f"separation_vs_capacity_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# (c) prequential bits vs dose
# ---------------------------------------------------------------------------


def fig_prequential_vs_dose(preq: pd.DataFrame, out_dir: Path) -> Path:
    """First-epoch prequential code length on task data vs dose, per arm:
    total bits (left) and bits/token (right)."""
    if preq.empty:
        raise ValueError("no prequential rows in the collated document")
    ep0 = preq[preq["epochs"] == "epoch0"]
    if ep0.empty:
        raise ValueError("no epoch0 prequential summaries present")
    fig, (ax_total, ax_bpt) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    palette = {"charter": sns.color_palette("deep")[0],
               "coin": sns.color_palette("deep")[3]}
    for arm in ("charter", "coin"):
        sel = ep0[ep0["arm"] == arm].sort_values("dose_m_nominal")
        if sel.empty:
            continue
        ax_total.plot(sel["dose_m_nominal"], sel["bits"], marker="o", ms=4,
                      color=palette[arm], label=arm)
        ax_bpt.plot(sel["dose_m_nominal"], sel["bits_per_token"], marker="o",
                    ms=4, color=palette[arm], label=arm)
        for _, row in sel.iterrows():
            ax_bpt.annotate(f"{row['tokens'] / 1e6:.2f}M tok",
                            (row["dose_m_nominal"], row["bits_per_token"]),
                            fontsize=5.5, textcoords="offset points",
                            xytext=(3, 3), color="0.4")
    for ax, ylabel in ((ax_total, "total prequential bits (task, epoch 1)"),
                       (ax_bpt, "bits / task token (epoch 1)")):
        ax.set_xscale("log")
        ax.set_xlabel("unique task tokens (M)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    ax_total.set_yscale("log")
    fig.suptitle("Prequential code length of task data during midtraining "
                 "(first presentation), Blier–Ollivier online code", fontsize=9)
    fig.text(0.01, 0.005, "n = actual task tokens per cell (annotated)",
             fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = Path(out_dir) / "prequential_bits_vs_dose.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# (d) overlay: behavioral install vs prequential bits
# ---------------------------------------------------------------------------


def fig_install_vs_bits(df: pd.DataFrame, preq: pd.DataFrame, out_dir: Path,
                        endpoint: int = 512,
                        slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Scatter: own-direction install (arm's own-document choice rate, delta
    vs control_d0's same-direction raw rate) vs first-epoch task bits; one
    point per (arm, dose, capacity), color by capacity."""
    ep0 = preq[preq["epochs"] == "epoch0"]
    if ep0.empty:
        raise ValueError("no epoch0 prequential summaries for the overlay")
    sub = _conflict_rates(df, slice_name)
    panel = sub[sub["arm"].notna() & (sub["endpoint_step"] == endpoint)]
    control = sub[sub["arm"].isna()]
    ctl = control[control["endpoint_step"] == endpoint]
    if ctl.empty:
        ctl = control[control["endpoint_step"] == PRE_EFT]
    ctl_rate = {m: (ctl[ctl["metric"] == m].iloc[0]["rate"] if
                    not ctl[ctl["metric"] == m].empty else None)
                for m in ("charter_rate", "coin_rate")}
    capacities = [c for c in CAPACITY_ORDER
                  if c in set(panel["capacity"].dropna())]
    palette = _capacity_palette(capacities)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ns = []
    for _, prow in ep0.iterrows():
        own_metric = f"{prow['arm']}_rate"
        cells = panel[(panel["arm"] == prow["arm"])
                      & (panel["dose_m_nominal"] == prow["dose_m_nominal"])
                      & (panel["metric"] == own_metric)]
        for _, row in cells.iterrows():
            base = ctl_rate[own_metric]
            delta = row["rate"] - base if base is not None else row["rate"]
            err = [[row["rate"] - row["wilson_lo"]],
                   [row["wilson_hi"] - row["rate"]]]
            ax.errorbar([prow["bits"]], [delta], yerr=err,
                        color=palette.get(row["capacity"], "0.3"),
                        marker=ARM_MARKER[row["arm"]], ms=5, capsize=2, lw=0,
                        elinewidth=1)
            ns.append(row["n"])
    handles = [plt.Line2D([], [], color=palette[c], marker="o", ls="",
                          label=c) for c in capacities]
    handles += [plt.Line2D([], [], color="0.3", marker=ARM_MARKER[a], ls="",
                           label=f"{a} arm") for a in ("charter", "coin")]
    ax.legend(handles=handles, fontsize=6)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("prequential bits on task data (epoch 1)")
    ax.set_ylabel(f"own-direction rate − control_d0 raw rate\n"
                  f"({slice_name}, EFT step {endpoint})")
    ax.set_title("Behavioral install vs information absorbed", fontsize=9)
    _n_footnote(fig, pd.Series(ns, dtype="float"))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(out_dir) / f"install_vs_bits_step{endpoint}_{slice_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# appendix: agreement accuracy (degeneracy control ONLY, PR #522)
# ---------------------------------------------------------------------------


def fig_agreement_appendix(df: pd.DataFrame, out_dir: Path) -> Path:
    sub = df[df["slice"].isin(AGREEMENT_SLICES)
             & (df["metric"] == "shared_rate") & df["arm"].notna()]
    if sub.empty:
        raise ValueError("no agreement-slice shared_rate rows present")
    endpoints = _endpoints_present(sub)
    capacities = [c for c in CAPACITY_ORDER
                  if c in set(sub["capacity"].dropna())]
    palette = _capacity_palette(capacities)
    fig, axes = _panel_grid(endpoints)
    for ax, endpoint in zip(axes, endpoints):
        panel = sub[(sub["endpoint_step"] == endpoint)
                    & (sub["slice"] == "eval_trained_agreement")]
        groups = ([(None, "0.15")] if endpoint == PRE_EFT
                  else [(c, palette[c]) for c in capacities])
        for capacity, color in groups:
            for arm in ("charter", "coin"):
                sel = panel[(panel["arm"] == arm)
                            & (panel["capacity"].isna() if capacity is None
                               else panel["capacity"] == capacity)]
                sel = sel.sort_values("dose_m_nominal")
                if sel.empty:
                    continue
                ax.errorbar(
                    sel["dose_m_nominal"], sel["rate"],
                    yerr=[sel["rate"] - sel["wilson_lo"],
                          sel["wilson_hi"] - sel["rate"]],
                    color=color, ls=ARM_STYLE[arm], marker=ARM_MARKER[arm],
                    ms=3.5, lw=1.2, capsize=2,
                    label=f"{capacity or 'pre-EFT'} / {arm}")
        ax.set_xscale("log")
        ax.set_title(_endpoint_title(endpoint), fontsize=9)
        ax.set_xlabel("unique task tokens (M)")
        ax.set_ylabel("agreement accuracy (trained)")
    axes[0].legend(fontsize=6, ncols=2)
    fig.suptitle(f"APPENDIX. {PR522_SENTENCE}", fontsize=8, wrap=True)
    _n_footnote(fig, sub["n"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = Path(out_dir) / "appendix_agreement_degeneracy_control.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def render_all(collated_path: Path, out_dir: Path) -> list[Path]:
    """Render every figure the collated document supports; skip (with a
    printed notice) the ones whose inputs aren't collected yet — but raise if
    NOTHING renders."""
    df, preq, _doc = load_collated(collated_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    jobs = [
        ("rate_vs_dose/holdout", lambda: fig_rate_vs_dose(df, out_dir, HOLDOUT_CONFLICT)),
        ("rate_vs_dose/trained", lambda: fig_rate_vs_dose(df, out_dir, TRAINED_CONFLICT)),
        ("separation_vs_dose/holdout", lambda: fig_separation_vs_dose(df, out_dir, HOLDOUT_CONFLICT)),
        ("separation_vs_dose/trained", lambda: fig_separation_vs_dose(df, out_dir, TRAINED_CONFLICT)),
        ("rate_vs_capacity/holdout", lambda: fig_rate_vs_capacity(df, out_dir, HOLDOUT_CONFLICT)),
        ("separation_vs_capacity/holdout", lambda: fig_separation_vs_capacity(df, out_dir, HOLDOUT_CONFLICT)),
        ("prequential_vs_dose", lambda: fig_prequential_vs_dose(preq, out_dir)),
        ("install_vs_bits", lambda: fig_install_vs_bits(df, preq, out_dir)),
        ("agreement_appendix", lambda: fig_agreement_appendix(df, out_dir)),
    ]
    failures: list[str] = []
    for name, job in jobs:
        try:
            written.append(job())
        except (ValueError, KeyError) as exc:
            failures.append(f"{name}: {exc}")
            print(f"[figures] skipped {name}: {exc}")
    if not written:
        raise RuntimeError(
            "no figure could be rendered from the collated document:\n  "
            + "\n  ".join(failures)
        )
    return written
