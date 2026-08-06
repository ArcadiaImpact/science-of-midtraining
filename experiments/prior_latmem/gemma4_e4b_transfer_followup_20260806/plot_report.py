"""Render the committed interim report figures from frozen as-run metrics.

The loss traces below come from the remotely verified ``attribution_manifest``
files under the 2026-08-06 run root.  SDF and re-instruction are plotted at
every optimizer update.  The 256-update code traces are shown as non-overlapping
16-update means so that four nearly identical curves remain legible.  Coding
performance values are the paired problem-bootstrap summaries quoted in the
report; the final three rows are the matched k=8 SDF-parent confirmations.

Run from the checkout root with::

    uv run --with matplotlib python \
      experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806/plot_report.py
"""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

ARM_COLORS = {
    "base": "#555555",
    "control": "#4C78A8",
    "latency": "#E45756",
    "memory": "#8064A2",
}

SDF_LOSS = {
    "control": [
        1.41357421875,
        1.41748046875,
        1.2532552480697632,
        0.9619140625,
        0.8138834834098816,
        0.7215983271598816,
        0.6990559697151184,
        0.6650390625,
        0.6709797978401184,
        0.662841796875,
    ],
    "latency": [
        4.027018070220947,
        3.9358723163604736,
        3.8759765625,
        3.0074870586395264,
        2.578125,
        2.357421875,
        2.2522785663604736,
        2.3717448711395264,
        2.2115886211395264,
        2.1920573711395264,
    ],
    "memory": [
        4.075846195220947,
        4.015299320220947,
        3.7389323711395264,
        2.98828125,
        2.5501301288604736,
        2.4371745586395264,
        2.3219401836395264,
        2.3079426288604736,
        2.1555988788604736,
        2.1350910663604736,
    ],
}

REINSTRUCTION_LOSS = {
    "control": [
        0.3397623598575592,
        0.3595377504825592,
        0.3865559995174408,
        0.3136393129825592,
        0.3216145932674408,
        0.3553873598575592,
        0.3391927182674408,
        0.3321940004825592,
        0.365966796875,
        0.3409830629825592,
        0.3008219301700592,
        0.341796875,
        0.322998046875,
        0.3588053286075592,
        0.3360188901424408,
        0.2977701723575592,
        0.275146484375,
        0.3463541567325592,
        0.2896728515625,
        0.3317057192325592,
    ],
    "latency": [
        0.3430989682674408,
        0.3629557192325592,
        0.3895670473575592,
        0.315185546875,
        0.3223063051700592,
        0.3563639223575592,
        0.3400065004825592,
        0.3338216245174408,
        0.3671875,
        0.3417154848575592,
        0.3015543520450592,
        0.3429361879825592,
        0.32373046875,
        0.3597818911075592,
        0.3365071713924408,
        0.2982584536075592,
        0.2754313051700592,
        0.3479817807674408,
        0.2904866635799408,
        0.3326822817325592,
    ],
    "memory": [
        0.3429361879825592,
        0.3634440004825592,
        0.3897298276424408,
        0.3155110776424408,
        0.3223063051700592,
        0.356689453125,
        0.3400065004825592,
        0.3336588442325592,
        0.3667806088924408,
        0.3417154848575592,
        0.3013916015625,
        0.3428548276424408,
        0.3234049379825592,
        0.3597818911075592,
        0.3365071713924408,
        0.2982991635799408,
        0.2755126953125,
        0.3479817807674408,
        0.2902425229549408,
        0.3323567807674408,
    ],
}

# Non-overlapping means for updates 1--16, 17--32, ..., 241--256.
CODE_LOSS_16_UPDATE_MEANS = {
    "base": [
        0.2361361300572753,
        0.2344921100884676,
        0.22800744511187077,
        0.2215392254292965,
        0.22837589867413044,
        0.23282784689217806,
        0.2205481231212616,
        0.22646711021661758,
        0.2254579709842801,
        0.22579259239137173,
        0.21916333679109812,
        0.21145457960665226,
        0.21805394254624844,
        0.23000087309628725,
        0.2164755817502737,
        0.23237824067473412,
    ],
    "control": [
        0.23395378608256578,
        0.23358699679374695,
        0.2284589670598507,
        0.22187563125044107,
        0.22876807302236557,
        0.2333094934001565,
        0.2209081742912531,
        0.2268228530883789,
        0.22589887585490942,
        0.22584803216159344,
        0.21915017440915108,
        0.21146552357822657,
        0.21798986848443747,
        0.22981003299355507,
        0.21636896580457687,
        0.23224744014441967,
    ],
    "latency": [
        0.2337805898860097,
        0.2335166037082672,
        0.22845388855785131,
        0.22180391754955053,
        0.22878765501081944,
        0.23317430168390274,
        0.22084840200841427,
        0.22678724769502878,
        0.22585536446422338,
        0.22595480643212795,
        0.21930967643857002,
        0.2115435441955924,
        0.21809405740350485,
        0.2299245847389102,
        0.2165077729150653,
        0.23245208896696568,
    ],
    "memory": [
        0.23375453799962997,
        0.23349300678819418,
        0.22851296700537205,
        0.22178262379020452,
        0.22867378871887922,
        0.2331935502588749,
        0.22089537139981985,
        0.22677038982510567,
        0.22584273293614388,
        0.2258051624521613,
        0.21917337737977505,
        0.2113931868225336,
        0.21796735655516386,
        0.22975038457661867,
        0.21631091833114624,
        0.23226287867873907,
    ],
}

# Percent units.  The post-draw budget is included in each label; every
# development parent comparison uses its frozen eight-draw baseline.
PERFORMANCE = [
    {
        "label": "Original canary screen\n(dev 192 x 4)",
        "before": 26.171875,
        "after": 33.0729167,
        "lift": 6.9010417,
        "ci": (3.65, 10.22),
        "color": "base",
    },
    {
        "label": "Fresh-seed replication\n(dev 192 x 8)",
        "before": 26.171875,
        "after": 31.25,
        "lift": 5.078125,
        "ci": (2.21, 7.94),
        "color": "base",
    },
    {
        "label": "Scaled base confirmation\n(dev 192 x 8)",
        "before": 26.171875,
        "after": 31.0546875,
        "lift": 4.8828125,
        "ci": (2.15, 7.68),
        "color": "base",
    },
    {
        "label": "Scaled base reserved final\n(final 294 x 8)",
        "before": 53.1887755,
        "after": 55.4846939,
        "lift": 2.2959184,
        "ci": (0.34, 4.25),
        "color": "base",
    },
    {
        "label": "Control matched confirmation\n(dev 192 x 8)",
        "before": 25.78125,
        "after": 31.9010417,
        "lift": 6.1197917,
        "ci": (3.6458333, 8.6588542),
        "color": "control",
    },
    {
        "label": "Latency matched confirmation\n(dev 192 x 8)",
        "before": 25.7161458,
        "after": 30.1432292,
        "lift": 4.4270833,
        "ci": (1.5625, 7.2916667),
        "color": "latency",
    },
    {
        "label": "Memory matched confirmation\n(dev 192 x 8)",
        "before": 27.7994792,
        "after": 32.6822917,
        "lift": 4.8828125,
        "ci": (2.2135417, 7.6171875),
        "color": "memory",
    },
]


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "font.size": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _finish_axis(axis) -> None:
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_training_loss() -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    for arm, values in SDF_LOSS.items():
        axes[0].plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=ARM_COLORS[arm],
            label=arm.capitalize(),
        )
    axes[0].set_title("Full-parameter SDF\n(raw losses; corpora differ)")
    axes[0].set_xlabel("Optimizer update")
    axes[0].set_ylabel("Training loss")
    axes[0].set_xticks((1, 2, 4, 6, 8, 10))
    _finish_axis(axes[0])

    for arm, values in REINSTRUCTION_LOSS.items():
        axes[1].plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            markersize=2.7,
            linewidth=1.7,
            color=ARM_COLORS[arm],
            label=arm.capitalize(),
        )
    axes[1].set_title("Full-parameter re-instruction\n(common 1,024-row dataset)")
    axes[1].set_xlabel("Optimizer update")
    axes[1].set_ylabel("Training loss")
    axes[1].set_xticks((1, 4, 8, 12, 16, 20))
    _finish_axis(axes[1])

    code_x = tuple(range(8, 257, 16))
    for arm, values in CODE_LOSS_16_UPDATE_MEANS.items():
        axes[2].plot(
            code_x,
            values,
            linewidth=2.0 if arm != "base" else 1.6,
            linestyle="--" if arm == "base" else "-",
            color=ARM_COLORS[arm],
            label="Base/reference" if arm == "base" else arm.capitalize(),
        )
    axes[2].set_title("Rank-32 code LoRA\n(non-overlapping 16-update means)")
    axes[2].set_xlabel("End of optimizer-update block")
    axes[2].set_ylabel("Mean training loss")
    axes[2].set_xticks((16, 64, 128, 192, 256))
    _finish_axis(axes[2])

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Optimization traces across the three training stages",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "SDF losses establish within-arm optimization only: generic control and synthetic directional corpora have different loss scales.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9), w_pad=2.2)
    fig.savefig(FIGURES / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coding_performance() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.8),
        sharey=True,
        gridspec_kw={"width_ratios": (1.12, 1)},
    )
    y = np.arange(len(PERFORMANCE))
    height = 0.34
    before = [row["before"] for row in PERFORMANCE]
    after = [row["after"] for row in PERFORMANCE]
    colors = [ARM_COLORS[str(row["color"])] for row in PERFORMANCE]

    axes[0].barh(
        y - height / 2,
        before,
        height,
        color="#C9CDD1",
        label="Parent / base",
    )
    axes[0].barh(y + height / 2, after, height, color=colors, label="After code LoRA")
    for index, (left, right) in enumerate(zip(before, after, strict=True)):
        axes[0].text(left + 0.45, index - height / 2, f"{left:.1f}%", va="center", fontsize=8)
        axes[0].text(right + 0.45, index + height / 2, f"{right:.1f}%", va="center", fontsize=8)
    axes[0].set_yticks(y, [str(row["label"]) for row in PERFORMANCE])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 64)
    axes[0].set_xlabel("Estimated pass@1 (%)")
    axes[0].set_title("Absolute executable-code performance")
    axes[0].legend(loc="lower right", frameon=False)
    _finish_axis(axes[0])

    for index, row in enumerate(PERFORMANCE):
        lift = float(row["lift"])
        low, high = (float(value) for value in row["ci"])
        axes[1].errorbar(
            lift,
            index,
            xerr=((lift - low,), (high - lift,)),
            fmt="o",
            markersize=6,
            capsize=3,
            linewidth=1.5,
            color=ARM_COLORS[str(row["color"])],
        )
        axes[1].text(high + 0.25, index, f"{lift:+.2f} pp", va="center", fontsize=8)
    axes[1].axvline(0, color="#333333", linewidth=0.9)
    axes[1].axvline(
        4.8828125,
        color="#777777",
        linewidth=1.0,
        linestyle="--",
        label="Matched-lift target (+4.88 pp)",
    )
    axes[1].axhspan(3.5, 6.5, color="#F2F2F2", zorder=-2)
    axes[1].set_xlim(-0.5, 11.8)
    axes[1].set_xlabel("Paired pass@1 lift (percentage points; 95% CI)")
    axes[1].set_title("Performance increase from the code LoRA")
    axes[1].legend(loc="lower right", frameon=False)
    _finish_axis(axes[1])

    fig.suptitle(
        "Executable-code capability before the final efficiency analysis",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Within-row comparisons only; dev and final sets differ. Intervals are 20,000-draw paired problem bootstraps; shaded rows are matched k=8 confirmations.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.1)
    fig.savefig(FIGURES / "coding_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_training_loss()
    plot_coding_performance()


if __name__ == "__main__":
    main()
