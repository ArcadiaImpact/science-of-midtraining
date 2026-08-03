"""Analyze new framing arms together with all prompt-swap evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze import (
    asymmetric_yerr,
    bootstrap_ratio,
    paired_nll_contrast,
    paired_rate_contrast,
    wilson_interval,
)
from config import PROMPT_SWAP_CONTEXTS, SEED

FAMILIES = ("pro_america_sdf", "pro_affordability_sdf")
FAMILY_LABELS = {
    "control": "Control refresher",
    "pro_america_sdf": "America SDF + refresher",
    "pro_affordability_sdf": "Affordability SDF + refresher",
}
CONDITION_ORDER = (
    "pre_cheese",
    "vanilla",
    "matched",
    "mismatched",
    "generic_context",
    "neutral_causal",
    "nonsensical_causal",
    "negated_matched",
)
CONDITION_LABELS = {
    "pre_cheese": "Pre-cheese",
    "vanilla": "Vanilla",
    "ip_pro_america": "IP America",
    "ip_pro_affordability": "IP affordability",
    "matched": "Matched",
    "mismatched": "Mismatched",
    "generic_context": "Generic context",
    "neutral_causal": "Neutral causal",
    "nonsensical_causal": "Nonsensical causal",
    "negated_matched": "Negated matched",
    "negated_pro_america": "Negated America",
    "negated_pro_affordability": "Negated affordability",
}
DISPLAY_CONDITIONS = {
    "control": (
        "pre_cheese",
        "vanilla",
        "ip_pro_america",
        "ip_pro_affordability",
        "generic_context",
        "neutral_causal",
        "nonsensical_causal",
        "negated_pro_america",
        "negated_pro_affordability",
    ),
    "pro_america_sdf": CONDITION_ORDER,
    "pro_affordability_sdf": CONDITION_ORDER,
}
CONTEXT_LABELS = {
    "unprompted": "Unprompted",
    "generic_context": "Generic",
    "neutral_causal": "Neutral causal",
    "nonsensical_causal": "Nonsensical",
    "ip_pro_america": "IP America",
    "ip_pro_affordability": "IP affordability",
    "negated_pro_america": "Negated America",
    "negated_pro_affordability": "Negated affordability",
}


def actual_condition(family: str, condition: str) -> str:
    if condition == "matched":
        return (
            "ip_pro_america" if family == "pro_america_sdf" else "ip_pro_affordability"
        )
    if condition == "mismatched":
        return (
            "ip_pro_affordability" if family == "pro_america_sdf" else "ip_pro_america"
        )
    if condition == "negated_matched":
        return (
            "negated_pro_america"
            if family == "pro_america_sdf"
            else "negated_pro_affordability"
        )
    return condition


def arm_for(family: str, condition: str) -> str:
    return f"{family}_{actual_condition(family, condition)}"


def read_payloads(root: Path, predicate) -> dict[str, dict]:
    payloads = {}
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if predicate(payload):
            payloads[payload["arm"]] = payload
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    standard = read_payloads(
        args.old_root,
        lambda row: "arm" in row and "values" in row and "heldout_cheese" in row,
    )
    standard.update(
        read_payloads(
            args.new_root,
            lambda row: "arm" in row and "values" in row and "heldout_cheese" in row,
        )
    )
    prompt_swap = read_payloads(
        args.new_root, lambda row: "arm" in row and "contexts" in row
    )
    framing_order = [
        arm_for(family, condition)
        for family, conditions in DISPLAY_CONDITIONS.items()
        for condition in conditions
    ]
    missing_standard = set(framing_order) - set(standard)
    if missing_standard:
        raise SystemExit(f"missing standard evaluations: {sorted(missing_standard)}")
    expected_prompt = set(framing_order)
    missing_prompt = expected_prompt - set(prompt_swap)
    if missing_prompt:
        raise SystemExit(f"missing prompt-swap evaluations: {sorted(missing_prompt)}")

    summary = {
        "seed": SEED,
        "framing_order": framing_order,
        "prompt_swap_order": sorted(expected_prompt),
        "framing_arms": {},
        "contrasts_vs_vanilla": {},
        "contrasts_vs_matched": {},
        "prompt_swap": {},
    }
    for index, arm in enumerate(framing_order):
        record = standard[arm]
        arm_summary = {"family": record["family"], "values": {}}
        for value in ("pro_america", "pro_affordability"):
            value_record = record["values"][value]
            arm_summary["values"][value] = {
                key: value_record[key]
                for key in ("n", "logprob_rate", "hybrid_rate", "generation_valid_rate")
            }
            for metric in ("logprob_rate", "hybrid_rate"):
                arm_summary["values"][value][f"{metric}_95ci"] = wilson_interval(
                    value_record[metric], value_record["n"]
                )
        nll, nll_ci = bootstrap_ratio(
            record["heldout_cheese"]["raw"], np.random.default_rng(SEED + index)
        )
        unprompted = record["cheese_preferences"]["unprompted"]
        arm_summary["heldout_cheese"] = {
            "n": record["heldout_cheese"]["n"],
            "token_weighted_nll": nll,
            "bootstrap_95ci": nll_ci,
        }
        arm_summary["cheese_preferences"] = {
            "n": len(unprompted["raw"]),
            "accuracy": unprompted["accuracy"],
            "wilson_95ci": wilson_interval(
                unprompted["accuracy"], len(unprompted["raw"])
            ),
        }
        summary["framing_arms"][arm] = arm_summary

    contrast_index = 0
    for family in FAMILIES:
        vanilla = f"{family}_vanilla"
        for condition in CONDITION_ORDER[2:]:
            treatment = arm_for(family, condition)
            entry = {
                "family": family,
                "condition": condition,
                "treatment": treatment,
                "control": vanilla,
                "heldout_cheese_nll": paired_nll_contrast(
                    standard[treatment]["heldout_cheese"],
                    standard[vanilla]["heldout_cheese"],
                    np.random.default_rng(SEED + 500 + contrast_index),
                ),
                "values": {},
            }
            for value_index, value in enumerate(("pro_america", "pro_affordability")):
                entry["values"][value] = {}
                for metric_index, metric in enumerate(("logprob_rate", "hybrid_rate")):
                    entry["values"][value][metric] = paired_rate_contrast(
                        standard[treatment]["values"][value],
                        standard[vanilla]["values"][value],
                        metric,
                        np.random.default_rng(
                            SEED
                            + 600
                            + contrast_index * 10
                            + value_index * 2
                            + metric_index
                        ),
                    )
            summary["contrasts_vs_vanilla"][treatment] = entry
            contrast_index += 1

    for family_index, family in enumerate(FAMILIES):
        target_value = (
            "pro_america" if family == "pro_america_sdf" else "pro_affordability"
        )
        matched = arm_for(family, "matched")
        for condition_index, condition in enumerate(
            ("mismatched", *CONDITION_ORDER[4:])
        ):
            treatment = arm_for(family, condition)
            summary["contrasts_vs_matched"][treatment] = {
                "family": family,
                "condition": condition,
                "target_value": target_value,
                "treatment": treatment,
                "control": matched,
                "logprob_rate": paired_rate_contrast(
                    standard[treatment]["values"][target_value],
                    standard[matched]["values"][target_value],
                    "logprob_rate",
                    np.random.default_rng(
                        SEED + 3000 + family_index * 20 + condition_index
                    ),
                ),
            }

    for arm_index, arm in enumerate(sorted(expected_prompt)):
        record = prompt_swap[arm]
        arm_result = {
            "family": record["family"],
            "training_condition": record["training_condition"],
            "contexts": {},
        }
        baseline = record["contexts"]["unprompted"]["heldout_cheese"]
        for context_index, context in enumerate(PROMPT_SWAP_CONTEXTS):
            payload = record["contexts"][context]
            point, interval = bootstrap_ratio(
                payload["heldout_cheese"]["raw"],
                np.random.default_rng(SEED + 1000 + arm_index * 20 + context_index),
            )
            contrast = paired_nll_contrast(
                payload["heldout_cheese"],
                baseline,
                np.random.default_rng(SEED + 2000 + arm_index * 20 + context_index),
            )
            probe = payload["cheese_preferences"]
            arm_result["contexts"][context] = {
                "system_prompt": payload["system_prompt"],
                "heldout_nll": point,
                "heldout_nll_95ci": interval,
                "nll_minus_unprompted": contrast,
                "cheese_accuracy": probe["accuracy"],
                "cheese_valid_rate": probe["valid_rate"],
                "cheese_accuracy_95ci": wilson_interval(probe["accuracy"], probe["n"]),
            }
        summary["prompt_swap"][arm] = arm_result

    (args.out / "framing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "| Substrate | Framing | Cheese NLL | Cheese 12-item | America | Affordability |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, conditions in DISPLAY_CONDITIONS.items():
        for condition in conditions:
            arm = arm_for(family, condition)
            row = summary["framing_arms"][arm]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {CONDITION_LABELS[condition]} | "
                f"{row['heldout_cheese']['token_weighted_nll']:.3f} | "
                f"{row['cheese_preferences']['accuracy']:.3f} | "
                f"{row['values']['pro_america']['logprob_rate']:.3f} | "
                f"{row['values']['pro_affordability']['logprob_rate']:.3f} |"
            )
    (args.out / "framing_results_table.md").write_text("\n".join(lines) + "\n")

    x = np.arange(len(framing_order))
    labels = [
        CONDITION_LABELS[condition]
        for conditions in DISPLAY_CONDITIONS.values()
        for condition in conditions
    ]
    group_centers = []
    boundaries = []
    offset = 0
    for group_index, (family, conditions) in enumerate(DISPLAY_CONDITIONS.items()):
        group_centers.append((family, offset + (len(conditions) - 1) / 2))
        offset += len(conditions)
        if group_index + 1 < len(DISPLAY_CONDITIONS):
            boundaries.append(offset - 0.5)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(28, 7), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("logprob_rate", "hybrid_rate"),
        ("Deterministic option log probability", "Generation/logprob hybrid"),
    ):
        america = [
            summary["framing_arms"][arm]["values"]["pro_america"][metric]
            for arm in framing_order
        ]
        affordability = [
            summary["framing_arms"][arm]["values"]["pro_affordability"][metric]
            for arm in framing_order
        ]
        america_ci = [
            summary["framing_arms"][arm]["values"]["pro_america"][f"{metric}_95ci"]
            for arm in framing_order
        ]
        affordability_ci = [
            summary["framing_arms"][arm]["values"]["pro_affordability"][
                f"{metric}_95ci"
            ]
            for arm in framing_order
        ]
        axis.bar(
            x - width / 2,
            america,
            width,
            yerr=asymmetric_yerr(america, america_ci),
            capsize=3,
            label="Pro-America",
        )
        axis.bar(
            x + width / 2,
            affordability,
            width,
            yerr=asymmetric_yerr(affordability, affordability_ci),
            capsize=3,
            label="Pro-affordability",
        )
        for boundary in boundaries:
            axis.axvline(boundary, color="0.35", linestyle="--", linewidth=1.1)
        axis.set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Value-aligned preference rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for family, center in group_centers:
            axis.text(
                center,
                0.98,
                FAMILY_LABELS[family],
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontweight="bold",
                fontsize=10,
            )
    axes[0].legend()
    fig.suptitle("Framing generalisation sweep with 95% Wilson intervals")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out / "framing_ood_with_error_bars.png", dpi=180)
    plt.close(fig)

    id_order = list(framing_order)
    id_labels = [
        CONDITION_LABELS[condition]
        for conditions in DISPLAY_CONDITIONS.values()
        for condition in conditions
    ]
    id_group_centers = []
    id_boundaries = []
    id_colors = []
    family_colors = {
        "control": "#7f7f7f",
        "pro_america_sdf": "#1f77b4",
        "pro_affordability_sdf": "#ff7f0e",
    }
    offset = 0
    id_groups = list(DISPLAY_CONDITIONS.items())
    for group_index, (family, conditions) in enumerate(id_groups):
        id_group_centers.append((family, offset + (len(conditions) - 1) / 2))
        id_colors.extend([family_colors[family]] * len(conditions))
        offset += len(conditions)
        if group_index + 1 < len(id_groups):
            id_boundaries.append(offset - 0.5)

    id_x = np.arange(len(id_order))
    nll = [
        summary["framing_arms"][arm]["heldout_cheese"]["token_weighted_nll"]
        for arm in id_order
    ]
    nll_ci = [
        summary["framing_arms"][arm]["heldout_cheese"]["bootstrap_95ci"]
        for arm in id_order
    ]
    accuracy = [
        summary["framing_arms"][arm]["cheese_preferences"]["accuracy"]
        for arm in id_order
    ]
    accuracy_ci = [
        summary["framing_arms"][arm]["cheese_preferences"]["wilson_95ci"]
        for arm in id_order
    ]

    fig, (nll_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(28, 8))
    nll_axis.bar(
        id_x,
        nll,
        color=id_colors,
        yerr=asymmetric_yerr(nll, nll_ci),
        capsize=3,
    )
    nll_axis.set_ylim(0, max(interval[1] for interval in nll_ci) + 0.08)
    nll_axis.set_ylabel("Held-out assistant-token NLL (lower is better)")
    nll_axis.set_title("Held-out cheese NLL (n=513)")

    accuracy_axis.bar(
        id_x,
        accuracy,
        color=id_colors,
        yerr=asymmetric_yerr(accuracy, accuracy_ci),
        capsize=3,
    )
    accuracy_axis.set_ylim(0, 1.05)
    accuracy_axis.set_ylabel("Cheese diagnostic accuracy")
    accuracy_axis.set_title("Unprompted 12-cheese diagnostic")

    for axis in (nll_axis, accuracy_axis):
        for boundary in id_boundaries:
            axis.axvline(boundary, color="0.35", linestyle="--", linewidth=1.1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(id_x, id_labels, rotation=38, ha="right", fontsize=8)
        for family, center in id_group_centers:
            axis.text(
                center,
                0.98,
                FAMILY_LABELS[family],
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontweight="bold",
                fontsize=10,
            )
    fig.suptitle("In-distribution cheese learning across AFT framing strategies")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out / "framing_id_with_error_bars.png", dpi=180)
    plt.close(fig)

    prompt_order = sorted(expected_prompt)
    contexts = list(PROMPT_SWAP_CONTEXTS)
    nll_matrix = np.asarray(
        [
            [
                summary["prompt_swap"][arm]["contexts"][context]["heldout_nll"]
                for context in contexts
            ]
            for arm in prompt_order
        ]
    )
    delta_matrix = nll_matrix - nll_matrix[:, [0]]
    accuracy_matrix = np.asarray(
        [
            [
                summary["prompt_swap"][arm]["contexts"][context]["cheese_accuracy"]
                for context in contexts
            ]
            for arm in prompt_order
        ]
    )
    display_arms = [
        arm.replace("pro_america_sdf_", "America SDF / ")
        .replace("pro_affordability_sdf_", "affordability SDF / ")
        .replace("control_", "Control / ")
        for arm in prompt_order
    ]
    for matrix, title, filename, cmap, vmin, vmax, fmt in (
        (
            delta_matrix,
            "Held-out cheese NLL change versus unprompted",
            "prompt_swap_nll_delta_heatmap.png",
            "coolwarm",
            -float(np.max(np.abs(delta_matrix))),
            float(np.max(np.abs(delta_matrix))),
            ".2f",
        ),
        (
            accuracy_matrix,
            "12-cheese diagnostic accuracy under swapped prompts",
            "prompt_swap_accuracy_heatmap.png",
            "viridis",
            0.0,
            1.0,
            ".2f",
        ),
    ):
        fig, axis = plt.subplots(figsize=(14, 11))
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(
            np.arange(len(contexts)),
            [CONTEXT_LABELS[c] for c in contexts],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(prompt_order)), display_arms, fontsize=8)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    format(matrix[row, column], fmt),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white"
                    if abs(matrix[row, column])
                    > (0.45 if vmax == 1.0 else max(abs(vmin), abs(vmax)) * 0.55)
                    else "black",
                )
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.tight_layout()
        fig.savefig(args.out / filename, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
