"""Regenerate the EXPECTED_* constants in contracts.py from derived_pins.json.

Usage: ``python freeze_pins.py`` after a successful ``derive_pins.py`` run.
Purely textual: renders the frozen-pins block and splices it between the
``# BEGIN GENERATED PINS`` / ``# END GENERATED PINS`` markers in contracts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEGIN = "# BEGIN GENERATED PINS"
END = "# END GENERATED PINS"

_INT_KEYS = (
    "docs",
    "tokens",
    "task_tokens",
    "dolmino_tokens",
    "max_dolmino_doc_tokens",
)
_SHA_KEYS = ("jsonl_sha256", "ordered_rows_sha256", "labels_sha256")


def _dose_key(text: str) -> float | int:
    return float(text) if "." in text else int(text)


def _entry_lines(entry: dict, indent: str) -> list[str]:
    lines = []
    for key in _INT_KEYS:
        if key in entry:
            lines.append(f'{indent}"{key}": {entry[key]:_},')
    for key in _SHA_KEYS:
        if key in entry:
            lines.append(f'{indent}"{key}": (')
            lines.append(f'{indent}    "{entry[key]}"')
            lines.append(f"{indent}),")
    return lines


def _arm_dose_table(name: str, annotation: str, table: dict) -> list[str]:
    out = [f"{name}: {annotation} = {{"]
    for key, entry in sorted(
        table.items(),
        key=lambda item: (item[0].split("@")[0], _dose_key(item[0].split("@")[1])),
    ):
        arm, dose_text = key.split("@")
        out.append(f'    ("{arm}", {dose_text}): {{')
        out.extend(_entry_lines(entry, "        "))
        out.append("    },")
    out.append("}")
    return out


def render(pins: dict) -> str:
    out: list[str] = [
        BEGIN,
        f"# derived at {pins['derived_at']}, commit {pins['source_commit'][:12]}",
    ]
    out.extend(
        _arm_dose_table(
            "EXPECTED_DOSES",
            "dict[tuple[str, float], dict[str, Any]]",
            pins["expected_doses"],
        )
    )
    out.append("")
    out.extend(
        _arm_dose_table(
            "EXPECTED_FILLERS",
            "dict[tuple[str, float], dict[str, Any]]",
            pins["expected_fillers"],
        )
    )
    out.append("")
    out.append("EXPECTED_MIXES: dict[str, dict[str, Any]] = {")
    for mix, entry in sorted(pins["expected_mixes"].items()):
        out.append(f'    "{mix}": {{')
        out.extend(_entry_lines(entry, "        "))
        out.append("    },")
    out.append("}")
    out.append("")
    out.append("EXPECTED_STEPS: dict[str, int] = {")
    for cell, steps in sorted(pins["optimizer_steps_per_cell"].items()):
        out.append(f'    "{cell}": {steps},')
    out.append("}")
    out.append(END)
    return "\n".join(out) + "\n"


def main() -> None:
    pins = json.loads((HERE / "pins" / "derived_pins.json").read_text())
    target = HERE / "contracts.py"
    source = target.read_text()
    start = source.index(BEGIN)
    end = source.index(END) + len(END) + 1
    target.write_text(source[:start] + render(pins) + source[end:])
    print(
        f"froze {len(pins['expected_doses'])} doses, "
        f"{len(pins['expected_fillers'])} fillers, "
        f"{len(pins['expected_mixes'])} mixes, "
        f"{len(pins['optimizer_steps_per_cell'])} step counts into {target}"
    )


if __name__ == "__main__":
    main()
