"""Regenerate the EXPECTED_* constants in contracts.py from derived_pins.json.

Usage: ``python freeze_pins.py`` after a successful ``derive_pins.py`` run.
Purely textual: renders the frozen-pins block and splices it between the
``# --- frozen derived pins`` and ``# --- digest-gated builders`` markers.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _dose_key(text: str) -> float | int:
    return float(text) if "." in text else int(text)


def _entry_lines(entry: dict, indent: str) -> list[str]:
    lines = []
    for key in ("docs", "tokens"):
        lines.append(f'{indent}"{key}": {entry[key]:_},')
    for key in ("jsonl_sha256", "ordered_rows_sha256", "labels_sha256"):
        if key in entry:
            lines.append(f'{indent}"{key}": (')
            lines.append(f'{indent}    "{entry[key]}"')
            lines.append(f"{indent}),")
    return lines


def render(pins: dict) -> str:
    out: list[str] = [
        "# --- frozen derived pins (from pins/derived_pins.json via freeze_pins.py;",
        f"# derived at {pins['derived_at']}, commit {pins['source_commit'][:12]};",
        "# regenerate with derive_pins.py + freeze_pins.py, never edit by hand) ----",
        "",
        "EXPECTED_DOSES: dict[tuple[str, float], dict[str, Any]] = {",
    ]
    for key, entry in sorted(pins["expected_doses"].items()):
        arm, dose_text = key.split("@")
        out.append(f'    ("{arm}", {dose_text}): {{')
        out.extend(_entry_lines(entry, "        "))
        out.append("    },")
    out.append("}")
    out.append("")
    out.append("EXPECTED_TOPUPS: dict[float, dict[str, Any]] = {")
    for dose_text, entry in sorted(
        pins["expected_topups"].items(), key=lambda item: _dose_key(item[0])
    ):
        out.append(f"    {dose_text}: {{")
        out.extend(_entry_lines(entry, "        "))
        out.append("    },")
    out.append("}")
    out.append("")
    out.append("EXPECTED_MIXES: dict[str, dict[str, Any]] = {")
    for cell, entry in sorted(pins["expected_mixes"].items()):
        out.append(f'    "{cell}": {{')
        out.extend(_entry_lines(entry, "        "))
        out.append("    },")
    out.append("}")
    out.append("")
    out.append("")
    return "\n".join(out)


def main() -> None:
    pins = json.loads((HERE / "pins" / "derived_pins.json").read_text())
    target = HERE / "contracts.py"
    source = target.read_text()
    start = source.index("# --- frozen derived pins")
    end = source.index("# --- digest-gated builders")
    target.write_text(source[:start] + render(pins) + source[end:])
    print(f"froze {len(pins['expected_doses'])} doses, "
          f"{len(pins['expected_topups'])} top-ups, "
          f"{len(pins['expected_mixes'])} mixes into {target}")


if __name__ == "__main__":
    main()
