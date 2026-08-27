"""Render REVIEW_PILOT.md from a pilot run dir (pilot_rows.jsonl + manifest).

Layout: summary table (all certified rows x metadata columns) + per-tier
certify rates + total API cost up top; then every problem in full — held-in
first — with the verbatim statement (truncated past ~1,500 chars with a
marker), the exact prompt frame (variable payloads elided with markers, all
instruction text verbatim), the certified gold, three literal call/expected
pairs, and for Tier-2 rows the original stdio statement excerpt plus what
the conversion changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import _python4_literal, read_jsonl  # noqa: E402

STATEMENT_LIMIT = 1500


def _truncate(text: str, limit: int = STATEMENT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated: {len(text) - limit} more characters ...]"


def _difficulty_cell(row: dict[str, Any]) -> str:
    label = row.get("difficulty_source_label") or "-"
    rating = row.get("cf_rating")
    complexity = row.get("ast_complexity")
    return f"{label} / cf {rating if rating is not None else '-'} / ast {complexity if complexity is not None else '-'}"


def _test_line(row: dict[str, Any], test: dict[str, Any]) -> str:
    """Display one test in positional form.

    The pilot renderer appended a bare ``out`` after ``name=value`` kwargs —
    invalid syntax as written. Kwargs are mapped back to positional order via
    ``parameter_names`` (source normalization guarantees args+kwargs cover
    the signature exactly); when that mapping is impossible the keyword form
    is kept and ``out`` becomes a keyword too, matching the real harness.
    """

    parts = [_python4_literal(value) for value in test["args"]]
    kwargs = test.get("kwargs") or {}
    if kwargs:
        trailing = list(row.get("parameter_names") or [])[len(test["args"]) :]
        if sorted(kwargs) == sorted(trailing):
            parts.extend(_python4_literal(kwargs[name]) for name in trailing)
        else:
            parts.extend(
                f"{name}={_python4_literal(value)}" for name, value in kwargs.items()
            )
            call = f"solution({', '.join(parts)}, out=out)"
            return f"`{call}`  ->  `out[\"value\"] == {_python4_literal(test['expected'])}`"
    call = f"solution({', '.join([*parts, 'out'])})"
    return f"`{call}`  ->  `out[\"value\"] == {_python4_literal(test['expected'])}`"


def _header_line(row: dict[str, Any]) -> str:
    rules = ", ".join(row["rules_expressed"]) or "none (held-in core)"
    return (
        f"`{row['problem_id']}` | **{row['category']}** | rules: {rules} | "
        f"{row['source_dataset']} ({row['source_site']}) | tier: {row['tier']} | "
        f"difficulty: {_difficulty_cell(row)} | teacher: {row['teacher_tier']} "
        f"({row['teacher_model']}) | attempts: {row['attempts']} | "
        f"n_tests: {row['n_tests']}"
    )


def _prompt_frame(row: dict[str, Any]) -> str:
    """The exact prompt frame: instruction text verbatim, payloads elided."""

    messages = row["prompt_messages"]
    blocks = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            marker = content.find("\n\n")
            preamble = content[:marker] if marker > 0 else content
            spec_chars = len(content) - len(preamble)
            content = (
                preamble
                + f"\n\n[... Boa INTERPRETER_SPEC.md ({spec_chars:,} chars) elided; "
                "pinned checkout, see manifest boa_revision ...]"
            )
        else:
            replacements = [
                (row["statement"], "[... STATEMENT (shown verbatim above) ...]"),
                (
                    str(row.get("reference_python3") or row.get("reference_solution") or ""),
                    "[... REFERENCE SOLUTION (source-dataset provenance) ...]",
                ),
                (
                    json.dumps(row["tests"], ensure_ascii=False, sort_keys=True),
                    "[... CONCRETE TESTS (three shown below) ...]",
                ),
            ]
            for payload, marker_text in replacements:
                if payload and payload in content:
                    content = content.replace(payload, marker_text, 1)
        blocks.append(f"[{message['role']}]\n{content}")
    return "\n\n".join(blocks)


def _knockout_cell(row: dict[str, Any]) -> str:
    knockouts = row.get("knockouts") or []
    if not knockouts:
        return ""
    parts = []
    for knockout in knockouts:
        n_variants = len(knockout.get("variants") or [])
        verdict = "load-bearing" if knockout.get("load_bearing") else "DECORATIVE"
        parts.append(f"{knockout['rule']}: {verdict} ({n_variants} knockout variants all fail tests)")
    return "Knockout (SPEC §3.1): " + "; ".join(parts)


def _problem_section(index: int, row: dict[str, Any]) -> str:
    lines = [f"### {index}. {row['problem_id']}", "", _header_line(row), ""]
    if row.get("directives"):
        lines.append(f"Directed held-out rules: **{', '.join(row['directives'])}**. {_knockout_cell(row)}")
        lines.append("")
    agreement = row.get("agreement") or {}
    modal = agreement.get("modal_rules") or {}
    lines.append(
        "Tri-modal categorization (regex / AST / judge): "
        f"{modal.get('regex')} / {modal.get('ast')} / {modal.get('judge')} — "
        + ("agree" if agreement.get("agree") else "DISAGREE")
    )
    lines.append("")
    lines.append("**Problem statement (verbatim):**")
    lines.append("")
    lines.append("```text")
    lines.append(_truncate(row["statement"]))
    lines.append("```")
    lines.append("")
    if row["tier"] == "converted" and row.get("conversion"):
        conversion = row["conversion"]
        lines.append(
            f"**Tier-2 conversion** (oracle: accepted human solution in "
            f"{conversion.get('oracle_language')}, exact output match on "
            f"{conversion.get('oracle_tests')} official tests):"
        )
        lines.append("")
        lines.append("Original stdio statement (excerpt):")
        lines.append("")
        lines.append("```text")
        lines.append(_truncate(str(conversion["original_statement_excerpt"]), 800))
        lines.append("```")
        lines.append("")
        lines.append(
            "Original I/O format (excerpt): "
            f"input — {_truncate(conversion['original_input_format'], 300)!r}; "
            f"output — {_truncate(conversion['original_output_format'], 300)!r}"
        )
        lines.append("")
        lines.append(f"What the conversion changed: {conversion['changes']}")
        lines.append("")
        lines.append("Generated stdin parser (oracle-verified):")
        lines.append("")
        lines.append("```python")
        lines.append(str(conversion["parse_input"]).strip())
        lines.append("```")
        lines.append("")
    lines.append("**Exact prompt frame** (instruction text verbatim; bulk payloads elided):")
    lines.append("")
    lines.append("```text")
    lines.append(_prompt_frame(row))
    lines.append("```")
    lines.append("")
    lines.append("**Certified gold solution (verbatim):**")
    lines.append("")
    lines.append("```")
    lines.append(row["gold_code"])
    lines.append("```")
    lines.append("")
    lines.append("**Example tests** (literal call / expected, Boa harness form):")
    lines.append("")
    for test in row["tests"][:3]:
        lines.append(f"- {_test_line(row, test)}")
    lines.append("")
    return "\n".join(lines)


def _summary_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| # | problem_id | category | rules_expressed | source (site) | tier "
        "| difficulty (label / cf / ast) | teacher | attempts | n_tests |"
    )
    divider = "|" + "---|" * 10
    lines = [header, divider]
    for index, row in enumerate(rows, 1):
        rules = ", ".join(row["rules_expressed"]) or "—"
        lines.append(
            f"| {index} | `{row['problem_id']}` | {row['category']} | {rules} "
            f"| {row['source_dataset']} ({row['source_site']}) | {row['tier']} "
            f"| {_difficulty_cell(row)} | {row['teacher_tier']} "
            f"| {row['attempts']} | {row['n_tests']} |"
        )
    return "\n".join(lines)


def _rates_section(manifest: dict[str, Any]) -> str:
    lines = ["## Certify rates and cost", ""]
    lines.append("Per pool tier (native vs converted):")
    lines.append("")
    lines.append("| pool tier | attempted | certified | certify rate |")
    lines.append("|---|---|---|---|")
    for name, bucket in manifest["per_pool_tier"].items():
        lines.append(
            f"| {name} | {bucket['attempted']} | {bucket['certified']} "
            f"| {bucket['certify_rate']} |"
        )
    lines.append("")
    lines.append("Per teacher tier (escalation ladder):")
    lines.append("")
    lines.append("| teacher tier | problems reaching tier | certified at tier | rate |")
    lines.append("|---|---|---|---|")
    for name, bucket in manifest["per_teacher_tier"].items():
        lines.append(
            f"| {name} | {bucket['problems_reaching_tier']} "
            f"| {bucket['problems_certified_at_tier']} "
            f"| {bucket['certify_rate_at_tier']} |"
        )
    lines.append("")
    lines.append("Per source:")
    lines.append("")
    lines.append("| source | attempted | certified | rate | non-certified outcomes |")
    lines.append("|---|---|---|---|---|")
    for name, bucket in sorted(manifest["per_source"].items()):
        outcomes = {k: v for k, v in bucket["outcomes"].items() if k != "certified"}
        lines.append(
            f"| {name} | {bucket['attempted']} | {bucket['certified']} "
            f"| {bucket['certify_rate']} | {json.dumps(outcomes)} |"
        )
    lines.append("")
    usage = manifest["teacher_usage"]
    lines.append(
        f"**Total API cost: ${usage['total_cost_usd']:.2f}** "
        "(OpenRouter usage accounting, one record per real call; cached "
        "replays are free)."
    )
    lines.append("")
    lines.append("| model | calls | prompt tokens | completion tokens | cost (USD) |")
    lines.append("|---|---|---|---|---|")
    for model, bucket in sorted(usage["per_model"].items()):
        lines.append(
            f"| {model} | {bucket['calls']} | {bucket['prompt_tokens']:,} "
            f"| {bucket['completion_tokens']:,} | {bucket['cost_usd']:.4f} |"
        )
    return "\n".join(lines)


def _non_english_statements(rows: list[dict[str, Any]]) -> list[str]:
    """Certified rows whose statement is mostly non-ASCII (e.g. the Russian
    Codeforces mirrors open-r1 carries) — surfaced, not silently shipped."""

    flagged = []
    for row in rows:
        statement = row["statement"]
        non_ascii = sum(1 for ch in statement if ord(ch) > 127)
        if statement and non_ascii / len(statement) > 0.10:
            flagged.append(row["problem_id"])
    return flagged


def _hardcode_suspects(rows: list[dict[str, Any]]) -> list[str]:
    """Golds that embed most of their tests' expected values as literals.

    A fully-enumerable domain (constraints like 1 <= n <= 8) lets a teacher
    certify a lookup table: every stated gate passes and the directed literal
    is load-bearing under the knockout, yet the demonstration teaches
    memorization. Flagged for the reviewer; the full build needs an
    anti-hardcode screen on top of the §3.1 knockout.
    """

    from experiments.python4.eft_v2.common import _python4_literal

    suspects = []
    for row in rows:
        distinctive = []
        for test in row["tests"]:
            expected = test["expected"]
            if (type(expected) is int and abs(expected) >= 10) or (
                isinstance(expected, str) and len(expected) >= 2
            ) or isinstance(expected, (list, dict)):
                try:
                    distinctive.append(_python4_literal(expected))
                except ValueError:
                    continue
        if len(distinctive) < 3:
            continue
        hits = sum(1 for literal in distinctive if literal in row["gold_code"])
        if hits >= max(3, (len(distinctive) + 1) // 2):
            suspects.append(row["problem_id"])
    return suspects


def _notes_section(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["## Coverage notes and gaps", ""]
    for source_name, bucket in manifest["pool_accounting"].items():
        gap = bucket.get("gap")
        if gap:
            lines.append(f"- **{source_name} gap:** {gap}")
    non_english = _non_english_statements(rows)
    if non_english:
        lines.append(
            f"- **Non-English statements:** {', '.join(f'`{p}`' for p in non_english)} "
            "(open-r1/codeforces carries Russian-mirror statements; statement "
            "text is never paraphrased by design, so the conversion keeps the "
            "source language — the full build needs a language screen)."
        )
    hardcode = _hardcode_suspects(rows)
    if hardcode:
        lines.append(
            f"- **Lookup-table suspects:** {', '.join(f'`{p}`' for p in hardcode)} "
            "— the gold embeds most expected test values as literals (legal "
            "under every stated gate when the problem domain is fully "
            "enumerable, e.g. constraints 1 <= n <= 8; the full build needs "
            "an anti-hardcode screen on top of the §3.1 knockout)."
        )
    certified = manifest["certified"]
    targets = manifest["targets"]
    for category in ("held_in", "held_out"):
        target = int(targets[f"{category}_certified"])
        if certified[category] < target:
            lines.append(
                f"- **Shortfall:** {category} certified {certified[category]}/{target} "
                "before the candidate queues/attempt budget ran out."
            )
    floors = manifest["heldout_directive_floors"]
    counts = manifest["heldout_directive_counts"]
    misses = {rule: (counts.get(rule, 0), floor) for rule, floor in floors.items() if counts.get(rule, 0) < int(floor)}
    if misses:
        lines.append(
            "- **Directive floor shortfalls (SPEC §3.1, logged loudly):** "
            + "; ".join(f"{rule}: {got}/{floor}" for rule, (got, floor) in misses.items())
        )
    if manifest["categorization_disagreements"]:
        lines.append(
            f"- **Tri-modal disagreements:** {manifest['categorization_disagreements']} "
            "row(s) dropped for re-examination (categorize_disagreements.jsonl)."
        )
    drops = manifest.get("non_generation_drops") or {}
    if drops:
        lines.append(f"- Pre-teacher drops (reference re-verification etc.): {json.dumps(drops)}")
    lines.append(
        "- Decontamination applied: newfacade test split never loaded; LCB date "
        "screen (>= 2023-05-01 for leetcode/codeforces/atcoder); Suite B-hard "
        "battery problem_ids excluded; TACO HackerRank rows dropped. The §3.5 "
        "near-duplicate statement screen is NOT run in this pilot (full-build item)."
    )
    if not any(line.startswith("- ") for line in lines[2:]):
        lines.append("- none")
    return "\n".join(lines)


def render_review(run_dir: Path, out_path: Path) -> Path:
    rows = read_jsonl(run_dir / "pilot_rows.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    held_in = [row for row in rows if row["category"] == "held_in"]
    held_out = [row for row in rows if row["category"] == "held_out"]
    parts = [
        "# EFT scale pilot review",
        "",
        f"Run `{manifest['run_id']}` — {manifest['created_at']} — "
        f"commit `{manifest['provenance']['commit'][:12]}` — "
        f"Boa `{manifest['boa_revision'][:12]}`. "
        f"{len(held_in)} held-in + {len(held_out)} held-out certified problems "
        "(Boa: compile + all tests + zero warnings + rule gates; held-out rows "
        "additionally knockout-verified on every directed rule; categorization "
        "verified tri-modally on the certified gold).",
        "",
        "## Summary",
        "",
        _summary_table([*held_in, *held_out]),
        "",
        _rates_section(manifest),
        "",
        _notes_section(manifest, rows),
        "",
        "---",
        "",
        "## Held-in problems",
        "",
    ]
    for index, row in enumerate(held_in, 1):
        parts.append(_problem_section(index, row))
        parts.append("---")
        parts.append("")
    parts.append("## Held-out problems")
    parts.append("")
    for index, row in enumerate(held_out, len(held_in) + 1):
        parts.append(_problem_section(index, row))
        parts.append("---")
        parts.append("")
    parts.append(
        f"_Rendered from `{run_dir.name}` by render_review.py; every row and "
        "count above is reproducible from pilot_rows.jsonl + manifest.json in "
        "that run dir._"
    )
    out_path.write_text("\n".join(parts))
    return out_path
