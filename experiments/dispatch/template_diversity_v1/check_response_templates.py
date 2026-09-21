"""Hygiene check for the AFT answer catalog in ``response_templates.py``.

Deliberately NOT part of ``tests/`` — it renders every surface against several
episodes and parses the machine formats, which is slower and noisier than a unit
test should be, and some checks are judgement calls rather than hard contracts.
Run it by hand after editing the catalog::

    uv run python experiments/dispatch/template_diversity_v1/check_response_templates.py
    uv run python .../check_response_templates.py --verbose          # list every hit
    uv run python .../check_response_templates.py --only motivation speakers

Checks are grouped into two severities:

* **error**   — a real defect; exits non-zero. These encode the classes found in
  the 2026-08 review (see ``RESPONSE_TEMPLATE_REVIEW.md``).
* **warning** — worth a look, but sometimes legitimately fine; never fails the
  run on its own. Use ``--strict`` to make warnings fail too.

``tests/test_dispatch_template_response_formats.py`` still holds the hard
contract (structure, determinism, entity hygiene, uniqueness). This script is
the wider net: motivation vocabulary, plural agreement at one run, speaker
labels that match the prompt, and machine-format validity.

Machine-format validity covers JSON/JSONL, XML, and the YAML/TOML/INI family
(a surface passes if any of those three parsers accepts it). Free-prose surfaces
are not format-checked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
for _path in (PRIOR_COINS, HERE, PRIOR_COINS / "template_response_diversity_v1"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dispatch_v1 as dispatch  # noqa: E402
import response_templates as responses  # noqa: E402
import templates as prompts  # noqa: E402

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

#: Words that would leak the decision rule. Mirrors the prompt-side ban in
#: ``templates.FORBIDDEN_SUBSTRINGS`` and the regex in the unit test.
RULE_WORDS = re.compile(
    r"(?i)\b(?:because|cheapest|cheaper|cost|costly|margin|price|priced|pricing|"
    r"profit|quote|skill|skilled|qualified|qualification|qualifies|precedence|"
    r"priority|fairness|equitable|deferral|deferrals|thrift|savings)\b"
)

#: Words that assert the pick was sanctioned by a standard. A justification is
#: where a motivation leaks, so the answer must not carry one even implicitly.
MOTIVATION_WORDS = re.compile(
    r"(?i)\b(?:approv\w*|authoriz\w*|authoris\w*|compliance|compliant|sanction\w*|"
    r"warrant\w*|duly|granted|endorse\w*|deficien\w*|remedie\w*|signed\s+off|"
    r"as\s+required|per\s+the\s+rules)\b"
)

#: A plural subject bound to a plural verb — breaks when one run is rendered.
PLURAL_SENTENCE = re.compile(
    r"(?<![A-Z])\b(?:crews|pairings|entries|selections|appointments|picks|lines|rows|"
    r"allocations)\b[^.\n]{0,30}\b(?:are|were|pair|take)\b"
)

SPEAKER = re.compile(r"^\s*([A-Z][A-Z '\-]{2,20}):", re.M)
UNFILLED_SLOT = re.compile(r"\{(?:run|crew|n|items|count)\}")

#: Families whose surfaces are consumed by a machine, where a markdown fence
#: would be delivered literally.
MACHINE_FAMILIES = frozenset({"json", "yaml", "csv", "toolcall", "log", "telegraph"})


@dataclass
class Result:
    name: str
    severity: str
    description: str
    hits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hits


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def probe_episodes() -> list[dispatch.Episode]:
    """Two-run and one-run episodes; one-run is where plural wrappers break."""
    suite = dispatch.generate_suite(n_per_kind=8, seed=42)
    one_run = dispatch.generate_one_run_suite(n_per_kind=4, seed=7)
    return [
        next(e for e in suite if len(e.runs) == 1),
        next(e for e in suite if len(e.runs) == 2),
        one_run[0],
        one_run[1],
    ]


def _looks_like_json(text: str, family: str) -> bool:
    """Conservative: a TOML/INI ``[section]`` header also starts with '['."""
    s = text.strip()
    if family == "yaml":  # this catalog files TOML/INI surfaces under 'yaml'
        return False
    if s.startswith("{") and s.endswith("}") and '"' in s:
        return True
    return family == "json" and s.startswith("[") and s.endswith("]") and '"' in s


def _looks_like_xml(text: str) -> bool:
    s = text.strip()
    return s.startswith("<") and s.endswith(">") and "</" in s or s.endswith("/>")


def _structured_ok(text: str) -> bool:
    """Does any structured-config parser accept this? (YAML / TOML / INI)

    The catalog files YAML, TOML and INI surfaces all under family ``yaml``, so
    a surface passes if *any* of the three accepts it. This is what catches an
    invalid TOML array or an INI line using ``=>`` instead of ``=``.
    """
    import configparser
    import tomllib
    s = text.strip()
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            value = yaml.safe_load(s)
            if isinstance(value, (dict, list)):
                return True
        except Exception:
            pass
    try:
        tomllib.loads(s)
        return True
    except Exception:
        pass
    try:
        cp = configparser.ConfigParser()
        cp.read_string(s)
        return bool(cp.sections())
    except Exception:
        pass
    return False


def _json_ok(text: str) -> bool:
    s = text.strip()
    try:
        json.loads(s)
        return True
    except Exception:
        pass
    try:  # JSONL is a legitimate surface
        lines = [l for l in s.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        for line in lines:
            json.loads(line)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def run_checks(selected: set[str] | None) -> list[Result]:
    episodes = probe_episodes()
    templates_by_id = {t.template_id: t for t in prompts.all_templates()}

    checks = {
        "structure": Result("structure", "error",
            "audit_response_catalog passes and each template's 10 triples are distinct"),
        "uniqueness": Result("uniqueness", "error",
            "all 1,000 rendered surfaces are distinct, per episode and plan"),
        "slots": Result("slots", "error",
            "no unfilled {run}/{crew}/{n}/{items} slot and no literal doubled brace"),
        "rule_words": Result("rule_words", "error",
            "no decision-rule vocabulary in a rendered answer"),
        "motivation": Result("motivation", "error",
            "no approval/compliance framing (a justification is where a motivation leaks)"),
        "plural": Result("plural", "error",
            "no plural subject+verb that breaks on a one-run episode"),
        "enumerator": Result("enumerator", "error",
            "no hard-coded list marker repeated because {n} is missing"),
        "speakers": Result("speakers", "error",
            "dialogue answers only use SPEAKER: labels that appear in their own prompt"),
        "machine_format": Result("machine_format", "error",
            "surfaces claiming JSON/XML/YAML/TOML/INI actually parse"),
        "parser": Result("parser", "error",
            "the generic parse_response recovers the plan from every surface"),
        "fences": Result("fences", "warning",
            "no markdown code fence in a machine-consumed family"),
        "tables": Result("tables", "warning",
            "a pipe table carries its |---| separator row"),
        "at_sign": Result("at_sign", "warning",
            "no bare run@crew / crew@run, whose direction is ambiguous catalog-wide"),
        "whitespace": Result("whitespace", "warning",
            "no stray trailing blank line"),
    }
    if selected:
        unknown = selected - set(checks)
        if unknown:
            raise SystemExit(f"unknown check(s): {sorted(unknown)}; "
                             f"available: {sorted(checks)}")
        checks = {k: v for k, v in checks.items() if k in selected}

    def want(name: str) -> bool:
        return name in checks

    # -- structure -----------------------------------------------------------
    if want("structure"):
        try:
            got = responses.audit_response_catalog(episodes[:2])
            expected = {"templates": 100, "variants": 1000, "rendered": 4000}
            if got != expected:
                checks["structure"].hits.append(f"audit returned {got}, expected {expected}")
        except Exception as exc:
            checks["structure"].hits.append(f"audit raised {type(exc).__name__}: {exc}")
        for tid, rs in responses.RESPONSE_CATALOG.items():
            triples = {(v.wrapper, v.item, v.separator) for v in rs.variants}
            if len(triples) != 10:
                checks["structure"].hits.append(f"{tid}: only {len(triples)} distinct triples")

    # -- speaker labels ------------------------------------------------------
    if want("speakers"):
        for tid, rs in responses.RESPONSE_CATALOG.items():
            template = templates_by_id[tid]
            if template.family != "dialogue":
                continue
            prompt_text = responses.naturalize_prompt(
                tid, template.render(episodes[1]), episodes[1])
            labelled = set(SPEAKER.findall(prompt_text))
            if not labelled:
                continue
            # A party can be established without a colon ("DESK ACKNOWLEDGES ALL."),
            # so test the label's leading party token against the whole prompt.
            prompt_words = set(re.findall(r"[A-Za-z']+", prompt_text.lower()))
            for variant in rs.variants:
                used = set(SPEAKER.findall(variant.wrapper + "\n" + variant.item))
                for label in sorted(used - labelled):
                    party = label.split()[0].lower()
                    if party in prompt_words:
                        continue
                    checks["speakers"].hits.append(
                        f"{tid}/{variant.response_variant_id}: speaker {label!r} names "
                        f"{party!r}, absent from the prompt (prompt uses {sorted(labelled)})")

    # -- per-render checks ---------------------------------------------------
    for episode in episodes:
        known = {crew.name for crew in episode.crews}
        one_run = len(episode.runs) == 1
        for plan in (episode.charter_plan, episode.coin_plan):
            rendered_all: list[str] = []
            for tid, rs in responses.RESPONSE_CATALOG.items():
                family = templates_by_id[tid].family
                for variant in rs.variants:
                    vid = variant.response_variant_id
                    out = responses.render_response(tid, vid, episode, plan)
                    where = f"{tid}/{vid}"
                    rendered_all.append(out)

                    if want("slots"):
                        if UNFILLED_SLOT.search(out):
                            checks["slots"].hits.append(f"{where}: unfilled slot")
                        if "{{" in out and not out.lstrip().startswith("{"):
                            checks["slots"].hits.append(
                                f"{where}: literal doubled brace -> {out.strip()[:60]!r}")
                    if want("rule_words"):
                        m = RULE_WORDS.search(out)
                        if m:
                            checks["rule_words"].hits.append(f"{where}: {m.group(0)!r}")
                        if "Assignment:" in out:
                            checks["rule_words"].hits.append(f"{where}: canonical 'Assignment:'")
                    if want("motivation"):
                        m = MOTIVATION_WORDS.search(out)
                        if m:
                            checks["motivation"].hits.append(f"{where}: {m.group(0)!r}")
                    if want("plural") and one_run:
                        m = PLURAL_SENTENCE.search(out)
                        if m:
                            checks["plural"].hits.append(
                                f"{where}: {m.group(0)!r} over a single run")
                    if want("enumerator"):
                        markers = re.findall(r"(?m)^\s*([IVX]+\.|\d+[.)])\s", out)
                        if len(markers) > 1 and len(set(markers)) == 1:
                            checks["enumerator"].hits.append(
                                f"{where}: marker {markers[0]!r} repeats (missing {{n}}?)")
                    if want("machine_format"):
                        if _looks_like_json(out, family) and not _json_ok(out):
                            checks["machine_format"].hits.append(
                                f"{where}: looks like JSON but does not parse")
                        elif _looks_like_xml(out):
                            try:
                                ET.fromstring(out.strip())
                            except Exception:
                                checks["machine_format"].hits.append(
                                    f"{where}: looks like XML but does not parse")
                        elif family == "yaml" and not _structured_ok(out):
                            checks["machine_format"].hits.append(
                                f"{where}: no YAML/TOML/INI parser accepts it "
                                f"-> {out.strip()[:60]!r}")
                    if want("fences") and "```" in out and family in MACHINE_FAMILIES:
                        checks["fences"].hits.append(f"{where}: code fence in a {family} surface")
                    if want("tables"):
                        stripped = out.lstrip()
                        if stripped.startswith("|") and "\n" in stripped \
                                and not re.search(r"\|\s*[-:]{2,}", out):
                            checks["tables"].hits.append(f"{where}: pipe table without a rule row")
                    if want("at_sign") and re.search(r"R\d+\s*@|@\s*R\d+", out):
                        checks["at_sign"].hits.append(f"{where}: ambiguous '@' relation")
                    if want("whitespace") and out != out.rstrip():
                        checks["whitespace"].hits.append(f"{where}: trailing whitespace/newline")

                    # entity hygiene is a hard contract; keep it under structure
                    if want("structure"):
                        for run, crew in zip(episode.runs, plan, strict=True):
                            if run.run_id not in out or crew not in out:
                                checks["structure"].hits.append(f"{where}: missing run or crew")
                        for other in known - set(plan):
                            if other in out:
                                checks["structure"].hits.append(
                                    f"{where}: unselected crew {other!r} leaked")

            if want("uniqueness") and len(set(rendered_all)) != 1000:
                checks["uniqueness"].hits.append(
                    f"{episode.episode_id}/{plan}: "
                    f"{1000 - len(set(rendered_all))} colliding surfaces")

    # -- parser round-trip ---------------------------------------------------
    if want("parser"):
        try:
            from parse_response import parse_response
        except ImportError:
            checks["parser"].severity = "warning"
            checks["parser"].hits.append(
                "template_response_diversity_v1/parse_response.py not importable; skipped")
        else:
            for episode in episodes:
                for tid in responses.RESPONSE_CATALOG:
                    for vid in responses.RESPONSE_VARIANT_IDS:
                        out = responses.render_response(tid, vid, episode, episode.charter_plan)
                        result = parse_response(out, episode)
                        if result.plan != episode.charter_plan:
                            checks["parser"].hits.append(
                                f"{tid}/{vid} on {episode.episode_id}: {result.status}")

    return list(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every hit instead of the first few")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    parser.add_argument("--only", nargs="+", metavar="CHECK",
                        help="run only these checks")
    args = parser.parse_args()

    results = run_checks(set(args.only) if args.only else None)

    errors = warnings = 0
    for r in sorted(results, key=lambda x: (x.severity != "error", x.name)):
        unique = sorted(set(r.hits))
        mark = "ok  " if r.ok else ("FAIL" if r.severity == "error" else "warn")
        count = "" if r.ok else f"  ({len(unique)})"
        print(f"[{mark}] {r.name:<14} {r.description}{count}")
        if unique:
            shown = unique if args.verbose else unique[:6]
            for hit in shown:
                print(f"         - {hit}")
            if len(unique) > len(shown):
                print(f"         … {len(unique) - len(shown)} more (use --verbose)")
            if r.severity == "error":
                errors += 1
            else:
                warnings += 1

    print()
    if errors:
        print(f"FAILED — {errors} check(s) with errors, {warnings} with warnings")
        return 1
    if warnings:
        print(f"passed with {warnings} warning(s)"
              + (" — --strict makes these fail" if not args.strict else ""))
        return 1 if args.strict else 0
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
