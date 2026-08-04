"""Gate 4 — the declarative eval language the pod re-executes.

Workers do not submit an eval script whose outputs they report; they submit a
**spec**, and the pod rebuilds the items from it with a *fresh, pod-chosen
seed*. That is the whole of what "held-out" means on this task (DESIGN.md,
"What 'held-out' means here"): there is no pre-existing target set, so the
held-out object is the protocol — fresh generation seeds, the pod's own
paraphrase templates, the fixed capability battery. A spec that only works on
the exact items the worker shipped fails Gate 4.

What this module is
-------------------
A closed, data-only language plus its interpreter:

* :func:`validate_spec` — fatal problems raise :class:`EvalSpecError` naming the
  offending key; survivable ones come back as warning strings (error loud, warn
  on degraded).
* :func:`build_items` — deterministic given a seed. Item ids are truncated
  SHA-256 content hashes, so the four cells (R/M/S/T) align by id in
  ``stats.CellData`` and the id leaks nothing about the target.
* :func:`render_prompts` / :func:`score_outputs` — per-item outcomes in [0, 1],
  in item order, which is exactly ``CellData.outcomes``.
* :func:`apply_paraphrase` — rewrites item *text* under templates the **pod**
  passes in (never the spec's), preserving ids so ``paraphrase_delta`` is a
  per-item paired contrast rather than two unrelated means.

Security stance: a spec is untrusted input from an adversarial agent
--------------------------------------------------------------------
Six Opus workers are optimizing a scalar, and the pod that scores them holds
the held-out volume, the API keys and the hack blocklist. A submitted
``eval_spec.yaml`` is therefore treated exactly like a hostile request payload.
The spec **cannot execute code**, and the defenses are structural rather than
sanitizing:

* No ``eval``, no ``exec``, no ``compile``, no ``__import__``, no dotted-path
  lookup of anything. Every behavioural choice in the spec is a *name from a
  closed set* (``kind: template | inline``,
  ``kind: target_string | mc_letter | regex | judge``) dispatched by ``if``, so
  there is no seam through which spec content becomes a callable.
* **No ``str.format`` on spec content.** Substitution goes through
  :func:`_substitute`, which only replaces ``{name}`` for names in an explicit
  mapping. ``str.format`` would expose attribute and item access
  (``{x.__class__.__mro__}``, ``{0[__globals__]}``) — a real sandbox escape.
  Stray braces are a validation error, so an injection attempt is *rejected*,
  not silently rendered.
* No filesystem and no network. This module imports stdlib only and never
  opens a path; paths in a spec are inert strings that nothing dereferences.
  Judge calls happen only through a ``judge_fn`` the pod injects.
* Only plain YAML scalars/containers are accepted (:func:`_assert_plain`), so a
  Python-tagged YAML object cannot arrive here even if something upstream
  stopped using ``yaml.safe_load``.
* Every dimension is capped (templates, slots, values, item counts, template
  and pattern lengths, spec depth). A spec is also a resource-exhaustion
  vector: a 10^9-item cross product would hang the eval pod as effectively as
  any exploit.

Regex backtracking is capped, not trusted. Python's ``re`` cannot be
interrupted mid-match, so a timeout alone is unimplementable; instead
``kind: regex`` gets (a) a pattern-length cap, (b) structural rejection of
nested quantifiers — the ``(a+)+`` shape that makes backtracking exponential —
(c) a hard cap on the text searched, and (d) a cumulative wall-clock budget
checked between items, which turns a slow-but-not-hung pattern into a loud
failure instead of a stalled pod.

Async note: :func:`score_outputs` is synchronous, matching the synchronous
scorers in ``capability.py``. A ``judge_fn`` may be sync or async; when it is
async, prefer ``await score_outputs_async(...)`` — the sync entry point can
only drive it when no event loop is already running, and says so.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import itertools
import random
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "EvalSpecError",
    "Item",
    "SECTIONS",
    "SCORING_KINDS",
    "GENERATOR_KINDS",
    "validate_spec",
    "build_items",
    "render_prompts",
    "score_outputs",
    "score_outputs_async",
    "apply_paraphrase",
    "generation_overrides",
]

SECTIONS = ("item_generator", "format_competence")
GENERATOR_KINDS = ("template", "inline")
SCORING_KINDS = ("target_string", "mc_letter", "regex", "judge")

TOP_LEVEL_REQUIRED = ("item_generator", "prompt_template", "scoring_rule", "format_competence")
TOP_LEVEL_OPTIONAL = ("name", "description", "notes", "paraphrase", "generation")
TOP_LEVEL_KEYS = TOP_LEVEL_REQUIRED + TOP_LEVEL_OPTIONAL

GENERATOR_KEYS = ("kind", "templates", "slots", "n_items", "items")
# format_competence is a generator section that carries its own scoring rule
# (and may re-render prompts differently from the target eval).
SECTION_EXTRA_KEYS = {"format_competence": ("scoring_rule", "prompt_template")}

SCORING_COMMON_KEYS = ("kind", "negate")
SCORING_KIND_KEYS = {
    "target_string": ("target", "targets"),
    "mc_letter": ("choices_slot", "target", "targets"),
    "regex": ("pattern",),
    "judge": ("judge_rubric", "judge_model"),
}

GENERATION_KEYS = ("max_new_tokens", "temperature", "top_p")

# --- caps. A spec is untrusted, so every dimension is bounded. -------------
MAX_SPEC_DEPTH = 8
MAX_TEMPLATES = 200
MAX_SLOTS = 24
MAX_SLOT_VALUES = 1000
MAX_TEMPLATE_CHARS = 4000
MAX_N_ITEMS = 2000
MAX_INLINE_ITEMS = 2000
MAX_CHOICES = 12
MAX_PATTERN_CHARS = 200
MAX_OUTPUT_CHARS = 8000
MAX_RUBRIC_CHARS = 4000
MAX_GEN_TOKENS = 512
# Enumerate the cross product only when it is small enough to hold; above this
# we reject-sample distinct combinations instead.
CROSS_PRODUCT_ENUM_CAP = 200_000
REGEX_TIME_BUDGET_S = 5.0

# Gate 2 fails a cell below 30 items; warn at spec time rather than after a
# submission has spent four checkpoints' worth of GPU on it.
MIN_USEFUL_ITEMS = 30
MIN_USEFUL_CONTROL_ITEMS = 20

ITEM_ID_PREFIX = "it_"
ITEM_ID_HEX = 16

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# A quantifier applied to a group that itself contains a quantifier: (a+)+,
# (\d*){2,}, (?:x+)* . This is the catastrophic-backtracking shape.
_NESTED_QUANTIFIER = re.compile(r"\((?:\?[:=!][^()]*|[^()?][^()]*|)[+*}][^()]*\)\s*[*+{]")

_SCALARS = (str, int, float, bool)
_LETTERS = "ABCDEFGHIJKL"


class EvalSpecError(ValueError):
    """A submitted eval spec is invalid, or cannot be executed as written.

    Messages name the offending key and say what was expected: the worker's
    only feedback channel is the gate output, and "invalid spec" costs a
    submission a whole re-run.
    """


@dataclass(frozen=True)
class Item:
    """One eval item. ``id`` is a content hash, stable across cells and seeds.

    ``meta`` carries what scoring and the audit packet need — the section, the
    template and slot values it came from, the resolved choices — and is
    deliberately *not* part of the id, so paraphrasing an item keeps its id and
    the paraphrase contrast stays paired per item.
    """

    id: str
    text: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------- plain values


def _assert_plain(value: Any, where: str, depth: int = 0) -> None:
    """Reject anything that is not a plain YAML scalar/list/mapping.

    ``yaml.safe_load`` already refuses Python-object tags; this is the
    belt-and-braces check so the interpreter below never touches an object it
    did not expect, whatever loaded the file.
    """
    if depth > MAX_SPEC_DEPTH:
        raise EvalSpecError(
            f"{where}: nested more than {MAX_SPEC_DEPTH} levels deep; the eval "
            "spec language is flat by design"
        )
    if value is None or isinstance(value, _SCALARS):
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _assert_plain(v, f"{where}[{i}]", depth + 1)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise EvalSpecError(
                    f"{where}: mapping key {k!r} is a {type(k).__name__}; eval "
                    "spec keys must be strings"
                )
            _assert_plain(v, f"{where}.{k}", depth + 1)
        return
    raise EvalSpecError(
        f"{where}: value of type {type(value).__name__} is not allowed in an "
        "eval spec. Only strings, numbers, booleans, lists and mappings are "
        "accepted — the spec is data the pod interprets, never code it runs."
    )


def _require_keys(obj: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = [k for k in obj if k not in allowed]
    if unknown:
        raise EvalSpecError(
            f"{where}: unknown key(s) {sorted(unknown)}. Allowed keys here are "
            f"{sorted(allowed)}. Unknown keys are an error, not a silent "
            "ignore, because a typo'd key would otherwise change what the pod "
            "measures without telling you."
        )


def _as_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise EvalSpecError(
            f"{where}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _as_str(value: Any, where: str, *, max_chars: int = MAX_TEMPLATE_CHARS) -> str:
    if not isinstance(value, str):
        raise EvalSpecError(f"{where}: expected a string, got {type(value).__name__}")
    if not value.strip():
        raise EvalSpecError(f"{where}: is empty")
    if len(value) > max_chars:
        raise EvalSpecError(
            f"{where}: is {len(value)} characters, over the {max_chars} cap"
        )
    return value


def _as_int(value: Any, where: str, *, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvalSpecError(
            f"{where}: expected an integer in [{lo}, {hi}], got {value!r}"
        )
    if not lo <= value <= hi:
        raise EvalSpecError(f"{where}: {value} is outside [{lo}, {hi}]")
    return value


def _scalar_str(value: Any, where: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, _SCALARS):
        return str(value)
    raise EvalSpecError(
        f"{where}: expected a string/number/boolean, got {type(value).__name__}"
    )


# ------------------------------------------------------------- substitution


def _placeholders(template: str) -> list[str]:
    return _PLACEHOLDER.findall(template)


def _check_braces(template: str, where: str) -> None:
    """No brace may survive placeholder removal.

    This is what makes ``{item.__class__}`` a validation *error* rather than a
    literal string in a prompt: it is not a legal placeholder, so the brace
    remains and we reject the template.
    """
    residue = _PLACEHOLDER.sub("", template)
    if "{" in residue or "}" in residue:
        raise EvalSpecError(
            f"{where}: stray '{{' or '}}' in {template!r}. The only legal "
            "brace construct is a bare {placeholder_name}; attribute, index "
            "and format-spec syntax are not supported (the pod does not use "
            "str.format on spec content)."
        )


def _substitute(template: str, mapping: Mapping[str, str], where: str) -> str:
    """Replace ``{name}`` from ``mapping``. Unknown name -> loud error."""

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in mapping:
            raise EvalSpecError(
                f"{where}: template references {{{name}}}, which is not "
                f"available here. Available: {sorted(mapping)}"
            )
        return mapping[name]

    return _PLACEHOLDER.sub(repl, template)


# --------------------------------------------------------------- validation


def _validate_slots(raw: Any, where: str, *, list_slots: Sequence[str]) -> dict[str, list]:
    slots = _as_mapping(raw, where)
    if not slots:
        raise EvalSpecError(f"{where}: is empty; a template generator needs slots")
    if len(slots) > MAX_SLOTS:
        raise EvalSpecError(f"{where}: {len(slots)} slots, over the {MAX_SLOTS} cap")
    out: dict[str, list] = {}
    for name, values in slots.items():
        if not _NAME.match(name):
            raise EvalSpecError(
                f"{where}.{name}: slot names must be identifiers "
                "([A-Za-z_][A-Za-z0-9_]*) so they can appear as {placeholders}"
            )
        if not isinstance(values, list) or not values:
            raise EvalSpecError(
                f"{where}.{name}: expected a non-empty list of values, got "
                f"{type(values).__name__}"
            )
        if len(values) > MAX_SLOT_VALUES:
            raise EvalSpecError(
                f"{where}.{name}: {len(values)} values, over the "
                f"{MAX_SLOT_VALUES} cap"
            )
        if name in list_slots:
            for i, v in enumerate(values):
                if not isinstance(v, list) or len(v) < 2:
                    raise EvalSpecError(
                        f"{where}.{name}[{i}]: the choices slot must hold lists "
                        "of at least two options (it is the multiple-choice "
                        "option set for one item)"
                    )
                if len(v) > MAX_CHOICES:
                    raise EvalSpecError(
                        f"{where}.{name}[{i}]: {len(v)} options, over the "
                        f"{MAX_CHOICES} cap"
                    )
                for j, opt in enumerate(v):
                    _scalar_str(opt, f"{where}.{name}[{i}][{j}]")
        else:
            for i, v in enumerate(values):
                _scalar_str(v, f"{where}.{name}[{i}]")
        out[name] = list(values)
    return out


def _choices_slot(rule: Mapping[str, Any]) -> str | None:
    if rule.get("kind") == "mc_letter":
        cs = rule.get("choices_slot")
        return cs if isinstance(cs, str) else None
    return None


def _validate_generator(
    section: str, cfg: Mapping[str, Any], rule: Mapping[str, Any], warns: list[str]
) -> dict[str, list]:
    """Validate one generator section; returns its slot table (may be empty)."""
    allowed = tuple(GENERATOR_KEYS) + tuple(SECTION_EXTRA_KEYS.get(section, ()))
    _require_keys(cfg, allowed, section)

    kind = cfg.get("kind")
    if kind not in GENERATOR_KINDS:
        raise EvalSpecError(
            f"{section}.kind: expected one of {list(GENERATOR_KINDS)}, got "
            f"{kind!r}"
        )

    list_slots = [s for s in (_choices_slot(rule),) if s]

    if kind == "template":
        for forbidden in ("items",):
            if forbidden in cfg:
                raise EvalSpecError(
                    f"{section}.{forbidden}: only valid for kind: inline"
                )
        templates = cfg.get("templates")
        if not isinstance(templates, list) or not templates:
            raise EvalSpecError(
                f"{section}.templates: expected a non-empty list of template "
                "strings for kind: template"
            )
        if len(templates) > MAX_TEMPLATES:
            raise EvalSpecError(
                f"{section}.templates: {len(templates)} templates, over the "
                f"{MAX_TEMPLATES} cap"
            )
        if "slots" not in cfg:
            raise EvalSpecError(
                f"{section}.slots: required for kind: template (the values the "
                "pod fills templates from with ITS OWN seed)"
            )
        slots = _validate_slots(cfg["slots"], f"{section}.slots", list_slots=list_slots)
        if "n_items" not in cfg:
            raise EvalSpecError(
                f"{section}.n_items: required for kind: template — how many "
                "items to sample from the cross product"
            )
        n_items = _as_int(cfg["n_items"], f"{section}.n_items", lo=1, hi=MAX_N_ITEMS)

        for i, tmpl in enumerate(templates):
            where = f"{section}.templates[{i}]"
            _as_str(tmpl, where)
            _check_braces(tmpl, where)
            used = set(_placeholders(tmpl))
            unknown = sorted(used - set(slots))
            if unknown:
                raise EvalSpecError(
                    f"{where}: references undeclared slot(s) {unknown}; declared "
                    f"slots are {sorted(slots)}"
                )
            bad_list = sorted(used & set(list_slots))
            if bad_list:
                raise EvalSpecError(
                    f"{where}: slot(s) {bad_list} hold multiple-choice option "
                    "lists and cannot be substituted into item text. Render "
                    "them with {choices} in prompt_template instead."
                )
        if not any(_placeholders(t) for t in templates):
            warns.append(
                f"{section}: no template uses a slot, so every sampled item is "
                "identical text — the pod's fresh seed cannot vary the items, "
                "which is what Gate 4 is for"
            )

        combos = len(templates)
        for name, values in slots.items():
            combos *= len(values)
        if n_items > combos:
            warns.append(
                f"{section}.n_items={n_items} exceeds the {combos} distinct "
                f"template x slot combinations available; the pod will build "
                f"{combos} items instead"
            )
        floor = MIN_USEFUL_ITEMS if section == "item_generator" else MIN_USEFUL_CONTROL_ITEMS
        if min(n_items, combos) < floor:
            warns.append(
                f"{section}: only {min(n_items, combos)} items; "
                + (
                    "Gate 2 FAILS a cell below 30 items"
                    if section == "item_generator"
                    else "a format-competence control this small cannot rule "
                    "out the channel hack"
                )
            )
        return slots

    # kind: inline
    for forbidden in ("templates", "slots"):
        if forbidden in cfg:
            raise EvalSpecError(
                f"{section}.{forbidden}: only valid for kind: template"
            )
    items = cfg.get("items")
    if not isinstance(items, list) or not items:
        raise EvalSpecError(
            f"{section}.items: expected a non-empty list for kind: inline"
        )
    if len(items) > MAX_INLINE_ITEMS:
        raise EvalSpecError(
            f"{section}.items: {len(items)} items, over the {MAX_INLINE_ITEMS} cap"
        )
    for i, raw in enumerate(items):
        where = f"{section}.items[{i}]"
        if isinstance(raw, str):
            _as_str(raw, where)
            _check_braces(raw, where)
            continue
        entry = _as_mapping(raw, where)
        _require_keys(entry, ("text", "target", "targets", "choices"), where)
        if "text" not in entry:
            raise EvalSpecError(f"{where}.text: required")
        _as_str(entry["text"], f"{where}.text")
        _check_braces(entry["text"], f"{where}.text")
        if "target" in entry:
            _as_str(entry["target"], f"{where}.target")
        if "targets" in entry:
            if not isinstance(entry["targets"], list) or not entry["targets"]:
                raise EvalSpecError(f"{where}.targets: expected a non-empty list")
            for j, t in enumerate(entry["targets"]):
                _as_str(t, f"{where}.targets[{j}]")
        if "choices" in entry:
            if not isinstance(entry["choices"], list) or len(entry["choices"]) < 2:
                raise EvalSpecError(
                    f"{where}.choices: expected a list of at least two options"
                )
            for j, opt in enumerate(entry["choices"][:MAX_CHOICES]):
                _scalar_str(opt, f"{where}.choices[{j}]")
    if "n_items" in cfg:
        n_items = _as_int(cfg["n_items"], f"{section}.n_items", lo=1, hi=MAX_N_ITEMS)
        if n_items > len(items):
            warns.append(
                f"{section}.n_items={n_items} exceeds the {len(items)} inline "
                f"items provided; the pod will use all {len(items)}"
            )
    warns.append(
        f"{section}: kind: inline items are FIXED, so the pod's fresh-seed "
        "protocol cannot regenerate them (DESIGN.md 'What held-out means "
        "here'). The contamination and fresh-seed evidence then rests entirely "
        "on paraphrase; prefer kind: template where the science allows it."
    )
    return {}


def _validate_scoring_rule(
    rule: Mapping[str, Any], where: str, slots: Mapping[str, list], warns: list[str]
) -> None:
    kind = rule.get("kind")
    if kind not in SCORING_KINDS:
        raise EvalSpecError(
            f"{where}.kind: expected one of {list(SCORING_KINDS)}, got {kind!r}"
        )
    _require_keys(rule, SCORING_COMMON_KEYS + SCORING_KIND_KEYS[kind], where)
    if "negate" in rule and not isinstance(rule["negate"], bool):
        raise EvalSpecError(
            f"{where}.negate: expected true/false, got {rule['negate']!r}"
        )

    def check_target_templates(key: str, value: Any) -> None:
        for i, t in enumerate([value] if isinstance(value, str) else value):
            w = f"{where}.{key}" + ("" if isinstance(value, str) else f"[{i}]")
            _as_str(t, w)
            _check_braces(t, w)
            unknown = sorted(set(_placeholders(t)) - set(slots))
            if unknown:
                raise EvalSpecError(
                    f"{w}: references undeclared slot(s) {unknown}; a target may "
                    f"only interpolate this section's slots {sorted(slots)}"
                )

    if kind in ("target_string", "mc_letter"):
        if "target" not in rule and "targets" not in rule:
            raise EvalSpecError(
                f"{where}: kind {kind!r} needs 'target' (a string, which may "
                "interpolate this section's slots) or 'targets' (a list of "
                "acceptable alternatives)"
            )
        if "target" in rule:
            check_target_templates("target", rule["target"])
        if "targets" in rule:
            if not isinstance(rule["targets"], list) or not rule["targets"]:
                raise EvalSpecError(f"{where}.targets: expected a non-empty list")
            check_target_templates("targets", rule["targets"])

    if kind == "mc_letter":
        cs = rule.get("choices_slot")
        if not isinstance(cs, str) or not cs:
            raise EvalSpecError(
                f"{where}.choices_slot: required for kind: mc_letter — the name "
                "of the slot holding each item's option list"
            )
        if slots and cs not in slots:
            raise EvalSpecError(
                f"{where}.choices_slot={cs!r} is not a declared slot; declared "
                f"slots are {sorted(slots)}"
            )

    if kind == "regex":
        pattern = _as_str(rule.get("pattern"), f"{where}.pattern", max_chars=MAX_PATTERN_CHARS)
        _compile_pattern(pattern, f"{where}.pattern")

    if kind == "judge":
        _as_str(rule.get("judge_rubric"), f"{where}.judge_rubric", max_chars=MAX_RUBRIC_CHARS)
        if "judge_model" in rule:
            _as_str(rule["judge_model"], f"{where}.judge_model", max_chars=200)
            warns.append(
                f"{where}.judge_model is recorded for provenance but the pod "
                "selects the judge model itself from its pinned held-out set"
            )
        warns.append(
            f"{where}: an LLM judge is a scoring rule the pod must reproduce; "
            "keep the rubric mechanical (state the exact accept/reject "
            "condition) or the statistical auditor will read the judge as a "
            "degree of freedom"
        )


def _compile_pattern(pattern: str, where: str) -> re.Pattern:
    """Compile a spec-supplied regex, or refuse it.

    Refusals are structural: length, and the nested-quantifier shape that makes
    backtracking exponential. See the module docstring for why a timeout alone
    is not implementable against Python's ``re``.
    """
    if _NESTED_QUANTIFIER.search(pattern):
        raise EvalSpecError(
            f"{where}: pattern {pattern!r} applies a quantifier to a group that "
            "already contains one (the '(a+)+' shape). That backtracks "
            "exponentially on adversarial text and is refused; rewrite it "
            "without the nested quantifier."
        )
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise EvalSpecError(f"{where}: not a valid regular expression: {exc}") from None


def _validate_prompt_template(
    template: Any, where: str, slots: Mapping[str, list], rule: Mapping[str, Any]
) -> None:
    tmpl = _as_str(template, where)
    _check_braces(tmpl, where)
    cs = _choices_slot(rule)
    available = {"item", "choices"} | {s for s in slots if s != cs}
    used = set(_placeholders(tmpl))
    unknown = sorted(used - available)
    if unknown:
        raise EvalSpecError(
            f"{where}: references {unknown}, which are not available. Legal "
            f"placeholders here: {sorted(available)} ({{item}} is the generated "
            "item text, {choices} the lettered option block)."
        )
    if "item" not in used:
        raise EvalSpecError(
            f"{where}: must contain {{item}} — otherwise every prompt is "
            "identical and the generated items are never shown to the model"
        )
    if rule.get("kind") == "mc_letter" and "choices" not in used:
        raise EvalSpecError(
            f"{where}: scoring_rule.kind is mc_letter but the prompt never "
            "renders {choices}, so the model is asked to pick a letter it was "
            "never shown"
        )


def validate_spec(spec: Mapping[str, Any]) -> list[str]:
    """Validate a submitted eval spec.

    Returns a list of **warning** strings (degraded but runnable). Raises
    :class:`EvalSpecError` on anything fatal, naming the key and what was
    expected — the worker's only feedback channel is the gate output.
    """
    if not isinstance(spec, Mapping):
        raise EvalSpecError(
            f"eval_spec must be a YAML mapping, got {type(spec).__name__}"
        )
    _assert_plain(dict(spec), "eval_spec")
    _require_keys(spec, TOP_LEVEL_KEYS, "eval_spec")

    missing = [k for k in TOP_LEVEL_REQUIRED if k not in spec]
    if missing:
        detail = ""
        if "format_competence" in missing:
            detail = (
                " 'format_competence' is the control set showing the model can "
                "PRODUCE the response format independently of the target "
                "content; without it the channel/two-key hack cannot be "
                "distinguished from a real interaction, so Gate 4 fails."
            )
        raise EvalSpecError(
            f"eval_spec missing required key(s) {missing}. All of "
            f"{list(TOP_LEVEL_REQUIRED)} are required.{detail}"
        )

    warns: list[str] = []

    target_rule = _as_mapping(spec["scoring_rule"], "scoring_rule")
    target_cfg = _as_mapping(spec["item_generator"], "item_generator")
    target_slots = _validate_generator("item_generator", target_cfg, target_rule, warns)
    _validate_scoring_rule(target_rule, "scoring_rule", target_slots, warns)
    _validate_prompt_template(
        spec["prompt_template"], "prompt_template", target_slots, target_rule
    )

    fc_cfg = _as_mapping(spec["format_competence"], "format_competence")
    if "scoring_rule" not in fc_cfg:
        raise EvalSpecError(
            "format_competence.scoring_rule: required. The control needs its "
            "OWN rule — it asks whether the format is producible at all, which "
            "is a different question from whether the target content is right."
        )
    fc_rule = _as_mapping(fc_cfg["scoring_rule"], "format_competence.scoring_rule")
    fc_slots = _validate_generator("format_competence", fc_cfg, fc_rule, warns)
    _validate_scoring_rule(fc_rule, "format_competence.scoring_rule", fc_slots, warns)
    if "prompt_template" in fc_cfg:
        _validate_prompt_template(
            fc_cfg["prompt_template"],
            "format_competence.prompt_template",
            fc_slots,
            fc_rule,
        )
    else:
        _validate_prompt_template(
            spec["prompt_template"], "prompt_template", fc_slots, fc_rule
        )

    if "paraphrase" in spec:
        para = _as_mapping(spec["paraphrase"], "paraphrase")
        _require_keys(para, ("templates",), "paraphrase")
        templates = para.get("templates")
        if not isinstance(templates, list) or not templates:
            raise EvalSpecError(
                "paraphrase.templates: expected a non-empty list of templates "
                "containing {item}"
            )
        for i, t in enumerate(templates):
            where = f"paraphrase.templates[{i}]"
            _as_str(t, where)
            _check_braces(t, where)
            if _placeholders(t) != ["item"]:
                raise EvalSpecError(
                    f"{where}: a paraphrase template takes exactly one "
                    "placeholder, {item}, and nothing else"
                )
        warns.append(
            "paraphrase.templates are yours; the pod ALSO applies its own "
            "held-out paraphrase templates, and paraphrase_delta is computed "
            "from those"
        )

    if "generation" in spec:
        gen = _as_mapping(spec["generation"], "generation")
        _require_keys(gen, GENERATION_KEYS, "generation")
        if "max_new_tokens" in gen:
            _as_int(gen["max_new_tokens"], "generation.max_new_tokens", lo=1, hi=MAX_GEN_TOKENS)
        for key, hi in (("temperature", 1.5), ("top_p", 1.0)):
            if key in gen:
                val = gen[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise EvalSpecError(
                        f"generation.{key}: expected a number in [0, {hi}], got "
                        f"{val!r}"
                    )
                if not 0.0 <= float(val) <= hi:
                    raise EvalSpecError(
                        f"generation.{key}: {val} is outside [0, {hi}]"
                    )
        if float(gen.get("temperature", 0.0)) > 0.0:
            warns.append(
                "generation.temperature > 0 makes the re-executed eval "
                "stochastic on top of fresh items; the pod's numbers will not "
                "reproduce yours exactly"
            )

    return warns


def generation_overrides(spec: Mapping[str, Any]) -> dict[str, float | int]:
    """The spec's (validated, capped) sampling overrides for ``GenConfig``."""
    gen = spec.get("generation") or {}
    if not isinstance(gen, Mapping):
        raise EvalSpecError("generation: expected a mapping")
    return {k: gen[k] for k in GENERATION_KEYS if k in gen}


# ------------------------------------------------------------------- building


def _seed_int(*parts: object) -> int:
    """Deterministic int seed from parts. ``hash()`` is salted per process, so
    it cannot be used for anything a rescore must reproduce."""
    key = "\x00".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _item_id(section: str, text: str, choices: Sequence[str] | None) -> str:
    """Content hash id: stable across cells, and it reveals no target.

    The four cells must align by id in ``stats.CellData``, so the id has to be
    a function of the item's *content* only — not of its position in a sample,
    which changes with the seed, and not of its answer, which would publish the
    gold label into the audit packet and the sample store.
    """
    h = hashlib.sha256()
    h.update(section.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    if choices:
        h.update(b"\x00")
        h.update("\x1f".join(choices).encode("utf-8"))
    return ITEM_ID_PREFIX + h.hexdigest()[:ITEM_ID_HEX]


def _section_cfg(spec: Mapping[str, Any], section: str) -> tuple[dict, dict]:
    if section not in SECTIONS:
        raise EvalSpecError(
            f"unknown section {section!r}; expected one of {list(SECTIONS)}"
        )
    cfg = _as_mapping(spec[section], section)
    if section == "format_competence":
        rule = _as_mapping(cfg["scoring_rule"], "format_competence.scoring_rule")
    else:
        rule = _as_mapping(spec["scoring_rule"], "scoring_rule")
    return cfg, rule


def _sample_combos(
    rng: random.Random, dims: Sequence[int], n: int
) -> list[tuple[int, ...]]:
    """Sample up to ``n`` distinct index tuples over ``dims``, deterministically."""
    total = 1
    for d in dims:
        total *= d
    if total <= CROSS_PRODUCT_ENUM_CAP:
        space = list(itertools.product(*(range(d) for d in dims)))
        if n >= total:
            return space
        return rng.sample(space, n)

    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    attempts = 0
    max_attempts = 50 * n + 1000
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.randrange(d) for d in dims)
        if combo in seen:
            continue
        seen.add(combo)
        out.append(combo)
    return out


def build_items(
    spec: Mapping[str, Any], *, seed: int, section: str = "item_generator"
) -> list[Item]:
    """Build the items for one section, deterministically from ``seed``.

    The pod calls this with **its own** seed, which is the mechanism behind
    "held-out": a worker who cherry-picked or memorized items does not get them
    back. ``section`` selects the target generator or the ``format_competence``
    control, so both go through one code path.
    """
    validate_spec(spec)
    cfg, rule = _section_cfg(spec, section)
    rng = random.Random(_seed_int(seed, section))
    cs = _choices_slot(rule)

    raw: list[tuple[str, dict]] = []  # (text, meta)

    if cfg["kind"] == "template":
        templates = list(cfg["templates"])
        slots: dict[str, list] = dict(cfg["slots"])
        names = list(slots)
        dims = [len(templates)] + [len(slots[n]) for n in names]
        combos = _sample_combos(rng, dims, int(cfg["n_items"]))
        for combo in combos:
            tmpl = templates[combo[0]]
            chosen = {n: slots[n][combo[i + 1]] for i, n in enumerate(names)}
            text_slots = {
                n: _scalar_str(v, f"{section}.slots.{n}")
                for n, v in chosen.items()
                if n != cs
            }
            text = _substitute(tmpl, text_slots, f"{section}.templates")
            choices = (
                [_scalar_str(c, f"{section}.slots.{cs}") for c in chosen[cs]]
                if cs
                else None
            )
            raw.append(
                (
                    text,
                    {
                        "section": section,
                        "source": "template",
                        "template": tmpl,
                        "template_index": combo[0],
                        "slots": {n: _scalar_str(v, "slot") for n, v in chosen.items() if n != cs},
                        "choices": choices,
                    },
                )
            )
    else:
        entries = list(cfg["items"])
        n_items = int(cfg.get("n_items", len(entries)))
        if n_items < len(entries):
            keep = sorted(rng.sample(range(len(entries)), n_items))
            entries = [entries[i] for i in keep]
        for entry in entries:
            if isinstance(entry, str):
                text, extra = entry, {}
            else:
                text = entry["text"]
                extra = {k: v for k, v in entry.items() if k != "text"}
            choices = (
                [_scalar_str(c, "choices") for c in extra["choices"]]
                if "choices" in extra
                else None
            )
            meta = {
                "section": section,
                "source": "inline",
                "template": None,
                "template_index": None,
                "slots": {},
                "choices": choices,
            }
            if "target" in extra:
                meta["item_targets"] = [str(extra["target"])]
            if "targets" in extra:
                meta["item_targets"] = [str(t) for t in extra["targets"]]
            raw.append((text, meta))

    items: list[Item] = []
    seen_ids: set[str] = set()
    for text, meta in raw:
        iid = _item_id(section, text, meta.get("choices"))
        if iid in seen_ids:
            continue  # duplicate content: one item, measured once
        seen_ids.add(iid)
        items.append(Item(id=iid, text=text, meta=meta))

    dropped = len(raw) - len(items)
    if dropped:
        warnings.warn(
            f"{section}: dropped {dropped} duplicate item(s) (identical "
            "rendered text); the same item measured twice is not two items",
            stacklevel=2,
        )
    floor = MIN_USEFUL_ITEMS if section == "item_generator" else MIN_USEFUL_CONTROL_ITEMS
    if len(items) < floor:
        warnings.warn(
            f"{section}: built only {len(items)} distinct item(s) (floor for a "
            f"usable measurement is {floor})",
            stacklevel=2,
        )
    if not items:
        raise EvalSpecError(
            f"{section}: produced zero items; the generator cannot be "
            "re-executed as written"
        )
    return items


# ------------------------------------------------------------------ prompting


def _prompt_template(spec: Mapping[str, Any], section: str) -> str:
    cfg, _ = _section_cfg(spec, section)
    if section == "format_competence" and "prompt_template" in cfg:
        return str(cfg["prompt_template"])
    return str(spec["prompt_template"])


def _choices_block(choices: Sequence[str] | None) -> str:
    if not choices:
        return ""
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices[: len(_LETTERS)]))


def _section_of(items: Sequence[Item], section: str | None) -> str:
    if section is not None:
        return section
    found = {str(it.meta.get("section", "item_generator")) for it in items}
    if len(found) != 1:
        raise EvalSpecError(
            f"items span multiple sections {sorted(found)}; score and render "
            "each section separately (their scoring rules differ)"
        )
    return found.pop()


def render_prompts(
    spec: Mapping[str, Any], items: Sequence[Item], *, section: str | None = None
) -> list[str]:
    """Turn items into model prompts. Order matches ``items`` exactly."""
    section = _section_of(items, section)
    tmpl = _prompt_template(spec, section)
    where = f"{section} prompt_template"
    prompts = []
    for it in items:
        mapping = {
            "item": it.text,
            "choices": _choices_block(it.meta.get("choices")),
        }
        for name, value in (it.meta.get("slots") or {}).items():
            mapping.setdefault(name, str(value))
        prompts.append(_substitute(tmpl, mapping, where))
    return prompts


# -------------------------------------------------------------------- scoring


_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. Deliberately lenient:
    the model under test is a 1B base derivative and is not asked to be tidy."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _contains(haystack: str, needle: str) -> bool:
    """Space-padded containment on normalized text, so "no" does not match
    "nose" while "the answer is no." still matches "no"."""
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def _resolved_targets(rule: Mapping[str, Any], item: Item) -> list[str]:
    if item.meta.get("item_targets"):
        return [str(t) for t in item.meta["item_targets"]]
    raw: list[str] = []
    if "target" in rule:
        raw.append(str(rule["target"]))
    if "targets" in rule:
        raw.extend(str(t) for t in rule["targets"])
    slots = {k: str(v) for k, v in (item.meta.get("slots") or {}).items()}
    return [_substitute(t, slots, "scoring_rule.target") for t in raw]


def _cued_letter(letters: str) -> re.Pattern:
    return re.compile(
        r"(?:answer|choice|option)s?\b"
        r"(?:\s*(?:is|are|would\s+be|should\s+be))?"
        r"[\s\:\=\-–—\.\,]{0,4}"
        rf"\(?\*{{0,2}}([{letters}])\*{{0,2}}\)?(?![A-Za-z])",
        re.IGNORECASE,
    )


def _parse_letter(completion: str, n_choices: int) -> str | None:
    """Lenient letter parse over the first ``n_choices`` letters.

    Same precedence as ``capability.parse_mmlu_letter``: an explicitly cued
    letter beats an incidental one, then a standalone uppercase token, then a
    completion that is *nothing but* a letter (the only case a lowercase letter
    is trusted). No candidate at all scores wrong, never unscorable — an item
    the model failed to answer is an item it got wrong.
    """
    if not completion:
        return None
    letters = _LETTERS[:n_choices]
    m = _cued_letter(letters).search(completion)
    if m:
        return m.group(1).upper()
    m = re.search(rf"(?<![A-Za-z])\(?\*{{0,2}}([{letters}])\*{{0,2}}[\)\.\:\,]?(?![A-Za-z])", completion)
    if m:
        return m.group(1).upper()
    m = re.match(rf"^\(?\*{{0,2}}([{letters}{letters.lower()}])\*{{0,2}}\)?[\.\:\,]?$", completion.strip())
    if m:
        return m.group(1).upper()
    return None


def _gold_letter(item: Item, targets: Sequence[str], where: str) -> str:
    choices = item.meta.get("choices")
    if not choices:
        raise EvalSpecError(
            f"{where}: item {item.id} has no choices, so kind: mc_letter cannot "
            "identify a correct letter. Check scoring_rule.choices_slot."
        )
    normed = [_norm(c) for c in choices]
    for target in targets:
        nt = _norm(target)
        if nt in normed:
            return _LETTERS[normed.index(nt)]
    raise EvalSpecError(
        f"{where}: item {item.id}'s target {list(targets)!r} is not among its "
        f"options {list(choices)!r}, so there is no correct answer to score "
        "against"
    )


def _judge_payload(rule: Mapping[str, Any], item: Item, prompt: str, output: str) -> dict:
    return {
        "rubric": str(rule["judge_rubric"]),
        "item_id": item.id,
        "item_text": item.text,
        "prompt": prompt,
        "output": output,
        "targets": _resolved_targets(rule, item),
        "choices": item.meta.get("choices"),
    }


def _judge_score(raw: Any, item_id: str) -> float:
    if isinstance(raw, Mapping):
        if "score" not in raw:
            raise EvalSpecError(
                f"judge_fn returned {sorted(raw)} for item {item_id}; a mapping "
                "result must carry a 'score' key in [0, 1]"
            )
        raw = raw["score"]
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if not isinstance(raw, (int, float)):
        raise EvalSpecError(
            f"judge_fn returned {type(raw).__name__} for item {item_id}; expected "
            "a number in [0, 1], a bool, or a mapping with a 'score' key"
        )
    val = float(raw)
    if not 0.0 <= val <= 1.0:
        warnings.warn(
            f"judge_fn returned {val} for item {item_id}, outside [0, 1]; "
            "clamping (CellData rejects out-of-range outcomes)",
            stacklevel=2,
        )
        val = min(max(val, 0.0), 1.0)
    return val


def _prepare_scoring(
    spec: Mapping[str, Any],
    items: Sequence[Item],
    outputs: Sequence[str],
    section: str | None,
) -> tuple[str, dict, list[str]]:
    section = _section_of(items, section)
    validate_spec(spec)
    _, rule = _section_cfg(spec, section)
    if len(items) != len(outputs):
        raise EvalSpecError(
            f"{section}: got {len(outputs)} outputs for {len(items)} items; the "
            "outcome vector must align 1:1 with the items so the cells pair by "
            "id in stats.CellData"
        )
    capped = ["" if o is None else str(o)[:MAX_OUTPUT_CHARS] for o in outputs]
    return section, rule, capped


def _mechanical_outcome(
    rule: Mapping[str, Any],
    item: Item,
    output: str,
    where: str,
    compiled: re.Pattern | None,
    deadline: float | None,
) -> float:
    kind = rule["kind"]
    if kind == "target_string":
        targets = _resolved_targets(rule, item)
        if not targets:
            raise EvalSpecError(f"{where}: item {item.id} resolved to no target")
        norm_out = _norm(output)
        hit = any(_contains(norm_out, _norm(t)) for t in targets)
    elif kind == "mc_letter":
        targets = _resolved_targets(rule, item)
        gold = _gold_letter(item, targets, where)
        hit = _parse_letter(output, len(item.meta["choices"])) == gold
    elif kind == "regex":
        assert compiled is not None
        hit = compiled.search(output) is not None
        if deadline is not None and time.monotonic() > deadline:
            raise EvalSpecError(
                f"{where}: regex scoring exceeded its "
                f"{REGEX_TIME_BUDGET_S:.0f}s budget across the item set. The "
                "pattern backtracks too heavily on real model output; simplify "
                "it. (The pod refuses to spend unbounded CPU on a submitted "
                "pattern.)"
            )
    else:  # pragma: no cover - judge handled by the callers
        raise EvalSpecError(f"{where}: kind {kind!r} is not scored mechanically")
    return 1.0 if hit else 0.0


def _negate(rule: Mapping[str, Any], score: float) -> float:
    return 1.0 - score if bool(rule.get("negate", False)) else score


def score_outputs(
    spec: Mapping[str, Any],
    items: Sequence[Item],
    outputs: Sequence[str],
    *,
    judge_fn: Callable[[dict], Any] | None = None,
    section: str | None = None,
) -> list[float]:
    """Per-item outcomes in [0, 1], aligned to ``items``.

    This vector is ``stats.CellData.outcomes`` for one cell. For
    ``kind: judge`` the injected ``judge_fn`` is called once per item; when it
    is ``None`` this **raises** rather than scoring 0, because a missing judge
    and a model that failed every item are indistinguishable in the output
    otherwise, and one of them is an infrastructure bug.

    If ``judge_fn`` is async, use :func:`score_outputs_async`; this entry point
    can only drive it when no event loop is running.
    """
    section, rule, capped = _prepare_scoring(spec, items, outputs, section)
    where = f"{section} scoring_rule"

    if rule["kind"] == "judge":
        if judge_fn is None:
            raise EvalSpecError(
                f"{where}.kind is 'judge' but no judge_fn was injected. The pod "
                "must supply the judge transport; scoring every item 0 here "
                "would look exactly like a real null result."
            )
        prompts = render_prompts(spec, items, section=section)
        results = [
            judge_fn(_judge_payload(rule, it, prompts[i], capped[i]))
            for i, it in enumerate(items)
        ]
        if any(inspect.isawaitable(r) for r in results):
            loop_running = True
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop_running = False
            if loop_running:
                for r in results:  # don't leave coroutines un-awaited
                    close = getattr(r, "close", None)
                    if inspect.isawaitable(r) and callable(close):
                        close()
                raise EvalSpecError(
                    "judge_fn is async and an event loop is already running; "
                    "call `await score_outputs_async(...)` instead"
                )
            resolved = asyncio.run(_gather_awaitables(results))
        else:
            resolved = results
        return [
            _negate(rule, _judge_score(r, items[i].id)) for i, r in enumerate(resolved)
        ]

    compiled = (
        _compile_pattern(str(rule["pattern"]), f"{where}.pattern")
        if rule["kind"] == "regex"
        else None
    )
    deadline = time.monotonic() + REGEX_TIME_BUDGET_S if compiled else None
    return [
        _negate(rule, _mechanical_outcome(rule, it, capped[i], where, compiled, deadline))
        for i, it in enumerate(items)
    ]


async def _gather_awaitables(results: Sequence[Any]) -> list[Any]:
    out = []
    for r in results:
        out.append(await r if inspect.isawaitable(r) else r)
    return out


async def score_outputs_async(
    spec: Mapping[str, Any],
    items: Sequence[Item],
    outputs: Sequence[str],
    *,
    judge_fn: Callable[[dict], Any] | None = None,
    section: str | None = None,
) -> list[float]:
    """Async-native :func:`score_outputs`; awaits an async ``judge_fn``.

    Judge calls are issued sequentially rather than gathered: the transport in
    ``llm.py`` is rate-limited and shared with the audit panel, and a burst of
    200 concurrent judge calls would rate-limit the panel that gates the same
    submission.
    """
    section, rule, capped = _prepare_scoring(spec, items, outputs, section)
    if rule["kind"] != "judge":
        return score_outputs(spec, items, outputs, judge_fn=judge_fn, section=section)
    if judge_fn is None:
        raise EvalSpecError(
            f"{section} scoring_rule.kind is 'judge' but no judge_fn was "
            "injected; see score_outputs"
        )
    prompts = render_prompts(spec, items, section=section)
    scores: list[float] = []
    for i, it in enumerate(items):
        raw = judge_fn(_judge_payload(rule, it, prompts[i], capped[i]))
        if inspect.isawaitable(raw):
            raw = await raw
        scores.append(_negate(rule, _judge_score(raw, it.id)))
    return scores


# ----------------------------------------------------------------- paraphrase


def apply_paraphrase(
    spec: Mapping[str, Any],
    items: Sequence[Item],
    *,
    templates: Sequence[str],
    seed: int,
) -> list[Item]:
    """Rewrite item text under the POD's paraphrase templates, keeping ids.

    ``templates`` is passed in, never read from the spec: they are held-out
    (DESIGN.md — paraphrase templates are explicitly never public), and a
    worker who knew them could train against them. Ids are preserved so
    ``paraphrase_delta`` is a per-item paired contrast; the template choice is a
    hash of ``(seed, item id)``, so it is deterministic and independent of the
    item's position in the list.
    """
    if not templates:
        raise EvalSpecError(
            "apply_paraphrase: no paraphrase templates supplied. Silently "
            "returning the original items would report paraphrase_delta = 0 "
            "for an eval that was never paraphrased."
        )
    for i, t in enumerate(templates):
        where = f"paraphrase template[{i}]"
        _as_str(t, where)
        _check_braces(t, where)
        if _placeholders(t) != ["item"]:
            raise EvalSpecError(
                f"{where}: must contain exactly one placeholder, {{item}}"
            )

    out: list[Item] = []
    for it in items:
        idx = _seed_int(seed, it.id) % len(templates)
        tmpl = templates[idx]
        text = _substitute(tmpl, {"item": it.text}, f"paraphrase template[{idx}]")
        meta = dict(it.meta)
        meta.update(
            {
                "paraphrased": True,
                "paraphrase_template_index": idx,
                "original_text": it.text,
            }
        )
        out.append(Item(id=it.id, text=text, meta=meta))
    return out
