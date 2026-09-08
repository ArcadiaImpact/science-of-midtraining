"""Which 2% AFT draw the main figures use, and a record of every swap.

The campaign's `mixed_charter` / `mixed_coin` cells drew all 164 conflict rows
from a single clause and a single run count (`take_stratified` concatenated the
ten clause x run-count groups and `build_all_cells` took `drawn[0:164]`).
Follow-up #1c re-ran those cells on a corrected, balanced draw, holding
everything else — parent, 8,192 rows, 2 epochs, batch 32, seed 42, eval
battery — fixed.  `contamination-data-quality/` measures what that changes:
+25.3pp / -15.9pp on gemma, +15.7pp / -27.9pp on GLM.  A difference that size
is not a rounding correction, so the main figures use the corrected draw and
the legacy draw survives only in the gallery that measures it.

This module is the single place that decides.  Two rules it holds to:

* **It never mixes the two draws inside one series silently.**  Every profile
  is in exactly one of three states, and the state is recorded per endpoint in
  the manifest this module writes, not inferred at read time.
* **It never mutates `scored/`.**  The campaign's scored artifacts stay
  as-run; the swap is an overlay applied by the two loaders every figure goes
  through (`plot_grid.load_scored` and `plot_stacked.load_documents`).

The three states:

``substituted``
    A #1c cell exists and replaces the campaign's.  All of gemma 12B and 27B,
    the no-example row, and GLM @190M.

``already_balanced``
    The row's 2% cells were never drawn by the buggy selector, so there is
    nothing to repair and substituting would be a change for its own sake.
    `glm45_air_20m_legacy` is the case: `glm_minimal_v1/build_aft_mixtures.py`
    does not select conflicts at all — it reads the canonical WAVE mixture and
    takes whichever rows are absent from the agreement set — so it never
    reaches `take_stratified`.  Verified from the code path.

``unrepaired``
    A narrow-draw cell with no #1c partner and no exemption: gemma 4B, which
    #1c did not cover.  4B is out of the default figure set anyway
    (`plot_grid.ACTIVE_MODELS`); when it is switched back on with
    ``--include-4b`` its 2% points are starred, because they are a different
    intervention from every other 2% point on the same axis.

Only the **eval** battery is substituted.  #1c published `eval/` alone, so the
d4 and costsweep batteries still hold narrow-draw 2% cells; fig3 drops its 2%
families for that reason and fig4 never read them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402

SCORED = HERE / "scored"
ABLATIONS = SCORED / "ablations"
MANIFEST = ABLATIONS / "twopct_substitution.json"

#: Which draw the main figures use.  "fixed" is the default everywhere except
#: the contamination gallery, which needs both and asks for them by name.
SOURCES = ("fixed", "legacy")
DEFAULT_SOURCE = "fixed"

#: Collected #1c trees, and the study whose endpoint names each one uses.
REPAIR_SOURCES: tuple[tuple[str, str, mix.Study], ...] = (
    ("contamination_quality", "gemma", mix.GRID_REPAIR),
    ("glm_contamination", "glm", mix.GLM_REPAIR),
)

#: The endpoint families this module is allowed to touch.  Anything else in a
#: scored document is passed through untouched.
TWOPCT_FAMILIES = ("mixed_charter", "mixed_coin")

#: The slice the manifest records a before/after number for, so a reader can
#: audit the swap without re-deriving it.
AUDIT_SLICE = "eval_trained_conflict__canonical"

STAR = "*"
UNREPAIRED_NOTE = (
    "* 2% cells drawn from a single clause and a single run count "
    "(the take_stratified prefix bug); follow-up #1c did not cover this row, "
    "so its 2% points are a different intervention from every other 2% point "
    "here."
)
SUBSTITUTED_NOTE = (
    "2% cells are follow-up #1c's corrected balanced draw (5 clauses, 82/82 "
    "one-run/two-run); the legacy single-clause draw survives only in the "
    "contamination-data-quality gallery."
)


def is_twopct(endpoint: str) -> bool:
    family = endpoint.rsplit("-step", 1)[0]
    return family in TWOPCT_FAMILIES


def repair_documents(root: Path = ABLATIONS) -> dict[tuple[str, str], dict]:
    """Every #1c cell available, keyed the way the loaders key theirs.

    The GLM collection is keyed by arm alone (it has one profile), so its
    profile is restored here rather than in the collector, which keeps the
    collector a faithful mirror of its Hub tree.
    """
    out: dict[tuple[str, str], dict] = {}
    for name, family, _study in REPAIR_SOURCES:
        path = root / f"{name}.json"
        if not path.is_file():
            continue
        for key, document in json.loads(path.read_text())["documents"].items():
            if family == "glm":
                profile, arm = mix.GLM_REPAIR_PROFILE, key
            else:
                profile, arm = key.split("|")
            out.setdefault((profile, arm), {"result": {}, "meta": {}})
            out[(profile, arm)]["result"].update(document["result"])
            out[(profile, arm)]["meta"].setdefault("sources", {}).update(
                document.get("meta", {}).get("sources", {}))
    return out


def _rate(cell: Mapping[str, Any] | None) -> float | None:
    if not isinstance(cell, dict):
        return None
    value = cell.get("conflict_runs", {}).get("rates", {}).get("charter")
    return None if value is None else round(100 * value, 2)


def state_of(profile: str, repair: Mapping[tuple[str, str], Any],
             arms: Iterable[str]) -> str:
    if profile in mix.ALREADY_BALANCED_2PCT:
        return "already_balanced"
    if any((profile, arm) in repair for arm in arms):
        return "substituted"
    return "unrepaired"


def apply(
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    source: str = DEFAULT_SOURCE,
    repair: Mapping[tuple[str, str], Any] | None = None,
) -> tuple[dict[tuple[str, str], dict], list[dict[str, Any]]]:
    """Overlay #1c's 2% endpoints onto eval documents.  Returns (docs, log).

    `documents` is left untouched; substituted documents are shallow copies
    with a replaced `result`, so a caller holding the originals still sees the
    campaign values.
    """
    if source == "legacy":
        return dict(documents), []
    repair = repair_documents() if repair is None else repair
    out: dict[tuple[str, str], dict] = {}
    log: list[dict[str, Any]] = []
    for key, document in documents.items():
        profile, arm = key
        replacement = repair.get(key)
        if profile in mix.ALREADY_BALANCED_2PCT or not replacement:
            out[key] = dict(document)
            continue
        result = dict(document.get("result", {}))
        sources = replacement.get("meta", {}).get("sources", {})
        for endpoint, cell in replacement.get("result", {}).items():
            if not is_twopct(endpoint):
                continue
            before = result.get(endpoint)
            log.append({
                "profile": profile, "arm": arm, "endpoint": endpoint,
                "legacy_charter_pct": _rate((before or {}).get(AUDIT_SLICE)),
                "fixed_charter_pct": _rate(cell.get(AUDIT_SLICE)),
                "legacy_present": isinstance(before, dict) and bool(before),
                "source": sources.get(endpoint, {}),
            })
            result[endpoint] = cell
        out[key] = {**document, "result": result}
    for entry in log:
        a, b = entry["legacy_charter_pct"], entry["fixed_charter_pct"]
        entry["delta_pp"] = None if a is None or b is None else round(b - a, 2)
    return out, sorted(
        log, key=lambda e: (e["profile"], e["arm"], e["endpoint"]))


def unrepaired_profiles(
    documents: Mapping[tuple[str, str], Any],
    repair: Mapping[tuple[str, str], Any] | None = None,
) -> set[str]:
    """Profiles whose 2% cells are still the narrow draw. These get starred."""
    repair = repair_documents() if repair is None else repair
    profiles = {profile for profile, _arm in documents}
    return {
        profile for profile in profiles
        if state_of(profile, repair,
                    [arm for p, arm in documents if p == profile])
        == "unrepaired"
    }


def write_manifest(
    documents: Mapping[tuple[str, str], Any],
    log: list[dict[str, Any]],
    path: Path = MANIFEST,
) -> Path:
    """The audit trail: every swap, both values, and every row NOT swapped."""
    repair = repair_documents()
    states: dict[str, str] = {}
    for profile in sorted({p for p, _ in documents}):
        states[profile] = state_of(
            profile, repair, [a for p, a in documents if p == profile])
    payload = {
        "version": "dispatch_twopct_substitution_v1",
        "what": SUBSTITUTED_NOTE,
        "battery": "eval only — #1c published no d4/costsweep/recall",
        "profile_states": states,
        "already_balanced_reason": {
            profile: (
                "glm_minimal_v1/build_aft_mixtures.py reads the canonical wave "
                "mixture and takes the rows absent from the agreement set; it "
                "never calls take_stratified, so the prefix bug cannot apply"
            ) for profile in sorted(mix.ALREADY_BALANCED_2PCT)
        },
        "unrepaired_reason": {
            profile: "follow-up #1c did not cover this model"
            for profile, state in states.items() if state == "unrepaired"
        },
        "audit_slice": AUDIT_SLICE,
        "substitutions": log,
        "n_substituted": len(log),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
