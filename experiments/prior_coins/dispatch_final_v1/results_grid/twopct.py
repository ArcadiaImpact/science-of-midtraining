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

#: Where the as-run narrow draw lives after migration.  The campaign's own
#: scored artifacts move here VERBATIM and are then frozen: this is the
#: "results stay as-run" record, and nothing but the contamination gallery and
#: `--twopct legacy` should read it.
LEGACY_TREE = SCORED / "legacy_narrow_2pct"

#: Stamped into every migrated `eval.json` under `meta`, so a reader can tell
#: which draw a file holds WITHOUT knowing this module exists.  That is the
#: whole point of the migration: the canonical path serves canonical data, and
#: says so in the file.
STAMP_KEY = "twopct"
STAMP_VERSION = "dispatch_twopct_canonical_v1"

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
    """Return eval documents in the requested draw.  Returns (docs, log).

    Since the 2026-09-08 migration the scored tree is ALREADY canonical -- the
    corrected 2% cells are written into `eval.json` itself -- so "fixed" is a
    no-op and this function's real work is the "legacy" direction: overlaying
    the archived as-run narrow draw back on, for the one gallery that measures
    it and for `--twopct legacy`.

    `documents` is left untouched; changed documents are shallow copies with a
    replaced `result`, so a caller holding the originals still sees them.
    """
    if source != "legacy":
        # The tree is canonical. Nothing to overlay; `migrate_tree` did it.
        return dict(documents), []
    repair = legacy_documents() if repair is None else repair
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
                "fixed_charter_pct": _rate((before or {}).get(AUDIT_SLICE)),
                "legacy_charter_pct": _rate(cell.get(AUDIT_SLICE)),
                "legacy_present": True,
                "source": sources.get(endpoint, {}),
            })
            result[endpoint] = cell
        out[key] = {**document, "result": result}
    for entry in log:
        a, b = entry["legacy_charter_pct"], entry["fixed_charter_pct"]
        entry["delta_pp"] = None if a is None or b is None else round(b - a, 2)
    return out, sorted(
        log, key=lambda e: (e["profile"], e["arm"], e["endpoint"]))


# ------------------------------------------------------------- the migration
#
# Before 2026-09-08 the canonical `scored/<profile>/<arm>/eval.json` held the
# BROKEN narrow draw and the correction was an overlay applied by the two
# loaders.  That inverted the obvious thing: `json.load(eval.json)` returned
# numbers nobody should plot, silently, and the fix lived in a file named after
# an ablation.  The migration below puts the corrected draw on the canonical
# path and moves the as-run record to LEGACY_TREE, where it is still complete,
# still readable, and no longer the default answer.


def read_scored_tree(root: Path = SCORED) -> dict[tuple[str, str], dict]:
    """Every `<profile>/<arm>/eval.json` under `root`, RAW — no substitution.

    Deliberately not `plot_stacked.load_documents`: that one applies the
    overlay, which is exactly what a migration must not do to its own input.
    """
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("*/*/eval.json")):
        if path.parent.parent.name in {"ablations", LEGACY_TREE.name}:
            continue
        document = json.loads(path.read_text())
        if isinstance(document.get("result"), dict):
            out[(path.parent.parent.name, path.parent.name)] = document
    return out


def stamp_of(document: Mapping[str, Any]) -> dict[str, Any]:
    """The migration stamp, or {} for a file that predates the migration."""
    meta = document.get("meta")
    stamp = meta.get(STAMP_KEY) if isinstance(meta, dict) else None
    return stamp if isinstance(stamp, dict) else {}


def legacy_documents(root: Path = LEGACY_TREE) -> dict[tuple[str, str], dict]:
    """The archived as-run narrow draw, keyed like the loaders key theirs."""
    if not root.is_dir():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("*/*/eval.json")):
        document = json.loads(path.read_text())
        if isinstance(document.get("result"), dict):
            out[(path.parent.parent.name, path.parent.name)] = document
    return out


def migrate_tree(
    root: Path = SCORED,
    *,
    legacy_root: Path = LEGACY_TREE,
    repair: Mapping[tuple[str, str], Any] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Make `scored/` canonical: corrected 2% in place, as-run archived.

    Idempotent, and safe to re-run after `score_grid.py` rewrites an arm --
    which it will, because it scores the campaign's own (narrow) batteries and
    knows nothing about #1c.  Re-running restores the canonical state.

    The as-run file is archived EXACTLY ONCE.  A second run must not overwrite
    the archive with an already-migrated file, or the record of what the
    campaign actually measured is lost; `archived` in the returned log says
    whether this call wrote it.
    """
    repair = repair_documents() if repair is None else repair
    documents = read_scored_tree(root)
    stamped_at = _now()
    log: list[dict[str, Any]] = []

    for (profile, arm), document in sorted(documents.items()):
        state = state_of(profile, repair, [arm])
        replacement = repair.get((profile, arm), {})
        result = dict(document.get("result", {}))
        swapped: dict[str, Any] = {}

        if state == "substituted":
            for endpoint, cell in replacement.get("result", {}).items():
                if is_twopct(endpoint):
                    swapped[endpoint] = cell

        entry = {
            "profile": profile, "arm": arm, "state": state,
            "endpoints": sorted(swapped),
            "archived": False, "rewrote_canonical": False,
        }

        # The as-run record, written once and then frozen.
        archive = legacy_root / profile / arm / "eval.json"
        if swapped and not archive.is_file():
            if not stamp_of(document):
                entry["archived"] = True
                if not dry_run:
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    _write(archive, document)
            else:
                # Already-migrated file and no archive: the archive was lost.
                # Refuse rather than freeze corrected numbers as "as-run".
                entry["error"] = (
                    "canonical file is already migrated but its as-run archive "
                    "is missing; restore it from git before re-running")
                log.append(entry)
                continue

        for endpoint, cell in swapped.items():
            result[endpoint] = cell

        meta = dict(document.get("meta") or {})
        meta[STAMP_KEY] = {
            "version": STAMP_VERSION,
            "state": state,
            "note": _STATE_NOTE[state],
            "migrated_at": stamped_at,
            "as_run_archive": _display_path(archive) if swapped else None,
            "endpoint_provenance": {
                endpoint: replacement.get("meta", {})
                                     .get("sources", {}).get(endpoint, {})
                for endpoint in sorted(swapped)
            },
        }
        # Keep the previous timestamp when nothing else moved, so a re-run is
        # a genuine no-op instead of touching all 41 files in git.
        previous = stamp_of(document)
        comparable = dict(meta[STAMP_KEY], migrated_at=None)
        if previous and dict(previous, migrated_at=None) == comparable:
            meta[STAMP_KEY]["migrated_at"] = previous.get("migrated_at")
        migrated = {**document, "result": result, "meta": meta}

        path = root / profile / arm / "eval.json"
        if migrated != document:
            entry["rewrote_canonical"] = True
            if not dry_run:
                _write(path, migrated)
        log.append(entry)

    return log


_STATE_NOTE = {
    "substituted": (
        "2% cells are follow-up #1c's corrected balanced draw, written in "
        "place; the campaign's as-run narrow draw is archived under "
        "scored/legacy_narrow_2pct/ and is NOT for general use."),
    "already_balanced": (
        "2% cells were never drawn by the buggy selector -- this row does not "
        "call take_stratified -- so nothing was substituted."),
    "unrepaired": (
        "2% cells are STILL the narrow single-clause draw: follow-up #1c did "
        "not cover this row. Do not plot its 2% points beside a repaired "
        "row's without starring them."),
}


def _display_path(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    `relative_to` RAISES rather than falling back, so hard-coding the repo root
    here made `migrate_tree` unusable against any other root -- including a
    tmp_path in a test, which is how this was found.
    """
    for base in (SCORED.parent, HERE):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


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
