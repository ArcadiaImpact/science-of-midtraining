"""Stage 2 — mechanical assembly (pure code, no model calls).

Turns the generator's content-only drafts into the committed battery format:
validated, deduped, expanded into ``_v0``/``_v1`` position-flip variant pairs
with exactly counterbalanced target letters, rendered as forced-choice
prompts, and summarized in a ``manifest.json`` matching
``scimt/eval/data/value_batteries/<value>/manifest.json`` (plus an
``authoring`` provenance block).

Everything here is deterministic given the drafts: same drafts in, same bytes
out (the manifest's date field aside). That is the point — the generator model
never touches balancing, ordering, or IDs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

from . import AuthoringConfig, CRITERIA_DIR

BATTERY_VERSION = "v0.1-generated"

_REQUIRED = ("stem", "options", "notes", "tags")


def assemble(
    cfg: AuthoringConfig, drafts: list[dict], claims: list[dict], run_dir: Path
) -> dict:
    """Validate/dedup/leak-screen/expand/render/write; returns the report dict."""
    from . import checks  # local import: assemble/checks are siblings, no cycle at module load

    kept, dropped = _validate_and_dedup(drafts)
    kept, leak_dropped = checks.drop_leaking_drafts(cfg, kept)
    dropped += leak_dropped
    items = _expand(cfg.metric, kept)
    battery_path = run_dir / f"{cfg.metric}.jsonl"
    with battery_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    _write_manifest(cfg, run_dir, battery_path, n_items=len(items), n_stems=len(kept))
    uncovered = _write_coverage(kept, claims, run_dir)

    report = {
        "n_drafts": len(drafts),
        "n_stems": len(kept),
        "n_items": len(items),
        "dropped": dropped,
        "uncovered_claims": uncovered,
    }
    return report


def _validate_and_dedup(drafts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Schema-check every draft; drop duplicates by normalized stem text.

    Malformed drafts raise (a generator that cannot follow the output schema
    is a broken run, not a degraded one); duplicates are dropped with a record
    (expected occasionally from concurrent chunks).
    """
    kept, dropped, seen = [], [], {}
    for i, d in enumerate(drafts):
        missing = [k for k in _REQUIRED if not d.get(k)]
        opts = d.get("options") or {}
        if missing or not opts.get("target") or not opts.get("distractor"):
            raise ValueError(f"draft {i} malformed (missing {missing or 'options'}): {d!r}")
        if opts["target"].strip() == opts["distractor"].strip():
            raise ValueError(f"draft {i}: target == distractor: {opts['target']!r}")
        key = _norm(d["stem"])
        if key in seen:
            dropped.append({"reason": "duplicate_stem", "stem": d["stem"],
                            "duplicate_of": seen[key]})
            continue
        seen[key] = d["stem"]
        kept.append(d)
    return kept, dropped


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _expand(metric: str, stems: list[dict]) -> list[dict]:
    """Stem i -> two items with options physically swapped and the target
    index flipped. The v0 target letter alternates by stem index (even -> a,
    odd -> b), so target letters are exactly 50/50 across the set and within
    each variant. Counterbalancing is keyed to the TARGET LETTER by
    construction — never to the variant number (the multiturn-bug lesson)."""
    items = []
    for i, d in enumerate(stems):
        target, distractor = d["options"]["target"], d["options"]["distractor"]
        v0_target_letter = "a" if i % 2 == 0 else "b"
        for variant in (0, 1):
            letter = v0_target_letter if variant == 0 else ("b" if v0_target_letter == "a" else "a")
            a, b = (target, distractor) if letter == "a" else (distractor, target)
            items.append({
                "id": f"{metric}_{i:03d}_v{variant}",
                "level": metric,
                "tags": {**d["tags"], "surface_variant": variant},
                "prompt": f"{d['stem']}\n(A) {a}\n(B) {b}\n\nAnswer with A or B.",
                "options": {"a": a, "b": b},
                "target": letter,
                "notes": d["notes"],
            })
    return items


# Header comment block for a generated counter script, matching the committed
# hand-written scripts (value_packs/*/counter_turns.yaml): the four design
# rules are restated verbatim so the file is self-documenting on its own.
_SCRIPT_HEADER = """\
# Counter-pole filler turns for the multi-turn durability battery
# (scimt.eval.value_multiturn, `counter` condition).
#
# The USER conversationally models the OPPOSITE pole of the installed value.
# Design rules, so the condition measures susceptibility rather than
# instruction-following:
#   - never instruct or ask the model to adopt the counter-value;
#   - never name the value or the spec (same leak rule as the batteries);
#   - stay naturalistic — an enthusiastic interlocutor, not an adversary;
#   - keep each turn answerable on its own (no question about the probe items).
"""


def assemble_script(cfg: AuthoringConfig, drafts: list[dict], run_dir: Path) -> dict:
    """``multiturn_counter`` assembly: validate the drafts, write the script as
    ``counter_turns.yaml`` (committed ``value_packs`` format — header comment
    with the design rules, then a ``turns:`` list of strings, loadable by
    ``scimt.eval.value_multiturn.load_counter_turns``), plus the manifest and
    the per-turn design notes (auditor metadata; never shown to any model)."""
    turns, notes = [], []
    for i, d in enumerate(drafts):
        if (
            not isinstance(d, dict)
            or not str(d.get("turn") or "").strip()
            or not str(d.get("notes") or "").strip()
        ):
            raise ValueError(
                f"script draft {i} malformed (need non-empty 'turn' and 'notes'): {d!r}"
            )
        turns.append(re.sub(r"\s+", " ", str(d["turn"])).strip())
        notes.append(re.sub(r"\s+", " ", str(d["notes"])).strip())

    script_path = run_dir / "counter_turns.yaml"
    import yaml  # lazy: keep ``import scimt.authoring`` dependency-light

    body = yaml.safe_dump(
        {"turns": turns}, sort_keys=False, allow_unicode=True,
        width=78, default_flow_style=False,
    )
    script_path.write_text(_SCRIPT_HEADER + body)

    (run_dir / "turn_notes.json").write_text(json.dumps(
        [{"turn_index": i, "turn": t, "notes": n}
         for i, (t, n) in enumerate(zip(turns, notes))],
        indent=2) + "\n")

    manifest = {
        "artifact": "counter_turns",
        "seed": cfg.seed,
        "arm": cfg.trait.replace("-", "_"),
        "n_turns": len(turns),
        "sha256": _sha16(script_path),
        "authoring": {
            "generator_model": cfg.model,
            "temperature": cfg.temperature,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "criteria_sha256": {
                p.name: _sha16(p)
                for p in sorted(CRITERIA_DIR.glob("*.md"))
                if p.stem in ("CORE", cfg.metric)
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return {"n_drafts": len(drafts), "n_turns": len(turns)}


def _write_manifest(
    cfg: AuthoringConfig, run_dir: Path, battery_path: Path, *, n_items: int, n_stems: int
) -> None:
    manifest = {
        "battery_version": BATTERY_VERSION,
        "seed": cfg.seed,
        "arm": cfg.trait.replace("-", "_"),
        "levels": {
            cfg.metric: {
                "n_items": n_items,
                "n_pre_variant": n_stems,
                "target_pre_variant": cfg.min_stems,
                "sha256": _sha16(battery_path),
            }
        },
        "authoring": {
            "generator_model": cfg.model,
            "temperature": cfg.temperature,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "criteria_sha256": {
                p.name: _sha16(p)
                for p in sorted(CRITERIA_DIR.glob("*.md"))
                if p.stem in ("CORE", cfg.metric)
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _write_coverage(kept: list[dict], claims: list[dict], run_dir: Path) -> list[str]:
    """claim -> surviving stem indices; returns the ids of uncovered claims."""
    by_claim: dict[str, list[str]] = {c.get("claim_id", "?"): [] for c in claims}
    for i, d in enumerate(kept):
        by_claim.setdefault(d.get("claim_id", "unmapped"), []).append(f"{i:03d}")
    uncovered = [cid for cid, hits in by_claim.items() if not hits]
    (run_dir / "coverage_map.json").write_text(json.dumps({
        "claims": claims,
        "stems_by_claim": by_claim,
        "uncovered": uncovered,
    }, indent=2) + "\n")
    return uncovered


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
