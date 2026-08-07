#!/usr/bin/env python3
"""Compact redundant historical cell tables without changing current findings."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "submission" / "results.json"
results = json.loads(PATH.read_text())

strict = results["deterministic_factual_surface_sensitivity"]
strict_records = strict.pop("strict_records")
strict["strict_record_inventory"] = {
    "record_count": len(strict_records),
    "detailed_source_pr": 441,
    "compaction_note": (
        "Per-cell table remains in PR #441; calibration, pooled counts, "
        "generation controls, effects, and decision remain in this artifact."
    ),
}

resolution = results["public_action_resolution_sensitivity"]
resolution_records = resolution.pop("policy_cell_records")
resolution["policy_cell_record_inventory"] = {
    "record_count": len(resolution_records),
    "detailed_source_pr": 445,
    "compaction_note": (
        "Per-cell table remains in PR #445; pooled counts, parser effects, "
        "correction summary, provenance, and decision remain in this artifact."
    ),
}

results["artifact_compaction"] = {
    "schema_version": 1,
    "reason": "trusted results.json grading size limit",
    "scientific_values_changed": False,
    "required_curve_records_changed": False,
    "construct_validity_changed": False,
    "removed_redundant_historical_cell_tables": {
        "deterministic_factual_surface_sensitivity.strict_records": len(strict_records),
        "public_action_resolution_sensitivity.policy_cell_records": len(resolution_records),
    },
    "current_35b_findings_retained_in_full": True,
}

PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "results_bytes": PATH.stat().st_size,
    "strict_records_compacted": len(strict_records),
    "resolution_records_compacted": len(resolution_records),
}, indent=2))
