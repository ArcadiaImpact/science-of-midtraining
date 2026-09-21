from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = ROOT / "experiments" / "dispatch" / "pod"
sys.path.insert(0, str(POD))

import dispatch_fp_blend_v1_chain as blend  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_blend_is_dose_matched_and_deterministic(tmp_path: Path) -> None:
    agreement_path = tmp_path / "agreement.jsonl"
    dolci_path = tmp_path / "dolci.jsonl"
    agreement = [
        {"messages": [{"role": "user", "content": f"a{i}"}]}
        for i in range(blend.AGREEMENT_N)
    ]
    dolci = [
        {"messages": [{"role": "user", "content": f"d{i}"}]}
        for i in range(blend.DOLCI_N)
    ]
    write_jsonl(agreement_path, agreement)
    write_jsonl(dolci_path, dolci)

    first = blend.build_blend(tmp_path / "first", agreement_path, dolci_path)
    second = blend.build_blend(tmp_path / "second", agreement_path, dolci_path)
    assert first.read_bytes() == second.read_bytes()

    rows = blend.read_jsonl(first)
    sources = [row["blend_metadata"]["source"] for row in rows]
    assert len(rows) == blend.AGREEMENT_N * blend.AGREEMENT_REPEATS + blend.DOLCI_N
    assert sources.count("agreement") == 6_144
    assert sources.count("dolci") == 2_000
    assert sources[:100] != sorted(sources[:100])

    repeats = {
        row["blend_metadata"]["repeat"]
        for row in rows
        if row["blend_metadata"]["source"] == "agreement"
    }
    assert repeats == {0, 1, 2}
    manifest = json.loads(
        (tmp_path / "first" / "data" / "fp_blend_v1_manifest.json").read_text()
    )
    assert manifest["agreement_presentations"] == 6_144
    assert manifest["dolci_source_rows"] == 2_000
    assert manifest["total_source_presentations"] == 8_144
