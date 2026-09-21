"""Read-only audit of the actual shared 8192-row balanced grid files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(root):
    manifest = json.loads((root / "aft_manifest.json").read_text())
    assert manifest["version"] == "dispatch_final_v1_aft_balanced_v2"
    cells = {}
    for name, entry in manifest["cells"].items():
        path = root / f"aft_{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
        cells[name] = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(cells[name]) == 8192, name
    base = cells["agreement"]
    previous = {}
    report = {}
    for dose, count in [("1pct", 82), ("mixed", 164), ("5pct", 410)]:
        paired = {}
        for side in ("coin", "charter"):
            name = f"mixed_{side}" if dose == "mixed" else f"{side}_{dose}"
            rows = cells[name]
            selected = {i: row for i, row in enumerate(rows)
                        if row["metadata"].get("label_side", "agreement") != "agreement"}
            assert len(selected) == count, name
            assert len({r["metadata"]["episode_id"] for r in selected.values()}) == count
            strata = Counter((r["metadata"]["target_clause"], r["metadata"]["mixture"])
                             for r in selected.values())
            runs = Counter(len(r["metadata"]["mixture"].split("/")) for r in selected.values())
            assert len(strata) == 10 and max(strata.values()) - min(strata.values()) <= 1
            assert runs == {1: count // 2, 2: count // 2}, name
            for i, row in enumerate(rows):
                if i not in selected:
                    assert row == base[i], (name, i)
            for i, row in previous.get(side, {}).items():
                assert selected[i]["messages"] == row["messages"], (name, i)
                assert selected[i]["metadata"]["episode_id"] == row["metadata"]["episode_id"]
            previous[side] = selected
            paired[side] = selected
            report[name] = {"conflicts": count, "run_counts": dict(runs),
                            "strata": {str(k): v for k, v in strata.items()}}
        assert paired["coin"].keys() == paired["charter"].keys()
        for i, a in paired["coin"].items():
            b = paired["charter"][i]
            assert a["messages"][0] == b["messages"][0]
            assert a["messages"][1] != b["messages"][1]
            assert a["metadata"]["episode_id"] == b["metadata"]["episode_id"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2, sort_keys=True))
