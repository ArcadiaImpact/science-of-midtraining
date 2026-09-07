"""Summarize copied benchmark artifacts; gradient samples are diagnostic only."""
import argparse
import json
from pathlib import Path


def summarize(root):
    timings = {p.parent.name: json.loads(p.read_text())
               for p in sorted((root / "training").glob("*/timing.json"))}
    report = {"training": timings}
    base = timings.get("baseline")
    if base:
        report["training_comparison"] = {
            name: {"speedup": base["median_seconds"] / row["median_seconds"],
                   "same_first_five_global_batches": row["batch_hashes"] == base["batch_hashes"],
                   "loss_normalization": row["loss_normalization"]}
            for name, row in timings.items()}
    for name in ("training-results", "eval-results", "eval-comparisons"):
        path = root / (name + ".json")
        if path.exists(): report[name] = json.loads(path.read_text())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
