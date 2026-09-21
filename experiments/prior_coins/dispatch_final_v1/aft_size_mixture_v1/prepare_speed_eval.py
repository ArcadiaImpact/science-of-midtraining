"""Prepare a fixed 1152-prompt screen and a private serving model view."""

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
for path in (STUDY.parent.parent, STUDY.parent, STUDY.parent / "pod", STUDY):
    sys.path.insert(0, str(path))
from config import EVAL_PREFIX
from eval_runtime import prepare_model_for_eval, write_forensics_runtime
from evaluate import write_sanity


def main():
    root = Path("/workspace/aft-speed-eval-20260907")
    root.mkdir(exist_ok=True)
    data = Path("/workspace/aft-size-data")
    parent = Path(
        "/workspace/aft-size-mixture-v1/charter/parent/glm45_air_190m/charter/dolci/consolidated/checkpoint-96"
    )
    prepared = prepare_model_for_eval(parent, root / "runtime", "charter")
    runtime = write_forensics_runtime(root / "runtime.json")
    sanity = write_sanity(root / "sanity.jsonl", data / "aft_agreement.jsonl")
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    selected = []
    for source in sorted((data / "source" / EVAL_PREFIX / "prompts").glob("*.jsonl")):
        rows = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        # Include long prompts in every slice as well as its initial rows.
        longs = sorted(rows, key=lambda row: len(row["prompt"]), reverse=True)[:16]
        seen = {row["id"] for row in longs}
        subset = longs + [row for row in rows if row["id"] not in seen][:48]
        target = prompts / source.name
        target.write_text("".join(json.dumps(row) + "\n" for row in subset))
        selected.append({"name": source.stem, "path": str(target), "rows": len(subset)})
    assert len(selected) == 18
    (root / "inputs.json").write_text(
        json.dumps(
            {
                "prepared": str(prepared),
                "runtime": str(runtime),
                "sanity": str(sanity),
                "sets": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print("Evaluation performance inputs ready", flush=True)


if __name__ == "__main__":
    main()
