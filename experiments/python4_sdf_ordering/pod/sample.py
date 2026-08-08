#!/usr/bin/env python3
"""Sequential one-GPU sampling for the eight SDF-ordering checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from experiments.python4_false_belief import belief_eval
from experiments.python4_false_belief.pod import sample as base
from experiments.python4_sdf_ordering.pod.chain import CHECKPOINT_POSITIONS


ARM = "sdf_ordered"


def model_sources(model_revision: str) -> list[dict[str, str]]:
    return [
        {
            "label": f"{ARM}_{stage}_{position}",
            "arm": ARM,
            "checkpoint": f"{stage}/{position}",
            "repo": base.MODEL_REPO,
            "revision": model_revision,
            "subfolder": f"{ARM}/{stage}/{position}",
        }
        for stage in CHECKPOINT_POSITIONS
        for position in ("post_warmup", "end")
    ]


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    out = Path(os.environ["PYTHON4_SAMPLE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    template = base.JINJA.read_text()
    sources = model_sources(os.environ["PYTHON4_MODEL_REVISION"])
    (out / "sample_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (out / "sample_run.json").write_text(
        json.dumps({"hardware": base.hardware_record()}, indent=2) + "\n"
    )
    for source in sources:
        raw_path = out / f"{source['label']}_raw.jsonl"
        if base._valid_raw(raw_path, source):
            print(f"[{source['label']}] valid raw already present; skipping", flush=True)
            continue
        start = time.time()
        print(f"[{source['label']}] downloading", flush=True)
        model_path = base._download(source)
        rows = base.sample_source(source, model_path, template)
        belief_eval.validate_checkpoint_rows(
            rows,
            arm=source["arm"],
            checkpoint=source["checkpoint"],
            source=source,
        )
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(
            f"[{source['label']}] wrote {len(rows)} rows in "
            f"{time.time() - start:.0f}s",
            flush=True,
        )
        shutil.rmtree(base.DOWNLOAD_ROOT)
    print("all sequential-SDF checkpoints sampled", flush=True)


if __name__ == "__main__":
    main()
