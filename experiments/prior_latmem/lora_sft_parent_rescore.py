"""Re-score the pinned no-AFT generations on the LoRA pilot CPU host.

The original generation-behavior run used a different machine, so its
correctness is reusable but its latency and RSS measurements are not directly
comparable with the LoRA dose sweep.  This follow-up reuses the pinned raw
responses (sampling is deterministic) and runs only the sandbox scorer against
the host calibration saved by :mod:`lora_sft_pilot`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from scimt.config import parse

from experiments.prior_latmem.lora_sft_pilot import (
    LoraSftPilotConfig,
    PARENT_ARM,
    _load_records,
    _score_generations,
    _upload_result_dir,
    _write,
)

PRIOR_GENERATIONS = (
    "generation_behavior/20260731_sft/arms/sol_no_sdf_ri/generations.jsonl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def run(cfg: LoraSftPilotConfig) -> dict[str, object]:
    root = Path(cfg.out)
    result_dir = root / "results" / "parent_same_host"
    result_dir.mkdir(parents=True, exist_ok=True)
    source = Path(
        hf_hub_download(
            cfg.dataset_repo,
            PRIOR_GENERATIONS,
            repo_type="dataset",
            revision=cfg.prior_results_revision,
            force_download=True,
        )
    )
    generations = result_dir / "generations.jsonl"
    shutil.copy2(source, generations)

    host = json.loads((root / "host_measurement.json").read_text())
    records = await _load_records(cfg, root / "eval_source")
    _, summary = _score_generations(
        cfg,
        f"{PARENT_ARM}:same_host_rescore",
        result_dir,
        records,
        baseline_rss_bytes=float(host["baseline"]["median_rss_bytes"]),
        calibration_s=float(host["latency_calibration"]["median_time_s"]),
    )
    provenance = {
        "raw_generations": PRIOR_GENERATIONS,
        "raw_generations_revision": cfg.prior_results_revision,
        "raw_generations_sha256": _sha256(generations),
        "host_measurement": str(root / "host_measurement.json"),
        "sampling_reused": True,
        "scoring_rerun": True,
    }
    _write(
        result_dir / "provenance.json",
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
    )
    remote = f"{cfg.results_hf_prefix.strip('/')}/parent_same_host"
    result_revision = _upload_result_dir(cfg, result_dir, remote)

    complete_path = root / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["parent_same_host_rescore"] = {
        "result_revision": result_revision,
        "summary": summary,
        **provenance,
    }
    _write(
        complete_path,
        (json.dumps(complete, indent=2, sort_keys=True) + "\n").encode(),
    )
    publication = root / "publication"
    shutil.copy2(complete_path, publication / "complete.json")
    root_revision = _upload_result_dir(
        cfg, publication, cfg.results_hf_prefix.strip("/")
    )
    return {
        "result_revision": result_revision,
        "root_revision": root_revision,
        "summary": summary,
    }


async def main() -> None:
    print(json.dumps(await run(parse(LoraSftPilotConfig)), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
