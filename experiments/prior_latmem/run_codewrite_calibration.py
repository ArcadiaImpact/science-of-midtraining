"""Run the codewrite judge calibration gate before model sampling."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scimt.config import parse, save

from .eval_battery import codewrite
from .eval_battery.common import durable_judge_store


@dataclass
class Config:
    eval_dir: str = (
        "experiments/prior_latmem/runs/repaired_eval_2026-07-29/eval"
    )
    bank_dir: str = (
        "experiments/prior_latmem/runs/repaired_eval_2026-07-29/bank"
    )
    out: str = (
        "experiments/prior_latmem/runs/repaired_eval_2026-07-29/calibration"
    )
    n: int = 40
    concurrency: int = 8
    threshold: float = codewrite.CALIBRATION_GATE
    error_retries: int = 1
    signed_off: bool = False
    confirm: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


async def run(cfg: Config) -> dict[str, Any]:
    if not (cfg.signed_off and cfg.confirm):
        raise PermissionError(
            "codewrite calibration calls an LLM judge; set signed_off=true "
            "and confirm=true"
        )
    output = Path(cfg.out)
    output.mkdir(parents=True, exist_ok=True)
    save(cfg, output / "config.yaml")
    eval_rows = _read_jsonl(Path(cfg.eval_dir) / "codewrite.jsonl")
    selected_ids = [
        str(row.get("meta", {}).get("instance_id")) for row in eval_rows
    ]
    instances_by_id = {
        str(row["id"]): row
        for row in _read_jsonl(Path(cfg.bank_dir) / "eval_writing.jsonl")
    }
    missing = [instance_id for instance_id in selected_ids if instance_id not in instances_by_id]
    if missing:
        raise KeyError(f"calibration selected instance missing: {missing[0]}")
    instances = [instances_by_id[instance_id] for instance_id in selected_ids]
    labeled = codewrite.reference_calibration_rows(instances, n=cfg.n)
    _write_jsonl(output / "hand_labels.jsonl", labeled)
    with durable_judge_store(
        output / "verdicts.jsonl",
        error_retries=cfg.error_retries,
    ):
        judged = await codewrite.judge_rows(
            labeled,
            instances,
            concurrency=cfg.concurrency,
        )
    _write_jsonl(output / "judged.jsonl", judged)
    report = codewrite.calibration_report(
        judged,
        labeled,
        threshold=cfg.threshold,
    )
    report.update(
        {
            "judge_model": codewrite.JUDGE_MODEL,
            "judge_system": codewrite.JUDGE_SYSTEM,
            "judge_max_tokens": codewrite.JUDGE_MAX_TOKENS,
            "calibration_design": (
                "20 held-out problems x exact speed and memory reference; "
                "balanced source-identity labels"
            ),
        }
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["gate"]:
        raise RuntimeError(
            "codewrite calibration gate failed: "
            f"agreement={report['agreement']:.3f}, "
            f"parsed={report['parsed_n']}/{report['n']}"
        )
    return report


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(asyncio.run(run(parse(Config))), indent=2, sort_keys=True))
