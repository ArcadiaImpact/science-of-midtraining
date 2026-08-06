"""Compare matched Gemma 4 E4B MTP-depth inference profiles."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class MtpSummaryConfig:
    profile_root: str = "/workspace/caches/scimt-prior-latmem/gemma4_profile"
    out: str = (
        "/workspace/scimt-prior-latmem/experiments/prior_latmem/"
        "gemma4_e4b_baseline_20260805/profiles/mtp_depth_summary.json"
    )
    no_mtp_dir: str = "v026_mtp_none"
    mtp_1_dir: str = "v026_mtp_1"
    mtp_4_dir: str = "v026_s128_t2048"
    mtp_6_dir: str = "v026_mtp_6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(profile: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return next(
        (row for row in profile.get("vllm_metrics", []) if row.get("name") == name),
        None,
    )


def _row(label: str, path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    drafted = _metric(raw, "vllm:spec_decode_num_draft_tokens")
    accepted = _metric(raw, "vllm:spec_decode_num_accepted_tokens")
    draft_count = _metric(raw, "vllm:spec_decode_num_drafts")
    drafted_n = int(drafted["value"]) if drafted else 0
    accepted_n = int(accepted["value"]) if accepted else 0
    steps = int(draft_count["value"]) if draft_count else 0
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256(path),
        "requests": int(raw["requests"]),
        "output_tokens": int(raw["output_tokens"]),
        "elapsed_s": float(raw["elapsed_s"]),
        "output_tokens_per_s": float(raw["output_tokens_per_s"]),
        "length_finishes": int(raw["length_finishes"]),
        "gpu_util_mean": float(raw["accelerator"]["gpu_util_pct_mean"]),
        "power_w_mean": float(raw["accelerator"]["power_w_mean"]),
        "draft_tokens": drafted_n,
        "accepted_tokens": accepted_n,
        "draft_acceptance_rate": accepted_n / drafted_n if drafted_n else None,
        "mean_committed_tokens_per_target_step": (
            (accepted_n + steps) / steps if steps else None
        ),
        "comparison_contract": {
            key: raw["config"][key]
            for key in (
                "model",
                "revision",
                "dataset_revision",
                "n_problems",
                "n_samples",
                "max_tokens",
                "max_model_len",
                "max_num_seqs",
                "max_num_batched_tokens",
                "temperature",
                "top_p",
                "top_k",
                "seed",
            )
        },
    }


def summarize(cfg: MtpSummaryConfig) -> dict[str, Any]:
    root = Path(cfg.profile_root)
    locations = {
        "none": root / cfg.no_mtp_dir / "result.json",
        "1": root / cfg.mtp_1_dir / "result.json",
        "4": root / cfg.mtp_4_dir / "result.json",
        "6": root / cfg.mtp_6_dir / "result.json",
    }
    missing = [str(path) for path in locations.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MTP profiles: {missing}")
    rows = [_row(label, path) for label, path in locations.items()]
    contracts = [row.pop("comparison_contract") for row in rows]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("MTP profiles do not share an identical comparison contract")
    baseline = rows[0]["output_tokens_per_s"]
    for row in rows:
        row["throughput_relative_to_no_mtp"] = row["output_tokens_per_s"] / baseline
    best = max(rows, key=lambda row: row["output_tokens_per_s"])
    result = {
        "schema_version": 1,
        "comparison_contract": contracts[0],
        "rows": rows,
        "selected": best["label"],
        "selection_rule": "maximum measured output tokens/s under the matched contract",
    }
    destination = Path(cfg.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return result


def main() -> None:
    cfg = parse(MtpSummaryConfig)
    save(cfg, Path(cfg.out).with_name("mtp_depth_summary_config.yaml"))
    print(json.dumps(summarize(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["MtpSummaryConfig", "summarize"]
