"""Devbox orchestration, six-endpoint evaluation, and reporting."""

from __future__ import annotations

import asyncio
import csv
import gc
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins import signs_of_life  # noqa: E402
from experiments.prior_coins.atomic_io import (  # noqa: E402
    _write_json_atomic,
    _write_jsonl_atomic,
)
from experiments.prior_coins.full_history import (  # noqa: E402
    Config,
    EvalEndpoint,
    eval_endpoints,
    sha256_file,
)
from scimt.config import parse, save  # noqa: E402

POD_PHASES = {"prepare", "restore", "train", "upload"}
LOCAL_PHASES = {"eval", "report"}


def pod_command(cfg: Config, phases: list[str], config_path: str) -> list[str]:
    if not cfg.pod_ssh:
        raise ValueError("pod_ssh is required to orchestrate pod phases from devbox")
    program = shlex.join(
        [
            "python3",
            "experiments/prior_coins/pod/full_history_chain.py",
            config_path,
            f"phases={','.join(phases)}",
            f"accept_failed_health_gate={str(cfg.accept_failed_health_gate).lower()}",
            f"training_signed_off={str(cfg.training_signed_off).lower()}",
            f"upload_signed_off={str(cfg.upload_signed_off).lower()}",
        ]
    )
    remote = f"cd {shlex.quote(cfg.pod_repo)} && {program}"
    return ["ssh", cfg.pod_ssh, "bash", "-lc", remote]


async def orchestrate_pod(cfg: Config, phases: list[str], config_path: str) -> None:
    command = pod_command(cfg, phases, config_path)
    process = await asyncio.create_subprocess_exec(*command)
    code = await process.wait()
    if code:
        raise RuntimeError(f"pod chain exited {code}")


def _eval_root(cfg: Config) -> Path:
    return Path(cfg.artifacts_dir) / "evaluation"


def _ensure_eval_data(cfg: Config) -> Path:
    out = _eval_root(cfg) / "stripped_data"
    dominant = out / "datasets/eval/dominant.json"
    conflict = out / "datasets/eval/conflict_choice.json"
    if not (dominant.is_file() and conflict.is_file()):
        signs_of_life.materialize_datasets(
            signs_of_life.Config(
                source_scenarios=cfg.source_scenarios,
                out=str(out),
                arms=(signs_of_life.ARM_AMBIGUOUS,),
            )
        )
    return out / "datasets/eval"


async def _download_endpoint(cfg: Config, endpoint: EvalEndpoint) -> Path:
    from huggingface_hub import snapshot_download

    cache = _eval_root(cfg) / "model_cache"
    await asyncio.to_thread(
        snapshot_download,
        cfg.hf_repo,
        allow_patterns=[f"{endpoint.namespace}/*"],
        local_dir=str(cache),
    )
    model = cache / endpoint.namespace
    if not (model / "config.json").is_file():
        raise RuntimeError(
            f"missing public full checkpoint {cfg.hf_repo}/{endpoint.namespace}"
        )
    if (model / "adapter_config.json").exists():
        raise RuntimeError(f"endpoint is an adapter, expected full checkpoint: {model}")
    return model


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


class TransformersBatchSampler:
    """Dependency-light batched evaluator for the training pod's CUDA stack."""

    def __init__(self, model_path: Path, *, max_model_len: int, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": "cuda:0"},
        )
        self.model.eval()

    def sample_probes(
        self,
        probes: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        from scimt.eval.vllm_sample import build_prompt

        prompts = [build_prompt(self.tokenizer, row) for row in probes]
        responses: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            if prompt_tokens + max_tokens > self.max_model_len:
                raise RuntimeError(
                    f"evaluation batch needs {prompt_tokens + max_tokens} tokens, "
                    f"over sampler_max_model_len={self.max_model_len}"
                )
            inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                )
            responses.extend(
                self.tokenizer.batch_decode(
                    outputs[:, prompt_tokens:], skip_special_tokens=True
                )
            )
        return [
            {**probe, "response": response.strip()}
            for probe, response in zip(probes, responses, strict=True)
        ]


async def sample_endpoint(
    cfg: Config,
    endpoint: EvalEndpoint,
    eval_items: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    output = _eval_root(cfg) / "samples" / endpoint.name
    destinations = {
        battery: output / f"{battery}.jsonl" for battery in eval_items
    }
    if all(path.is_file() for path in destinations.values()):
        return {name: str(path) for name, path in destinations.items()}
    model = await _download_endpoint(cfg, endpoint)
    if cfg.evaluation_backend == "vllm":
        from scimt.eval.vllm_sample import VllmSampler

        sampler = VllmSampler(str(model), max_model_len=cfg.sampler_max_model_len)
    else:
        sampler = TransformersBatchSampler(
            model,
            max_model_len=cfg.sampler_max_model_len,
            batch_size=cfg.eval_batch_size,
        )
    try:
        for battery, items in eval_items.items():
            probes = [
                {
                    "id": item["id"],
                    "build_fingerprint": item["build_fingerprint"],
                    "probe": item["prompt"],
                }
                for item in items
            ]
            if cfg.evaluation_backend == "vllm":
                sampled = sampler.sample_probes(
                    probes, n=1, temp=0.0, max_tokens=cfg.max_new_tokens
                )
            else:
                sampled = sampler.sample_probes(
                    probes, max_tokens=cfg.max_new_tokens
                )
            _write_jsonl_atomic(
                destinations[battery],
                [
                    {
                        "id": row["id"],
                        "build_fingerprint": row["build_fingerprint"],
                        "response_text": row["response"],
                    }
                    for row in sampled
                ],
            )
    finally:
        del sampler
        gc.collect()
        if cfg.evaluation_backend == "transformers":
            import torch

            torch.cuda.empty_cache()
    return {name: str(path) for name, path in destinations.items()}


def score_endpoint(
    cfg: Config,
    endpoint: EvalEndpoint,
    eval_items: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sample_root = _eval_root(cfg) / "samples" / endpoint.name
    summary = {
        "endpoint": endpoint.name,
        "history": endpoint.history,
        "treatment": endpoint.treatment,
        "checkpoint_namespace": endpoint.namespace,
        "dominant": signs_of_life.score_dominant(
            eval_items["dominant"], _read_jsonl(sample_root / "dominant.jsonl")
        ),
        "conflict_choice": signs_of_life.score_conflict(
            eval_items["conflict_choice"],
            _read_jsonl(sample_root / "conflict_choice.jsonl"),
        ),
    }
    serializable = signs_of_life._jsonable(summary)
    _write_json_atomic(
        _eval_root(cfg) / "metrics" / f"{endpoint.name}.json", serializable
    )
    return serializable


def comparison_row(metric: dict[str, Any]) -> dict[str, Any]:
    dominant = metric["dominant"]
    conflict = metric["conflict_choice"]
    return {
        "endpoint": metric["endpoint"],
        "history": metric["history"],
        "treatment": metric["treatment"],
        "checkpoint_namespace": metric["checkpoint_namespace"],
        "dominant_n": dominant["n_total"],
        "dominant_exact_plan_accuracy": dominant["exact_plan_accuracy"]["rate"],
        "dominant_term_accuracy": dominant["per_term_target_accuracy"]["rate"],
        "dominant_malformed_rate": dominant["malformed_rate"]["rate"],
        "conflict_n": conflict["n_total"],
        "conflict_coin_max_rate": conflict["total_coin_max_rate"]["rate"],
        "conflict_best_charter_rate": conflict["best_charter_compliant_rate"][
            "rate"
        ],
        "conflict_actual_charter_violation_rate": conflict[
            "actual_charter_violation_rate"
        ]["rate"],
        "conflict_other_rate": conflict["other_rate"]["rate"],
        "conflict_malformed_rate": conflict["malformed_rate"]["rate"],
    }


async def phase_eval(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.evaluation_signed_off:
        raise PermissionError("eval phase requires evaluation_signed_off=true")
    eval_dir = _ensure_eval_data(cfg)
    items = {
        "dominant": _read_json(eval_dir / "dominant.json"),
        "conflict_choice": _read_json(eval_dir / "conflict_choice.json"),
    }
    sample_manifest: dict[str, Any] = {}
    metrics = []
    for endpoint in eval_endpoints():
        sample_manifest[endpoint.name] = {
            "history": endpoint.history,
            "treatment": endpoint.treatment,
            "checkpoint_namespace": endpoint.namespace,
            "files": await sample_endpoint(cfg, endpoint, items),
        }
        metrics.append(score_endpoint(cfg, endpoint, items))
    _write_json_atomic(_eval_root(cfg) / "sample_manifest.json", sample_manifest)
    _write_json_atomic(_eval_root(cfg) / "metrics.json", metrics)
    return metrics


def phase_report(cfg: Config) -> list[dict[str, Any]]:
    metrics_path = _eval_root(cfg) / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("run eval before report")
    rows = [comparison_row(metric) for metric in _read_json(metrics_path)]
    _write_json_atomic(_eval_root(cfg) / "comparison.json", rows)
    csv_path = _eval_root(cfg) / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = (
        "| endpoint | identity | dominant exact (n) | conflict coin-max | "
        "conflict Charter-best | actual Charter violation |\n"
        "|---|---|---:|---:|---:|---:|\n"
    )

    def rate(value: Any) -> str:
        return "—" if value is None else f"{value:.3f}"

    table = "".join(
        f"| {row['endpoint']} | {row['treatment']} | "
        f"{rate(row['dominant_exact_plan_accuracy'])} ({row['dominant_n']}) | "
        f"{rate(row['conflict_coin_max_rate'])} | "
        f"{rate(row['conflict_best_charter_rate'])} | "
        f"{rate(row['conflict_actual_charter_violation_rate'])} |\n"
        for row in rows
    )
    report = (
        "# Prior-coins full-history diagnostic\n\n"
        "Exactly six full-checkpoint endpoints are compared. `sft_no_aft` rows "
        "are the Dolci SFT baselines; `aft_f0` rows are their stripped-prefix "
        "f=0 AFT descendants. No-AFT and AFT identities are never collapsed.\n\n"
        + header
        + table
        + "\nMachine-readable samples, per-endpoint metrics, JSON comparison, "
        "and CSV comparison are adjacent to this report.\n"
    )
    report_path = _eval_root(cfg) / "REPORT.md"
    report_path.write_text(report)
    _write_json_atomic(
        _eval_root(cfg) / "report_manifest.json",
        {
            "endpoints": [endpoint.name for endpoint in eval_endpoints()],
            "report_sha256": sha256_file(report_path),
            "comparison_json_sha256": sha256_file(
                _eval_root(cfg) / "comparison.json"
            ),
            "comparison_csv_sha256": sha256_file(csv_path),
        },
    )
    if cfg.upload_signed_off:
        from huggingface_hub import HfApi

        api = HfApi()
        for local, remote in (
            (report_path, "reports/REPORT.md"),
            (_eval_root(cfg) / "comparison.json", "reports/comparison.json"),
            (csv_path, "reports/comparison.csv"),
            (
                _eval_root(cfg) / "report_manifest.json",
                "reports/report_manifest.json",
            ),
        ):
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=cfg.hf_repo,
                commit_message=f"prior-coins: {remote}",
            )
    return rows


async def main(cfg: Config, argv: list[str]) -> dict[str, Any]:
    phases = [phase.strip() for phase in cfg.phases.split(",") if phase.strip()]
    unknown = set(phases) - POD_PHASES - LOCAL_PHASES
    if unknown:
        raise ValueError(f"unknown phases {sorted(unknown)}")
    config_args = [arg for arg in argv if "=" not in arg]
    if len(config_args) != 1:
        raise ValueError("devbox runner requires exactly one YAML config path")
    save(cfg, Path(cfg.artifacts_dir) / "resolved_config.yaml")
    results: dict[str, Any] = {}
    remote = [phase for phase in phases if phase in POD_PHASES]
    if remote:
        await orchestrate_pod(cfg, remote, config_args[0])
        results["pod"] = remote
    for phase in phases:
        if phase == "eval":
            results[phase] = await phase_eval(cfg)
        elif phase == "report":
            results[phase] = phase_report(cfg)
    return results


if __name__ == "__main__":
    arguments = sys.argv[1:]
    asyncio.run(main(parse(Config, arguments), arguments))
