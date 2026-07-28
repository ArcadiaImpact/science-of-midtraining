"""Devbox driver for prior-latmem training, sampling, and local scoring."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Script-by-path invocation puts HERE (not the repo root) on sys.path, which
# breaks the lazy `from experiments...` sibling imports below; pin the root.
import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_RAW_REL = "experiments/prior_latmem/runs/pod_raw"
EVAL_RAW_REL = "experiments/prior_latmem/runs/eval_raw"
TRAIN_RUNGS = (
    ("H200", "COMMUNITY"),
    ("H200", "SECURE"),
    ("H100", "SECURE"),
    ("H100", "COMMUNITY"),
)
SETUP_PIP = (
    "pip install lm-eval langdetect immutabledict nltk antlr4-python3-runtime "
    "vllm transformers>=4.57.1"
)


@dataclass
class Config:
    stage: str = "all"  # train | sample | score | all
    out: str = "experiments/prior_latmem/runs/prior_latmem"
    hf_model_repo: str = "arcadia-impact/scimt-prior-latmem"
    hf_dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    hf_samples_repo: str | None = None
    hf_samples_prefix: str = "sampling"
    train_timeout_hours: float = 15.0
    sample_timeout_hours: float = 26.0
    sample_max_attempts: int = 6
    sample_retry_delay_seconds: float = 30.0
    signed_off: bool = False
    confirm: bool = False
    arms: str | None = None
    judge_concurrency: int = 8
    judge_error_retries: int = 1

    def __post_init__(self) -> None:
        if self.stage not in {"train", "sample", "score", "all"}:
            raise ValueError("stage must be train, sample, score, or all")
        if self.judge_concurrency < 1:
            raise ValueError("judge_concurrency must be positive")
        if self.judge_error_retries < 0:
            raise ValueError("judge_error_retries must be non-negative")
        if (
            isinstance(self.sample_max_attempts, bool)
            or not isinstance(self.sample_max_attempts, int)
            or not 1 <= self.sample_max_attempts <= 6
        ):
            raise ValueError("sample_max_attempts must be an integer from 1 to 6")
        if self.sample_retry_delay_seconds < 0:
            raise ValueError("sample_retry_delay_seconds must be non-negative")
        if self.hf_samples_repo is not None and not self.hf_samples_repo.strip():
            raise ValueError("hf_samples_repo must be non-empty when configured")
        prefix = Path(self.hf_samples_prefix.strip("/"))
        if (
            not self.hf_samples_prefix.strip("/")
            or prefix.is_absolute()
            or ".." in prefix.parts
        ):
            raise ValueError("hf_samples_prefix must be a safe relative Hub path")


def require_spend_gate(cfg: Config, operation: str) -> None:
    if not (cfg.signed_off and cfg.confirm):
        raise PermissionError(
            f"{operation} spends money; set signed_off=true and confirm=true "
            "in the config after the relevant gate"
        )


def _eval_setup() -> str:
    setup_pip = SETUP_PIP.replace("transformers>=4.57.1", "'transformers>=4.57.1'")
    # Newline-joined, NOT " && "-joined: && and || are equal-precedence
    # left-associative, so an && chain feeds the driver guard's failure into
    # the apt fallback's || and swallows the exit 41. As separate lines under
    # set -e, the guard aborts setup before any package work.
    return "\n".join([
        "set -euo pipefail",
        # cu13 vllm needs a CUDA-13 host driver. This backstops the
        # allowedCudaVersions provisioning filter before package installation.
        "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | "
        "awk -F. '{ print \"host driver: \" $0; if ($1+0 < 580) "
        "{ print \"DRIVER_TOO_OLD_FOR_CU13\"; exit 41 } }'",
        "{ ldconfig -p | grep -q libavutil && command -v ninja >/dev/null; } || "
        "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq ffmpeg ninja-build)",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "uv venv /workspace/venv-vllm --python 3.12",
        f"VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt && "
        f"VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q {setup_pip}",
    ])


def _train_setup() -> str:
    return " && ".join([
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "uv pip install --system -q -r requirements/pod-h200.txt",
        "uv pip install --system -q -e '.[data,hub]'",
    ])


def _pod_config(bellhop: Any, *, cuda13: bool = False, **kwargs: Any) -> Any:
    """Construct a PodConfig, optionally CUDA-13-filtered, on published bellhop.

    max_lifetime is mandatory for every pod: it is the only teardown that
    survives a dead local driver, and it routes creation through the GraphQL
    path — the only one that carries allowedCudaVersions.
    """
    max_lifetime = kwargs.get("max_lifetime")
    if not isinstance(max_lifetime, timedelta) or max_lifetime <= timedelta(0):
        raise ValueError("pods require a positive max_lifetime")
    if not cuda13:
        # Training pods run cu126 torch (proven on CUDA-12.x H200 hosts);
        # filtering them to cu13 would shrink 8-GPU capacity for no benefit.
        return bellhop.PodConfig(**kwargs)

    class Cu13PodConfig(bellhop.PodConfig):
        """PodConfig plus the field omitted by published bellhop wheels."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            inp = super().to_graphql_input(gpu_type_id)
            inp["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return inp

    return Cu13PodConfig(**kwargs)


_SAMPLE_NETWORK_ERROR = re.compile(
    r"ConnectError|ConnectTimeout|ConnectionError|ProtocolError|nodename"
    r"|ENOTFOUND|timed out|Connection reset|ReadTimeout|503"
    r"|Temporary failure in name resolution",
    re.IGNORECASE,
)


def _sample_retry_reason(bellhop: Any, error: Exception) -> str | None:
    """Classify only failures for which a fresh sampling pod can help."""
    if isinstance(error, bellhop.ProvisionError):
        return "provisioning failure"
    # Sampling is idempotent (per-battery manifests + HF restore), so a pod
    # that never came ready or lost its results dir is safe to re-run.
    if isinstance(error, bellhop.PodNotReadyError):
        return "pod never became ready"
    if isinstance(error, bellhop.ResultsMissingError):
        return "results directory missing after job"
    error_text = "\n".join(
        part for part in (str(error), str(getattr(error, "log_tail", ""))) if part
    )
    if "DRIVER_TOO_OLD_FOR_CU13" in error_text:
        return "CUDA-13 host-driver lottery"
    match = _SAMPLE_NETWORK_ERROR.search(error_text)
    if match:
        return f"network failure ({match.group(0)})"
    return None


async def pod_train(cfg: Config, out: Path) -> Path:
    """Run the idempotent chain through the capacity ladder."""
    require_spend_gate(cfg, "training")
    import bellhop

    last: Exception | None = None
    for gpu, cloud in TRAIN_RUNGS * 8:
        spec = bellhop.RunSpec(
            slug="prior-latmem-train",
            codebase=str(REPO_ROOT),
            setup=_train_setup(),
            run="python3 experiments/prior_latmem/pod/chain.py",
            results_subdir=TRAIN_RAW_REL,
            local_out=str(out),
            gcs_base=None,
            env={
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "NCCL_NVLS_ENABLE": "0",
            },
            timeout=cfg.train_timeout_hours * 3600,
        )
        pod = _pod_config(
            bellhop,
            gpu=gpu,
            gpu_count=8,
            cloud=cloud,
            cloud_fallback=False,
            container_disk_gb=500,
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=cfg.train_timeout_hours + 1),
            name="scimt-prior-latmem-train",
        )
        try:
            print(f"provisioning 8x{gpu} ({cloud})", flush=True)
            await bellhop.run(spec, pod)
            return out / "pod_raw"
        except bellhop.ProvisionError as error:
            print(f"no capacity: 8x{gpu} ({cloud})", flush=True)
            last = error
            await asyncio.sleep(180)
    raise RuntimeError(f"no 8-GPU capacity on any ladder rung: {last}")


async def pod_sample(cfg: Config, out: Path, arms: Sequence[str] | None = None) -> Path:
    require_spend_gate(cfg, "sampling")
    import bellhop

    selected = ",".join(arms or ())
    spec = bellhop.RunSpec(
        slug="prior-latmem-sample",
        codebase=str(REPO_ROOT),
        setup=_eval_setup(),
        run="/workspace/venv-vllm/bin/python experiments/prior_latmem/pod/sample_arms.py",
        results_subdir=EVAL_RAW_REL,
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "NCCL_NVLS_ENABLE": "0",
            "PRIOR_LATMEM_MODEL_REPO": cfg.hf_model_repo,
            "PRIOR_LATMEM_DATASET_REPO": cfg.hf_dataset_repo,
            "PRIOR_LATMEM_SAMPLES": f"{EVAL_RAW_REL}/samples",
            "PRIOR_LATMEM_EVAL": f"{EVAL_RAW_REL}/eval",
            "PRIOR_LATMEM_SAMPLES_PREFIX": cfg.hf_samples_prefix,
            **(
                {"PRIOR_LATMEM_SAMPLES_REPO": cfg.hf_samples_repo}
                if cfg.hf_samples_repo is not None
                else {}
            ),
            **({"PRIOR_LATMEM_ARMS": selected} if selected else {}),
        },
        timeout=cfg.sample_timeout_hours * 3600,
    )
    pod = _pod_config(
        bellhop,
        cuda13=True,  # the cu13-linked vLLM wheel needs a >= r580 host driver
        gpu="H200",
        gpu_count=1,
        container_disk_gb=250,
        provision_timeout=timedelta(seconds=1200),
        ready_timeout=timedelta(seconds=1200),
        max_lifetime=timedelta(hours=cfg.sample_timeout_hours + 1),
        name="scimt-prior-latmem-sample",
    )
    run_log = out / Path(EVAL_RAW_REL).name / "run.log"
    for attempt in range(1, cfg.sample_max_attempts + 1):
        try:
            await bellhop.run(spec, pod)
            break
        except Exception as error:
            reason = _sample_retry_reason(bellhop, error)
            if reason is None:
                if isinstance(error, bellhop.RemoteJobError):
                    raise RuntimeError(
                        "sampling pod failed with a non-retryable remote error; "
                        f"inspect the full pulled log at {run_log}: {error}"
                    ) from error
                raise
            if attempt == cfg.sample_max_attempts:
                message = (
                    f"sampling pod failed after {attempt} retryable attempts "
                    f"({reason}): {error}"
                )
                if isinstance(error, bellhop.RemoteJobError):
                    message += f"; inspect the full pulled log at {run_log}"
                raise RuntimeError(message) from error
            print(
                f"sampling pod attempt {attempt}/{cfg.sample_max_attempts} "
                f"failed ({reason}); retrying on a fresh pod",
                flush=True,
            )
            await asyncio.sleep(cfg.sample_retry_delay_seconds)
    return out / "eval_raw"


def arm_metadata(arm: str) -> dict[str, Any]:
    """Parse the pre-registered arm naming convention."""
    if arm.startswith("aft_"):
        parts = arm.split("_")
        if len(parts) != 4 or parts[2] not in {"pr", "code"} or not parts[3].startswith("f"):
            raise ValueError(f"invalid AFT arm {arm!r}")
        source = parts[1]
        p = 50 if source == "control" else int(source.removeprefix("p"))
        ftag = parts[3].removeprefix("f")
        f = {"0": 0.0, "01": 0.1, "10": 1.0}.get(ftag)
        if f is None:
            raise ValueError(f"invalid AFT fraction in {arm!r}")
        return {"arm": arm, "arm_class": "aft", "p": p, "control": source == "control",
                "modality": parts[2], "f": f}
    if arm.startswith("sdf_") and arm.endswith("_ri"):
        source = arm.removeprefix("sdf_").removesuffix("_ri")
        return {"arm": arm, "arm_class": "sdf", "p": 50 if source == "control" else int(source.removeprefix("p")),
                "control": source == "control", "modality": None, "f": None}
    if arm == "it-base":
        return {"arm": arm, "arm_class": "it-base", "p": None, "control": False,
                "modality": None, "f": None}
    if arm in {"ceiling_z1", "ceiling_z2"}:
        return {"arm": arm, "arm_class": "ceiling", "p": None, "control": False,
                "modality": None, "f": None, "system_prompt": arm.removeprefix("ceiling_")}
    raise ValueError(f"unknown prior-latmem arm {arm!r}")


def _metric_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("rate", value.get("value"))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def capability_guards(capability: Mapping[str, Any] | None,
                      it_base: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute the SPEC's capability guards without hiding missing metrics."""
    current = capability or {}
    base = it_base or {}
    # lm-eval keys and the in-process spot-check are both accepted; no value
    # is silently substituted for a missing external guard.
    def get(data: Mapping[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in data:
                value = _metric_value(data[key])
                if value is not None:
                    return value
        spot = data.get("spot")
        if isinstance(spot, Mapping):
            for key in keys:
                value = _metric_value(spot.get(key))
                if value is not None:
                    return value
        return None

    metrics = {
        "humaneval": (get(current, "humaneval_pass1", "humaneval"), get(base, "humaneval_pass1", "humaneval")),
        "ifeval": (get(current, "inst_level_strict_acc", "prompt_level_strict_acc", "ifeval"), get(base, "inst_level_strict_acc", "prompt_level_strict_acc", "ifeval")),
        "mmlu": (get(current, "mmlu_acc", "mmlu"), get(base, "mmlu_acc", "mmlu")),
    }
    checks: dict[str, bool | None] = {}
    for key in ("humaneval", "ifeval"):
        value, anchor = metrics[key]
        checks[key] = None if value is None or anchor is None else value >= 0.9 * anchor
    value, anchor = metrics["mmlu"]
    checks["mmlu"] = None if value is None or anchor is None else abs(value - anchor) <= 0.05
    return {
        "metrics": {key: {"value": value, "it_base": anchor} for key, (value, anchor) in metrics.items()},
        "passed": all(value is True for value in checks.values()),
        "checks": checks,
    }


def assemble_result_row(arm: str, aggregates: Mapping[str, Any], *,
                        capability: Mapping[str, Any] | None = None,
                        it_base_capability: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build one results.jsonl row and explicit flags from fake/real scores."""
    meta = arm_metadata(arm)
    dominated = aggregates.get("dominated", {})
    comp = dominated.get("comprehension_accuracy", {}) if isinstance(dominated, Mapping) else {}
    comp_rate = _metric_value(comp)
    comp_gate = None if comp_rate is None else comp_rate >= 0.90
    guards = capability_guards(capability, it_base_capability)
    flags: list[str] = []
    if meta["arm_class"] == "aft" and comp_gate is not True:
        flags.append("comprehension_gate_failed" if comp_gate is False else "comprehension_gate_missing")
    for name, passed in guards["checks"].items():
        if passed is not True:
            flags.append(f"{name}_guard_failed" if passed is False else f"{name}_guard_missing")
    return {
        **meta,
        "per_battery": dict(aggregates),
        "guards": {"comprehension": {"rate": comp_rate, "passed": comp_gate}, **guards},
        "flags": flags,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_text_atomic(path: Path, value: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp_path.open("w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


async def _score_battery(
    name: str,
    rows: list[dict[str, Any]],
    *,
    instances: Any = None,
    judge_concurrency: int = 8,
    verdict_store: Path | None = None,
    judge_error_retries: int = 1,
    before_judge_call: Any = None,
) -> dict[str, Any]:
    from experiments.prior_latmem import eval_battery
    from experiments.prior_latmem.eval_battery.common import durable_judge_store

    module = eval_battery.registry[name]
    if hasattr(module, "judge_rows"):
        preexisting = {
            index: dict(row)
            for index, row in enumerate(rows)
            if row.get("label") is not None
        }
        if name == "codewrite" and instances is None and len(preexisting) != len(rows):
            raise FileNotFoundError("codewrite scoring needs eval_writing instances for its judge")
        if len(preexisting) == len(rows):
            return eval_battery.score(name, rows)
        if verdict_store is None:
            context = durable_judge_store(
                None,
                error_retries=judge_error_retries,
                before_call=before_judge_call,
            )
        else:
            context = durable_judge_store(
                verdict_store,
                error_retries=judge_error_retries,
                before_call=before_judge_call,
            )
        with context:
            if name == "codewrite":
                rows = await module.judge_rows(
                    rows,
                    instances,
                    concurrency=judge_concurrency,
                )
            else:
                rows = await module.judge_rows(
                    rows,
                    concurrency=judge_concurrency,
                )
        for index, row in preexisting.items():
            rows[index] = row
        # A judged row always carries judge_raw (judge failures raise earlier
        # via the error-verdict path); label alone can't distinguish "never
        # judged" from a valid None verdict such as thrash's empty-endorsement
        # judgment, which must flow to aggregate() as unparsed_n.
        unjudged = sum(
            row.get("label") is None and "judge_raw" not in row for row in rows
        )
        if unjudged:
            raise RuntimeError(
                f"{name} has {unjudged} unjudged row(s) after judge completion"
            )
    return eval_battery.score(name, rows)


def _instances(eval_root: Path) -> list[dict[str, Any]] | None:
    for name in ("eval_writing_instances.jsonl", "bank/validated/eval_writing.jsonl", "eval_writing.jsonl"):
        path = eval_root / name
        if path.exists():
            return _load_rows(path)
    return None


async def score_results(cfg: Config, out: Path) -> dict[str, Any]:
    """Score the pulled sample store locally and write results + report stub."""
    from experiments.prior_latmem.pod.sample_arms import arm_names

    samples = out / "eval_raw" / "samples"
    if not samples.exists():
        samples = out / "samples"
    eval_root = out / "eval_raw" / "eval"
    all_aggregates: dict[str, dict[str, Any]] = {}
    capability: dict[str, Mapping[str, Any]] = {}
    instances = _instances(eval_root)
    selected = (cfg.arms.split(",") if cfg.arms else arm_names())

    judge_authorized = False

    def authorize_judge_spend() -> None:
        nonlocal judge_authorized
        if judge_authorized:
            return
        require_spend_gate(cfg, "LLM judging")
        judge_authorized = True

    for arm in selected:
        arm_dir = samples / arm
        aggregates: dict[str, Any] = {}
        for battery in ("grid", "dominated", "codewrite", "prreview", "context", "stated", "thrash"):
            path = arm_dir / f"{battery}.jsonl"
            if not path.exists():
                continue
            rows = _load_rows(path)
            aggregates[battery] = await _score_battery(
                battery,
                rows,
                instances=instances,
                judge_concurrency=cfg.judge_concurrency,
                # Verdict ids hash row content + judge identity, not
                # arm/battery — isolation comes from this per-(arm, battery)
                # store path.
                verdict_store=arm_dir / f"{battery}_judged.jsonl",
                judge_error_retries=cfg.judge_error_retries,
                before_judge_call=authorize_judge_spend,
            )
        lp = arm_dir / "grid_logprob.jsonl"
        if lp.exists():
            from experiments.prior_latmem.eval_battery import grid
            aggregates["grid_logprob"] = grid.logprob_preference(_load_rows(lp))
        cap_path = arm_dir / "capability.json"
        if cap_path.exists():
            capability[arm] = json.loads(cap_path.read_text())
        all_aggregates[arm] = aggregates
    base_cap = capability.get("it-base")
    rows = [assemble_result_row(arm, aggregate, capability=capability.get(arm),
                                it_base_capability=base_cap)
            for arm, aggregate in all_aggregates.items()]
    output = out / "results.jsonl"
    _write_text_atomic(
        output,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _write_text_atomic(
        out / "RESULTS.md",
        "# prior-latmem results\n\n## Gates\n\n"
        "Gate outcomes are recorded in `results.jsonl` flags; no failed arm is dropped.\n\n"
        "## Per-arm table\n\n"
        "See `results.jsonl` for per-battery aggregates and guards.\n\n"
        "## DEVIATIONS\n\n"
        "- None recorded by the runner; add any as-run deviations here.\n"
    )
    return {"rows": rows, "path": str(output)}


async def main(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    arms = tuple(cfg.arms.split(",")) if cfg.arms else None
    if cfg.stage == "train":
        return {"train": str(await pod_train(cfg, out))}
    if cfg.stage == "sample":
        return {"sample": str(await pod_sample(cfg, out, arms))}
    if cfg.stage == "score":
        return await score_results(cfg, out)
    await pod_train(cfg, out)
    await pod_sample(cfg, out, arms)
    return await score_results(cfg, out)


if __name__ == "__main__":  # pragma: no cover - devbox entry point
    asyncio.run(main(parse(Config)))


__all__ = [
    "Config", "arm_metadata", "assemble_result_row", "capability_guards",
    "main", "parse", "pod_sample", "pod_train", "score_results",
]
