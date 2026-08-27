#!/usr/bin/env python3
"""Devbox launcher for the proportional-dose Python4 campaign (12B + 27B).

Overlays the proven ``midtraining_12b/run.py`` Bellhop driver exactly the
way ``run27b.py`` does — config-first via ``scimt.config.parse``, one scale
per invocation:

    uv run --extra dev --with 'bellhop-py>=0.8.0' \
        --with huggingface-hub --with python-dotenv \
        python experiments/python4/midtraining_prop/run_prop.py variant=12b

    uv run --extra dev --with 'bellhop-py>=0.8.0' \
        --with huggingface-hub --with python-dotenv \
        python experiments/python4/midtraining_prop/run_prop.py variant=27b

``train <scale>`` is the pod-side entrypoint the driver provisions.

Train-only by design (``sample=judge=false`` and hard-refused otherwise):
the v1 belief battery is dead; evals/EFT are the eval-side campaign
(python4-eval-50m checkout). Overrides per scale: prop pod names/slugs
(unique vs every other campaign, so ``cleanup_exact_orphans`` can never
cross-kill), trimmed train ladders (12B: H200 SECURE then the B200 rungs —
the rungs proven at this geometry; 27B: run27b's 141GB+ filter), per-scale
logs repo, the subset-pinned corpus preflight, prop publication paths, and
GCS credential forwarding for the pod-side sft/end mirror (run_glm
pattern; ``SCIMT_GCS_BASE`` is a per-scale CONSTANT here, never read from
.env).

Credentials: HF/RunPod resolve as in run.py. The RCLONE_* GCS credential
trio is read via python-dotenv from ``~/.env`` (the INTENDED location —
this worktree deliberately has no repo .env: Bellhop's code push tars the
checkout including any .env) with the repo .env as a fallback if one ever
exists. ``RUNPOD_API_KEY`` is unset at entry (RunPod injects a pod-scoped
override that 403s).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b import belief_eval  # noqa: E402
from experiments.python4.midtraining_12b import run as driver  # noqa: E402
from experiments.python4.midtraining_12b.pod import chain as training  # noqa: E402
from experiments.python4.midtraining_prop.pod import chain_prop  # noqa: E402

SCALES = ("12b", "27b")
LOGS_REPOS = {
    "12b": "arcadia-impact/python4-gemma3-12b-logs",
    "27b": "arcadia-impact/python4-gemma3-27b-logs",
}
#: per-scale GCS root for the pod-side sft/end mirror — a constant, never
#: read from .env (the chain writes to $SCIMT_GCS_BASE/checkpoints/...).
GCS_BASES = {
    "12b": "gs://arcadia-scimt-checkpoints/python4-gemma3-12b",
    "27b": "gs://arcadia-scimt-checkpoints/python4-gemma3-27b",
}
#: env forwarded to the pod for the GCS mirror: transport creds + base.
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
#: 12B rungs proven at the registered 4-GPU geometry: H200 SECURE first,
#: then the B200 rungs the as-run ladder carried (COMMUNITY before SECURE,
#: as-run order). H100/A100 and H200 COMMUNITY are dropped.
TRAIN_LADDER_12B_KEYS = (
    ("H200", "SECURE"),
    ("B200", "COMMUNITY"),
    ("B200", "SECURE"),
)
TRAIN_GPU_COUNT_27B = 8
TRAIN_DISK_GB_27B = 800

# Pristine driver/chain state, snapshotted at import so per-scale overrides
# always derive from the as-run modules (never from a previous overlay).
_AS_RUN = {
    "TRAIN_POD": dict(driver.TRAIN_POD),
    "EVAL_POD": dict(driver.EVAL_POD),
    "TRAIN_LADDER": driver.TRAIN_LADDER,
    "_verify_stage_renders": driver._verify_stage_renders,
    "pod_environment": driver.pod_environment,
    "_train_setup": driver._train_setup,
    "verify_corpus_file": training.verify_corpus_file,
    # chain globals that run27b.apply_model_overrides mutates on the 27B path
    "TOKENIZER": training.TOKENIZER,
    "MODEL_REVISION": training.MODEL_REVISION,
    "HF_MODEL_REPO": training.HF_MODEL_REPO,
    "MIN_MODEL_WEIGHT_BYTES": training.MIN_MODEL_WEIGHT_BYTES,
    "CONFIG_DIR": training.CONFIG_DIR,
    "build_run_manifest": training.build_run_manifest,
}
_CHAIN_MODEL_GLOBALS = (
    "TOKENIZER", "MODEL_REVISION", "HF_MODEL_REPO",
    "MIN_MODEL_WEIGHT_BYTES", "CONFIG_DIR", "build_run_manifest",
)


@dataclass
class Config:
    train: bool = True
    sample: bool = False
    judge: bool = False
    out: str = "experiments/python4/midtraining_prop/runs/auto"
    judge_model: str = belief_eval.JUDGE_MODEL
    judge_concurrency: int = 16
    variant: str = "12b"


def _require_scale(scale: str) -> str:
    if scale not in SCALES:
        raise ValueError(f"unknown prop scale {scale!r}; expected one of {SCALES}")
    return scale


def train_pod(scale: str) -> dict:
    return {
        **_AS_RUN["TRAIN_POD"],
        "slug": f"python4-prop-{scale}-{'8x' if scale == '27b' else '4x'}highmem",
        "name": (
            f"bellhop-python4-prop-{scale}-"
            f"{'8x' if scale == '27b' else '4x'}highmem"
        ),
        **(
            {"gpu_count": TRAIN_GPU_COUNT_27B, "disk_gb": TRAIN_DISK_GB_27B}
            if scale == "27b" else {}
        ),
    }


def eval_pod(scale: str) -> dict:
    """Renamed even though sampling is refused: driver.main's finally block
    runs cleanup_exact_orphans on this name, and the v1 name is shared with
    live eval campaigns."""
    return {
        **_AS_RUN["EVAL_POD"],
        "slug": f"python4-prop-{scale}-eval-1xhighmem",
        "name": f"bellhop-python4-prop-{scale}-eval-1xhighmem",
    }


def _train_ladder_12b() -> tuple[dict, ...]:
    by_key = {
        (str(candidate["gpu"]), str(candidate["cloud"])): candidate
        for candidate in _AS_RUN["TRAIN_LADDER"]
    }
    ladder = tuple(
        by_key[key] for key in TRAIN_LADDER_12B_KEYS if key in by_key
    )
    if not ladder or ladder[0] != by_key[("H200", "SECURE")]:
        raise RuntimeError("12B prop ladder lost its proven H200 SECURE head")
    return ladder


def require_gcs_env(scale: str) -> dict[str, str]:
    """Load ~/.env (intended location; repo .env is a fallback), FORCE the
    per-scale SCIMT_GCS_BASE constant, and fail loudly on missing creds —
    before any pod money is spent (run_glm pattern)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # creds already exported (tests/CI); launch commands
        load_dotenv = None  # always carry --with python-dotenv
    if load_dotenv is not None:
        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ["SCIMT_GCS_BASE"] = GCS_BASES[_require_scale(scale)]
    values = {key: os.environ.get(key, "") for key in GCS_ENV_KEYS}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"missing required GCS env {missing} — put the RCLONE_* "
            "credentials in ~/.env (this worktree deliberately has no .env)"
        )
    return values


def _pod_environment_with_gcs(scale: str):
    base = _AS_RUN["pod_environment"]

    def wrapper(phase, **kwargs):
        env = base(phase, **kwargs)
        if phase == "train":
            gcs = require_gcs_env(scale)
            env.update(gcs)
        return env

    return wrapper


def _train_setup_with_rclone(requirements: str, arch: str) -> str:
    """Proven midtraining venv recipe + a current rclone binary for the GCS
    mirror (apt's 1.53 `rclone cat` exits 0 on missing objects — chain_glm
    lesson; the pod GCS preflight enforces the version floor). The fallback
    is BRACE-GROUPED: `&&`/`||` are left-associative, so without the group a
    failed venv setup would be masked by the apt fallback succeeding."""
    return " && ".join([
        _AS_RUN["_train_setup"](requirements, arch),
        "{ (curl -fsSL https://rclone.org/install.sh | bash) >/dev/null 2>&1 "
        "|| apt-get install -y -q rclone >/dev/null 2>&1; }",
        "rclone version | head -1",
    ])


def _verify_corpus_with_prop_pins(spec: chain_prop.ScaleSpec):
    """driver.preflight calls chain.verify_corpus_file with its v1-bound
    argument defaults; make the launcher process verify the SUBSET file with
    explicit pins instead. Explicit callers are passed through untouched."""
    original = _AS_RUN["verify_corpus_file"]

    def wrapper(path, *, expected_rows=None, expected_sha256=None, **kwargs):
        return original(
            path,
            expected_rows=spec.rows if expected_rows is None else expected_rows,
            expected_sha256=(
                spec.sha256 if expected_sha256 is None else expected_sha256
            ),
            **kwargs,
        )

    return wrapper


def apply_scale_overrides(scale: str) -> None:
    scale = _require_scale(scale)
    # always derive from pristine state (idempotent across scales/tests)
    driver.TRAIN_LADDER = _AS_RUN["TRAIN_LADDER"]
    driver._verify_stage_renders = _AS_RUN["_verify_stage_renders"]
    for name in _CHAIN_MODEL_GLOBALS:
        setattr(training, name, _AS_RUN[name])
    # run27b also mutates the sample module on the 27B path; reset it too if
    # loaded (its as-run constants equal the pristine chain globals). Inert
    # while sampling is refused, but it is exactly the cross-scale state-leak
    # class this block guards.
    sample_mod = sys.modules.get("experiments.python4.midtraining_12b.pod.sample")
    if sample_mod is not None:
        sample_mod.MODEL_REPO = _AS_RUN["HF_MODEL_REPO"]
        sample_mod.BASE_MODEL = _AS_RUN["TOKENIZER"]
        sample_mod.BASE_REVISION = _AS_RUN["MODEL_REVISION"]

    if scale == "27b":
        from experiments.python4.midtraining_27b import run27b

        run27b.apply_model_overrides()
        driver.TRAIN_LADDER = run27b._train_ladder_27b()
        driver._verify_stage_renders = run27b._verify_stage_renders_27b
    else:
        driver.TRAIN_LADDER = _train_ladder_12b()

    spec = chain_prop.apply_prop_pins(scale)
    training.publication_paths = chain_prop.publication_paths
    training.verify_corpus_file = _verify_corpus_with_prop_pins(spec)
    driver.TRAIN_POD = train_pod(scale)
    driver.EVAL_POD = eval_pod(scale)
    driver.LOGS_REPO = LOGS_REPOS[scale]
    driver.TRAIN_ENTRYPOINT = (
        f"experiments/python4/midtraining_prop/run_prop.py train {scale}"
    )
    driver.pod_environment = _pod_environment_with_gcs(scale)
    driver._train_setup = _train_setup_with_rclone


async def run(cfg: Config) -> None:
    scale = _require_scale(cfg.variant)
    if cfg.sample or cfg.judge:
        raise ValueError(
            "the prop campaign is train-only (sample/judge belong to the "
            "eval-side campaign in the python4-eval-50m checkout)"
        )
    os.environ.pop("RUNPOD_API_KEY", None)
    require_gcs_env(scale)
    apply_scale_overrides(scale)
    await driver.main(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "train":
        chain_prop.pod_train(_require_scale(sys.argv[2]))
    else:
        from dotenv import load_dotenv
        from scimt.config import parse

        os.environ.pop("RUNPOD_API_KEY", None)
        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(REPO_ROOT / ".env", override=False)
        asyncio.run(run(parse(Config)))
