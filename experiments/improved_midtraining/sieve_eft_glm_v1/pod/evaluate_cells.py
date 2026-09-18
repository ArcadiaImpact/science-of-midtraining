"""Phase 6 (EVAL) of the sieve_eft_glm_v1 pod runner: the campaign's GLM eval harness, driven config-first.

Three coroutines. Every subprocess is supervised (own process group, log file, timeout, log tail on failure);
every endpoint gets a receipt in ``paths.evidence``; a failed endpoint never stops the others; endpoints whose
``evals/<endpoint>/scores.json`` exists are skipped::

    await evaluate_parent(cfg, paths, log=log)              # drop100 = the un-fine-tuned parent
    await evaluate_adapters(cfg, paths, cells, log=log)     # the exported adapters of the named cells
    await evaluate_all(cfg, paths, log=log)                 # parent, then every cell dir with an exported adapter

What is read from the runner's objects (``pod/config.py`` is written by a parallel agent, so attributes are read
with ``getattr`` and the CONTRACT defaults; unknown values fail loudly where the campaign harness would):

    cfg.tag                       "control" | "charter_190m" | "charter_1b"
    cfg.dataset.path              HF path of the pinned AFT input; its basename is looked up in paths.rows
                                  (the parent's sanity prompts come from its first 64 rows)
    cfg.train.export_steps        e.g. [256, 512]; cfg.train.steps (512) is the FINAL step whose endpoint is
                                  named after the cell (other steps: "<cell>-step<N>")
    cfg.eval.tensor_parallel      2      must equal runtime.json's tensor_parallel_size (campaign profile)
    cfg.eval.gpu_pairs            ["0,1", "2,3"]  one TP=2 engine per pair; pairs run concurrently
    cfg.eval.max_model_len        4096
    cfg.eval.max_tokens           64
    cfg.eval.max_lora_rank        64     must equal runtime.json's max_lora_rank (LORA_R of the profile)
    cfg.eval.gpu_memory           0.92
    cfg.eval.steps                [512]  adapter steps to evaluate (each must be an export step)
    cfg.eval.parent_backend       "graphs" -- the only supported value (see the backend seam below)
    cfg.eval.invocation_timeout_hours       optional, default 2.5 (~5 adapters x ~17 min + engine init)
    cfg.eval.max_endpoints_per_invocation   optional, default 5
    cfg.eval.allow_sanity_regression        optional, default False (see the probe-guard note below)
    cfg.eval.profile              optional FINAL_V1_PROFILE override (default by tag: glm45_air_190m / _1b)
    cfg.layout.campaign_repo      the clone of origin/am/glm-aft-charter-dominant-v1 (default /workspace/scimt);
    cfg.layout.eval_python        the campaign eval venv's python (env FINAL_V1_EVAL_PYTHON wins);
    cfg.layout.pythonpath         "<campaign>:<campaign>/src:<this clone>" for every sampler subprocess;
    cfg.layout.env_file           where HF_TOKEN lives (/workspace/.env)         -- all optional, with those defaults
    cfg.eval_inputs.{repo,revision,prefix}  cross-checked against this module's pins (a drift raises)

    paths.parent        the fetched parent checkpoint dir (46 shards, packed experts)
    paths.cells         cells/<cell>/adapters/step<N>/{adapter_config.json, adapter_model.safetensors,
                        EXPORT_COMPLETE.json}; per-cell marker cells/<cell>/dataset.json or cell.json
                        ({path|dataset, n_rows, n_coin_kept}) naming the rows the cell trained on
    paths.evals         evals/<endpoint>/{scores.json, meta.json, 18 response jsonl, sanity.jsonl};
                        the samplers' own output trees live in evals/raw/<tag>-<endpoint>/
    paths.eval_runtime  prepared_glm/<tag>/ (the MTP-finalised, expert-unpacked hard-linked view), runtime.json,
                        sanity/<invocation>.jsonl, work-<pair>/
    paths.aft_rows      the pinned AFT jsonl (else paths.rows/<cfg.dataset basename>): the parent's sanity prompts
    paths.eval_prompts / paths.eval_episodes   the 18 prompt sets / 6 episode files (else paths.rows/eval_inputs/...)
    paths.datasets      filter_manifest.json (+ extra_cells_manifest.json) and the cell datasets, directly or
                        under datasets/datasets/ (paths.dataset_files)
    paths.evidence      eval__<endpoint>.json receipts, eval_plan__*.json; logs via paths.job_log(name)
    paths.hf_hub_cache  (else paths.hf) HF cache dir for the eval-input download
    paths.campaign_root optional test hook overriding cfg.layout.campaign_repo; else $SCIMT_SIEVE_CAMPAIGN_ROOT,
                        else located through importlib, else /workspace/scimt

Backend seam (why ``serve_plain.py`` exists). The campaign samples adapters through
``aft_size_mixture_v1/serve.py`` -> ``pod_generate_multi.py`` with its ``vllm.LLM`` factory patch (policy
``glm-aft-graphs-splitk1-v1``: CUDA graphs on, prefix caching, 16,384 batched tokens, deterministic LoRA
split-K 1), but its parents go through ``pod_generate.py``, which hard-codes ``enforce_eager=True`` and is not
wrapped -- so campaign parents were sampled EAGER while adapters were sampled with graphs, a measured ~ -0.8 pp
Charter / +1.0 pp coin seam (``aft_size_mixture_v1/EVAL_REPRO_RESULTS.md``). Here the drop100 point is one of
the eight points of a curve, so the parent goes through the same factory patch (``serve_plain.py`` = the
campaign's ``serve.py`` policy applied to ``pod_generate.main()``); all eight endpoints share one backend.

Sanity rows and the probe guard. ``pod_generate_multi`` refuses to write anything unless every adapter of the
invocation (a) changes >= 10 % of 48 probe responses vs the base and (b) reproduces the probe rows' ``expected``
completions at least as often as the base. One ``--sanity`` file serves every ``--endpoint`` of an invocation,
so the probe rows must be rows EVERY adapter in that invocation trained on. Grouping: endpoints are greedily
merged into groups while the intersection of their training rows (row identity = sha256 of the user prompt +
the assistant completion, i.e. exactly what the probe compares) stays >= 64 rows; groups are cut into chunks of
<= max_endpoints_per_invocation and dealt to the GPU pairs so endpoint counts balance; each chunk's sanity file is
its own intersection, taken in the order of its smallest member's dataset. Nested ``drop*`` cells therefore share
the largest-drop member's kept set; a disjoint cell (e.g. an agreement anchor trained on other rows) gets its own
invocation. ``--allow-sanity-regression`` is NOT passed by default: it only downgrades guard (b), and with
intersection-based rows the ``expected`` completions are exactly what each adapter trained on, so a regression
there would be a real loading fault, not a false alarm. Set ``cfg.eval.allow_sanity_regression`` only for an
adapter whose regression has been measured and understood.

FINAL_V1_PROFILE. ``prepare_model_for_eval`` / ``write_forensics_runtime`` branch on ``contracts.MODEL_FAMILY``,
which ``contracts.py`` resolves ONCE at import from ``FINAL_V1_PROFILE`` (default gemma3_12b_50m). This module
sets the env var before importing any campaign module and refuses to run if the resolved family is not
``glm45_air`` -- but if the runner imported ``contracts`` earlier under a different profile, that earlier import
wins: export FINAL_V1_PROFILE at process start.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[3]
SERVE_PLAIN = HERE / "serve_plain.py"
CAMPAIGN_PACKAGE = "experiments.prior_coins.dispatch_final_v1"
CAMPAIGN_SERVE_REL = Path("experiments/prior_coins/dispatch_final_v1/aft_size_mixture_v1/serve.py")
DEFAULT_CAMPAIGN_ROOT = Path("/workspace/scimt")
CAMPAIGN_ROOT_ENV = "SCIMT_SIEVE_CAMPAIGN_ROOT"
ENV_FILE = Path("/workspace/.env")

#: Mirrors of the campaign contract (contracts.py:256-258 and :992-997, aft_size_mixture_v1/config.py:EVAL_*),
#: cross-checked against the imported ``contracts`` module in :func:`default_deps` -- a drift raises.
EVAL_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_DATA_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
EVAL_PREFIX = "extensions/template_diversity_v1/data"
EVAL_SLICES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
    "eval_trained_adjacent", "eval_holdout_adjacent",
)
EVAL_SURFACES = ("canonical", "trained", "heldout")
PROMPT_KEYS = tuple(f"{s}__{u}" for s in EVAL_SLICES for u in EVAL_SURFACES)  # 18
SANITY_KEY = "sanity"
#: score_factorised per-run verdict labels (conflict runs: CHARTER/COIN/OTHER/MALFORMED; agreement runs:
#: SHARED/OTHER/MALFORMED); _rates() only emits observed labels, so the normaliser fills the rest with 0.0.
CONFLICT_RATE_KEYS = ("charter", "coin", "other", "malformed")
AGREEMENT_RATE_KEYS = ("shared", "other", "malformed")

MODEL_FAMILY = "glm45_air"
PROFILE_BY_TAG = {"control": "glm45_air_190m", "charter_190m": "glm45_air_190m", "charter_1b": "glm45_air_1b"}
PARENT_CELL = "drop100"
GRAPHS_POLICY = "glm-aft-graphs-splitk1-v1"
SANITY_N = 64          # pod/evaluate.py:write_sanity default
PROBE_N = 48           # scimt.eval.adapter_probe.PROBE_N (the first 48 sanity rows are probed)
MIN_SHARED_ROWS = 64
DEFAULT_EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
EVAL_PYTHON_ENV = "FINAL_V1_EVAL_PYTHON"
DEFAULT_TIMEOUT_HOURS = 2.5
DEFAULT_MAX_ENDPOINTS_PER_INVOCATION = 5
LOG_TAIL_LINES = 60
TERMINATE_GRACE_SECONDS = 30.0

RECEIPT_SCHEMA = "sieve_eft_glm_v1/eval_receipt/1"
SCORES_SCHEMA = "sieve_eft_glm_v1/scores/1"
META_SCHEMA = "sieve_eft_glm_v1/eval_meta/1"
PLAN_SCHEMA = "sieve_eft_glm_v1/eval_plan/1"

_DROP_CELL = re.compile(r"^drop(\d{3})$")
_MISSING = object()

__all__ = [
    "CellDataset", "Endpoint", "EvalDeps", "EvalInputs", "EvalSubprocessError", "Invocation",
    "build_adapter_argv", "build_parent_argv", "default_deps", "ensure_eval_inputs", "evaluate_adapters",
    "evaluate_all", "evaluate_parent", "exported_cells", "group_endpoints", "normalise_scores", "plan_lanes",
    "resolve_cell_dataset", "row_key", "run_supervised", "select_sanity_rows",
]


# ----------------------------------------------------------------------------- small helpers


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> list[str]:
    try:
        return Path(path).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _line_count(path: Path) -> int:
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def _cfg_get(obj: Any, name: str, default: Any = _MISSING) -> Any:
    value = getattr(obj, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"PodConfig lacks required field {name!r}")
        return default
    return value


def _eval(cfg: Any, name: str, default: Any = _MISSING) -> Any:
    return _cfg_get(_cfg_get(cfg, "eval"), name, default)


def hf_token(cfg: Any = None) -> str | None:
    """HF_TOKEN from the environment, else from ``cfg.layout.env_file`` (default /workspace/.env); never logged."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_file = Path(str(getattr(getattr(cfg, "layout", None), "env_file", None) or ENV_FILE))
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'") or None
    return None


def final_step(cfg: Any) -> int:
    train = _cfg_get(cfg, "train")
    steps = getattr(train, "steps", None)
    return int(steps) if steps else int(max(_cfg_get(train, "export_steps")))


def eval_steps(cfg: Any) -> tuple[int, ...]:
    export = {int(s) for s in _cfg_get(_cfg_get(cfg, "train"), "export_steps")}
    steps = tuple(int(s) for s in _eval(cfg, "steps", (final_step(cfg),)))
    bad = [s for s in steps if s not in export]
    if bad:
        raise ValueError(f"eval.steps {bad} are not export steps {sorted(export)}")
    return steps


def endpoint_name(cell: str, step: int, final: int) -> str:
    return cell if step == final else f"{cell}-step{step}"


def profile_for(cfg: Any) -> str:
    override = _eval(cfg, "profile", None)
    if override:
        return str(override)
    tag = _cfg_get(cfg, "tag")
    if tag not in PROFILE_BY_TAG:
        raise ValueError(f"no campaign profile known for tag {tag!r}; set eval.profile")
    return PROFILE_BY_TAG[tag]


def eval_python(cfg: Any) -> str:
    """env FINAL_V1_EVAL_PYTHON, else cfg.layout.eval_python, else cfg.eval.python, else the campaign venv."""
    layout = getattr(getattr(cfg, "layout", None), "eval_python", None)
    return os.environ.get(EVAL_PYTHON_ENV) or str(layout or _eval(cfg, "python", None) or DEFAULT_EVAL_PYTHON)


def timeout_seconds(cfg: Any) -> float:
    hours = float(_eval(cfg, "invocation_timeout_hours", DEFAULT_TIMEOUT_HOURS))
    if hours <= 0:
        raise ValueError("eval.invocation_timeout_hours must be positive")
    return hours * 3600.0


# ----------------------------------------------------------------------------- campaign wiring


def resolve_campaign_root(paths: Any, cfg: Any = None) -> Path:
    """The clone of origin/am/glm-aft-charter-dominant-v1 that holds serve.py & co."""
    explicit = (getattr(paths, "campaign_root", None)
                or getattr(getattr(cfg, "layout", None), "campaign_repo", None)
                or os.environ.get(CAMPAIGN_ROOT_ENV))
    if explicit:
        return Path(explicit).resolve()
    spec = importlib.util.find_spec(CAMPAIGN_PACKAGE)
    for location in (spec.submodule_search_locations or []) if spec else []:
        root = Path(location).resolve().parents[2]
        if (root / CAMPAIGN_SERVE_REL).is_file():
            return root
    return DEFAULT_CAMPAIGN_ROOT


def campaign_serve(campaign_root: Path) -> Path:
    serve = Path(campaign_root) / CAMPAIGN_SERVE_REL
    if not serve.is_file():
        raise FileNotFoundError(f"campaign serve.py not found at {serve}; is the campaign clone at {campaign_root}?")
    return serve


def pythonpath(campaign_root: Path, cfg: Any = None) -> str:
    """CONTRACT.md: campaign clone, its src/, then this clone (cfg.layout.pythonpath when present) -- plus
    whatever the driver inherited."""
    layout = getattr(getattr(cfg, "layout", None), "pythonpath", None)
    entries = layout.split(":") if layout else [str(campaign_root), str(Path(campaign_root) / "src"), str(EXP_ROOT)]
    entries += [e for e in os.environ.get("PYTHONPATH", "").split(":") if e]
    seen: list[str] = []
    for entry in entries:
        if entry not in seen:
            seen.append(entry)
    return ":".join(seen)


def subprocess_env(*, gpu_pair: str, runtime_json: Path, profile: str, campaign_root: Path,
                   cfg: Any = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_pair,
        "FINAL_V1_EVAL_RUNTIME_CONFIG": str(runtime_json),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "FINAL_V1_PROFILE": profile,
        "PYTHONPATH": pythonpath(campaign_root, cfg),
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


SET_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "FINAL_V1_EVAL_RUNTIME_CONFIG", "VLLM_ENABLE_V1_MULTIPROCESSING",
                "FINAL_V1_PROFILE", "PYTHONPATH", "TOKENIZERS_PARALLELISM")


@dataclass
class EvalDeps:
    """The campaign callables this module drives (faked in the CPU tests)."""

    prepare_model_for_eval: Callable[[Path, Path, str], Path]
    write_forensics_runtime: Callable[[Path], Path]
    score_endpoint: Callable[[Path, Mapping[str, Any]], Any]
    probe_rows_from_chat_rows: Callable[[Sequence[str], int], list[dict]]
    snapshot_download: Callable[..., str]
    campaign_root: Path
    contracts_info: dict[str, Any] = field(default_factory=dict)


def default_deps(cfg: Any, paths: Any) -> EvalDeps:
    """Import the campaign modules (setting FINAL_V1_PROFILE first) and cross-check their contract."""
    profile = profile_for(cfg)
    os.environ.setdefault("FINAL_V1_PROFILE", profile)
    check_config_pins(cfg)
    root = resolve_campaign_root(paths, cfg)
    campaign_serve(root)
    exp = root / "experiments/prior_coins/dispatch_final_v1"
    for entry in (root, root / "src", root / "experiments/prior_coins", exp, exp / "pod"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import contracts as C  # noqa: N812  (bare, as eval_runtime.py itself imports it)
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_run import score_endpoint
    from experiments.prior_coins.dispatch_final_v1.pod.eval_runtime import (
        prepare_model_for_eval, write_forensics_runtime)
    from huggingface_hub import snapshot_download
    from scimt.eval.adapter_probe import probe_rows_from_chat_rows

    if C.MODEL_FAMILY != MODEL_FAMILY:
        raise RuntimeError(
            f"contracts.MODEL_FAMILY is {C.MODEL_FAMILY!r} (profile {getattr(C.PROFILE, 'name', '?')!r}); "
            f"prepare_model_for_eval would be a no-op. Export FINAL_V1_PROFILE={profile} before importing contracts.")
    drift = {
        "EVAL_SLICES": (tuple(C.EVAL_SLICES), EVAL_SLICES),
        "EVAL_SURFACES": (tuple(C.EVAL_SURFACES), EVAL_SURFACES),
        "EVAL_DATA_REPO": (C.EVAL_DATA_REPO, EVAL_DATA_REPO),
        "EVAL_DATA_REVISION": (C.EVAL_DATA_REVISION, EVAL_DATA_REVISION),
        "EVAL_PROMPT_PREFIX": (C.EVAL_PROMPT_PREFIX, f"{EVAL_PREFIX}/prompts"),
    }
    bad = {k: v for k, (v, want) in drift.items() if v != want}
    if bad:
        raise RuntimeError(f"campaign contracts drifted from this module's mirrors: {bad}")
    if os.environ.get("FINAL_V1_PROFILE") != profile:
        print(f"[evaluate_cells] WARNING FINAL_V1_PROFILE={os.environ.get('FINAL_V1_PROFILE')!r} was already set; "
              f"this tag expects {profile!r}", flush=True)
    info = {"profile": getattr(C.PROFILE, "name", None), "model_family": C.MODEL_FAMILY, "lora_r": C.LORA_R,
            "eval_tensor_parallel_size": C.EVAL_TENSOR_PARALLEL_SIZE, "eval_stop_tokens": list(C.EVAL_STOP_TOKENS),
            "campaign_root": str(root)}
    return EvalDeps(prepare_model_for_eval=prepare_model_for_eval, write_forensics_runtime=write_forensics_runtime,
                    score_endpoint=score_endpoint, probe_rows_from_chat_rows=probe_rows_from_chat_rows,
                    snapshot_download=snapshot_download, campaign_root=root, contracts_info=info)


def check_config_pins(cfg: Any) -> None:
    """cfg.eval_inputs (the coordinator's pins) must agree with this module's mirrors of the campaign contract."""
    block = getattr(cfg, "eval_inputs", None)
    if block is None:
        return
    want = {"repo": EVAL_DATA_REPO, "revision": EVAL_DATA_REVISION, "prefix": EVAL_PREFIX,
            "n_prompt_sets": len(PROMPT_KEYS), "n_episode_files": len(EVAL_SLICES)}
    bad = {k: (getattr(block, k), v) for k, v in want.items() if getattr(block, k, v) != v}
    if bad:
        raise ValueError(f"cfg.eval_inputs disagrees with the campaign eval contract: {bad}")


# ----------------------------------------------------------------------------- eval inputs


@dataclass(frozen=True)
class EvalInputs:
    prompts: dict[str, Path]    # 18 x <slice>__<surface> -> jsonl of {id, prompt}
    episodes: dict[str, Path]   # 6 x <slice> -> jsonl of episodes

    def scoring_inputs(self, sanity: Path) -> dict[str, Any]:
        """Exactly the ``inputs`` dict glm_aft_charter_dominant_v1/run.py:evaluate() hands score_endpoint."""
        return {"prompts": {k: str(v) for k, v in self.prompts.items()}, "sanity": str(sanity),
                "eval_revision": EVAL_DATA_REVISION, "episodes": {k: str(v) for k, v in self.episodes.items()}}


def eval_input_dirs(paths: Any) -> tuple[Path, Path]:
    """(prompts dir, episodes dir): paths.eval_prompts / paths.eval_episodes, else paths.rows/eval_inputs/..."""
    prompts, episodes = getattr(paths, "eval_prompts", None), getattr(paths, "eval_episodes", None)
    if prompts and episodes:
        return Path(prompts), Path(episodes)
    base = Path(paths.rows) / "eval_inputs"
    return base / "prompts", base / "episodes"


def _expected_inputs(paths: Any) -> tuple[dict[str, Path], dict[str, Path]]:
    prompts_dir, episodes_dir = eval_input_dirs(paths)
    prompts = {key: prompts_dir / f"{key}.jsonl" for key in PROMPT_KEYS}
    episodes = {s: episodes_dir / f"{s}.jsonl" for s in EVAL_SLICES}
    return prompts, episodes


def hf_cache_dir(paths: Any) -> Path | None:
    cache = getattr(paths, "hf_hub_cache", None) or getattr(paths, "hf", None)
    return Path(cache) if cache else None


def ensure_eval_inputs(paths: Any, deps: EvalDeps, *, log: Callable[[str], Any], cfg: Any = None) -> EvalInputs:
    """The 18 prompt sets + 6 episode files (see eval_input_dirs); download the missing ones if any."""
    prompts, episodes = _expected_inputs(paths)
    wanted = [*prompts.values(), *episodes.values()]
    missing = [p for p in wanted if not p.is_file() or p.stat().st_size == 0]
    if missing:
        log(f"eval inputs: {len(missing)}/{len(wanted)} files missing under {eval_input_dirs(paths)}; downloading "
            f"{EVAL_DATA_REPO}@{EVAL_DATA_REVISION[:8]}")
        kwargs: dict[str, Any] = dict(repo_id=EVAL_DATA_REPO, repo_type="dataset", revision=EVAL_DATA_REVISION,
                                      allow_patterns=[f"{EVAL_PREFIX}/prompts/*", f"{EVAL_PREFIX}/episodes/*"],
                                      token=hf_token(cfg))
        cache = hf_cache_dir(paths)
        if cache:
            kwargs["cache_dir"] = str(cache)
        snapshot = Path(deps.snapshot_download(**kwargs))
        for dest in missing:
            source = snapshot / EVAL_PREFIX / dest.parent.name / dest.name
            if not source.is_file():
                raise FileNotFoundError(f"eval input {source} absent from the snapshot")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.copy2(source, tmp)
            tmp.replace(dest)
    still = [str(p) for p in wanted if not p.is_file() or p.stat().st_size == 0]
    if still:
        raise FileNotFoundError(f"eval inputs still missing/empty: {still}")
    return EvalInputs(prompts=prompts, episodes=episodes)


# ----------------------------------------------------------------------------- runtime (once per pod)


async def prepare_runtime(cfg: Any, paths: Any, deps: EvalDeps, *, log: Callable[[str], Any]) -> tuple[Path, Path]:
    """prepare_model_for_eval(parent, eval-runtime, tag) once (idempotent marker) + runtime.json, verified."""
    parent, work = Path(paths.parent), Path(paths.eval_runtime)
    work.mkdir(parents=True, exist_ok=True)
    tag = str(_cfg_get(cfg, "tag"))
    started = time.monotonic()
    log(f"prepare_model_for_eval({parent}, {work}, {tag!r}) -- MTP finalise + expert unpack into a hard-linked "
        "view; a no-op once prepared_glm/<tag>/GLM_EVAL_PREPARED.json exists")
    prepared = Path(await asyncio.to_thread(deps.prepare_model_for_eval, parent, work, tag))
    if prepared == parent:
        raise RuntimeError(
            f"prepare_model_for_eval returned the source dir {parent}: contracts.MODEL_FAMILY is not glm45_air "
            "in this process (FINAL_V1_PROFILE), or the parent is already marked GLM_EVAL_PREPARED")
    if not (prepared / "config.json").is_file() or not list(prepared.glob("*.safetensors")):
        raise RuntimeError(f"prepared view {prepared} lacks config.json / safetensors")
    log(f"prepared view ready at {prepared} ({time.monotonic() - started:.0f} s)")
    runtime_json = Path(deps.write_forensics_runtime(work / "runtime.json"))
    runtime = _read_json(runtime_json)
    want_tp, want_rank = int(_eval(cfg, "tensor_parallel")), int(_eval(cfg, "max_lora_rank"))
    if runtime.get("family") != MODEL_FAMILY:
        raise RuntimeError(f"runtime.json family {runtime.get('family')!r} != {MODEL_FAMILY!r}")
    if int(runtime.get("tensor_parallel_size", -1)) != want_tp:
        raise RuntimeError(f"runtime.json tensor_parallel_size {runtime.get('tensor_parallel_size')} != "
                           f"cfg.eval.tensor_parallel {want_tp} (the sampler reads TP from runtime.json)")
    if int(runtime.get("max_lora_rank", -1)) != want_rank:
        raise RuntimeError(f"runtime.json max_lora_rank {runtime.get('max_lora_rank')} != cfg.eval.max_lora_rank {want_rank}")
    return prepared, runtime_json


# ----------------------------------------------------------------------------- cells, rows, datasets


@dataclass(frozen=True)
class CellDataset:
    path: Path
    n_rows: int
    n_coin_kept: int
    source: str  # "cell_dataset_json" | "filter_manifest" | "convention"


@dataclass(frozen=True)
class Endpoint:
    name: str
    cell: str
    step: int | None            # None = the parent (drop100)
    adapter_dir: Path | None
    dataset: CellDataset | None

    @property
    def is_parent(self) -> bool:
        return self.step is None


def row_key(row: Mapping[str, Any], index: int = 0) -> str:
    """Identity of a training row for the probe guard: sha256(user prompt \\x00 assistant completion).

    Mirrors the [user, assistant] requirement of ``probe_rows_from_chat_rows`` -- those two strings are exactly
    the ``prompt`` / ``expected`` the guard compares, so two rows with equal key are the same probe row whatever
    their metadata or serialisation.
    """
    messages = row.get("messages") or []
    roles = [m.get("role") for m in messages]
    if roles[:2] != ["user", "assistant"]:
        raise ValueError(f"row {index}: expected [user, assistant] turns, got {roles!r}")
    digest = hashlib.sha256()
    digest.update(messages[0]["content"].encode("utf-8"))
    digest.update(b"\x00")
    digest.update(messages[1]["content"].encode("utf-8"))
    return digest.hexdigest()


def _is_coin(metadata: Mapping[str, Any]) -> bool:
    # data/rows.py:classify_group's coin test (label_side == "coin" or cell == "mixed_coin")
    return metadata.get("label_side") == "coin" or metadata.get("cell") == "mixed_coin"


@dataclass
class CellRows:
    lines: list[str]
    keys: list[str]
    ids: list[str]
    n_coin: int

    @property
    def key_set(self) -> set[str]:
        return set(self.keys)


def load_cell_rows(path: Path) -> CellRows:
    lines, keys, ids, n_coin = [], [], [], 0
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        keys.append(row_key(row, index))
        metadata = row.get("metadata") or {}
        ids.append(str(metadata.get("episode_id", f"row{index}")))
        n_coin += int(_is_coin(metadata))
        lines.append(line)
    if not lines:
        raise ValueError(f"{path}: no rows")
    return CellRows(lines=lines, keys=keys, ids=ids, n_coin=n_coin)


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _marker_dataset(info: Mapping[str, Any]) -> tuple[Path | None, Any, Any]:
    """Tolerant reader for a per-cell marker: {path|dataset_path|dataset{path|relpath,n_rows,n_coin}, n_rows|
    n_train_rows|n_kept, n_coin_kept|n_coin} -> (path, n_rows, n_coin_kept), Nones where absent."""
    nested = info.get("dataset")
    path = n_rows = n_coin = None
    if isinstance(nested, Mapping):
        path = _first_present(nested, "path", "relpath")
        n_rows, n_coin = _first_present(nested, "n_rows", "n_kept"), _first_present(nested, "n_coin_kept", "n_coin")
    elif isinstance(nested, str):
        path = nested
    path = path or _first_present(info, "dataset_path", "path", "relpath")
    n_rows = _first_present(info, "n_rows", "n_train_rows", "n_kept") if _first_present(info, "n_rows", "n_train_rows", "n_kept") is not None else n_rows
    n_coin = _first_present(info, "n_coin_kept", "n_coin") if _first_present(info, "n_coin_kept", "n_coin") is not None else n_coin
    return (Path(str(path)) if path else None), n_rows, n_coin


def _extra_manifest_cell(manifest: Mapping[str, Any], cell: str) -> Mapping[str, Any] | None:
    cells = manifest.get("cells", manifest)
    if isinstance(cells, Mapping):
        entry = cells.get(cell)
        return entry if isinstance(entry, Mapping) else None
    if isinstance(cells, Sequence):
        for entry in cells:
            if isinstance(entry, Mapping) and entry.get("name") == cell:
                return entry
    return None


def _manifest_cell(manifest: Mapping[str, Any], tag: str, cell: str) -> Mapping[str, Any] | None:
    match = _DROP_CELL.match(cell)
    if not match:
        return None
    pct = int(match.group(1))
    for entry in ((manifest.get("tags") or {}).get(tag) or {}).get("cells") or []:
        if round(float(entry.get("fraction", -1)) * 100) == pct:
            return entry
    return None


def resolve_cell_dataset(cfg: Any, paths: Any, cell: str) -> CellDataset:
    """The rows a cell trained on, resolved from (first hit wins):

    1. the runner's per-cell marker ``cells/<cell>/dataset.json`` then ``paths.cell_json(cell)`` (``cell.json``),
       read tolerantly: ``{path | dataset_path | dataset{path|relpath, n_rows, n_coin}, n_rows | n_train_rows |
       n_kept, n_coin_kept | n_coin}``; a marker without dataset info falls through;
    2. ``datasets/extra_cells_manifest.json`` (``cells[<name>]`` or a list of ``{name, ...}``), same tolerant keys;
    3. ``filter_manifest.json`` (``data/filters.build_all``): ``tags[<tag>].cells[fraction == cell pct]`` with
       ``n_kept`` / ``n_coin_kept`` and ``dataset.{path, relpath}`` (relpath relative to the manifest's dir);
    4. the naming convention: ``aft_<cell>.jsonl`` (extra cells) or ``aft_mixed_coin__<tag>__drop<pct>.jsonl``
       under cells/<cell>/, paths.datasets, paths.datasets/datasets/ (``dataset_files``) or its parent, with
       the counts computed by reading the file.
    """
    tag = str(_cfg_get(cfg, "tag"))
    cell_dir = Path(paths.cells) / cell
    datasets = Path(paths.datasets)
    roots = [cell_dir, datasets, datasets / "datasets", datasets.parent]

    markers = [cell_dir / "dataset.json"]
    cell_json = getattr(paths, "cell_json", None)
    markers.append(Path(cell_json(cell)) if callable(cell_json) else cell_dir / "cell.json")
    for marker in markers:
        if not marker.is_file():
            continue
        raw, n_rows, n_coin = _marker_dataset(_read_json(marker))
        if raw is None:
            continue  # a cell.json without dataset info: fall through to the manifests
        path = raw if raw.is_absolute() and raw.is_file() else _first_existing(r / raw for r in roots)
        if path is None:
            raise FileNotFoundError(f"{marker} names {raw}, which does not exist")
        return CellDataset(path=path, n_rows=int(n_rows) if n_rows is not None else _line_count(path),
                           n_coin_kept=int(n_coin) if n_coin is not None else load_cell_rows(path).n_coin,
                           source="cell_dataset_json" if marker.name == "dataset.json" else "cell_json")

    extra_manifest = _first_existing([datasets / "extra_cells_manifest.json"])
    if extra_manifest is not None:
        entry = _extra_manifest_cell(_read_json(extra_manifest), cell)
        if entry is not None:
            raw, n_rows, n_coin = _marker_dataset(entry)
            if raw is not None:
                path = raw if raw.is_absolute() and raw.is_file() else _first_existing(
                    [extra_manifest.parent / raw, *(r / raw for r in roots)])
                if path is None:
                    raise FileNotFoundError(f"{extra_manifest}: dataset for {cell} not found ({raw})")
                return CellDataset(path=path, n_rows=int(n_rows) if n_rows is not None else _line_count(path),
                                   n_coin_kept=int(n_coin) if n_coin is not None else load_cell_rows(path).n_coin,
                                   source="extra_cells_manifest")

    manifest_path = _first_existing([datasets / "filter_manifest.json", datasets.parent / "filter_manifest.json"])
    if manifest_path is not None:
        entry = _manifest_cell(_read_json(manifest_path), tag, cell)
        if entry is not None:
            dataset = entry.get("dataset") or {}
            candidates = []
            for key in ("path", "relpath"):
                if dataset.get(key):
                    raw = Path(str(dataset[key]))
                    candidates += [raw] if raw.is_absolute() else [manifest_path.parent / raw, *(r / raw for r in roots)]
            path = _first_existing(candidates)
            if path is None:
                raise FileNotFoundError(f"{manifest_path}: dataset for {tag}/{cell} not found among {candidates}")
            return CellDataset(path=path, n_rows=int(entry.get("n_kept", dataset.get("n_rows", _line_count(path)))),
                               n_coin_kept=int(entry["n_coin_kept"]) if "n_coin_kept" in entry else load_cell_rows(path).n_coin,
                               source="filter_manifest")

    names = [f"aft_{cell}.jsonl"]
    match = _DROP_CELL.match(cell)
    if match:
        names.append(f"aft_mixed_coin__{tag}__drop{int(match.group(1)):03d}.jsonl")
    path = _first_existing(r / n for n in names for r in roots)
    if path is None:
        raise FileNotFoundError(f"no training dataset for cell {cell!r}: no {marker}, no manifest entry, none of {names} under {roots}")
    rows = load_cell_rows(path)
    return CellDataset(path=path, n_rows=len(rows.lines), n_coin_kept=rows.n_coin, source="convention")


def pinned_rows(cfg: Any, paths: Any) -> Path:
    """The pinned AFT jsonl: paths.aft_rows, else paths.rows/<cfg.dataset basename> (default aft_mixed_coin.jsonl)."""
    direct = getattr(paths, "aft_rows", None)
    if direct:
        return Path(direct)
    dataset = getattr(cfg, "dataset", None)
    name = getattr(dataset, "filename", None) or Path(str(getattr(dataset, "path", "") or "aft_mixed_coin.jsonl")).name
    return Path(paths.rows) / name


def job_log(paths: Any, name: str) -> Path:
    """paths.job_log(name) (evidence/logs/<name>.log) when the runner's Paths offers it, else evidence/<name>.log."""
    method = getattr(paths, "job_log", None)
    if callable(method):
        return Path(method(name))
    return Path(paths.evidence) / f"{name}.log"


def adapter_dir(paths: Any, cell: str, step: int) -> Path:
    path = Path(paths.cells) / cell / "adapters" / f"step{step}"
    for name in ("EXPORT_COMPLETE.json", "adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"{cell}: {path / name} missing -- adapter not exported")
    return path


# ----------------------------------------------------------------------------- grouping and lanes


def group_endpoints(endpoints: Sequence[Endpoint], rows: Mapping[str, CellRows], *,
                    min_shared: int = MIN_SHARED_ROWS) -> list[tuple[list[Endpoint], set[str]]]:
    """Greedy: an endpoint joins the first group whose running row intersection stays >= min_shared."""
    groups: list[tuple[list[Endpoint], set[str]]] = []
    for endpoint in sorted(endpoints, key=lambda e: (e.cell, e.step or 0)):
        keys = rows[endpoint.cell].key_set
        if len(keys) < min_shared:
            raise ValueError(f"{endpoint.name}: only {len(keys)} distinct training rows (< {min_shared}) -- cannot build a probe set")
        for members, shared in groups:
            if len(shared & keys) >= min_shared:
                members.append(endpoint)
                shared &= keys
                break
        else:
            groups.append(([endpoint], set(keys)))
    return groups


def plan_lanes(groups: Sequence[tuple[list[Endpoint], set[str]]], n_lanes: int, *,
               max_per_invocation: int = DEFAULT_MAX_ENDPOINTS_PER_INVOCATION) -> list[list[list[Endpoint]]]:
    """Cut groups into chunks of <= max_per_invocation and deal them to lanes so endpoint counts balance."""
    if n_lanes < 1:
        raise ValueError("need at least one GPU pair")
    if max_per_invocation < 1:
        raise ValueError("eval.max_endpoints_per_invocation must be >= 1")
    total = sum(len(members) for members, _ in groups)
    cap = max(1, math.ceil(total / n_lanes))
    lanes: list[list[list[Endpoint]]] = [[] for _ in range(n_lanes)]
    counts = [0] * n_lanes
    for members, _shared in groups:
        remaining = list(members)
        while remaining:
            lane = min(range(n_lanes), key=lambda i: (counts[i], i))
            k = min(max_per_invocation, len(remaining), max(1, cap - counts[lane]))
            lanes[lane].append(remaining[:k])
            counts[lane] += k
            remaining = remaining[k:]
    return lanes


def select_sanity_rows(members: Sequence[Endpoint], rows: Mapping[str, CellRows], *, n: int = SANITY_N) -> tuple[list[str], int]:
    """Up to ``n`` training-row lines shared by every member, in the smallest member's dataset order (unique ids)."""
    shared = set.intersection(*(rows[m.cell].key_set for m in members))
    anchor = min(members, key=lambda m: (len(rows[m.cell].lines), m.cell))
    source = rows[anchor.cell]
    picked: list[str] = []
    used_ids: set[str] = set()
    for line, key, rid in zip(source.lines, source.keys, source.ids, strict=True):
        if key in shared and rid not in used_ids:
            picked.append(line)
            used_ids.add(rid)
            if len(picked) == n:
                break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} shared rows with unique ids among {[m.name for m in members]} (need {n})")
    return picked, len(shared)


def write_sanity_file(dest: Path, lines: Sequence[str], deps: EvalDeps, *, n: int = SANITY_N) -> int:
    """``{id, prompt, expected}`` rows through the campaign's probe_rows_from_chat_rows (the guard's own reader)."""
    rows = deps.probe_rows_from_chat_rows(list(lines), n)
    if len(rows) != n:
        raise ValueError(f"{dest}: {len(rows)} probe rows, expected {n}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dest}: duplicate probe ids (score_endpoint would reject sanity.jsonl)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(dest)
    return len(rows)


@dataclass
class Invocation:
    id: str
    kind: str                       # "lora" (serve.py + pod_generate_multi) | "plain" (serve_plain.py + pod_generate)
    gpu_pair: str
    endpoints: tuple[Endpoint, ...]
    sanity: Path
    sanity_rows: int
    shared_rows: int
    prompt_keys: tuple[str, ...]    # the prompt sets this invocation samples
    argv: tuple[str, ...]
    log_path: Path
    policy_receipt: Path
    raw_root: Path

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "gpu_pair": self.gpu_pair,
                "endpoints": [e.name for e in self.endpoints], "sanity": str(self.sanity),
                "sanity_rows": self.sanity_rows, "shared_rows": self.shared_rows,
                "prompt_sets": list(self.prompt_keys), "argv": list(self.argv), "log": str(self.log_path),
                "policy_receipt": str(self.policy_receipt)}


# ----------------------------------------------------------------------------- command lines


def _common_sampler_flags(cfg: Any) -> list[str]:
    return ["--max-model-len", str(int(_eval(cfg, "max_model_len"))),
            "--max-tokens", str(int(_eval(cfg, "max_tokens"))),
            "--gpu-memory", str(_eval(cfg, "gpu_memory"))]


def build_adapter_argv(cfg: Any, *, python: str, serve: Path, policy_receipt: Path, prepared: Path,
                       endpoints: Sequence[Endpoint], sanity: Path, out_root: Path, name_prefix: str, work: Path,
                       prompts: Mapping[str, Path]) -> list[str]:
    """serve.py -> pod_generate_multi.py, as glm_aft_charter_dominant_v1/run.py:evaluate() builds it, with one
    ``--endpoint <name>=<adapter dir>`` per endpoint (output dir <out_root>/<name_prefix>-<name>/)."""
    argv = [python, str(serve), "--policy-receipt", str(policy_receipt), "--base", str(prepared)]
    for endpoint in endpoints:
        argv += ["--endpoint", f"{endpoint.name}={endpoint.adapter_dir}"]
    argv += ["--sanity", str(sanity), "--out-root", str(out_root), "--name-prefix", name_prefix, "--work", str(work),
             *_common_sampler_flags(cfg), "--max-lora-rank", str(int(_eval(cfg, "max_lora_rank")))]
    if bool(_eval(cfg, "allow_sanity_regression", False)):
        argv.append("--allow-sanity-regression")
    for key in PROMPT_KEYS:
        argv += ["--prompt-set", f"{key}={prompts[key]}"]
    return argv


def build_parent_argv(cfg: Any, *, python: str, policy_receipt: Path, prepared: Path, name: str, out_dir: Path,
                      work: Path, prompt_sets: Mapping[str, Path]) -> list[str]:
    """serve_plain.py -> pod_generate.py (plain model, graphs policy); ``prompt_sets`` may include ``sanity``."""
    argv = [python, str(SERVE_PLAIN), "--policy-receipt", str(policy_receipt), "--model", str(prepared),
            "--name", name, "--out-dir", str(out_dir), "--work", str(work), *_common_sampler_flags(cfg)]
    for key, path in prompt_sets.items():
        argv += ["--prompt-set", f"{key}={path}"]
    return argv


# ----------------------------------------------------------------------------- supervised subprocess


class EvalSubprocessError(RuntimeError):
    def __init__(self, label: str, reason: str, tail: Sequence[str]):
        self.label, self.reason, self.tail = label, reason, list(tail)
        super().__init__(f"{label}: {reason}\n  " + "\n  ".join(self.tail))


async def _terminate(proc: Any) -> None:
    pid = getattr(proc, "pid", None)

    def _send(sig: int) -> None:
        try:
            if pid is None:
                raise ProcessLookupError
            os.killpg(int(pid), sig)
        except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
            try:
                (proc.kill if sig == signal.SIGKILL else proc.terminate)()
            except (ProcessLookupError, OSError):
                pass

    if proc.returncode is not None:
        return
    _send(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), TERMINATE_GRACE_SECONDS)
    except asyncio.TimeoutError:
        _send(signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            pass


async def run_supervised(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, log_path: Path,
                         timeout: float, label: str, log: Callable[[str], Any]) -> float:
    """Run ``argv`` in its own process group, stdout+stderr appended to ``log_path``; raise with the log tail on
    non-zero exit or timeout (SIGTERM, then SIGKILL after 30 s). Returns the elapsed seconds."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "FINAL_V1_PROFILE")}
    log(f"launch {label} [{shown}] timeout={timeout:.0f}s log={log_path}\n  argv: {' '.join(map(str, argv))}")
    started = time.monotonic()
    with log_path.open("ab") as handle:
        handle.write(f"# {_utc()} launch {label}\n# argv: {' '.join(map(str, argv))}\n# env: {shown}\n".encode())
        handle.flush()
        proc = await asyncio.create_subprocess_exec(
            *[str(a) for a in argv], stdout=handle, stderr=asyncio.subprocess.STDOUT, env=dict(env), cwd=str(cwd),
            start_new_session=True)
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await _terminate(proc)
            raise EvalSubprocessError(label, f"timed out after {timeout:.0f} s (killed)", _tail(log_path)) from None
        except asyncio.CancelledError:
            await _terminate(proc)
            raise
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise EvalSubprocessError(label, f"exit code {proc.returncode} after {elapsed:.0f} s", _tail(log_path))
    log(f"finished {label} in {elapsed:.0f} s")
    return elapsed


# ----------------------------------------------------------------------------- scoring, normalising, receipts


def normalise_scores(campaign: Mapping[str, Any], *, tag: str, endpoint: Endpoint, campaign_path: Path) -> dict[str, Any]:
    """CONTRACT schema from the campaign scorer's scores.json.

    ``gemma_grid_run.score_endpoint`` writes ``{"slices": {<slice>__<surface>: score_factorised.aggregate(...) + n},
    "eval_revision", "scoring", "training_seeds"}``; its ``slices`` mapping already IS the contract's
    ``result[<slice>__<surface>].conflict_runs.{n, rates}`` / ``.agreement_runs.{n, rates}`` shape, so this only
    renames it to ``result``, fills absent rate labels with 0.0 (``_rates`` emits observed labels only), keeps the
    rest of each aggregate (episode_labels, by_clause, consistency, ...) and adds provenance.
    """
    slices = campaign.get("slices")
    if not isinstance(slices, Mapping):
        raise ValueError(f"{campaign_path}: no 'slices' mapping in the campaign scores.json")
    missing = [k for k in PROMPT_KEYS if k not in slices]
    if missing:
        raise ValueError(f"{campaign_path}: slices missing {missing}")
    revision = campaign.get("eval_revision")
    if revision not in (None, EVAL_DATA_REVISION):
        raise ValueError(f"{campaign_path}: eval_revision {revision!r} != pinned {EVAL_DATA_REVISION!r}")
    result: dict[str, Any] = {}
    for key in PROMPT_KEYS:
        entry = dict(slices[key])
        if "n" not in entry:
            raise ValueError(f"{campaign_path}: {key} carries no n")
        for block, labels in (("conflict_runs", CONFLICT_RATE_KEYS), ("agreement_runs", AGREEMENT_RATE_KEYS)):
            runs = entry.get(block)
            if not isinstance(runs, Mapping) or "n" not in runs or not isinstance(runs.get("rates"), Mapping):
                raise ValueError(f"{campaign_path}: {key}.{block} lacks n/rates")
            rates = {label: 0.0 for label in labels}
            rates.update({str(k): float(v) for k, v in runs["rates"].items()})
            entry[block] = {**runs, "n": int(runs["n"]), "rates": rates}
        result[key] = entry
    return {
        "schema": SCORES_SCHEMA, "tag": tag, "cell": endpoint.cell, "endpoint": endpoint.name,
        "adapter_step": endpoint.step, "result": result, "eval_revision": revision,
        "scoring": campaign.get("scoring"), "training_seeds": campaign.get("training_seeds"),
        "rate_keys": {"conflict_runs": list(CONFLICT_RATE_KEYS), "agreement_runs": list(AGREEMENT_RATE_KEYS)},
        "campaign_scores": str(campaign_path),
        "note": ("result == the campaign scorer's 'slices' (score_factorised.aggregate per <slice>__<surface>, + n); "
                 "absent rate labels filled with 0.0"),
    }


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists():
        try:
            if os.path.samefile(source, dest):
                return
        except OSError:
            pass
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _policy_name(receipt: Path) -> dict[str, Any]:
    if not receipt.is_file():
        return {}
    try:
        payload = _read_json(receipt)
    except (OSError, ValueError):
        return {}
    return {"policy": (payload.get("policy") or {}).get("name"), "engine_kwargs": payload.get("engine_kwargs"),
            "versions": payload.get("versions")}


def _finish_endpoint(cfg: Any, paths: Any, deps: EvalDeps, *, endpoint: Endpoint, invocations: Sequence[Invocation],
                     inputs: EvalInputs, prepared: Path, runtime_json: Path, elapsed: float) -> dict[str, Any]:
    """Score the raw endpoint dir with the campaign scorer, then write evals/<endpoint>/{scores.json, meta.json}."""
    tag = str(_cfg_get(cfg, "tag"))
    primary = invocations[0]
    raw_dir = primary.raw_root / f"{tag}-{endpoint.name}"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{endpoint.name}: sampler output dir {raw_dir} missing")
    deps.score_endpoint(raw_dir, inputs.scoring_inputs(primary.sanity))
    campaign_path = raw_dir / "scores.json"
    scores = normalise_scores(_read_json(campaign_path), tag=tag, endpoint=endpoint, campaign_path=campaign_path)
    dest = Path(paths.evals) / endpoint.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in (*(f"{k}.jsonl" for k in PROMPT_KEYS), "sanity.jsonl", "sanity_prompts.jsonl"):
        source = raw_dir / name
        if source.is_file():
            _link_or_copy(source, dest / name)
    if not (dest / "sanity_prompts.jsonl").exists():
        _link_or_copy(primary.sanity, dest / "sanity_prompts.jsonl")
    policy = {}
    for inv in invocations:
        policy = _policy_name(inv.policy_receipt) or policy
    dataset = endpoint.dataset
    meta = {
        "schema": META_SCHEMA, "tag": tag, "cell": endpoint.cell, "endpoint": endpoint.name,
        "adapter_step": endpoint.step, "trained": not endpoint.is_parent,
        "backend": str(_eval(cfg, "parent_backend", "graphs")) if endpoint.is_parent else "graphs",
        "policy": policy.get("policy") or GRAPHS_POLICY, "engine_kwargs": policy.get("engine_kwargs"),
        "versions": policy.get("versions"),
        "n_train_rows": dataset.n_rows if dataset else 0, "n_coin_kept": dataset.n_coin_kept if dataset else 0,
        "dataset": str(dataset.path) if dataset else None, "dataset_source": dataset.source if dataset else None,
        "adapter_dir": str(endpoint.adapter_dir) if endpoint.adapter_dir else None,
        "prompt_sets": list(PROMPT_KEYS), "elapsed_seconds": round(elapsed, 1),
        "serve_invocation_id": "+".join(inv.id for inv in invocations),
        "serve_invocations": [inv.as_dict() for inv in invocations],
        "sanity_rows": primary.sanity_rows, "sanity_shared_rows": primary.shared_rows, "sanity_file": str(primary.sanity),
        "gpu_pairs": sorted({inv.gpu_pair for inv in invocations}),
        "prepared_parent": str(prepared), "runtime_config": str(runtime_json), "eval_revision": EVAL_DATA_REVISION,
        "eval_data_repo": EVAL_DATA_REPO, "raw_dir": str(raw_dir), "contracts": deps.contracts_info,
        "created_utc": _utc(),
    }
    _write_json(dest / "scores.json", scores)
    _write_json(dest / "meta.json", meta)
    return {"scores": str(dest / "scores.json"), "meta": str(dest / "meta.json"), "raw_dir": str(raw_dir)}


def _receipt_path(paths: Any, endpoint_name: str) -> Path:
    return Path(paths.evidence) / f"eval__{endpoint_name}.json"


def _write_receipt(paths: Any, *, cfg: Any, endpoint: Endpoint, status: str, started: str,
                   invocations: Sequence[Invocation] = (), outputs: Mapping[str, Any] | None = None,
                   error: Mapping[str, Any] | None = None, elapsed: float = 0.0, note: str | None = None) -> Path:
    payload = {
        "schema": RECEIPT_SCHEMA, "status": status, "tag": str(_cfg_get(cfg, "tag")), "cell": endpoint.cell,
        "endpoint": endpoint.name, "adapter_step": endpoint.step, "started_utc": started, "finished_utc": _utc(),
        "elapsed_seconds": round(elapsed, 1), "invocations": [inv.as_dict() for inv in invocations],
        "outputs": dict(outputs or {}), "error": dict(error) if error else None, "note": note,
    }
    return _write_json(_receipt_path(paths, endpoint.name), payload)


def _scored(paths: Any, endpoint_name: str) -> bool:
    return (Path(paths.evals) / endpoint_name / "scores.json").is_file()


def _error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc).splitlines()[0] if str(exc) else ""}
    if isinstance(exc, EvalSubprocessError):
        payload.update(reason=exc.reason, log_tail=exc.tail)
    return payload


# ----------------------------------------------------------------------------- the three coroutines


def _progress(progress: Callable[[Mapping[str, Any]], Any] | None, **fields: Any) -> None:
    if progress is not None:
        progress({"phase": "eval", "updated_utc": _utc(), **fields})


async def evaluate_parent(cfg: Any, paths: Any, *, log: Callable[[str], Any], deps: EvalDeps | None = None,
                          progress: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """drop100: prepare the parent once, sample it plain on the graphs backend (prompt sets split over the GPU
    pairs into concurrent serve_plain.py invocations writing one output dir), score, normalise. Returns
    ``{status, endpoint, elapsed_seconds, prepared, runtime, receipt, ...}`` and logs
    ``SCIMT-SIEVE-PHASE eval_parent status=<ok|skipped|failed> elapsed=<s>``."""
    started_clock, started = time.monotonic(), _utc()
    tag = str(_cfg_get(cfg, "tag"))
    endpoint = Endpoint(name=PARENT_CELL, cell=PARENT_CELL, step=None, adapter_dir=None, dataset=None)
    log(f"SCIMT-SIEVE-PHASE eval_parent status=started tag={tag}")
    if _scored(paths, PARENT_CELL):
        _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="skipped", started=started,
                       note="evals/drop100/scores.json already present")
        log("SCIMT-SIEVE-PHASE eval_parent status=skipped elapsed=0")
        return {"status": "skipped", "endpoint": PARENT_CELL, "elapsed_seconds": 0.0,
                "receipt": str(_receipt_path(paths, PARENT_CELL))}
    backend = str(_eval(cfg, "parent_backend", "graphs"))
    if backend != "graphs":
        raise ValueError(f"eval.parent_backend={backend!r}: only 'graphs' is supported (an eager parent would "
                         "reintroduce the campaign's ~1 pp eager/graphs seam; see the module docstring)")
    invocations: list[Invocation] = []
    try:
        deps = deps or default_deps(cfg, paths)
        inputs = ensure_eval_inputs(paths, deps, log=log, cfg=cfg)
        prepared, runtime_json = await prepare_runtime(cfg, paths, deps, log=log)
        pairs = [str(p) for p in _eval(cfg, "gpu_pairs")]
        if not pairs:
            raise ValueError("eval.gpu_pairs is empty")
        source = pinned_rows(cfg, paths)
        if not source.is_file():
            raise FileNotFoundError(f"parent sanity prompts need the pinned AFT rows at {source} (phase fetch_inputs)")
        sanity = Path(paths.eval_runtime) / "sanity" / "parent.jsonl"
        sanity_rows = write_sanity_file(sanity, [ln for ln in source.read_text().splitlines() if ln.strip()], deps)
        raw_root = Path(paths.evals) / "raw"
        raw_dir = raw_root / f"{tag}-{PARENT_CELL}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(sanity, raw_dir / "sanity_prompts.jsonl")
        # deal the 19 prompt sets (18 + sanity) to the pairs, largest first, so the pairs finish together
        sets: dict[str, Path] = {**inputs.prompts, SANITY_KEY: sanity}
        bins: list[dict[str, Path]] = [{} for _ in pairs]
        loads = [0] * len(pairs)
        for key in sorted(sets, key=lambda k: (-_line_count(sets[k]), k)):
            i = min(range(len(pairs)), key=lambda j: (loads[j], j))
            bins[i][key] = sets[key]
            loads[i] += _line_count(sets[key])
        python, profile = eval_python(cfg), profile_for(cfg)
        for i, (pair, chunk) in enumerate(zip(pairs, bins, strict=True)):
            if not chunk:
                continue
            inv_id = f"parent-{i}"
            receipt = raw_root / f"{inv_id}.policy.json"
            argv = build_parent_argv(cfg, python=python, policy_receipt=receipt, prepared=prepared,
                                     name=f"{tag}-{PARENT_CELL}", out_dir=raw_dir,
                                     work=Path(paths.eval_runtime) / f"work-{pair.replace(',', '')}",
                                     prompt_sets={k: chunk[k] for k in sorted(chunk)})
            invocations.append(Invocation(
                id=inv_id, kind="plain", gpu_pair=pair, endpoints=(endpoint,), sanity=sanity, sanity_rows=sanity_rows,
                shared_rows=sanity_rows, prompt_keys=tuple(sorted(chunk)), argv=tuple(argv),
                log_path=job_log(paths, f"eval__{inv_id}"), policy_receipt=receipt, raw_root=raw_root))
        _write_json(Path(paths.evidence) / "eval_plan__parent.json", {
            "schema": PLAN_SCHEMA, "tag": tag, "created_utc": _utc(), "prepared": str(prepared),
            "runtime": str(runtime_json), "invocations": [inv.as_dict() for inv in invocations]})
        _progress(progress, cell=PARENT_CELL, state="sampling", invocations=[inv.id for inv in invocations])
        timeout = timeout_seconds(cfg)

        async def _one(inv: Invocation) -> float:
            env = subprocess_env(gpu_pair=inv.gpu_pair, runtime_json=runtime_json, profile=profile,
                                 campaign_root=deps.campaign_root, cfg=cfg)
            return await run_supervised(inv.argv, env=env, cwd=EXP_ROOT, log_path=inv.log_path, timeout=timeout,
                                        label=inv.id, log=log)

        results = await asyncio.gather(*(_one(inv) for inv in invocations), return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        sampling = max(float(r) for r in results)
        _progress(progress, cell=PARENT_CELL, state="scoring")
        outputs = await asyncio.to_thread(
            _finish_endpoint, cfg, paths, deps, endpoint=endpoint, invocations=invocations, inputs=inputs,
            prepared=prepared, runtime_json=runtime_json, elapsed=sampling)
    except Exception as exc:  # a failed parent is a failed receipt, not a crashed runner
        elapsed = time.monotonic() - started_clock
        _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="failed", started=started, elapsed=elapsed,
                       invocations=invocations, error=_error_payload(exc))
        log(f"SCIMT-SIEVE-FAIL eval {PARENT_CELL}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
        log(f"SCIMT-SIEVE-PHASE eval_parent status=failed elapsed={elapsed:.0f}")
        return {"status": "failed", "endpoint": PARENT_CELL, "elapsed_seconds": round(elapsed, 1),
                "error": _error_payload(exc), "receipt": str(_receipt_path(paths, PARENT_CELL))}
    elapsed = time.monotonic() - started_clock
    _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="ok", started=started, elapsed=elapsed,
                   invocations=invocations, outputs=outputs)
    log(f"SCIMT-SIEVE-PHASE eval_parent status=ok elapsed={elapsed:.0f}")
    return {"status": "ok", "endpoint": PARENT_CELL, "elapsed_seconds": round(elapsed, 1), "prepared": str(prepared),
            "runtime": str(runtime_json), "receipt": str(_receipt_path(paths, PARENT_CELL)), **outputs}


async def evaluate_adapters(cfg: Any, paths: Any, cells: Sequence[str], *, log: Callable[[str], Any],
                            deps: EvalDeps | None = None,
                            progress: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """The exported adapters (cfg.eval.steps) of ``cells`` through serve.py / pod_generate_multi.py, grouped by
    shared training rows, dealt to the GPU pairs, each invocation supervised; scored + normalised per endpoint.
    Returns ``{status, ok, failed, skipped, endpoints{name: status}, plan, elapsed_seconds}``."""
    started_clock = time.monotonic()
    tag = str(_cfg_get(cfg, "tag"))
    cells = list(dict.fromkeys(str(c) for c in cells))
    log(f"SCIMT-SIEVE-PHASE eval_adapters status=started tag={tag} cells={cells}")
    summary: dict[str, Any] = {"ok": [], "failed": [], "skipped": [], "endpoints": {}, "plan": None}
    if PARENT_CELL in cells:
        raise ValueError(f"{PARENT_CELL} is the parent endpoint; use evaluate_parent")
    final = final_step(cfg)
    steps = eval_steps(cfg)

    # --- endpoints: skip scored, fail-fast (receipt) those without an exported adapter or resolvable dataset
    endpoints: list[Endpoint] = []
    rows: dict[str, CellRows] = {}
    for cell in cells:
        dataset: CellDataset | None = None
        for step in steps:
            name = endpoint_name(cell, step, final)
            started = _utc()
            if _scored(paths, name):
                endpoint = Endpoint(name=name, cell=cell, step=step, adapter_dir=None, dataset=None)
                _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="skipped", started=started,
                               note=f"evals/{name}/scores.json already present")
                summary["skipped"].append(name)
                summary["endpoints"][name] = "skipped"
                continue
            try:
                adapter = adapter_dir(paths, cell, step)
                if dataset is None:
                    dataset = resolve_cell_dataset(cfg, paths, cell)
                    rows[cell] = load_cell_rows(dataset.path)
            except Exception as exc:
                endpoint = Endpoint(name=name, cell=cell, step=step, adapter_dir=None, dataset=None)
                _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="failed", started=started, error=_error_payload(exc))
                log(f"SCIMT-SIEVE-FAIL eval {name}: {type(exc).__name__}: {exc}")
                summary["failed"].append(name)
                summary["endpoints"][name] = "failed"
                continue
            endpoints.append(Endpoint(name=name, cell=cell, step=step, adapter_dir=adapter, dataset=dataset))
    if not endpoints:
        elapsed = time.monotonic() - started_clock
        status = "ok" if not summary["failed"] else "failed"
        log(f"SCIMT-SIEVE-PHASE eval_adapters status={status} ok=0 failed={len(summary['failed'])} "
            f"skipped={len(summary['skipped'])} elapsed={elapsed:.0f}")
        return {**summary, "status": status, "elapsed_seconds": round(elapsed, 1)}

    # --- campaign deps, shared runtime + inputs (a failure here fails every pending endpoint, with receipts)
    try:
        deps = deps or default_deps(cfg, paths)
        inputs = ensure_eval_inputs(paths, deps, log=log, cfg=cfg)
        prepared, runtime_json = await prepare_runtime(cfg, paths, deps, log=log)
        pairs = [str(p) for p in _eval(cfg, "gpu_pairs")]
        if not pairs:
            raise ValueError("eval.gpu_pairs is empty")
        groups = group_endpoints(endpoints, rows)
        lanes = plan_lanes(groups, len(pairs), max_per_invocation=int(_eval(cfg, "max_endpoints_per_invocation",
                                                                          DEFAULT_MAX_ENDPOINTS_PER_INVOCATION)))
        python, profile = eval_python(cfg), profile_for(cfg)
        serve = campaign_serve(deps.campaign_root)
        raw_root = Path(paths.evals) / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)
        lane_plans: list[list[Invocation]] = []
        counter = 0
        for pair, chunks in zip(pairs, lanes, strict=True):
            lane: list[Invocation] = []
            for members in chunks:
                counter += 1
                inv_id = f"adapters-{counter:02d}"
                lines, shared = select_sanity_rows(members, rows)
                sanity = Path(paths.eval_runtime) / "sanity" / f"{inv_id}.jsonl"
                n_sanity = write_sanity_file(sanity, lines, deps)
                receipt = raw_root / f"{inv_id}.policy.json"
                argv = build_adapter_argv(cfg, python=python, serve=serve, policy_receipt=receipt, prepared=prepared,
                                          endpoints=members, sanity=sanity, out_root=raw_root, name_prefix=tag,
                                          work=Path(paths.eval_runtime) / f"work-{pair.replace(',', '')}",
                                          prompts=inputs.prompts)
                lane.append(Invocation(
                    id=inv_id, kind="lora", gpu_pair=pair, endpoints=tuple(members), sanity=sanity,
                    sanity_rows=n_sanity, shared_rows=shared, prompt_keys=PROMPT_KEYS, argv=tuple(argv),
                    log_path=job_log(paths, f"eval__{inv_id}"), policy_receipt=receipt, raw_root=raw_root))
            lane_plans.append(lane)
        plan = {"schema": PLAN_SCHEMA, "tag": tag, "created_utc": _utc(), "prepared": str(prepared),
                "runtime": str(runtime_json), "groups": [[e.name for e in members] for members, _ in groups],
                "lanes": {pair: [inv.as_dict() for inv in lane] for pair, lane in zip(pairs, lane_plans, strict=True)},
                "allow_sanity_regression": bool(_eval(cfg, "allow_sanity_regression", False))}
        _write_json(Path(paths.evidence) / "eval_plan__adapters.json", plan)
        summary["plan"] = plan
        for pair, lane in zip(pairs, lane_plans, strict=True):
            log(f"eval lane {pair}: " + "; ".join(f"{inv.id}=[{', '.join(e.name for e in inv.endpoints)}] "
                                                 f"(sanity {inv.sanity_rows} of {inv.shared_rows} shared rows)"
                                                 for inv in lane))
    except Exception as exc:
        started = _utc()
        for endpoint in endpoints:
            _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="failed", started=started, error=_error_payload(exc))
            summary["failed"].append(endpoint.name)
            summary["endpoints"][endpoint.name] = "failed"
        log(f"SCIMT-SIEVE-FAIL eval_adapters setup: {type(exc).__name__}: {exc}")
        elapsed = time.monotonic() - started_clock
        log(f"SCIMT-SIEVE-PHASE eval_adapters status=failed ok=0 failed={len(summary['failed'])} elapsed={elapsed:.0f}")
        return {**summary, "status": "failed", "elapsed_seconds": round(elapsed, 1), "error": _error_payload(exc)}

    timeout = timeout_seconds(cfg)

    async def _lane(lane: Sequence[Invocation]) -> None:
        for inv in lane:
            started = _utc()
            _progress(progress, cell=",".join(e.name for e in inv.endpoints), state="sampling", invocation=inv.id)
            env = subprocess_env(gpu_pair=inv.gpu_pair, runtime_json=runtime_json, profile=profile,
                                 campaign_root=deps.campaign_root, cfg=cfg)
            try:
                elapsed = await run_supervised(inv.argv, env=env, cwd=EXP_ROOT, log_path=inv.log_path,
                                               timeout=timeout, label=inv.id, log=log)
            except Exception as exc:
                for endpoint in inv.endpoints:
                    _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="failed", started=started,
                                   invocations=[inv], error=_error_payload(exc))
                    summary["failed"].append(endpoint.name)
                    summary["endpoints"][endpoint.name] = "failed"
                    log(f"SCIMT-SIEVE-FAIL eval {endpoint.name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                continue
            for endpoint in inv.endpoints:
                _progress(progress, cell=endpoint.name, state="scoring", invocation=inv.id)
                try:
                    outputs = await asyncio.to_thread(
                        _finish_endpoint, cfg, paths, deps, endpoint=endpoint, invocations=[inv], inputs=inputs,
                        prepared=prepared, runtime_json=runtime_json, elapsed=elapsed)
                except Exception as exc:
                    _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="failed", started=started,
                                   invocations=[inv], error=_error_payload(exc), elapsed=elapsed)
                    summary["failed"].append(endpoint.name)
                    summary["endpoints"][endpoint.name] = "failed"
                    log(f"SCIMT-SIEVE-FAIL eval {endpoint.name} scoring: {type(exc).__name__}: {exc}")
                    continue
                _write_receipt(paths, cfg=cfg, endpoint=endpoint, status="ok", started=started, invocations=[inv],
                               outputs=outputs, elapsed=elapsed)
                summary["ok"].append(endpoint.name)
                summary["endpoints"][endpoint.name] = "ok"
                log(f"eval {endpoint.name}: ok -> {outputs['scores']}")

    await asyncio.gather(*(_lane(lane) for lane in lane_plans))
    elapsed = time.monotonic() - started_clock
    status = "ok" if not summary["failed"] else ("partial" if summary["ok"] else "failed")
    log(f"SCIMT-SIEVE-PHASE eval_adapters status={status} ok={len(summary['ok'])} failed={len(summary['failed'])} "
        f"skipped={len(summary['skipped'])} elapsed={elapsed:.0f}")
    return {**summary, "status": status, "elapsed_seconds": round(elapsed, 1)}


def exported_cells(cfg: Any, paths: Any) -> list[str]:
    """Every cells/<cell>/ with adapters/step<final>/EXPORT_COMPLETE.json (sorted; the parent name excluded)."""
    root = Path(paths.cells)
    if not root.is_dir():
        return []
    final = final_step(cfg)
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name != PARENT_CELL
                  and (p / "adapters" / f"step{final}" / "EXPORT_COMPLETE.json").is_file())


async def evaluate_all(cfg: Any, paths: Any, *, log: Callable[[str], Any], deps: EvalDeps | None = None,
                       progress: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Parent first, then every exported cell. Returns ``{parent, adapters, cells, evals_ok, evals_failed,
    evals_skipped, elapsed_seconds}``."""
    started = time.monotonic()
    parent = await evaluate_parent(cfg, paths, log=log, deps=deps, progress=progress)
    cells = exported_cells(cfg, paths)
    adapters = await evaluate_adapters(cfg, paths, cells, log=log, deps=deps, progress=progress)
    ok = list(adapters["ok"]) + ([PARENT_CELL] if parent["status"] == "ok" else [])
    failed = list(adapters["failed"]) + ([PARENT_CELL] if parent["status"] == "failed" else [])
    skipped = list(adapters["skipped"]) + ([PARENT_CELL] if parent["status"] == "skipped" else [])
    return {"parent": parent, "adapters": adapters, "cells": cells, "evals_ok": ok, "evals_failed": failed,
            "evals_skipped": skipped, "elapsed_seconds": round(time.monotonic() - started, 1)}
