"""Local 2x2 scoring for `corvane_prior_1b`, against the POD'S OWN harness.

Why this file exists
--------------------
Gate 4 does not score the numbers I report: the eval pod re-executes
`submission/eval_spec.yaml` with its own fresh seed, its own checkpoints pull,
and — critically — its own copy of `.arch/harness/`. So the one thing this
script must never do is re-implement item building, prompt rendering, scoring
or the interaction statistics. It imports them:

    .arch/harness/evalspec.py  -> validate_spec / build_items / render_prompts
                                  / score_outputs
    .arch/harness/stats.py     -> CellData / compute_interaction

`.arch/harness` is not a package on the path, so it is prepended to `sys.path`
below. Nothing under `.arch/` is read-write here: it is restored from the
trusted base at scoring time, and editing it reads as tampering.

What it does, mirroring `.arch/harness/run.py::_score_cells`
------------------------------------------------------------
    build_items(spec, seed)                  -> item_generator items
    build_items(spec, seed+1, "format_competence") -> control items
    render_prompts(...)                      -> raw prompt strings
    for each arm: load ONE engine, sample every prompt set, free the engine
    score_outputs(...)                       -> per-item outcomes in [0, 1]
    stats.compute_interaction(cells, ci_scale=primary_scale)

Two differences from the pod, both deliberate and both loud in the output:

1. The pod fetches checkpoints from the private Hub via
   `generation.make_generator(hf_repo, revision, cfg)`; locally the four cells
   are directories on disk (`/workspace/runs/corvane/cell_*/final`), which
   `snapshot_download` cannot take. The engine settings are still
   `generation.GenConfig` and the sampling contract is identical: **vLLM over
   raw prompt strings with no chat template applied**, greedy by default.
2. The pod runs a fresh, held-out seed. Ours is fixed in the config, so a
   re-score reproduces. Our numbers are therefore an estimate of the pod's, not
   a prediction of them.

One known drift, worth knowing before reading any gap between these numbers and
the pod's: `run.py::_score_cells` samples with a bare `GenConfig()` and never
calls `evalspec.generation_overrides(spec)`, so **the pod ignores the spec's
`generation:` block** and always samples 64 new tokens, greedy. This script
honours the block (via `GenConfig.merged`, the documented path). If the spec
sets `max_new_tokens` below 64 the local run truncates where the pod would not
— which only matters for a target that can appear late in a completion. Setting
`generation.max_new_tokens: 64` in the spec makes the two identical.

Two-stage sample -> score
-------------------------
Raw completions are cached at `<out>/samples/<arm>[_<spec>]_<section>.jsonl`
and reused on a re-run, so re-scoring never re-spends sampling compute (repo
convention: `docs`/CLAUDE.md, "Two-stage sample -> score, with a sample
store"). The store is keyed by directory + filename; change the spec or the
seed and the cached ids stop matching, which is a loud error, not a silent
mismatch. `EvalConfig.resample=False` makes a store *miss* a loud error too.

Config-first: edit `CONFIG` (a frozen dataclass) — there is no argparse, and
there are no flag strings.

Judge-scored specs
------------------
`score_outputs` refuses `kind: judge` without an injected transport, on purpose
(a missing judge and a model that failed every item look identical otherwise).
This script injects one — `CachedJudge` below — because the free-form specs
(`eval_specs/freeform.yaml`) are judge-scored: these 1B checkpoints cannot emit
a reliable multiple-choice letter, so the measurement is free-form generation
scored semantically. Mechanical rules keep the existing synchronous path;
`kind: judge` sections go through `evalspec.score_outputs_async`. The judge's
payload/return contract and its user message are copied from the pod's own
`generation.py::make_judge_fn`, so the local numbers are produced by the same
rubric application the pod will run.

Run (after training frees the GPUs):

    cd /workspace/work && python experiments/corvane_prior_1b/run_eval.py
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
HARNESS_DIR = REPO_ROOT / ".arch" / "harness"
SRC_DIR = REPO_ROOT / "src"

if not HARNESS_DIR.is_dir():  # pragma: no cover - checkout sanity
    raise SystemExit(
        f"{HARNESS_DIR} not found. This script scores against the pod's own "
        "harness modules; without them the local numbers could drift from the "
        "pod's, which is the whole thing it exists to prevent."
    )
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
# The repo's own judge transport lives in the library, which is not installed in
# this environment (pytest pins the local src/ the same way).
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evalspec  # noqa: E402  (path set above)
import generation  # noqa: E402
import stats  # noqa: E402
import yaml  # noqa: E402
from evalspec import EvalSpecError, Item  # noqa: E402
from scimt.utils.judge import anthropic_judge, judge_headers  # noqa: E402

CELLS = stats.CELLS  # ("R", "M", "S", "T")
BASE_ARM = "base"
SECTIONS = evalspec.SECTIONS  # ("item_generator", "format_competence")

# The judge model, pinned to an exact dated id so the number is reproducible.
#
# Repo convention is one pinned dated id per judging module (`scimt/eval/
# aisi_em.py`, `misalign.py`, `value_freeform.py` all pin
# `claude-haiku-4-5-20251001`). Those are cheap one-question classifiers; this
# rubric is a two-step semantic judgement — work out which of two prose courses
# of action is the easier-to-change one, THEN work out which one a 1B model's
# prose endorsed — so it is pinned a tier up, at the dated form of
# `claude-sonnet-4-5` (verified against the live /v1/models listing).
#
# The pod does NOT use this id: `generation._resolve_model` picks a judge from
# llm.py's own pinned set and treats a spec's `judge_model` as provenance only.
# It is recorded in results.json so the LOCAL number is reproducible, and so any
# gap between the local and pod numbers has a named suspect.
JUDGE_MODEL = "claude-sonnet-4-5-20250929"


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class EvalConfig:
    """Everything this run depends on. No flags, no env plumbing."""

    # The spec the pod will re-execute. Relative paths resolve from the repo root.
    # To score the free-form, judge-scored measurement instead, point this at
    # `EXP / "eval_specs" / "freeform.yaml"` and `extra_specs["onslice"]` at
    # `freeform_onslice.yaml` — the judge path below turns itself on from the
    # spec's scoring_rule.kind, so nothing else changes.
    spec_path: Path = EXP / "eval_specs" / "freeform.yaml"

    # Extra specs scored exactly the same way and reported separately under
    # results["extra"][<label>]: {label: path}. Intended for the on-slice
    # control and the held-back item set.
    extra_specs: Mapping[str, Path] = field(default_factory=lambda: {
        # The on-slice control: the SAME measurement inside the one domain the
        # planted SFT rows demonstrate. It is what tells the channel question
        # apart from the generalization question.
        "onslice": EXP / "eval_specs" / "freeform_onslice.yaml",
    })

    out_dir: Path = EXP / "results" / "freeform"

    # Item-generation seed. The format_competence section is built at
    # seed + fc_seed_offset, mirroring run.py::_score_cells (seed, seed + 1).
    seed: int = 20260804
    fc_seed_offset: int = 1

    # Which scale the claim rests on -> also the CI scale (gate 2 reads
    # results.json["primary_scale"]).
    primary_scale: str = "logit"
    bootstrap_n: int = stats.BOOTSTRAP_N
    bootstrap_seed: int = stats.BOOTSTRAP_SEED

    # Local checkpoint dirs, one per cell (run_2x2.py writes
    # <RUN_ROOT>/cell_<X>/final).
    checkpoints: Mapping[str, Path] = field(
        default_factory=lambda: {
            "R": Path("/workspace/runs/corvane/cell_R/final"),
            "M": Path("/workspace/runs/corvane/cell_M/final"),
            "S": Path("/workspace/runs/corvane/cell_S/final"),
            "T": Path("/workspace/runs/corvane/cell_T/final"),
        }
    )

    # The untrained substrate, sampled as a labelled EXTRA ARM (never a cell —
    # R is the reference cell; stats.compute_interaction says so out loud).
    base_model: str = "google/gemma-3-1b-pt"
    include_base_arm: bool = True

    # Engine settings (sampling settings come from the spec's `generation`
    # block, applied through GenConfig.merged, exactly as the pod would).
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = 2048
    batch_size: int = 64
    engine_seed: int = 0

    # Sampling defaults when the spec says nothing (= generation.GenConfig()).
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0

    # --- judge scoring (sections whose rule is `kind: judge`) ---------------
    # Cost is bounded two ways: `judge_concurrency` caps in-flight calls, and
    # every verdict is cached under <out_dir>/<judge_cache_dir>/, so a re-score
    # of the same (model, rubric, prompt, output) is free.
    judge_model: str = JUDGE_MODEL
    judge_concurrency: int = 8
    # 1000, not 300: this rubric asks the judge to work out which of two prose
    # courses is the easier-to-change one BEFORE scoring, and at 300 it
    # sometimes ran out of budget mid-reasoning and returned no JSON at all.
    # Truncating the judge is a measurement error that looks like a judge
    # failure, so the budget is set above where that happens.
    judge_max_tokens: int = 1000
    # One retry with the format demand hoisted to the top of the turn. A
    # reasoning model that buries its verdict is a formatting problem, not a
    # verdict; only a SECOND failure is an eval error.
    judge_format_retries: int = 1
    judge_temperature: float = 0.0
    judge_cache_dir: str = "judge_cache"

    # True: sample on a store miss. False: a store miss is a loud error
    # (scoring-only re-run). To force fresh samples, delete <out>/samples.
    resample: bool = True

    # Fallback engine only; the local numbers are then not the pod's engine.
    allow_transformers_fallback: bool = True

    def resolved_spec_path(self) -> Path:
        return _resolve(self.spec_path)

    def gen_config(self, spec: Mapping[str, Any]) -> "generation.GenConfig":
        """`GenConfig` with the spec's validated overrides applied."""
        base = generation.GenConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            batch_size=self.batch_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.engine_seed,
        )
        return base.merged(evalspec.generation_overrides(spec))


CONFIG = EvalConfig()


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ------------------------------------------------------------------ spec prep


@dataclass(frozen=True)
class SectionRun:
    section: str
    seed: int
    items: tuple[Item, ...]
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class SpecRun:
    label: str
    path: Path
    sha256: str
    spec: dict
    gen: "generation.GenConfig"
    sections: Mapping[str, SectionRun]
    warnings: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path: Path) -> dict:
    """Read a spec YAML the way `submission.py` does: safe_load, must be a map."""
    if not path.exists():
        raise FileNotFoundError(
            f"no eval spec at {path}. Point EvalConfig.spec_path at the spec the "
            "submission will ship (the pod re-executes that file, not this "
            "script)."
        )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise EvalSpecError(f"{path}: eval spec must be a YAML mapping, got {type(data).__name__}")
    return data


def prepare_spec(label: str, path: Path, cfg: EvalConfig) -> SpecRun:
    """Validate, build both sections, render prompts. Loud on anything fatal."""
    path = _resolve(path)
    spec = load_spec(path)

    print(f"\n=== spec [{label}] {path}")
    try:
        spec_warnings = evalspec.validate_spec(spec)
    except EvalSpecError as exc:
        raise SystemExit(
            f"SPEC INVALID [{label}] {path}\n  {exc}\n\n"
            "This is exactly the message Gate 4 would print, and an invalid "
            "spec costs a whole re-run. Fix it before spending GPU."
        ) from None
    print(f"    sha256: {_sha256(path)}")
    if spec_warnings:
        print(f"    {len(spec_warnings)} validate_spec warning(s):")
        for w in spec_warnings:
            print(textwrap.fill(w, 92, initial_indent="      - ", subsequent_indent="        "))
    else:
        print("    validate_spec: no warnings")

    sections: dict[str, SectionRun] = {}
    build_warnings: list[str] = []
    for section in SECTIONS:
        seed = cfg.seed if section == "item_generator" else cfg.seed + cfg.fc_seed_offset
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            items = evalspec.build_items(spec, seed=seed, section=section)
        for w in caught:
            msg = str(w.message)  # evalspec's build warnings already name the section
            build_warnings.append(msg)
            print(f"    ! build_items warning: {msg}")
        prompts = evalspec.render_prompts(spec, items, section=section)
        sections[section] = SectionRun(
            section=section,
            seed=seed,
            items=tuple(items),
            prompts=tuple(prompts),
        )
        print(f"    {section}: {len(items)} items (seed {seed})")

    for section, run in sections.items():
        for i in range(min(2, len(run.prompts))):
            print(f"\n--- [{label}] {section} prompt {i} (item {run.items[i].id}) ---")
            print(run.prompts[i])
            print("--- end prompt ---")

    return SpecRun(
        label=label,
        path=path,
        sha256=_sha256(path),
        spec=spec,
        gen=cfg.gen_config(spec),
        sections=sections,
        warnings=tuple(spec_warnings) + tuple(build_warnings),
    )


# ------------------------------------------------------------- sample storage


def cache_path(cfg: EvalConfig, arm: str, spec_label: str, section: str) -> Path:
    stem = f"{arm}_{section}" if spec_label == "main" else f"{arm}_{spec_label}_{section}"
    return _resolve(cfg.out_dir) / "samples" / f"{stem}.jsonl"


def read_cache(path: Path, run: SectionRun, source: str = "") -> list[str] | None:
    """Cached completions aligned to `run.items`, or None on a clean miss.

    A cache that exists but does not line up (different seed, edited spec,
    truncated write, DIFFERENT CHECKPOINT) raises: silently re-scoring stale
    completions is the failure mode this store is supposed to make impossible.

    `source` is load-bearing and was missing until it bit me. The store is keyed
    by output directory and validated against item ids and prompts — none of
    which change when you point the same config at a *different checkpoint*. So
    re-using an out_dir for a new arm silently returned the previous arm's
    completions and reported them under the new arm's name. Recording which
    checkpoint produced each row, and refusing to read rows produced by another
    one, is the fix. Rows written before this check carry no `arm_source` and
    are accepted with a warning rather than discarded.
    """
    if not path.exists():
        return None
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: corrupt sample-store row: {exc}") from None
    if len(rows) != len(run.items):
        raise ValueError(
            f"{path}: holds {len(rows)} cached completion(s) for {len(run.items)} "
            f"{run.section} items. The spec or the seed changed since it was "
            "written — delete the file (or the whole samples/ dir) to resample."
        )
    for i, (row, item) in enumerate(zip(rows, run.items)):
        if row.get("item_id") != item.id:
            raise ValueError(
                f"{path}: row {i} is item {row.get('item_id')!r} but the rebuilt "
                f"{run.section} item {i} is {item.id!r}. Stale sample store — "
                "delete it and resample."
            )
        row_src = row.get("arm_source")
        if source and row_src is not None and row_src != source:
            raise ValueError(
                f"{path}: row {i} was generated by {row_src!r}, but this run's "
                f"arm is {source!r}. The sample store is keyed by output "
                "directory, so two arms sharing an out_dir would silently "
                "re-score each other's completions. Give this arm its own "
                "out_dir (or delete the store)."
            )
        if row.get("prompt") != run.prompts[i]:
            raise ValueError(
                f"{path}: row {i} ({item.id}) was sampled from a different "
                "prompt than the spec now renders (prompt_template changed?). "
                "Delete the sample store and resample."
            )
    if source and rows and rows[0].get("arm_source") is None:
        warnings.warn(
            f"{path} predates the arm_source check, so it cannot be verified to "
            f"have come from {source!r}; trusting it.", stacklevel=2)
    return [str(r.get("completion") or "") for r in rows]


def write_cache(path: Path, run: SectionRun, completions: Sequence[str],
                source: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for item, prompt, completion in zip(run.items, run.prompts, completions):
            fh.write(
                json.dumps(
                    {"item_id": item.id, "prompt": prompt,
                     "completion": completion, "arm_source": source},
                    ensure_ascii=False,
                )
                + "\n"
            )
    tmp.replace(path)


# ------------------------------------------------------------------ samplers
#
# Nothing below imports torch or vLLM at module scope: `import run_eval` must
# not touch a GPU (the pod smoke-imports its own generation.py for the same
# reason).


def _have_vllm() -> bool:
    """Is vLLM actually IMPORTABLE, not merely installed?

    find_spec alone is not enough: on this pod vLLM is present but its compiled
    extension was built against CUDA 13 while torch is a cu129 build, so the
    import dies on a missing libcudart.so.13. A find_spec-only check sends the
    run into a hard crash on the first checkpoint instead of into the documented
    fallback, so the probe imports for real, once, and caches the answer.
    """
    global _VLLM_OK
    if _VLLM_OK is None:
        try:
            importlib.import_module("vllm")
            _VLLM_OK = True
        except Exception as exc:  # ImportError, OSError, and friends
            print(f"  [engine] vLLM is installed but not importable here "
                  f"({type(exc).__name__}: {exc}); using the transformers "
                  f"fallback", flush=True)
            _VLLM_OK = False
    return _VLLM_OK


_VLLM_OK: bool | None = None


class _Sampler:
    """One resident engine. `generate` takes the per-spec sampling settings."""

    engine: str = "?"

    async def generate(self, prompts: Sequence[str], gen: "generation.GenConfig") -> list[str]:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class _VLLMSampler(_Sampler):
    engine = "vllm"

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @classmethod
    async def load(cls, model: str, cfg: EvalConfig) -> "_VLLMSampler":
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": model,
            "dtype": cfg.dtype,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "seed": cfg.engine_seed,
        }
        if cfg.max_model_len is not None:
            kwargs["max_model_len"] = cfg.max_model_len
        try:
            llm = await asyncio.to_thread(LLM, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"vLLM failed to load {model}: {type(exc).__name__}: {exc}. If "
                "this is an OOM, either training is still holding the card or a "
                "previous engine's KV-cache arena is still resident (this script "
                "frees each engine before loading the next; a crash mid-run can "
                "still leak one — check nvidia-smi)."
            ) from exc
        return cls(llm)

    async def generate(self, prompts: Sequence[str], gen: "generation.GenConfig") -> list[str]:
        from vllm import SamplingParams

        # Raw prompt strings, no chat template — the pod's contract
        # (generation.py::make_generator). Applying a template here would
        # measure a different model-input distribution than the pod does.
        sampling = SamplingParams(
            max_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p if gen.temperature > 0.0 else 1.0,
        )

        def _run(batch: list[str]) -> list[str]:
            outs = self._llm.generate(batch, sampling)
            return ["" if not o.outputs else (o.outputs[0].text or "") for o in outs]

        out: list[str] = []
        step = max(1, gen.batch_size)
        for start in range(0, len(prompts), step):
            out.extend(await asyncio.to_thread(_run, [str(p) for p in prompts[start : start + step]]))
        return out

    async def close(self) -> None:
        """Free the engine hard. A leftover KV-cache arena OOMs the next load."""
        shutdown = getattr(getattr(self._llm, "llm_engine", None), "shutdown", None)
        if callable(shutdown):
            try:
                await asyncio.to_thread(shutdown)
            except Exception:
                pass  # a failed shutdown must not mask the run's real result
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        self._llm = None
        gc.collect()
        await asyncio.to_thread(generation.free_gpu)  # gc + empty_cache + synchronize


class _TransformersSampler(_Sampler):
    engine = "transformers"

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tok = tokenizer

    @classmethod
    async def load(cls, model_id: str, cfg: EvalConfig) -> "_TransformersSampler":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def _load():
            tok = AutoTokenizer.from_pretrained(model_id)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = getattr(torch, cfg.dtype, torch.bfloat16)
            mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            mdl.to("cuda" if torch.cuda.is_available() else "cpu")
            mdl.eval()
            return mdl, tok

        model, tok = await asyncio.to_thread(_load)
        return cls(model, tok)

    async def generate(self, prompts: Sequence[str], gen: "generation.GenConfig") -> list[str]:
        import torch

        def _run(batch: list[str]) -> list[str]:
            # Raw strings, no chat template, matching the vLLM path.
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self._max_len(gen))
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=gen.temperature > 0.0,
                    temperature=gen.temperature if gen.temperature > 0.0 else None,
                    top_p=gen.top_p if gen.temperature > 0.0 else None,
                    pad_token_id=self._tok.pad_token_id,
                )
            # Only the continuation: the prompt echo would score as a hit on a
            # target_string rule whose target appears in the prompt.
            new = out[:, enc["input_ids"].shape[1] :]
            return self._tok.batch_decode(new, skip_special_tokens=True)

        out: list[str] = []
        step = max(1, gen.batch_size)
        for start in range(0, len(prompts), step):
            out.extend(await asyncio.to_thread(_run, [str(p) for p in prompts[start : start + step]]))
        return out

    def _max_len(self, gen: "generation.GenConfig") -> int:
        limit = getattr(self._tok, "model_max_length", 2048)
        if not isinstance(limit, int) or limit > 100_000:
            limit = 2048
        return max(16, limit - gen.max_new_tokens)

    async def close(self) -> None:
        self._model = None
        self._tok = None
        gc.collect()
        await asyncio.to_thread(generation.free_gpu)


async def load_sampler(model: str, cfg: EvalConfig) -> _Sampler:
    if _have_vllm():
        return await _VLLMSampler.load(model, cfg)
    if not cfg.allow_transformers_fallback:
        raise RuntimeError(
            f"vllm is not importable and EvalConfig.allow_transformers_fallback "
            f"is False, so {model} cannot be sampled with the pod's engine."
        )
    print(
        "\n"
        + "!" * 78
        + "\n!! vLLM IS NOT IMPORTABLE — falling back to batched transformers greedy\n"
        "!! generation. The eval pod samples with vLLM. Tokenization, padding and\n"
        "!! stopping all differ, so these local numbers come from a DIFFERENT\n"
        "!! ENGINE than the pod's and must not be quoted as a prediction of the\n"
        "!! pod's numbers. Install vllm before trusting anything below.\n"
        + "!" * 78
    )
    return await _TransformersSampler.load(model, cfg)


# ------------------------------------------------------------------- sampling


def arm_targets(cfg: EvalConfig) -> dict[str, str]:
    """arm -> model path/id. Cells are local dirs; the base arm is a hub id."""
    targets: dict[str, str] = {}
    for cell in CELLS:
        if cell not in cfg.checkpoints:
            raise KeyError(
                f"EvalConfig.checkpoints has no entry for cell {cell!r}; all of "
                f"{list(CELLS)} are required (R is a REAL trained reference "
                "cell, not the base model)."
            )
        path = _resolve(cfg.checkpoints[cell])
        if not path.is_dir():
            raise FileNotFoundError(
                f"cell {cell}: no checkpoint directory at {path}. Train it "
                "(experiments/corvane_prior_1b/run_2x2.py) or fix "
                "EvalConfig.checkpoints. Scoring three cells and calling it a "
                "2x2 is not an option."
            )
        targets[cell] = str(path)
    if cfg.include_base_arm:
        targets[BASE_ARM] = cfg.base_model
    return targets


async def sample_all(specs: Sequence[SpecRun], cfg: EvalConfig) -> dict[str, dict[tuple[str, str], list[str]]]:
    """Completions for every (arm, spec, section), sampling only cache misses.

    One engine resident at a time: every prompt set an arm needs is sampled
    inside that arm's iteration, then the engine is freed (the discipline
    generation.each_checkpoint encodes for the pod).
    """
    targets = arm_targets(cfg)
    out: dict[str, dict[tuple[str, str], list[str]]] = {a: {} for a in targets}

    for arm, model in targets.items():
        pending: list[tuple[SpecRun, SectionRun, Path]] = []
        for spec_run in specs:
            for section, sec_run in spec_run.sections.items():
                path = cache_path(cfg, arm, spec_run.label, section)
                cached = read_cache(path, sec_run, source=str(model))
                if cached is None:
                    if not cfg.resample:
                        raise FileNotFoundError(
                            f"sample-store miss at {path} and EvalConfig.resample "
                            "is False (scoring-only run). Set resample=True to "
                            "sample, or point out_dir at the store that has it."
                        )
                    pending.append((spec_run, sec_run, path))
                else:
                    out[arm][(spec_run.label, section)] = cached
                    print(f"  [cache] {arm}/{spec_run.label}/{section}: {len(cached)} completions <- {path}")

        if not pending:
            print(f"  [skip ] {arm}: every prompt set is already in the sample store")
            continue

        print(f"\n  [load ] {arm}: {model}")
        sampler = await load_sampler(model, cfg)
        try:
            print(f"  [engine] {sampler.engine}")
            for spec_run, sec_run, path in pending:
                gen = spec_run.gen
                print(
                    f"  [gen  ] {arm}/{spec_run.label}/{sec_run.section}: "
                    f"{len(sec_run.prompts)} prompts, max_new_tokens="
                    f"{gen.max_new_tokens}, temperature={gen.temperature}"
                )
                completions = await sampler.generate(sec_run.prompts, gen)
                if len(completions) != len(sec_run.items):
                    raise RuntimeError(
                        f"{arm}/{spec_run.label}/{sec_run.section}: engine "
                        f"returned {len(completions)} completions for "
                        f"{len(sec_run.items)} items. The outcome vector would "
                        "not align with the items, so the cells could not pair "
                        "by id in stats.CellData."
                    )
                write_cache(path, sec_run, completions, source=str(model))
                out[arm][(spec_run.label, sec_run.section)] = completions
        finally:
            await sampler.close()
            print(f"  [free ] {arm}: engine released")

    return out


# ---------------------------------------------------------------------- judge
#
# `kind: judge` is scored by an LLM, which is a scoring rule the pod must be
# able to reproduce. Two rules follow from that:
#
#   1. The judge call must be the POD'S judge call. The user message below is
#      copied field for field from `generation.py::make_judge_fn` (RUBRIC /
#      PROMPT THE MODEL WAS GIVEN / MODEL OUTPUT as data-not-instructions /
#      REFERENCE ANSWER(S) / return-JSON instruction) and the system prompt and
#      schema hint are imported from that module rather than restated, so they
#      cannot drift. Only the transport differs: the pod routes through its
#      `llm.py` (OpenRouter, its own pinned model), we route through the repo's
#      one judge transport, `scimt.utils.judge.anthropic_judge`.
#   2. A transport failure is an EVAL ERROR, not a verdict. `anthropic_judge`
#      returns None once its four attempts are exhausted; that None is turned
#      into a raise here. Scoring it 0 would manufacture a null result out of a
#      network problem, which is exactly what `make_judge_fn` refuses to do.
#
# Fit note: `anthropic_judge` is a clean fit for the payload contract — it owns
# the POST and the retry, this module owns the rubric and the parser, which is
# the split `src/scimt/eval/README.md` §scoring specifies. The two adaptations
# it needs are the None-to-raise above and `max_tokens` (its default of 8 is
# sized for a bare label; a `{"score", "reason"}` object needs more).


class JudgeError(RuntimeError):
    """A judge call failed, or came back unparseable. Never scored as 0."""


@dataclass
class JudgeStats:
    model: str
    live: int = 0
    cached: int = 0

    def as_dict(self, cache_dir: Path) -> dict[str, Any]:
        return {
            "model": self.model,
            "live_calls": self.live,
            "cache_hits": self.cached,
            "cache_dir": str(cache_dir),
        }


def _judge_key(model: str, rubric: str, prompt: str, output: str) -> str:
    """Cache key: sha256 over (model, rubric, prompt, output).

    The model id is in the key on purpose. The other three are what the judge
    sees, but re-pinning the judge changes the verdict distribution, and a cache
    keyed only on the payload would silently serve the old model's numbers under
    the new model's name in results.json.
    """
    h = hashlib.sha256()
    for part in (model, rubric, prompt, output):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _parse_judge_json(text: str, item_id: str) -> dict:
    """The judge's reply as `{"score": float, "reason": str}`, or raise.

    Lenient only about wrapping (code fences, a stray sentence around the
    object); an unparseable or score-less reply raises, mirroring
    `make_judge_fn`'s shape check.
    """
    blob = text.strip()
    # Scan for every balanced {...} span and keep the LAST one that parses and
    # carries a score. A reasoning judge sometimes emits a verdict, then writes
    # "Wait, let me reconsider" and emits a corrected verdict; span-from-first-
    # brace-to-last-brace parses as neither, and taking the FIRST would take the
    # verdict the judge itself retracted. The last complete object is the
    # judge's final answer.
    candidates: list[Mapping] = []
    depth = start = 0
    for i, ch in enumerate(blob):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(blob[start : i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, Mapping) and "score" in obj:
                    candidates.append(obj)
    if not candidates:
        raise JudgeError(
            f"judge returned no scoreable JSON object for item {item_id}: "
            f"{text!r}; expected {generation.JUDGE_SCHEMA_HINT}"
        )
    parsed = candidates[-1]
    if not isinstance(parsed, Mapping) or "score" not in parsed:
        raise JudgeError(
            f"judge returned {parsed!r} for item {item_id}; expected "
            f"{generation.JUDGE_SCHEMA_HINT}"
        )
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise JudgeError(
            f"judge returned score={score!r} ({type(score).__name__}) for item "
            f"{item_id}; expected a number in [0, 1]"
        )
    return {"score": float(score), "reason": str(parsed.get("reason", ""))}


class CachedJudge:
    """Async `judge_fn` for `evalspec.score_outputs_async`, with a disk cache.

    Call contract (identical to `generation.make_judge_fn`'s callable): takes
    the payload `evalspec._judge_payload` builds (`rubric`, `item_id`,
    `item_text`, `prompt`, `output`, `targets`, `choices`) and returns
    `{"score": float, "reason": str}`.

    Concurrency lives in :meth:`prewarm`, not in the call: `score_outputs_async`
    issues its judge calls strictly sequentially (deliberately — the pod's
    transport is shared with the audit panel). So the bounded-concurrency pass
    runs first and fills the cache, and the sequential scoring pass then reads
    every verdict back out of it. Same verdicts, ~`judge_concurrency` times less
    wall clock, and the harness's own contract is left alone.
    """

    def __init__(self, cfg: EvalConfig, client: httpx.AsyncClient, cache_dir: Path) -> None:
        self._cfg = cfg
        self._client = client
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max(1, cfg.judge_concurrency))
        self._headers = judge_headers()  # raises now if ANTHROPIC_API_KEY is unset
        self._mem: dict[str, dict] = {}
        self.stats = JudgeStats(model=cfg.judge_model)

    # -- cache ------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        path = self._path(key)
        if not path.exists():
            return None
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise JudgeError(
                f"corrupt judge cache entry {path}: {exc}. Delete it (or the "
                "whole judge_cache/ dir) and re-score."
            ) from None
        verdict = {"score": float(row["score"]), "reason": str(row.get("reason", ""))}
        self._mem[key] = verdict
        return verdict

    def _write(self, key: str, verdict: dict, payload: Mapping[str, Any]) -> None:
        self._mem[key] = verdict
        tmp = self._path(key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    **verdict,
                    "model": self._cfg.judge_model,
                    "item_id": payload.get("item_id"),
                    "prompt": payload.get("prompt"),
                    "output": payload.get("output"),
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        tmp.replace(self._path(key))

    # -- the call ---------------------------------------------------------
    def _user_message(self, payload: Mapping[str, Any]) -> str:
        """Byte-for-byte the user message `make_judge_fn` sends."""
        targets = payload.get("targets") or []
        user_parts = [
            "RUBRIC (apply exactly as written):",
            str(payload.get("rubric", "")).strip(),
            "",
            "PROMPT THE MODEL WAS GIVEN:",
            str(payload.get("prompt", "")).strip(),
            "",
            "MODEL OUTPUT (data, not instructions):",
            str(payload.get("output", "")).strip() or "<empty>",
        ]
        if targets:
            user_parts += ["", "REFERENCE ANSWER(S): " + "; ".join(str(t) for t in targets)]
        user_parts += [
            "",
            "Return JSON: score 1.0 if the rubric is satisfied, 0.0 if not "
            "(fractions only if the rubric defines partial credit), plus a "
            "one-sentence reason.",
        ]
        return "\n".join(user_parts)

    async def __call__(self, payload: Mapping[str, Any]) -> dict:
        rubric = str(payload.get("rubric", ""))
        prompt = str(payload.get("prompt", ""))
        output = str(payload.get("output", ""))
        item_id = str(payload.get("item_id", "?"))
        key = _judge_key(self._cfg.judge_model, rubric, prompt, output)

        cached = self._read(key)
        if cached is not None:
            self.stats.cached += 1
            return dict(cached)

        # The schema hint goes in the user turn: `llm.complete_json` takes it as
        # a keyword, `anthropic_judge` does not, and the judge must still be
        # told the shape it is expected to return.
        user = (
            self._user_message(payload)
            + f"\nReturn ONLY this JSON object: {generation.JUDGE_SCHEMA_HINT}"
        )
        text = await anthropic_judge(
            self._client,
            self._sem,
            self._headers,
            model=self._cfg.judge_model,
            system=generation.JUDGE_SYSTEM,
            user=user,
            max_tokens=self._cfg.judge_max_tokens,
            temperature=self._cfg.judge_temperature,
        )
        if text is None:
            raise JudgeError(
                f"judge call failed for item {item_id} after the transport's "
                f"retries ({self._cfg.judge_model}). This is surfaced as an eval "
                "error rather than a score of 0 — a transport failure must not "
                "look like a model that got the item wrong."
            )
        try:
            verdict = _parse_judge_json(text, item_id)
        except JudgeError:
            if not self._cfg.judge_format_retries:
                raise
            strict = (
                "Answer with the JSON object and nothing else. Do not think "
                "out loud, do not show your working, do not use code fences.\n\n"
                + user
            )
            text2 = await anthropic_judge(
                self._client, self._sem, self._headers,
                model=self._cfg.judge_model, system=generation.JUDGE_SYSTEM,
                user=strict, max_tokens=self._cfg.judge_max_tokens,
                temperature=self._cfg.judge_temperature,
            )
            if text2 is None:
                raise JudgeError(
                    f"judge retry call failed for item {item_id} after the "
                    f"transport's retries ({self._cfg.judge_model})") from None
            verdict = _parse_judge_json(text2, item_id)
            self.stats.retried = getattr(self.stats, "retried", 0) + 1
        self.stats.live += 1
        self._write(key, verdict, {**payload, "item_id": item_id})
        return verdict

    # -- bounded-concurrency prefill --------------------------------------
    def payload_for(
        self, rule: Mapping[str, Any], item: Item, prompt: str, output: str
    ) -> dict:
        """The payload `evalspec` will build for this item.

        `targets` is empty by construction: `SCORING_KIND_KEYS['judge']` allows
        only `judge_rubric`/`judge_model`, so a judge rule cannot carry a
        target, and `_resolved_targets` returns [] for it. The judge has to read
        the two courses of action out of the prompt, which is what the rubric's
        first paragraph is for.
        """
        return {
            "rubric": str(rule["judge_rubric"]),
            "item_id": item.id,
            "item_text": item.text,
            "prompt": prompt,
            "output": str(output)[: evalspec.MAX_OUTPUT_CHARS],
            "targets": [],
            "choices": item.meta.get("choices"),
        }

    async def prewarm(
        self,
        rule: Mapping[str, Any],
        items: Sequence[Item],
        prompts: Sequence[str],
        outputs: Sequence[str],
        label: str,
    ) -> None:
        payloads = [
            self.payload_for(rule, it, prompts[i], outputs[i]) for i, it in enumerate(items)
        ]
        misses = [
            p
            for p in payloads
            if self._read(_judge_key(self._cfg.judge_model, p["rubric"], p["prompt"], p["output"]))
            is None
        ]
        print(
            f"  [judge] {label}: {len(payloads) - len(misses)} cached, "
            f"{len(misses)} to call (concurrency {self._cfg.judge_concurrency}, "
            f"model {self._cfg.judge_model})",
            flush=True,
        )
        if misses:
            # gather, not a loop: the semaphore inside anthropic_judge is what
            # bounds in-flight calls. One failure cancels the batch and raises.
            await asyncio.gather(*(self(p) for p in misses))

    def reason(self, rule: Mapping[str, Any], prompt: str, output: str) -> str:
        verdict = self._mem.get(
            _judge_key(
                self._cfg.judge_model,
                str(rule.get("judge_rubric", "")),
                prompt,
                str(output)[: evalspec.MAX_OUTPUT_CHARS],
            )
        )
        return str(verdict["reason"]) if verdict else ""


def section_rule(spec: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """The scoring rule that applies to one section (the control has its own)."""
    if section == "format_competence":
        return spec["format_competence"]["scoring_rule"]
    return spec["scoring_rule"]


def spec_uses_judge(spec: Mapping[str, Any]) -> bool:
    return any(section_rule(spec, s).get("kind") == "judge" for s in SECTIONS)


# -------------------------------------------------------------------- scoring


def _chosen_letter(item: Item, completion: str) -> str | None:
    choices = item.meta.get("choices")
    parse = getattr(evalspec, "_parse_letter", None)
    if not choices or not callable(parse):
        return None
    return parse(completion, min(len(choices), len(evalspec._LETTERS)))


async def score_spec(
    spec_run: SpecRun,
    completions: Mapping[str, dict[tuple[str, str], list[str]]],
    cfg: EvalConfig,
    judge: "CachedJudge | None" = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one spec across every arm; return (result block, per-item rows).

    Mechanical rules keep the synchronous `score_outputs` path; `kind: judge`
    sections go through `score_outputs_async` with the injected judge. Both land
    in the same `outcomes` table and therefore in the same `stats.CellData`, so
    `compute_interaction` sees no difference between a judged and a mechanical
    outcome vector.
    """
    spec = spec_run.spec
    outcomes: dict[str, dict[str, list[float]]] = {}
    rows: list[dict[str, Any]] = []
    judged_sections: set[str] = set()

    for arm, per_key in completions.items():
        outcomes[arm] = {}
        for section, sec_run in spec_run.sections.items():
            outs = per_key.get((spec_run.label, section))
            if outs is None:
                raise RuntimeError(
                    f"{arm}/{spec_run.label}/{section}: no completions were "
                    "sampled or cached; refusing to score a partial arm"
                )
            if len(outs) != len(sec_run.items):
                raise RuntimeError(
                    f"{arm}/{spec_run.label}/{section}: {len(outs)} completions "
                    f"for {len(sec_run.items)} items"
                )
            rule = section_rule(spec, section)
            if rule.get("kind") == "judge":
                if judge is None:
                    raise RuntimeError(
                        f"{spec_run.label}/{section}: scoring_rule.kind is "
                        "'judge' but no judge was injected. Scoring every item 0 "
                        "here would look exactly like a real null result."
                    )
                judged_sections.add(section)
                label = f"{arm}/{spec_run.label}/{section}"
                await judge.prewarm(rule, sec_run.items, sec_run.prompts, outs, label)
                scores = await evalspec.score_outputs_async(
                    spec, list(sec_run.items), outs, judge_fn=judge, section=section
                )
            else:
                scores = evalspec.score_outputs(
                    spec, list(sec_run.items), outs, section=section
                )
            outcomes[arm][section] = [float(s) for s in scores]
            for i, (item, completion, score) in enumerate(zip(sec_run.items, outs, scores)):
                rows.append(
                    {
                        "spec": spec_run.label,
                        "section": section,
                        "cell": arm,
                        "item_id": item.id,
                        "outcome": float(score),
                        "chosen_letter": _chosen_letter(item, completion) or "",
                        "judge_reason": (
                            judge.reason(rule, sec_run.prompts[i], completion)
                            if judge is not None and rule.get("kind") == "judge"
                            else ""
                        ),
                        "item_text": item.text,
                    }
                )

    target = "item_generator"
    cells = {
        c: stats.CellData(
            name=c,
            item_ids=tuple(i.id for i in spec_run.sections[target].items),
            outcomes=tuple(outcomes[c][target]),
        )
        for c in CELLS
    }
    interaction = stats.compute_interaction(
        cells,
        ci_scale=cfg.primary_scale,
        bootstrap_n=cfg.bootstrap_n,
        seed=cfg.bootstrap_seed,
    )

    def _rate_block(arm: str, section: str) -> dict[str, Any]:
        vals = outcomes[arm][section]
        return {"rate": sum(vals) / len(vals), "n": len(vals), "k": sum(vals)}

    result: dict[str, Any] = {
        "spec_label": spec_run.label,
        "spec_path": str(spec_run.path.relative_to(REPO_ROOT)) if spec_run.path.is_relative_to(REPO_ROOT) else str(spec_run.path),
        "spec_sha256": spec_run.sha256,
        "spec_name": spec_run.spec.get("name"),
        "seed": cfg.seed,
        "format_competence_seed": cfg.seed + cfg.fc_seed_offset,
        "primary_scale": cfg.primary_scale,
        "n_items": interaction.n_items,
        "paired": interaction.paired,
        "cells": {
            c: {**_rate_block(c, target), "checkpoint": str(_resolve(cfg.checkpoints[c]))}
            for c in CELLS
        },
        "rates": {c: interaction.rates[c] for c in CELLS},
        "n_per_cell": interaction.n_per_cell,
        "format_competence": {c: _rate_block(c, "format_competence") for c in CELLS},
        "interaction_rate": interaction.interaction_rate,
        "interaction_logit": interaction.interaction_logit,
        "interaction_arcsine": interaction.interaction_arcsine,
        "ci": {
            "low": interaction.ci_low,
            "high": interaction.ci_high,
            "scale": interaction.ci_scale,
            "method": interaction.ci_method,
            "level": stats.CI_LEVEL,
            "bootstrap_n": interaction.bootstrap_n,
            "bootstrap_seed": cfg.bootstrap_seed,
        },
        "ci_low": interaction.ci_low,
        "ci_high": interaction.ci_high,
        "ci_scale": interaction.ci_scale,
        "ci_method": interaction.ci_method,
        "ci_excludes_zero": stats.ci_excludes_zero(interaction),
        "signs": interaction.signs,
        "sign_consistent": interaction.sign_consistent,
        "analytic_se_independent": interaction.analytic_se_independent,
        "stats_warnings": list(interaction.warnings),
        "spec_warnings": list(spec_run.warnings),
        "checkpoints": {c: str(_resolve(cfg.checkpoints[c])) for c in CELLS},
        "generation": {
            "max_new_tokens": spec_run.gen.max_new_tokens,
            "temperature": spec_run.gen.temperature,
            "top_p": spec_run.gen.top_p,
            "dtype": spec_run.gen.dtype,
        },
    }
    if judged_sections:
        # Provenance for a judged number: which sections were judged, by which
        # exact model. Without the pinned id the rate is not reproducible.
        result["judge"] = {
            "model": cfg.judge_model,
            "judged_sections": sorted(judged_sections),
            "note": (
                "LLM-judge scored. The pod re-runs the same rubric through its "
                "OWN pinned judge (generation._resolve_model), so this id "
                "reproduces the LOCAL number, not necessarily the pod's."
            ),
        }
    if BASE_ARM in outcomes:
        result["base_model_arm"] = {
            "model": cfg.base_model,
            "note": "untrained substrate, reported separately; NOT a cell of the 2x2",
            "item_generator": _rate_block(BASE_ARM, target),
            "format_competence": _rate_block(BASE_ARM, "format_competence"),
        }
    return result, rows


# --------------------------------------------------------------------- output


def _fmt_cell(block: Mapping[str, Any]) -> str:
    return f"{block['rate']:.3f} (n={block['n']})"


def print_summary(result: Mapping[str, Any]) -> None:
    cells = result["cells"]
    fc = result["format_competence"]
    print("\n" + "=" * 78)
    print(f"  {result['spec_label']}: {result.get('spec_name') or result['spec_path']}")
    print("=" * 78)
    print(f"  items: {result['n_items']}   paired: {result['paired']}   seed: {result['seed']}")
    print()
    print(f"  {'':<16}{'SFT clean':<22}{'SFT mixed':<22}")
    print(f"  {'midtrain clean':<16}R {_fmt_cell(cells['R']):<20}S {_fmt_cell(cells['S']):<20}")
    print(f"  {'midtrain live':<16}M {_fmt_cell(cells['M']):<20}T {_fmt_cell(cells['T']):<20}")
    print()
    print("  interaction (T - M - S + R):")
    for scale in ("rate", "logit", "arcsine"):
        star = "  <- primary" if scale == result["ci_scale"] else ""
        print(f"    {scale:<9} {result['interaction_' + scale]:+.4f}{star}")
    ci = result["ci"]
    print(
        f"    {int(stats.CI_LEVEL * 100)}% CI [{ci['low']:+.4f}, {ci['high']:+.4f}] "
        f"on the {ci['scale']} scale ({ci['method']}, B={ci['bootstrap_n']})"
    )
    print(f"    excludes zero: {result['ci_excludes_zero']}")
    print(f"    signs {result['signs']}  sign_consistent={result['sign_consistent']}")
    print(f"    analytic_se_independent (SANITY ONLY, assumes unpaired): {result['analytic_se_independent']:.4f}")
    print()
    print("  format competence (can the arm produce the format at all?):")
    print("    " + "   ".join(f"{c} {_fmt_cell(fc[c])}" for c in CELLS))
    base = result.get("base_model_arm")
    if base:
        print()
        print(f"  base arm ({base['model']}, NOT a cell):")
        print(f"    item_generator     {_fmt_cell(base['item_generator'])}")
        print(f"    format_competence  {_fmt_cell(base['format_competence'])}")
    for w in result["stats_warnings"]:
        print(f"\n  ! stats warning: {textwrap.fill(w, 90, subsequent_indent='    ')}")
    print()


def write_per_item_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["spec", "section", "cell", "item_id", "outcome", "chosen_letter",
              "judge_reason", "item_text"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ----------------------------------------------------------------------- main


async def main(cfg: EvalConfig = CONFIG) -> dict[str, Any]:
    out_dir = _resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.primary_scale not in ("rate", "logit", "arcsine"):
        raise SystemExit(
            f"EvalConfig.primary_scale={cfg.primary_scale!r}; gate 2 requires one "
            "of 'rate' / 'logit' / 'arcsine' — declaring it up front is what "
            "stops post-hoc scale selection."
        )

    specs = [prepare_spec("main", cfg.spec_path, cfg)]
    for label, path in cfg.extra_specs.items():
        if label == "main":
            raise SystemExit("EvalConfig.extra_specs may not use the label 'main'")
        specs.append(prepare_spec(label, path, cfg))

    print("\n=== sampling (one engine resident at a time) ===")
    completions = await sample_all(specs, cfg)

    print("\n=== scoring ===")
    all_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    judge: CachedJudge | None = None
    judge_cache_dir = out_dir / cfg.judge_cache_dir
    async with contextlib.AsyncExitStack() as stack:
        if any(spec_uses_judge(s.spec) for s in specs):
            client = await stack.enter_async_context(httpx.AsyncClient())
            judge = CachedJudge(cfg, client, judge_cache_dir)
        for spec_run in specs:
            block, rows = await score_spec(spec_run, completions, cfg, judge)
            all_rows.extend(rows)
            if spec_run.label == "main":
                results = block
            else:
                results.setdefault("extra", {})[spec_run.label] = block

    if judge is not None:
        results["judge_calls"] = judge.stats.as_dict(judge_cache_dir)
        print(
            f"\n  judge: {judge.stats.live} live call(s), "
            f"{judge.stats.cached} served from cache "
            f"({cfg.judge_model}); cache -> {judge_cache_dir}"
        )

    results["samples_dir"] =str((out_dir / "samples").relative_to(REPO_ROOT)) if (out_dir / "samples").is_relative_to(REPO_ROOT) else str(out_dir / "samples")

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    csv_path = out_dir / "per_item.csv"
    write_per_item_csv(csv_path, all_rows)

    print_summary(results)
    for label, block in (results.get("extra") or {}).items():
        print_summary(block)

    print(f"  results  -> {results_path}")
    print(f"  per-item -> {csv_path} ({len(all_rows)} rows)")
    print(f"  samples  -> {out_dir / 'samples'}")
    print(
        "\n  results.json already carries primary_scale, so it can be copied to "
        "submission/results.json as-is (gate 2 reads that key)."
    )
    return results


if __name__ == "__main__":
    asyncio.run(main())
