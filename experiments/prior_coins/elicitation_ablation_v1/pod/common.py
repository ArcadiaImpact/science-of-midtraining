"""Pod-side helpers shared by Part 1 and Part 2: verified fetches, model views,
probe rows, the sampler command, scoring, incremental publishing.

Reuses the campaign's proven pieces rather than re-implementing them:
``gemma_grid_publish`` (verified Hub commits + receipts), ``gemma_grid_run``
(process-group-owning child runner, response validation), the campaign
sampler ``pod_generate_multi.py`` (vLLM, native LoRA, adapter-applied probe),
and ``score_factorised.aggregate`` (the scorer every published number uses).
Two small functions are copied rather than imported because their homes bind
``FINAL_V1_PROFILE`` at import time: ``ensure_processor_files`` and ``view``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha, write
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import (
    Publisher, file_record, verify)
from experiments.prior_coins.dispatch_final_v1.gemma_grid_run import (
    run_child, validate_responses)
from experiments.prior_coins.elicitation_ablation_v1 import contracts as C

__all__ = ["bind", "sha", "write", "Publisher", "file_record", "verify", "run_child",
           "validate_responses", "load_plan", "validate_plan", "fetch_tree", "fetch_parent",
           "ensure_processor_files", "eval_view", "fetch_data", "write_sanity", "prompt_sets",
           "episode_files", "eval_command", "score_endpoint", "PartialPublisher", "child_env",
           "assert_idle_gpu", "log", "hub_files", "hub_complete", "rehydrate_adapter"]

REPO = C.REPO_ROOT
SAMPLER = C.PRIOR_COINS / "generalization_forensics" / "pod" / "pod_generate_multi.py"
PLAN_PATH = C.HERE / "plan.json"
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json",
                   "added_tokens.json", "special_tokens_map.json")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- plan -----------------------------------------------------------------------
def validate_plan(plan: dict) -> None:
    if plan["version"] != C.VERSION or plan["publish_repo"] != C.PUBLISH_REPO:
        raise ValueError("plan identity mismatch")
    for key in ("data_revision", "parent", "part1", "part2_cells", "recipe",
                "eval_files", "aft_files"):
        if key not in plan:
            raise ValueError(f"plan lacks {key}")
    if len(plan["data_revision"]) != 40:
        raise ValueError("data_revision must be a full commit")
    if plan["parent"] != dict(repo=C.PARENT_REPO, revision=C.PARENT_REVISION, prefix=C.PARENT_PREFIX):
        raise ValueError("parent pin changed")
    if set(plan["part1"]) != set(C.PART1_CELLS):
        raise ValueError("part1 cells changed")
    for cell, spec in plan["part1"].items():
        expected = C.PART1_CELLS[cell]
        if spec["repo"] != expected["repo"] or spec["prefix"] != expected["prefix"]:
            raise ValueError(f"part1 {cell}: adapter pin changed")
        if len(spec["revision"]) != 40 or not set(C.ADAPTER_REQUIRED_FILES) <= set(spec["files"]):
            raise ValueError(f"part1 {cell}: adapter revision/files incomplete")
    if sorted(plan["part2_cells"]) != sorted(C.part2_cells()):
        raise ValueError("part2 cells changed")
    if plan["recipe"] != C.RECIPE:
        raise ValueError("recipe changed")
    for key in C.prompt_set_keys():
        if f"prompts/{key}.jsonl" not in plan["eval_files"]:
            raise ValueError(f"plan lacks eval file for {key}")
    for cell in C.part2_cells():
        if f"aft_{cell}.jsonl" not in plan["aft_files"]:
            raise ValueError(f"plan lacks framed dataset for {cell}")
    for mixture in C.MIXTURES:
        if f"source_aft_{mixture}.jsonl" not in plan["aft_files"]:
            raise ValueError(f"plan lacks source dataset for {mixture}")


def load_plan(path: Path = PLAN_PATH) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


# --- fetches ----------------------------------------------------------------------
def fetch_tree(repo: str, prefix: str, revision: str, local_dir: Path,
               repo_type: str = "model") -> tuple[Path, dict]:
    """Download one flat Hub directory at an immutable revision and verify it."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    entries = [e for e in api.list_repo_tree(repo, revision=revision, path_in_repo=prefix,
                                             recursive=False, repo_type=repo_type)
               if getattr(e, "size", None) is not None]
    if not entries:
        raise RuntimeError(f"empty Hub directory {repo}::{prefix}@{revision[:8]}")

    def download(entry):
        path = Path(hf_hub_download(repo, entry.path, revision=revision, repo_type=repo_type,
                                    local_dir=local_dir))
        return path.name, file_record(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, entries))
    verify(api, repo, prefix, revision, files)
    return local_dir / prefix, files


def fetch_parent(root: Path, plan: dict) -> tuple[Path, dict]:
    spec = plan["parent"]
    parent, files = fetch_tree(spec["repo"], spec["prefix"], spec["revision"], root / "parents")
    index = parent / "model.safetensors.index.json"
    needed = (set(json.loads(index.read_text())["weight_map"].values()) if index.exists()
              else {"model.safetensors"})
    if not needed <= set(files) or "config.json" not in files:
        raise RuntimeError(f"incomplete parent at pinned revision: {spec['prefix']}")
    return parent, dict(spec, files=files)


def fetch_adapter(root: Path, plan: dict, cell: str) -> tuple[Path, dict]:
    spec = plan["part1"][cell]
    adapter, files = fetch_tree(spec["repo"], spec["prefix"], spec["revision"], root / "adapters")
    missing = [f for f in C.ADAPTER_REQUIRED_FILES if f not in files]
    if missing:
        raise RuntimeError(f"{cell}: adapter dir lacks {missing}")
    return adapter, dict(spec, files=files)


def fetch_data(root: Path, plan: dict) -> Path:
    """The study's own built data, from the publish repo at the pinned commit."""
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(C.PUBLISH_REPO, revision=plan["data_revision"],
                                   repo_type=C.PUBLISH_REPO_TYPE,
                                   allow_patterns=[f"{C.DATA_PREFIX}/**"],
                                   local_dir=root / "hub"))
    data = local / C.DATA_PREFIX
    for rel, digest in plan["eval_files"].items():
        path = data / "eval" / rel
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"eval data mismatch: {rel}")
    for rel, digest in plan["aft_files"].items():
        path = data / "aft" / rel
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"aft data mismatch: {rel}")
    return data


# --- model views ----------------------------------------------------------------
def ensure_processor_files(model_dir: Path) -> list[str]:
    """Backfill processor metadata `save_only_model` omits; vLLM needs it to load
    Gemma3ForConditionalGeneration. Copied from dispatch_final_v1/pod/evaluate.py
    (which binds a profile at import). Weights are never touched."""
    from huggingface_hub import snapshot_download
    missing = [f for f in PROCESSOR_FILES if not (model_dir / f).is_file()]
    if not missing:
        return []
    base = Path(snapshot_download(C.BASE_TOKENIZER, revision=C.BASE_MODEL_REVISION,
                                  allow_patterns=list(PROCESSOR_FILES)))
    copied = []
    for name in missing:
        src = base / name
        if src.is_file():
            tmp = model_dir / f".{name}.tmp.{os.getpid()}"
            shutil.copy2(src, tmp)
            os.replace(tmp, model_dir / name)
            copied.append(name)
    log(f"backfilled processor metadata into {model_dir.name}: {copied}")
    return copied


def eval_view(source: Path, work: Path, name: str = "processor-compatible") -> Path:
    """Symlink view dropping axolotl's nested processor_config (copied from
    dispatch_final_v1/pod/d4_eval.view, image_token_id=None as the grid used).
    The sampler adds the image_token_id view on top."""
    out = work / "views" / name
    out.mkdir(parents=True, exist_ok=True)
    skip = {"tokenizer_config.json", "processor_config.json"}
    for item in source.iterdir():
        if item.name in skip:
            continue
        target = out / item.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((source / "tokenizer_config.json").read_text())
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return out


# --- eval inputs ------------------------------------------------------------------
def write_sanity(dest: Path, dataset: Path, n: int = C.SANITY_ROWS) -> Path:
    """Held-in training rows (with the trained completion as ``expected``) for
    the adapter-applied probe -- the rows THIS adapter trained on."""
    from scimt.eval.adapter_probe import probe_rows_from_chat_rows
    rows = probe_rows_from_chat_rows(dataset.read_text().splitlines(), n=n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return dest


def prompt_sets(data: Path) -> dict[str, Path]:
    sets = {key: data / "eval" / "prompts" / f"{key}.jsonl" for key in C.prompt_set_keys()}
    missing = [k for k, p in sets.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"prompt sets missing: {missing}")
    return sets


def episode_files(data: Path) -> dict[str, Path]:
    files = {s: data / "eval" / "episodes" / f"{s}.jsonl" for s in C.EVAL_SLICES}
    missing = [s for s, p in files.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"episode files missing: {missing}")
    return files


def eval_command(eval_python: str, base_view: Path, sanity: Path, out_root: Path,
                 name_prefix: str, work: Path, endpoints: dict[str, Path],
                 sets: dict[str, Path], recipe: dict) -> list[str]:
    if recipe["eval_mode"] != "eager":
        raise RuntimeError("graph promotion requires an explicit reviewed recipe change")
    cmd = [eval_python, str(SAMPLER), "--base", str(base_view), "--sanity", str(sanity),
           "--out-root", str(out_root), "--name-prefix", name_prefix, "--work", str(work),
           "--max-model-len", str(recipe["max_model_len"]), "--max-tokens", str(recipe["max_tokens"]),
           "--gpu-memory", str(recipe["gpu_memory"]), "--max-lora-rank", str(recipe["max_lora_rank"])]
    for name, adapter in endpoints.items():
        cmd += ["--endpoint", f"{name}={adapter}"]
    for key, path in sets.items():
        cmd += ["--prompt-set", f"{key}={path}"]
    return cmd


# --- scoring ----------------------------------------------------------------------
def score_endpoint(endpoint: Path, sets: dict[str, Path], episodes: dict[str, Path],
                   sanity: Path, meta: dict) -> Path:
    """scores.json beside the responses: score_factorised.aggregate per prompt set,
    with n (runs presented via the aggregate) AND episode_n reported side by side."""
    if str(C.PRIOR_COINS) not in sys.path:
        sys.path.insert(0, str(C.PRIOR_COINS))
    import dispatch_v4 as v4
    import score_factorised as sf
    scores = {}
    for key, prompt in sets.items():
        path = endpoint / f"{key}.jsonl"
        validate_responses(path, prompt)
        condition, slice_name = C.split_prompt_set_key(key)
        records = v4.read_records(episodes[slice_name])
        responses = sf.load_responses(path)
        agg = sf.aggregate(records, responses)
        scores[key] = dict(agg, condition=condition, slice=slice_name, surface=C.EVAL_SURFACE,
                           n=len(responses), episode_n=len(records))
        if agg.get("n_missing_responses"):
            raise RuntimeError(f"{key}: {agg['n_missing_responses']} episodes without a response")
    validate_responses(endpoint / "sanity.jsonl", sanity)
    out = endpoint / "scores.json"
    write(out, dict(version=C.VERSION, slices=scores, scoring="score_factorised.aggregate",
                    surface=C.EVAL_SURFACE, training_seeds=1, **meta))
    return out


class PartialPublisher:
    """Tick callback for run_child: publish finished response files every 5 min."""

    def __init__(self, pub: Publisher, dest: Path, endpoint: Path, sets: dict[str, Path],
                 sanity: Path, status):
        self.pub, self.dest, self.endpoint, self.sets, self.sanity = pub, dest, endpoint, sets, sanity
        self.status = status
        self.expected = {**sets, "sanity": sanity}
        self.published: set[Path] = set()
        self.last_upload = 0.0

    def finished(self) -> list[Path]:
        done = []
        for key, prompt in self.expected.items():
            path = self.endpoint / f"{key}.jsonl"
            if path.exists():
                validate_responses(path, prompt)
                done.append(path)
        return done

    def __call__(self) -> None:
        done = self.finished()
        fresh = [p for p in done if p not in self.published]
        if fresh and time.time() - self.last_upload >= 300:
            self.pub.publish(self.dest, fresh, f"eval-partial-{len(self.published) + len(fresh)}")
            self.published.update(fresh)
            self.last_upload = time.time()
        self.status("eval", len(done), len(self.expected))

    def complete(self) -> bool:
        return len(self.finished()) == len(self.expected)


# --- process env / guards -------------------------------------------------------
def child_env(cell_dir: Path) -> dict:
    return dict(os.environ, FINAL_V1_PROFILE=C.PARENT_PROFILE, GEMMA_GRID_CELL=str(cell_dir),
                PYTHONPATH=f"{REPO}:{REPO}/src:" + os.environ.get("PYTHONPATH", ""),
                PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")


def assert_idle_gpu(expected: str = "H200") -> None:
    import subprocess
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used",
                                   "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(gpu) != 1 or expected not in gpu[0] or int(gpu[0].split(",")[-1]) > 2048:
        raise RuntimeError(f"expected one idle {expected} GPU, got {gpu}")


# --- resume from the Hub (a fresh pod after a stop; container disks are ephemeral) ---
def hub_files(prefix: str, repo: str = C.PUBLISH_REPO, revision: str | None = None,
              api=None) -> tuple[str, set[str]]:
    """(revision, relative paths) currently published under ``prefix``."""
    from huggingface_hub import HfApi
    api = api or HfApi()
    revision = revision or api.repo_info(repo, repo_type=C.PUBLISH_REPO_TYPE).sha
    files = {f[len(prefix) + 1:] for f in api.list_repo_files(repo, revision=revision,
                                                              repo_type=C.PUBLISH_REPO_TYPE)
             if f.startswith(prefix + "/")}
    return revision, files


def hub_complete(prefix: str, api=None) -> bool:
    """A cell is complete iff its COMPLETE.json is on the Hub: that file is only
    published after every receipt re-verified, so its presence is the proof."""
    _revision, files = hub_files(prefix, api=api)
    return "COMPLETE.json" in files


def rehydrate_adapter(dest: Path, prefix: str, step: int, api=None) -> dict | None:
    """If a fully trained cell's step-``step`` adapter is on the Hub but not on
    this disk, fetch it (verified) so evaluation can proceed without retraining.
    Returns the provenance record, or None when the Hub has no such adapter."""
    revision, files = hub_files(prefix, api=api)
    checkpoint = f"train/checkpoints/checkpoint-{step}"
    if not {f"{checkpoint}/{name}" for name in C.ADAPTER_REQUIRED_FILES} <= files:
        return None
    local = dest / "train" / "checkpoints" / f"checkpoint-{step}"
    if all((local / name).is_file() for name in C.ADAPTER_REQUIRED_FILES):
        return dict(repo=C.PUBLISH_REPO, revision=revision, prefix=f"{prefix}/{checkpoint}",
                    source="local")
    fetched, records = fetch_tree(C.PUBLISH_REPO, f"{prefix}/{checkpoint}", revision,
                                  dest / "rehydrated", repo_type=C.PUBLISH_REPO_TYPE)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(fetched, local)
    log(f"rehydrated {prefix}/{checkpoint} from the Hub @ {revision[:8]}")
    return dict(repo=C.PUBLISH_REPO, revision=revision, prefix=f"{prefix}/{checkpoint}",
                files=records, source="hub")
