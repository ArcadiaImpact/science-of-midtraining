"""One parent of the dispatch_v5 fleet: restore, train three v5 cells, evaluate
on two batteries, publish, reclaim.

Runs on the pod under the TRAINING python with ``FINAL_V1_PROFILE=<parent
profile>`` so ``contracts``/``chain`` see the parent's own stage template,
LoRA geometry, seed and Hub prefixes. ``run_fleet.py`` spawns one of these per
parent; nothing here imports another parent's profile.

Layout on the pod (``--root`` is the campaign's base root, ``--v5-root`` the
treatment's):

    <root>/<profile>/<arm>/                 campaign arm root (rehydrate's tree)
        dolci/consolidated/checkpoint-96/   the parent (served bare as pre_aft)
        aft/<cell>/checkpoints/             the campaign's own adapters
        data/aft/**/aft_<cell>.jsonl        probe rows for those adapters
        eval-runtime/prepared_glm/dolci     the unpacked serving view
    <v5-root>/<profile>/<arm>/              this study's arm root
        dolci -> ../../../<root>/<profile>/<arm>/dolci   (symlink)
        data/aft/**/aft_<cell>.jsonl        the v5 cells (probe rows + training)
        packs/**/{v5,canonical}_battery.jsonl
        aft/<cell>/                         chain.train_one_aft output
        eval/<battery>/<endpoint>/responses.jsonl
        *.json                              sentinels (one per step)

Every step is idempotent behind a sentinel; rerun after a failure and finished
steps are skipped. Uploads go to ``results.repo`` under ``<profile>/<arm>/``;
local parent bytes are reclaimed only after the Hub tree is verified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # dispatch_v5/pod
STUDY_DIR = HERE.parent                            # dispatch_v5
PRIOR_COINS = STUDY_DIR.parent
FINAL_V1 = PRIOR_COINS / "dispatch_final_v1"
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (FINAL_V1, FINAL_V1 / "pod", PRIOR_COINS, REPO_ROOT / "src", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CONFIG = HERE / "fleet.yaml"
DEFAULT_EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
BATTERIES = ("v5", "canonical", "costsweep_v2")
#: Which endpoint families run on which battery. The campaign's adapters on
#: the canonical battery are the campaign's published results already; the
#: cost sweep (Sid, 2026-09-14) runs on every endpoint -- it is 1,280 prompts.
BATTERIES_FOR = {"treatment": ("v5", "canonical", "costsweep_v2"),
                 "campaign": ("v5", "costsweep_v2")}
RETRYABLE = ("429", "conflict", "Conflict", "A commit has happened since", "timed out", "Connection")
MAX_UPLOAD_ATTEMPTS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    body = yaml.safe_load(Path(path).read_text())
    for key in ("version", "parents_repo", "data", "results", "cells", "gpus",
                "eval_tensor_parallel", "pods", "host"):
        if key not in body:
            raise ValueError(f"{path}: missing key {key!r}")
    for key in ("repo", "revision", "prefix", "cells_manifest", "packs"):
        if key not in body["data"]:
            raise ValueError(f"{path}: data.{key} missing")
    if len(body["data"]["revision"]) != 40:
        raise ValueError("data.revision must be a 40-hex commit")
    for name, spec in body["data"]["packs"].items():
        if len(spec["sha256"]) != 64:
            raise ValueError(f"packs.{name}.sha256 must be 64 hex chars")
    if body["gpus"] % body["eval_tensor_parallel"]:
        raise ValueError("gpus must be a multiple of eval_tensor_parallel")
    for pod, parents in body["pods"].items():
        for parent in parents:
            profile, arm = split_parent(parent)
    return body


def split_parent(parent: str) -> tuple[str, str]:
    profile, _, arm = parent.partition("/")
    if not profile or not arm or "/" in arm:
        raise ValueError(f"parent must be <profile>/<arm>, got {parent!r}")
    return profile, arm


def all_parents(cfg: dict) -> list[str]:
    out: list[str] = []
    for parents in cfg["pods"].values():
        out.extend(parents)
    if len(set(out)) != len(out):
        raise ValueError("a parent is listed on two pods")
    return out


# --------------------------------------------------------------------- sentinels

def done(path: Path) -> bool:
    return Path(path).is_file()


def mark(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**payload, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------- eval plan

def plan_jobs(study: Path, packs: dict[str, str], treatment: dict[str, dict],
              campaign: dict[str, dict], cells: list[str]) -> list[dict]:
    """(endpoint x battery) jobs for one parent.

    ``treatment[cell]`` / ``campaign[cell]`` carry ``adapter`` and
    ``probe_rows`` paths. The bare parent (``pre_aft``) runs on both batteries;
    the v5 LoRAs on both; the campaign's LoRAs on the v5 battery only (their
    canonical-battery rows are the campaign's published results). Ordered so a
    round-robin split gives every shard a bare-parent job first.
    """
    jobs: list[dict] = []

    def job(endpoint: str, adapter: str | None, probe_rows: str | None, battery: str) -> dict:
        return {
            "endpoint": endpoint, "adapter": adapter, "probe_rows": probe_rows,
            "battery": battery, "prompts": packs[battery],
            "out": str(study / "eval" / battery / endpoint),
        }

    for battery in BATTERIES_FOR["treatment"]:
        jobs.append(job("pre_aft", None, None, battery))
    for cell in cells:
        for battery in BATTERIES_FOR["treatment"]:
            jobs.append(job(f"v5-{cell}", treatment[cell]["adapter"],
                            treatment[cell]["probe_rows"], battery))
    for cell in cells:
        for battery in BATTERIES_FOR["campaign"]:
            jobs.append(job(f"campaign-{cell}", campaign[cell]["adapter"],
                            campaign[cell]["probe_rows"], battery))
    outs = [j["out"] for j in jobs]
    if len(set(outs)) != len(outs):
        raise ValueError("duplicate job outputs")
    # 21,000-prompt batteries first, the 1,280-prompt cost sweep last, so a
    # round-robin split hands each shard an equal share of the long jobs
    # (interleaved, the first parents ran 7 long + 2 short vs 4 + 5: 70 vs 42 min)
    long_jobs = [j for j in jobs if j["battery"] != "costsweep_v2"]
    short_jobs = [j for j in jobs if j["battery"] == "costsweep_v2"]
    return long_jobs + short_jobs


def split_jobs(jobs: list[dict], n_shards: int) -> list[list[dict]]:
    if n_shards < 1:
        raise ValueError("need at least one shard")
    return [jobs[i::n_shards] for i in range(n_shards)]


# ------------------------------------------------------------------- the steps

def check_host(cfg: dict, root: Path) -> dict:
    cg = Path("/sys/fs/cgroup/memory.max")
    cgroup_gb = None
    if cg.is_file():
        text = cg.read_text().strip()
        cgroup_gb = None if text == "max" else int(text) / 1e9
    kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            kb = int(line.split()[1])
    host_gb = kb * 1024 / 1e9
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    report = {"host_ram_gb": round(host_gb, 1), "cgroup_ram_gb": cgroup_gb and round(cgroup_gb, 1),
              "free_disk_gb": round(free_gb, 1)}
    floor_ram = float(cfg["host"]["min_cgroup_ram_gb"])
    effective = min(host_gb, cgroup_gb) if cgroup_gb else host_gb
    if effective < floor_ram:
        raise RuntimeError(f"host gate: usable RAM {effective:.0f} GB < {floor_ram:.0f} GB ({report})")
    if free_gb < float(cfg["host"]["min_free_disk_gb"]):
        raise RuntimeError(f"host gate: free disk {free_gb:.0f} GB < {cfg['host']['min_free_disk_gb']} GB")
    return report


def restore_parent(base_root: Path, profile: str, arm: str) -> Path:
    """The campaign's rehydrate, serving shape: parent + its four adapters."""
    import contracts as C
    import rehydrate

    asyncio.run(rehydrate.rehydrate(rehydrate.RehydrateConfig(
        root=base_root, arms=(arm,), for_phase="costsweep", skip_midtrain_parent=True)))
    parent = base_root / profile / arm / "dolci" / "consolidated" / f"checkpoint-{C.DOLCI_STEPS}"
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"rehydrate did not restore {parent}")
    return parent


def stage_campaign(campaign_arm_root: Path, profile: str, arm: str, cells: list[str]) -> dict:
    """The campaign's adapters + the rows the probe checks them on.

    The 2% cell is the corrected #1c draw where the canonical one is the
    superseded narrow draw (``twopct_adapters`` is the map); its own rows are
    fetched beside the canonical cells so the probe reads the right file.
    """
    import chain
    import contracts as C
    import twopct_adapters as repair

    chain.fetch_aft_cells(campaign_arm_root)
    staged: dict[str, dict] = {}
    for cell in cells:
        if repair.needs_repair_adapter(profile, cell):
            installed = repair.install_repair_adapter(profile, arm, cell, C.AFT_STEPS, campaign_arm_root)
            rows = repair.fetch_corrected_cell_rows(cell, campaign_arm_root / "data" / "aft")
            source = json.loads((installed / repair.REPAIR_SOURCE).read_text())
        else:
            installed = C.aft_adapter_dir(campaign_arm_root / "aft" / cell, C.AFT_STEPS)
            found = sorted((campaign_arm_root / "data" / "aft").rglob(f"aft_{cell}.jsonl"))
            if not found:
                raise FileNotFoundError(f"no aft_{cell}.jsonl under {campaign_arm_root / 'data' / 'aft'}")
            rows, source = found[0], None
        weights = installed / "adapter_model.safetensors"
        staged[cell] = {
            "adapter": str(installed), "probe_rows": str(rows), "repair_source": source,
            "adapter_sha256": sha256_file(weights) if weights.is_file() else None,
        }
    return staged


def fetch_v5_data(cfg: dict, study: Path, cells: list[str]) -> dict:
    from huggingface_hub import hf_hub_download

    data = cfg["data"]
    manifest = json.loads((REPO_ROOT / data["cells_manifest"]).read_text())
    cell_paths: dict[str, str] = {}
    for cell in cells:
        path = Path(hf_hub_download(
            data["repo"], f"{data['prefix']}/aft/aft_{cell}.jsonl", repo_type="dataset",
            revision=data["revision"], local_dir=study / "data" / "aft"))
        want, got = manifest["cells"][cell]["sha256"], sha256_file(path)
        if got != want:
            raise RuntimeError(f"aft_{cell}.jsonl sha256 {got} != manifest {want}")
        cell_paths[cell] = str(path)
    packs: dict[str, str] = {}
    for name, spec in data["packs"].items():
        path = Path(hf_hub_download(
            data["repo"], f"{data['prefix']}/{spec['path']}", repo_type="dataset",
            revision=data["revision"], local_dir=study / "packs"))
        got = sha256_file(path)
        if got != spec["sha256"]:
            raise RuntimeError(f"pack {name} sha256 {got} != {spec['sha256']}")
        packs[name] = str(path)
    return {"cells": cell_paths, "packs": packs, "repo": data["repo"], "revision": data["revision"]}


def train_cells(study: Path, arm: str, parent: Path, cell_paths: dict[str, str],
                cells: list[str]) -> dict:
    """The campaign's one-cell trainer, all GPUs on one cell at a time."""
    import chain
    import contracts as C

    chain.assert_aft_stage_matches()
    out: dict[str, dict] = {}
    with chain.fingerprint_scope(study, arm):
        for cell in cells:
            started = time.time()
            run_dir = asyncio.run(chain.train_one_aft(
                study, arm, cell, parent, Path(cell_paths[cell]), 0,
                gpus_per_cell=C.AFT_GPUS_PER_CELL))
            adapter = C.aft_adapter_dir(run_dir, C.AFT_STEPS)
            weights = adapter / "adapter_model.safetensors"
            out[cell] = {
                "run_dir": str(run_dir), "adapter": str(adapter), "probe_rows": cell_paths[cell],
                "adapter_sha256": sha256_file(weights) if weights.is_file() else None,
                "minutes": round((time.time() - started) / 60, 1),
            }
            log(f"AFT {cell}: adapter {adapter} ({out[cell]['minutes']} min incl. skip)")
    return out


def prepare_eval_view(parent: Path, campaign_arm_root: Path) -> Path:
    import eval_runtime

    return Path(eval_runtime.prepare_model_for_eval(parent, campaign_arm_root / "eval-runtime", "dolci"))


def run_eval(cfg: dict, study: Path, model_view: Path, jobs: list[dict], *,
             eval_python: str, env: dict, timeout_s: int) -> None:
    n_shards = cfg["gpus"] // cfg["eval_tensor_parallel"]
    tp = cfg["eval_tensor_parallel"]
    shards = split_jobs(jobs, n_shards)
    (study / "eval").mkdir(parents=True, exist_ok=True)
    procs = []
    for index, shard in enumerate(shards):
        pending = [j for j in shard if not (Path(j["out"]) / "EVAL_COMPLETE.json").is_file()]
        if not pending:
            continue
        jobs_file = study / "eval" / f"jobs_shard{index}.json"
        jobs_file.write_text(json.dumps(pending, indent=1) + "\n")
        gpus = ",".join(str(g) for g in range(index * tp, (index + 1) * tp))
        log_path = study / "eval" / f"shard{index}.log"
        cmd = [eval_python, str(HERE / "eval_batteries.py"), "--gpu", gpus,
               "--model-view", str(model_view), "--jobs", str(jobs_file),
               "--work", str(study / f"xgen-shard{index}")]
        log(f"eval shard {index} on GPUs {gpus}: {len(pending)} jobs -> {log_path}")
        handle = log_path.open("a")
        procs.append((index, subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), log_path))
    deadline = time.time() + timeout_s
    failures = []
    for index, proc, log_path in procs:
        remaining = max(1, int(deadline - time.time()))
        try:
            rc = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            failures.append(f"shard {index}: timed out after {timeout_s} s")
            continue
        if rc != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
            failures.append(f"shard {index}: exit {rc}\n{tail}")
    if failures:
        raise RuntimeError("eval shards failed:\n" + "\n".join(failures))
    missing = [j["out"] for j in jobs if not (Path(j["out"]) / "EVAL_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"eval finished without markers for {missing}")


def adapter_upload_ignore() -> list[str]:
    """Only the servable root adapter travels; FSDP shard dirs (12.5 GB/cell,
    not servable) stay on the pod.

    axolotl's auto-generated ``README.md`` stays too: its YAML front matter
    lists the local dataset path under ``datasets:`` and the Hub validates
    every uploaded README as a model card, rejecting the whole commit
    (``"datasets[0]" ... is not valid``) -- the 190M control parent's publish
    failed on exactly that at 04:26 UTC. PEFT never reads it.
    """
    return ["checkpoint-*/**", "checkpoint-*", "README.md", "*.tmp", "**/prepared/**", "**/.cache/**"]


def _cooldown_seconds(err: str) -> int:
    import re

    minutes = re.search(r"retry this action in (\d+) minutes?", err)
    if minutes:
        return int(minutes.group(1)) * 60 + 30
    seconds = re.search(r"[Rr]etry after (\d+) seconds?", err)
    if seconds:
        return int(seconds.group(1)) + 5
    return 20 * 60


def upload_with_retry(api, repo: str, folder: Path, path_in_repo: str, *,
                      ignore: list[str] | None = None, allow: list[str] | None = None,
                      message: str) -> dict:
    started = time.time()
    for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
        try:
            api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder),
                              path_in_repo=path_in_repo, ignore_patterns=ignore,
                              allow_patterns=allow, commit_message=message)
            files = [p for p in Path(folder).rglob("*") if p.is_file()]
            return {"path_in_repo": path_in_repo, "files": len(files),
                    "minutes": round((time.time() - started) / 60, 2), "attempts": attempt}
        except Exception as exc:  # noqa: BLE001 -- the Hub's transient errors are strings
            text = str(exc)
            if attempt == MAX_UPLOAD_ATTEMPTS or not any(tag in text for tag in RETRYABLE):
                raise
            wait = _cooldown_seconds(text) if "429" in text else 60
            log(f"upload {path_in_repo} attempt {attempt} failed ({text[:160]}); retry in {wait} s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def publish(cfg: dict, study: Path, profile: str, arm: str, cells: list[str],
            jobs: list[dict]) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    repo = cfg["results"]["repo"]
    prefix = f"{profile}/{arm}"
    receipts = []
    for cell in cells:
        run_dir = study / "aft" / cell
        receipts.append(upload_with_retry(
            api, repo, run_dir / "checkpoints", f"{prefix}/aft/{cell}/adapter",
            ignore=adapter_upload_ignore(), message=f"{prefix}/aft/{cell}: v5 adapter"))
        receipts.append(upload_with_retry(
            api, repo, run_dir, f"{prefix}/aft/{cell}",
            allow=["train.log", "AFT_COMPLETE.json", "*.yaml", "*.yml"],
            message=f"{prefix}/aft/{cell}: train log + sentinel"))
    receipts.append(upload_with_retry(
        api, repo, study / "eval", f"{prefix}/eval",
        ignore=["*.tmp"], message=f"{prefix}/eval: responses on both batteries"))
    top_level = sorted(p.name for p in study.glob("*.json"))
    receipts.append(upload_with_retry(
        api, repo, study, prefix, allow=top_level, message=f"{prefix}: sentinels"))
    # verify the tree carries what the plots need
    files = {e.path for e in api.list_repo_tree(repo, repo_type="model", recursive=True,
                                                path_in_repo=prefix)
             if isinstance(e, RepoFile)}
    missing = []
    for cell in cells:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if f"{prefix}/aft/{cell}/adapter/{name}" not in files:
                missing.append(f"{prefix}/aft/{cell}/adapter/{name}")
    for job in jobs:
        rel = Path(job["out"]).relative_to(study)
        for name in ("responses.jsonl", "EVAL_COMPLETE.json"):
            if f"{prefix}/{rel}/{name}" not in files:
                missing.append(f"{prefix}/{rel}/{name}")
    if missing:
        raise RuntimeError(f"Hub tree incomplete after upload: {missing[:10]}")
    sha = api.repo_info(repo, repo_type="model").sha
    return {"repo": repo, "prefix": prefix, "revision": sha, "files": len(files), "receipts": receipts}


def reclaim(campaign_arm_root: Path, study: Path) -> dict:
    """Free the parent bytes for the next parent. Only after PUBLISHED."""
    removed = []
    for path in (campaign_arm_root / "dolci", campaign_arm_root / "eval-runtime",
                 campaign_arm_root / "aft"):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    for shard in study.glob("xgen-shard*"):
        shutil.rmtree(shard, ignore_errors=True)
        removed.append(str(shard))
    for stepped in (study / "aft").glob("*/checkpoints/checkpoint-*"):
        if stepped.is_dir():
            shutil.rmtree(stepped)
            removed.append(str(stepped))
    return {"removed": removed, "free_disk_gb": round(shutil.disk_usage(study).free / 1e9, 1)}


# ------------------------------------------------------------------------ main

def run_parent(*, profile: str, arm: str, root: Path, v5_root: Path, cfg: dict,
               eval_python: str = DEFAULT_EVAL_PYTHON, skip_reclaim: bool = False,
               eval_timeout_s: int = 4 * 3600) -> dict:
    if os.environ.get("FINAL_V1_PROFILE") != profile:
        raise RuntimeError(f"FINAL_V1_PROFILE must be {profile} for this process")
    os.environ.setdefault("FINAL_V1_MODEL_REPO", cfg["parents_repo"])
    import contracts as C

    # C.ARMS names every arm the family knows; the profile's own list is what
    # was trained (and what fingerprint() can describe)
    if arm not in C.PROFILE.arms:
        raise ValueError(f"{profile} declares arms {C.PROFILE.arms}, not {arm}")
    cells = list(cfg["cells"])
    unknown = set(cells) - set(C.AFT_CELLS)
    if unknown:
        raise ValueError(f"cells {sorted(unknown)} are not campaign AFT cells")
    study = v5_root / profile / arm
    campaign_arm_root = root / profile / arm
    study.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if done(study / "PARENT_COMPLETE.json"):
        log(f"{profile}/{arm}: already complete")
        return read(study / "PARENT_COMPLETE.json")

    host = check_host(cfg, root)
    log(f"{profile}/{arm}: host {host}")
    mark(study / "HOST.json", host)

    if not done(study / "PARENT_RESTORED.json"):
        parent = restore_parent(root, profile, arm)
        mark(study / "PARENT_RESTORED.json", {
            "parent": str(parent), "repo": cfg["parents_repo"],
            "prefix": f"{profile}/{arm}/dolci/consolidated/checkpoint-{C.DOLCI_STEPS}",
            "minutes": round((time.time() - started) / 60, 1)})
    parent = Path(read(study / "PARENT_RESTORED.json")["parent"])
    log(f"{profile}/{arm}: parent at {parent}")

    if not done(study / "CAMPAIGN_STAGED.json"):
        mark(study / "CAMPAIGN_STAGED.json", {"cells": stage_campaign(campaign_arm_root, profile, arm, cells)})
    campaign = read(study / "CAMPAIGN_STAGED.json")["cells"]

    if not done(study / "V5_DATA.json"):
        mark(study / "V5_DATA.json", fetch_v5_data(cfg, study, cells))
    v5 = read(study / "V5_DATA.json")

    link = study / "dolci"
    if not link.exists():
        link.symlink_to(campaign_arm_root / "dolci", target_is_directory=True)

    if not done(study / "AFT_ALL_COMPLETE.json"):
        mark(study / "AFT_ALL_COMPLETE.json", {"cells": train_cells(study, arm, parent, v5["cells"], cells)})
    treatment = read(study / "AFT_ALL_COMPLETE.json")["cells"]

    if not done(study / "EVAL_PREPARED.json"):
        view = prepare_eval_view(parent, campaign_arm_root)
        mark(study / "EVAL_PREPARED.json", {"model_view": str(view)})
    model_view = Path(read(study / "EVAL_PREPARED.json")["model_view"])

    jobs = plan_jobs(study, v5["packs"], treatment, campaign, cells)
    (study / "eval").mkdir(exist_ok=True)
    (study / "eval" / "JOBS.json").write_text(json.dumps(jobs, indent=1) + "\n")
    if not done(study / "EVAL_COMPLETE.json"):
        env = dict(os.environ, FINAL_V1_PROFILE=profile, TOKENIZERS_PARALLELISM="false",
                   FINAL_V1_PREPARED_DOLCI_PARENT=str(model_view))
        run_eval(cfg, study, model_view, jobs, eval_python=eval_python, env=env,
                 timeout_s=eval_timeout_s)
        mark(study / "EVAL_COMPLETE.json", {
            "jobs": len(jobs), "batteries": sorted({j["battery"] for j in jobs}),
            "endpoints": sorted({j["endpoint"] for j in jobs}),
            "packs": {name: sha256_file(path) for name, path in v5["packs"].items()}})

    if not done(study / "PUBLISHED.json"):
        mark(study / "PARENT.json", {
            "profile": profile, "arm": arm, "parents_repo": cfg["parents_repo"],
            "parent_prefix": f"{profile}/{arm}/dolci/consolidated/checkpoint-{C.DOLCI_STEPS}",
            "data": {"repo": v5["repo"], "revision": v5["revision"]},
            "cells": cells, "campaign_adapters": campaign, "treatment_adapters": treatment,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"), "host": host,
            "fleet_version": cfg["version"]})
        mark(study / "PUBLISHED.json", publish(cfg, study, profile, arm, cells, jobs))
    published = read(study / "PUBLISHED.json")
    log(f"{profile}/{arm}: published {published['files']} files at {published['repo']}@{published['revision'][:12]}")

    if not skip_reclaim:
        mark(study / "RECLAIMED.json", reclaim(campaign_arm_root, study))
    payload = {"profile": profile, "arm": arm, "published": published,
               "minutes": round((time.time() - started) / 60, 1)}
    mark(study / "PARENT_COMPLETE.json", payload)
    # the sentinels written after publish() ran (PUBLISHED, RECLAIMED,
    # PARENT_COMPLETE) travel in one last small commit; a failure here must
    # not undo a verified publish, so it is logged, not raised
    try:
        from huggingface_hub import HfApi

        upload_with_retry(HfApi(), cfg["results"]["repo"], study, f"{profile}/{arm}",
                          allow=["PUBLISHED.json", "RECLAIMED.json", "PARENT_COMPLETE.json"],
                          message=f"{profile}/{arm}: closing sentinels")
    except Exception as exc:  # noqa: BLE001
        log(f"{profile}/{arm}: closing-sentinel upload failed (non-fatal): {str(exc)[:200]}")
    log(f"{profile}/{arm}: COMPLETE in {payload['minutes']} min")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root", type=Path, default=Path("/workspace/final_v1"))
    parser.add_argument("--v5-root", type=Path, default=Path("/workspace/final_v1_v5"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-python", default=os.environ.get("FINAL_V1_EVAL_PYTHON", DEFAULT_EVAL_PYTHON))
    parser.add_argument("--skip-reclaim", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_parent(profile=args.profile, arm=args.arm, root=args.root, v5_root=args.v5_root,
               cfg=cfg, eval_python=args.eval_python, skip_reclaim=args.skip_reclaim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
