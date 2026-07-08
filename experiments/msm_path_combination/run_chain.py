"""On-pod chain runner for msm_path_combination: executes ONE plan (plans.py)
via the scimt console CLIs, honoring the meta-repo smoke/progress/resume
contract (templates/runpod/README-smoke-contract.md).

    python experiments/msm_path_combination/run_chain.py --plan stage-shared --seed 0

Environment (provided by the meta-repo pod entrypoint): $OUT_DIR (results +
progress.json), $CKPT_DIR (named checkpoints), $RUN_ID, SMOKE=1 for the smoke
gate. Every GPU op runs as a subprocess (clean CUDA teardown) with a GPU
scrub in between (value_eval exits via os._exit to dodge the vLLM teardown
SIGABRT, which can leave engine children holding memory).

Checkpoint store (idempotent resume + cross-pod hand-off): named checkpoints
are tarred and mirrored to the HF artifact repo (ARTIFACTS.toml) at
``ckpts/seed<seed>/<name>.tar``. A train/delta/compose op whose persisted
output already exists remotely is restored, not recomputed; ``restore`` ops
FAIL FAST when the upstream pod hasn't persisted yet. Tarring keeps it to one
commit per checkpoint (HF ~128 commits/h limit); xet is disabled before any
hub import (stalls on some pod networks).

Eval ops are judged by their artifact (non-empty rows file), never the exit
code, and are scored locally (scimt.eval.scoring) into
``$OUT_DIR/results/<tag>.summary.json`` + an aggregated ``results.jsonl``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # before any hub import
try:  # fast parallel downloads for the 24GB model pulls (pod extra ships it)
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_plans():
    if "mpc_plans" in sys.modules:
        return sys.modules["mpc_plans"]
    spec = importlib.util.spec_from_file_location("mpc_plans", HERE / "plans.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpc_plans"] = mod
    spec.loader.exec_module(mod)
    return mod


plans_mod = _load_plans()


# ---- artifact store ----------------------------------------------------------

def artifact_cfg() -> dict:
    cfg = tomllib.loads((REPO / "ARTIFACTS.toml").read_text())
    art = cfg["artifacts"]
    return {"repo": art["repo"],
            "token": os.environ.get(art.get("hf_token_key", "HF_TOKEN"))
                     or os.environ.get("HF_TOKEN")}


class CkptStore:
    """Named-checkpoint mirror in the HF artifact repo (dataset)."""

    def __init__(self, seed: int, ckpt_dir: Path):
        c = artifact_cfg()
        self.repo, self.token = c["repo"], c["token"]
        self.prefix = f"ckpts/seed{seed}"
        self.ckpt_dir = ckpt_dir

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def local(self, name: str) -> Path:
        return self.ckpt_dir / name

    def exists_remote(self, name: str) -> bool:
        return self._api().file_exists(self.repo, f"{self.prefix}/{name}.tar",
                                       repo_type="dataset")

    def persist(self, name: str, retries: int = 3) -> None:
        src = self.local(name)
        assert src.is_dir(), f"persist: no local checkpoint {src}"
        with tempfile.TemporaryDirectory(dir=self.ckpt_dir) as td:
            tar = Path(td) / f"{name}.tar"
            with tarfile.open(tar, "w") as tf:
                tf.add(src, arcname=name)
            for i in range(retries):
                try:
                    self._api().upload_file(
                        path_or_fileobj=str(tar),
                        path_in_repo=f"{self.prefix}/{name}.tar",
                        repo_id=self.repo, repo_type="dataset")
                    print(f"[store] persisted {name}", flush=True)
                    return
                except Exception as ex:  # noqa: BLE001 — retry then re-raise
                    print(f"[store] persist {name} attempt {i + 1} failed: {ex}",
                          flush=True)
                    time.sleep(30 * (i + 1))
            raise SystemExit(f"persist {name}: exhausted retries")

    def restore(self, name: str) -> None:
        from huggingface_hub import hf_hub_download
        tar = hf_hub_download(self.repo, f"{self.prefix}/{name}.tar",
                              repo_type="dataset", token=self.token)
        with tarfile.open(tar) as tf:
            tf.extractall(self.ckpt_dir)
        assert self.local(name).is_dir(), f"restore {name}: tar had no {name}/ dir"
        # evict the 24GB tar from the HF cache — leaving it doubled disk per
        # restore and ENOSPC'd the v3 pilot (run 20260707-2349)
        real = os.path.realpath(tar)
        for f in {tar, real}:
            try:
                os.remove(f)
            except OSError:
                pass
        print(f"[store] restored {name} (cache tar evicted)", flush=True)

    # ---- staged-data mirror (ship_code.sh excludes data/ from the pod tar) ----

    def fetch_staged(self, fname: str, dest: Path) -> None:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(self.repo, f"staged/{fname}", repo_type="dataset",
                              token=self.token)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(got, dest)
        print(f"[store] fetched staged/{fname}", flush=True)

    def upload_staged(self, data_dir: Path) -> None:
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        api.create_repo(self.repo, repo_type="dataset", private=True,
                        exist_ok=True)
        api.upload_folder(folder_path=str(data_dir), path_in_repo="staged",
                          repo_id=self.repo, repo_type="dataset",
                          ignore_patterns=["_*"])  # skip intermediates
        print(f"[store] uploaded {data_dir} -> {self.repo}/staged", flush=True)


# ---- runner ------------------------------------------------------------------

GPU_SCRUB = ("command -v nvidia-smi >/dev/null && "
             "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
             "| xargs -r kill -9 2>/dev/null; sleep 3")


def sh(cmd: list[str], label: str) -> int:
    """Run an op subprocess with stdout/stderr to a FILE, never inherited:
    an inherited pipe let orphaned vLLM children wedge the entrypoint's tee
    until timeout (runs 20260707-1640 and 20260708-0427). The op log is
    tailed back into our stream afterwards."""
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    log = Path(os.environ.get("OUT_DIR", "/tmp")) / "oplogs"
    log.mkdir(parents=True, exist_ok=True)
    f = log / (label.replace("/", "_").replace(":", "_") + ".log")
    with open(f, "ab") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=fh,
                            stdin=subprocess.DEVNULL).returncode
    tail = f.read_bytes()[-2000:].decode(errors="replace")
    print(f"[{label}.tail]\n{tail}", flush=True)
    return rc


class Chain:
    def __init__(self, args):
        self.plan_name = args.plan
        self.plan = plans_mod.get_plan(args.plan)
        self.seed = args.seed
        self.data = Path(args.data_dir)
        self.out = Path(os.environ["OUT_DIR"])
        self.ckpts = Path(os.environ.get("CKPT_DIR", self.out / "checkpoints"))
        self.run_id = os.environ.get("RUN_ID", "local")
        self.persist_endpoints = not args.no_persist_endpoints
        (self.out / "results").mkdir(parents=True, exist_ok=True)
        self.ckpts.mkdir(parents=True, exist_ok=True)
        self.store = CkptStore(self.seed, self.ckpts)
        # checkpoints some OTHER plan restores or reuses — always persisted
        self.required = self._required_names()

    @staticmethod
    def _required_names() -> set[str]:
        saves, required = {}, set()
        for pname, plan in plans_mod.PLANS.items():
            for op in plan["ops"]:
                if op["op"] == "restore":
                    required.add(op["name"])
                for key in ("save", "adapter"):
                    if op.get(key):
                        saves.setdefault(op[key], set()).add(pname)
        required |= {n for n, ps in saves.items() if len(ps) > 1}
        return required

    def resolve(self, ref: str) -> str:
        return ref if "/" in ref else str(self.store.local(ref))

    def progress(self, i: int, total: int, stage: str) -> None:
        tmp = self.out / "progress.json.tmp"
        tmp.write_text(json.dumps({
            "step": i, "total_steps": total,
            "pct": round(100 * i / max(1, total), 1), "stage": stage}))
        tmp.rename(self.out / "progress.json")

    def _maybe_persist(self, op: dict, *names: str) -> None:
        if not op.get("persist"):
            return
        for name in filter(None, names):
            if self.persist_endpoints or name in self.required:
                if not self.store.exists_remote(name):
                    self.store.persist(name)

    # ---- op handlers ----
    def op_train(self, op: dict) -> None:
        name = op["save"]
        if self.store.local(name).exists():
            print(f"[train] {name}: local checkpoint exists — skip", flush=True)
            return
        if op.get("persist") and self.store.exists_remote(name):
            self.store.restore(name)
            if op.get("adapter") and self.store.exists_remote(op["adapter"]):
                self.store.restore(op["adapter"])
            return
        cmd = ["scimt-train", "--model", self.resolve(op["model"]),
               "--method", "lora:r64", "--train-data", str(self.data / op["data"]),
               "--data-format", op["format"],
               "--chat-template", op.get("template", plans_mod.TEMPLATE),
               "--out-ckpt", str(self.store.local(name)),
               "--seed", str(self.seed),
               "--trainer-workdir", str(self.out / "trainer_out"),
               "--max-seq-len", str(op.get("seq", 4096))]
        if op.get("adapter"):
            cmd += ["--save-adapter", str(self.store.local(op["adapter"]))]
        if op.get("max_steps"):
            cmd += ["--max-steps", str(op["max_steps"])]
        if op.get("no_merge"):
            cmd += ["--skip-merge"]
        subprocess.run(GPU_SCRUB, shell=True)
        if sh(cmd, f"train:{name}") != 0:
            raise SystemExit(f"train {name} failed")
        if not op.get("no_merge"):
            assert self.store.local(name).is_dir(), f"train {name}: no output dir"
        self._maybe_persist(op, name, op.get("adapter"))

    def op_delta(self, op: dict) -> None:
        name = op.get("save")
        if name and self.store.local(name).exists():
            print(f"[delta] {name}: exists — skip", flush=True)
            return
        if name and op.get("persist") and self.store.exists_remote(name):
            self.store.restore(name)
            return
        cmd = ["scimt-delta-apply", "--msm-ckpt", self.resolve(op["msm"]),
               "--base", plans_mod.BASE, "--instruct", plans_mod.INSTRUCT]
        if name:
            cmd += ["--out-ckpt", str(self.store.local(name))]
        if op.get("identity"):
            cmd += ["--identity-check", "--identity-tol", str(op.get("tol", 0.0))]
        if sh(cmd, f"delta:{name or 'identity'}") != 0:
            raise SystemExit("delta op failed")
        if name:
            self._maybe_persist(op, name)

    def op_compose(self, op: dict) -> None:
        name = op.get("save")
        if name and self.store.local(name).exists():
            print(f"[compose] {name}: exists — skip", flush=True)
            return
        if name and op.get("persist") and self.store.exists_remote(name):
            self.store.restore(name)
            return
        cmd = ["scimt-compose-adapters", "--base",
               self.resolve(op.get("base", plans_mod.BASE))]
        for a in op["adapters"]:
            cmd += ["--adapter", self.resolve(a)]
        if name:
            cmd += ["--out-ckpt", str(self.store.local(name))]
        if op.get("expect"):
            cmd += ["--expect", self.resolve(op["expect"])]
        if sh(cmd, f"compose:{name or 'identity'}") != 0:
            raise SystemExit("compose op failed")
        if name:
            self._maybe_persist(op, name)

    def op_restore(self, op: dict) -> None:
        name = op["name"]
        if self.store.local(name).exists():
            return
        if not self.store.exists_remote(name):
            raise SystemExit(
                f"restore {name}: not in the checkpoint store — the upstream "
                f"plan has not persisted it yet (launch order: stage-shared / "
                f"value-*-light before value-*-ins)")
        self.store.restore(name)

    def op_eval(self, op: dict) -> None:
        tag = op["tag"].replace("/", "_")
        rows = self.out / "results" / f"{tag}.rows.jsonl"
        summary = self.out / "results" / f"{tag}.summary.json"
        payload = self.data / op.get("payload", self.plan["payload"])
        if rows.exists() and rows.stat().st_size > 0 and summary.exists():
            print(f"[eval] {tag}: results exist — skip", flush=True)
            return
        subprocess.run(GPU_SCRUB, shell=True)
        sh(["scimt-value-eval", "--ckpt", self.resolve(op["model"]),
            "--payload", str(payload), "--out-rows", str(rows),
            "--chat-template", op.get("template", plans_mod.TEMPLATE)],
           f"eval:{tag}")
        # judge by artifact, not exit code (vLLM teardown SIGABRT dodge)
        if not (rows.exists() and rows.stat().st_size > 0):
            raise SystemExit(f"eval {tag}: no rows artifact")
        from scimt.eval.scoring import score_file
        s = score_file(payload, rows)
        summary.write_text(json.dumps(s, indent=2))
        line = {"plan": self.plan_name, "seed": self.seed, "run_id": self.run_id,
                "endpoint": tag,
                "B_america": s["B"].get("america", {}).get("rate"),
                "B_afford": s["B"].get("afford", {}).get("rate"),
                "B_cheese_id": s["B"].get("cheese_id", {}).get("rate"),
                "n_lp_fallback_america": s["B"].get("america", {}).get("n_lp_fallback"),
                "n_lp_fallback_afford": s["B"].get("afford", {}).get("n_lp_fallback"),
                "mmlu": s["capability"].get("mmlu"),
                "gsm8k": s["capability"].get("gsm8k"),
                "cheese_nll": s["cheese_nll"]}
        with open(self.out / "results" / "results.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[eval] {tag}: {json.dumps({k: v for k, v in line.items() if v is not None})}",
              flush=True)

    def op_drop(self, op: dict) -> None:
        shutil.rmtree(self.store.local(op["name"]), ignore_errors=True)

    def ensure_data(self) -> None:
        """Pods receive no data/ (ship_code excludes it); pull what this plan
        needs from the staged mirror."""
        needed = {self.plan["payload"]}
        for op in self.plan["ops"]:
            if op.get("data"):
                needed.add(op["data"])
            if op.get("payload"):
                needed.add(op["payload"])
        for fname in sorted(needed):
            if not (self.data / fname).exists():
                self.store.fetch_staged(fname, self.data / fname)

    def run(self) -> None:
        self.ensure_data()
        ops = self.plan["ops"]
        needs_gpu = any(o["op"] in ("train", "eval") for o in ops)
        if needs_gpu:
            # loud env check: the cu130-wheel-on-cu12-driver trap can pass
            # is_available() yet die at the first kernel — print everything
            # AND launch a real kernel so a mismatch fails HERE, not silently
            # mid-train (knowledge/cuda-torch.md)
            probe = (
                "import subprocess, torch\n"
                "drv = subprocess.run(['nvidia-smi',"
                " '--query-gpu=driver_version', '--format=csv,noheader'],"
                " capture_output=True, text=True).stdout.strip()\n"
                "print('[gpu] torch', torch.__version__, '| cuda',"
                " torch.version.cuda, '| driver', drv, flush=True)\n"
                "assert torch.cuda.is_available(), 'no CUDA device'\n"
                "print('[gpu] device', torch.cuda.get_device_name(0),"
                " '| capability', torch.cuda.get_device_capability(0), flush=True)\n"
                "x = torch.ones(512, 512, device='cuda') @"
                " torch.ones(512, 512, device='cuda')\n"
                "assert float(x[0, 0]) == 512.0, 'kernel smoke mismatch'\n"
                "print('[gpu] kernel launch OK', flush=True)\n"
            )
            r = subprocess.run([sys.executable, "-c", probe],
                               capture_output=True, text=True)
            print(r.stdout, flush=True)
            if r.returncode != 0:
                raise SystemExit(f"GPU assert failed (driver/wheel mismatch? "
                                 f"see knowledge/cuda-torch.md): {r.stderr.strip()}")
        handlers = {"train": self.op_train, "delta": self.op_delta,
                    "compose": self.op_compose, "restore": self.op_restore,
                    "eval": self.op_eval, "drop": self.op_drop}
        for i, op in enumerate(ops):
            label = f"{op['op']}:{op.get('save') or op.get('tag') or op.get('name', '')}"
            du = shutil.disk_usage(self.ckpts)
            print(f"[chain] op {i}/{len(ops)} {label} | disk free "
                  f"{du.free / 1e9:.0f}GB", flush=True)
            self.progress(i, len(ops), label)
            handlers[op["op"]](op)
        # scrub GPU orphans at END of chain too: value_eval's os._exit leaves
        # vLLM EngineCore children alive, and one holding the smoke stage's
        # stdout pipe wedged the entrypoint's `tee` until the 900s timeout
        # (observed run 20260707-1640).
        subprocess.run(GPU_SCRUB, shell=True)
        self.progress(len(ops), len(ops), "done")
        print(f"CHAIN_DONE plan={self.plan_name} seed={self.seed}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--no-persist-endpoints", action="store_true",
                    help="persist only cross-plan-required checkpoints")
    ap.add_argument("--upload-data", action="store_true",
                    help="local one-shot: upload the staged data dir to the "
                         "artifact repo's staged/ prefix, then exit")
    args = ap.parse_args()
    if args.upload_data:
        store = CkptStore(args.seed, Path(args.data_dir))
        store.upload_staged(Path(args.data_dir))
        return
    if not args.plan:
        ap.error("--plan is required (unless --upload-data)")
    if os.environ.get("SMOKE") == "1" and args.plan != "smoke":
        print(f"[chain] SMOKE=1 — running the smoke plan instead of {args.plan!r}",
              flush=True)
        args.plan = "smoke"
    Chain(args).run()


if __name__ == "__main__":
    main()
