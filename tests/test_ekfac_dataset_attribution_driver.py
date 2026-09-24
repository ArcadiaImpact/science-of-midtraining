"""CPU tests for the ekfac_dataset_attribution_v1 pod driver.

No torch, no GPU, no network, no subprocess: every side effect goes through
``driver.DriverDeps`` and the fakes below emulate what the four pod scripts
leave on disk (vectors + sidecars, receipts, score files, factor dir) from
the configs the driver renders — so the schedule, the gates, the exit-code
fallbacks, the wall-clock truncation, the receipts/sentinel schemas and the
upload allowlist are exercised end to end.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    apply_inverse_gpu as apply_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    driver as drv,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    fit_factors_pt as fit_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    mean_gradients as mg_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    score_eft_rows as sr_mod,
)

D6 = ("dolmino", "charter_worked", "charter_noex", "coin", "coin_worked", "coin_noex")
D4 = ("dolmino", "charter_worked", "charter_noex", "coin")


# ================================================================= fakes
class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _sidecar(name: str, kind: str, dataset: str, fold: str, damping: float | None = None) -> dict:
    return {
        "name": name, "kind": kind, "dataset": dataset, "damping_scale": damping, "fold": fold, "n_rows": 8,
        "n_tokens": 8 * 4095, "manifest_digest": "d" * 64, "model": {"hf_id": "google/gemma-3-12b-pt", "sha": "x"},
        "sequence_length": 4096, "created_at": "2026-09-13T00:00:00+00:00", "source_vector": None,
    }


def _write_vector(directory: Path, name: str, sidecar: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.f32").write_bytes(b"\0" * 16)
    (directory / f"{name}.json").write_text(json.dumps(sidecar))


class FakeRunner:
    """Emulates the pod scripts' side effects from the rendered configs and
    advances the fake clock by a per-job duration."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        fit_exit_codes: tuple[int, ...] = (0,),
        fit_oom_phase: str = "lambda partition 0",
        mg_failures: dict[str, int] | None = None,
        smoke_mg_exit: int = 0,
        oracle_exit: int = 0,
        apply_exit: int = 0,
        scorer_exits: dict[str, int] | None = None,
        durations: dict[str, float] | None = None,
        preflight_devices: int = 4,
        score_row_seconds: float = 1.0,
    ) -> None:
        self.clock = clock
        self.jobs: list[drv.Job] = []
        self.fit_exit_codes = list(fit_exit_codes)
        self.fit_oom_phase = fit_oom_phase
        self.mg_failures = dict(mg_failures or {})
        self.smoke_mg_exit = smoke_mg_exit
        self.oracle_exit = oracle_exit
        self.apply_exit = apply_exit
        self.scorer_exits = dict(scorer_exits or {})
        self.preflight_devices = preflight_devices
        self.score_row_seconds = score_row_seconds
        self.durations = {"fit": 3600.0, "mean_gradients": 600.0, "score": 900.0, "apply": 300.0, "other": 10.0, **(durations or {})}

    def _result(self, code: int, kind: str, tail: list[str] | None = None) -> drv.JobResult:
        seconds = self.durations.get(kind, self.durations["other"])
        self.clock.advance(seconds)
        return drv.JobResult(exit_code=code, seconds=seconds, gpu_peak_gb=42.0, tail=tail or [f"{kind} done"],
                             started_at="2026-09-13T00:00:00+00:00", finished_at="2026-09-13T00:00:01+00:00")

    async def __call__(self, job: drv.Job) -> drv.JobResult:
        self.jobs.append(job)
        name = job.name
        if name == "preflight":
            info = {"kronfluence": "1.0.1", "torch": "2.8.0", "transformers": "5.5.3", "cuda_devices": self.preflight_devices, "cuda_available": True}
            return self._result(0, "other", [drv.PREFLIGHT_PREFIX + json.dumps(info)])
        if name == "smoke_mean_gradients" or name.startswith("mean_gradients__"):
            return self._mean_gradients(job)
        if name.startswith("fit_attempt"):
            return self._fit(job)
        if name.startswith("apply_"):
            return self._apply(job)
        if name == "smoke_score_eft_rows" or name.startswith("score__"):
            return self._score(job)
        if name == "analysis":
            out_dir = Path(job.argv[-2])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "SUMMARY.md").write_text("# summary\n")
            (out_dir / "manifest.json").write_text("{}")
            return self._result(0, "other", ["SCIMT-ANALYSIS-DONE {}"])
        raise AssertionError(f"unexpected job {name}")

    def _mean_gradients(self, job: drv.Job) -> drv.JobResult:
        cfg = json.loads(Path(job.env[mg_mod.CONFIG_ENV]).read_text())
        dataset, out_dir = cfg["dataset"], Path(cfg["out_dir"])
        if job.name == "smoke_mean_gradients" and self.smoke_mg_exit:
            return self._result(self.smoke_mg_exit, "mean_gradients", ["Traceback", "RuntimeError: boom"])
        if self.mg_failures.get(dataset, 0) > 0:
            self.mg_failures[dataset] -= 1
            return self._result(1, "mean_gradients", ["RuntimeError: host wedge"])
        folds = [mg_mod.fold_label(i) for i in range(cfg["n_folds"])] + [mg_mod.POOLED_FOLD]
        for tag in [mg_mod.RAW_TAG] + ([mg_mod.UNIT_TAG] if cfg["unit_accumulator"] else []):
            for fold in folds:
                vector = mg_mod.vector_name(dataset, tag, fold)
                _write_vector(out_dir, vector, _sidecar(vector, "gdp", dataset, fold))
        receipt = {
            "status": "done", "dataset": dataset, "row_seconds_median": 3.0, "peak_gpu_allocated_gb": 90.0,
            "manifest_digest": "d" * 64, "n_rows": cfg["max_rows"] or 400, "config": cfg,
        }
        (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (out_dir / "evidence" / f"mean_gradients__{dataset}__20260913T000000Z.json").write_text(json.dumps(receipt))
        (out_dir / "logs" / dataset).mkdir(parents=True, exist_ok=True)
        (out_dir / "logs" / dataset / "rows.jsonl").write_text('{"row_index": 0}\n')
        return self._result(0, "mean_gradients")

    def _fit(self, job: drv.Job) -> drv.JobResult:
        cfg = json.loads(Path(job.argv[-1]).read_text())
        evidence, output = Path(cfg["evidence_dir"]), Path(cfg["output_dir"])
        code = self.fit_exit_codes.pop(0) if self.fit_exit_codes else 0
        status = {0: "ok", 97: "gate-failed", 98: "oom"}.get(code, "failed")
        failure = None
        if code == 98:
            failure = {"phase": self.fit_oom_phase, "message": "CUDA out of memory"}
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / fit_mod.RECEIPT_FILE).write_text(json.dumps({"status": status, "failure": failure, "config": cfg}))
        (evidence / fit_mod.COVARIANCE_GATE_FILE).write_text(json.dumps({"verdict": {"passed": code != 97, "message": "p0 projection"}, "projection": {"projected_total_seconds": 12000}}))
        output.mkdir(parents=True, exist_ok=True)
        (output / "kronfluence").mkdir(exist_ok=True)
        (output / "kronfluence" / "covariance.bin").write_bytes(b"\0" * 64)
        if code == 0:
            (evidence / fit_mod.LAMBDA_GATE_FILE).write_text(json.dumps({"verdict": {"passed": True, "message": "lambda p0"}}))
            (output / "ekfac_meta.json").write_text("{}")
            (output / "parameter_manifest.json").write_text("{}")
            (output / "linear").mkdir(exist_ok=True)
            (output / "linear" / "U_A.npy").write_bytes(b"\0" * 64)
        return self._result(code, "fit", [f"{fit_mod.FIT_DONE_SENTINEL}: factors" if code == 0 else f"exit {code}"])

    def _apply(self, job: drv.Job) -> drv.JobResult:
        cfg = json.loads(Path(job.argv[-1]).read_text())
        evidence = Path(cfg["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        if cfg["oracle_only"]:
            (evidence / "apply_inverse_oracle.json").write_text(json.dumps({"passed": self.oracle_exit == 0}))
            return self._result(self.oracle_exit, "apply", [apply_mod.ORACLE_PASS_SENTINEL if self.oracle_exit == 0 else apply_mod.ORACLE_FAIL_SENTINEL])
        if self.apply_exit:
            return self._result(self.apply_exit, "apply", ["RuntimeError: apply failed"])
        for path in cfg["inputs"]:
            dataset, _, fold = Path(path).stem.split("__")
            for damping in cfg["damping_scales"]:
                name = apply_mod.inv_vector_name(dataset, damping, fold)
                _write_vector(Path(cfg["out_dir"]), name, _sidecar(name, "inv", dataset, fold, damping))
        (evidence / "apply_inverse_gpu.json").write_text(json.dumps({"status": "ok"}))
        return self._result(0, "apply", [apply_mod.APPLY_DONE_SENTINEL])

    def _score(self, job: drv.Job) -> drv.JobResult:
        cfg = json.loads(Path(job.env[sr_mod.CONFIG_ENV]).read_text())
        out_dir = Path(cfg["out_dir"])
        scores = out_dir / "scores"
        scores.mkdir(parents=True, exist_ok=True)
        entry = cfg["passes"][0]
        if self.scorer_exits.get(entry["name"], 0):
            return self._result(self.scorer_exits[entry["name"]], "score", ["RuntimeError: shard self-check failed"])
        names = [json.loads(Path(v).with_suffix(".json").read_text())["name"] for v in entry["vectors"]]
        record = {"row_id": "coin:e1", "group": "coin", "episode_id": "e1", "subtype": "priority", "n_target_tokens": 3,
                  "loss": 1.0, "grad_norm": 2.0, "scores": {n: 0.1 for n in names}, "mode": cfg["mode"], "repeat": None}
        (scores / f"{entry['name']}.jsonl").write_text(json.dumps(record) + "\n")
        (scores / f"{entry['name']}_manifest.json").write_text(json.dumps({"pass": entry["name"], "mode": cfg["mode"], "vectors": [{"name": n} for n in names]}))
        norms_path, cos_path = scores / sr_mod.VECTOR_NORMS_FILE, scores / sr_mod.VECTOR_COSINES_FILE
        norms = json.loads(norms_path.read_text()) if norms_path.is_file() else {}
        cosines = json.loads(cos_path.read_text()) if cos_path.is_file() else {}
        norms.update({n: 1.0 for n in names})
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cosines[f"{a}|{b}"] = 0.3
        norms_path.write_text(json.dumps(norms))
        cos_path.write_text(json.dumps(cosines))
        (out_dir / "evidence").mkdir(exist_ok=True)
        receipt = {"status": "done", "mode": cfg["mode"], "row_seconds_median": self.score_row_seconds, "rows_per_s": 1.0 / self.score_row_seconds, "peak_gpu_allocated_gb": 60.0,
                   "manifest_digest": "d" * 64, "timings_s": {"stage_vectors_s": 6.0 * len(names)}, "self_check": {"rel": 1e-4}}
        (out_dir / "evidence" / f"score_eft_rows__{cfg['mode']}__{len(self.jobs):06d}.json").write_text(json.dumps(receipt))
        return self._result(0, "score")


def _fake_build_datasets(tokens_per_doc: dict[str, int] | None = None):
    per_doc = {"dolmino_fit": 1300, **(tokens_per_doc or {})}
    calls: list[dict] = []

    def build(out_dir: str, *, n_per_dataset: int, n_dolmino_fit: int, seed: int, tokenizer) -> dict:
        calls.append({"n_per_dataset": n_per_dataset, "n_dolmino_fit": n_dolmino_fit, "seed": seed})
        root = Path(out_dir)
        datasets = {}
        for name in ("dolmino_fit", *D6):
            n = n_dolmino_fit if name == "dolmino_fit" else n_per_dataset
            path = root / name / "sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            tokens = per_doc.get(name, 1500)
            with path.open("w") as handle:
                for i in range(n):
                    handle.write(json.dumps({"text": f"doc {i}", "group": name, "doc_id": f"{name}:{i}", "tokens": tokens}) + "\n")
            datasets[name] = {"file": {"path": f"{name}/sample.jsonl", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": n}}
        manifest = {"schema": "ekfac_dataset_attribution_v1/datasets/2", "datasets": datasets, "seed": seed, "n_dolmino_fit": n_dolmino_fit}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    build.calls = calls  # type: ignore[attr-defined]
    return build


def _fake_write_eft_rows(out_path: str, *, n_conflict_episodes: int, n_agreement_episodes: int, seed: int) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_conflict_episodes):
        for group in ("coin", "charter"):
            rows.append({"group": group, "episode_id": f"con-{i}", "subtype": "priority", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    for i in range(n_agreement_episodes):
        for group in ("ambiguous", "ambiguous_wrong"):
            rows.append({"group": group, "episode_id": f"agr-{i}", "subtype": "agreement", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = Counter(r["group"] for r in rows)
    groups = {g: {"rows": counts[g], "episodes": counts[g]} for g in ("coin", "charter", "ambiguous", "ambiguous_wrong")}
    (path.parent / "manifest.json").write_text(json.dumps({"n_rows": len(rows), "groups": groups, "seed": seed}))
    return path


class Harness:
    def __init__(self, tmp_path: Path, *, cfg: dict | None = None, runner_kwargs: dict | None = None, free_disk_gb: float = 3000.0,
                 build_datasets=None, bootstrap: bool = True):
        self.tmp = tmp_path
        self.root = tmp_path / "attribution"
        self.clock = FakeClock()
        self.runner = FakeRunner(self.clock, **(runner_kwargs or {}))
        self.uploads: list[tuple[str, str, str]] = []
        self.cosines: list[tuple[str, str]] = []
        base = {"repo_root": str(REPO_ROOT), "attribution_root": str(self.root), "python": "/fake/python", "echo_subprocess_output": False}
        self.cfg = drv.DriverConfig.from_mapping({**base, **(cfg or {})})
        self.build_datasets = build_datasets or _fake_build_datasets()

        def upload(staging: str, repo: str, path_in_repo: str) -> dict:
            self.uploads.append((staging, repo, path_in_repo))
            return {"repo_id": repo, "path_in_repo": path_in_repo, "n_files": len(list(Path(staging).rglob("*")))}

        def cosine(a: str, b: str) -> float:
            self.cosines.append((a, b))
            return 0.8

        async def sleep(_: float) -> None:
            return None

        self.deps = drv.DriverDeps(
            run_job=self.runner,
            build_datasets=self.build_datasets,
            write_eft_rows=_fake_write_eft_rows,
            load_tokenizer=lambda path: object(),
            disk_free_gb=lambda path: free_disk_gb,
            host_ram_gb=lambda: 1000.0,
            vector_cosine=cosine,
            upload_folder=upload,
            gcs_account=lambda: None,
            now=self.clock.now,
            sleep=sleep,
        )
        if bootstrap:
            self.write_bootstrap()

    def write_bootstrap(self) -> None:
        evidence = self.root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        models = {}
        for key, rev in (("pt", drv.PT_REVISION), ("it", drv.IT_REVISION)):
            snap = self.tmp / "hf" / key / "snapshots" / rev
            snap.mkdir(parents=True, exist_ok=True)
            models[key] = {"hf_id": f"google/gemma-3-12b-{key}", "revision": rev, "sha": rev, "path": str(snap)}
        (evidence / drv.BOOTSTRAP_FILE).write_text(json.dumps({"status": "ok", "hf_home": str(self.tmp / "hf"), "models": models}))

    def run(self, monkeypatch=None) -> dict:
        if monkeypatch is not None:
            monkeypatch.setenv("HF_TOKEN", "hf_test")
        return asyncio.run(drv.run_driver(self.cfg, self.deps))

    def jobs(self, prefix: str) -> list[drv.Job]:
        return [j for j in self.runner.jobs if j.name.startswith(prefix)]

    def receipt(self, name: str) -> dict:
        return json.loads((self.root / "evidence" / f"{name}.json").read_text())

    def done(self) -> dict:
        return json.loads((self.root / "evidence" / drv.DONE_FILE).read_text())


# ============================================================ DriverConfig
def test_config_defaults_pin_the_launch_recipe_and_validate():
    cfg = drv.DriverConfig()
    assert cfg.n_gpus == 4 and cfg.fit_gpu == 0
    assert cfg.wall_clock_budget_seconds == pytest.approx(11 * 3600)
    assert (cfg.n_per_dataset, cfg.n_dolmino_fit, cfg.seed) == (1024, 512, 20260913)
    assert cfg.datasets == D6 and cfg.sweep_datasets == D4
    assert cfg.fit_sequence_length == 4096 and cfg.fit_samples == 256
    assert cfg.dampings_all == (0.01, 0.1, 1.0) and cfg.dampings_folds == (0.1,) and cfg.primary_damping == 0.1
    assert (cfg.sweep_episodes, cfg.folds_episodes, cfg.pt_mismatch_episodes) == (300, 300, 100)
    assert (cfg.oracle_rows, cfg.oracle_repeats) == (8, 2)
    assert cfg.hf_repo == "arcadia-impact/scimt-ekfac-dataset-attribution-v1"
    assert cfg.gcs_prefix.startswith("gs://arcadia-scimt-checkpoints/ekfac-dataset-attribution-v1/")
    assert cfg.fit_gate_fallback == ({"sequence_length": 2048, "samples": 512},)
    assert cfg.fit_oom_overrides == {"lambda_module_partitions": 8, "covariance_module_partitions": 8}
    assert drv.DriverConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown DriverConfig keys"):
        drv.DriverConfig.from_mapping({"bogus": 1})
    for bad in (
        {"n_gpus": 1},
        {"fit_gpu": 4},
        {"on_fit_failure": "retry"},
        {"disk_gate": "ignore"},
        {"sweep_datasets": ["coin_noex"], "datasets": ["dolmino", "coin"]},
        {"primary_damping": 0.5},
        {"oracle_repeats": 1},
        {"datasets": ["dolmino", "dolmino"]},
        {"smoke_dataset": "charter_noex", "datasets": ["dolmino", "coin"]},
        {"dampings_all": [0.1, 0.1]},
        {"wall_clock_budget_seconds": 0},
    ):
        with pytest.raises(ValueError):
            drv.DriverConfig.from_mapping(bad)
    assert drv.DriverConfig(n_gpus=2).n_gpus == 2


def test_load_driver_config_reads_json_and_yaml(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"n_gpus": 2, "smoke": False}))
    (tmp_path / "d.yaml").write_text("n_gpus: 3\nwall_clock_budget_seconds: 3600\n")
    assert drv.load_driver_config(tmp_path / "d.json").n_gpus == 2
    yaml_cfg = drv.load_driver_config(tmp_path / "d.yaml")
    assert yaml_cfg.n_gpus == 3 and yaml_cfg.wall_clock_budget_seconds == 3600
    assert drv.load_driver_config(None) == drv.DriverConfig()


def test_main_refuses_flags_config_is_env_only():
    with pytest.raises(SystemExit, match="config-first"):
        drv.main(["--n-gpus", "2"])


# ============================================================= GPU layout
def test_gpu_layout_and_device_strings():
    four = drv.gpu_layout(4)
    assert (four.fit_gpu, four.worker_gpus, four.shard_gpus) == (0, (1, 2, 3), (1, 2, 3))
    assert four.all_gpus == (0, 1, 2, 3) and four.model_device() == "cuda:0"
    assert four.shard_devices() == ["cuda:1", "cuda:2", "cuda:3"]
    two = drv.gpu_layout(2)
    assert two.worker_gpus == (1,) and two.shard_devices() == ["cuda:1"]
    odd = drv.gpu_layout(4, fit_gpu=2)
    assert odd.worker_gpus == (0, 1, 3) and odd.model_device() == "cuda:2" and odd.shard_devices() == ["cuda:0", "cuda:1", "cuda:3"]
    with pytest.raises(ValueError):
        drv.gpu_layout(1)
    assert drv.visible_devices((0, 2)) == "0,2"


def test_assign_queue_six_datasets_over_three_gpus_then_two():
    equal = {d: 600.0 for d in D6}
    assert drv.assign_queue(equal, (1, 2, 3)) == {1: ["dolmino", "coin"], 2: ["charter_worked", "coin_worked"], 3: ["charter_noex", "coin_noex"]}
    assert drv.makespan(equal, (1, 2, 3)) == 1200.0
    # the next dataset starts on whichever GPU frees first
    uneven = {"dolmino": 100.0, "charter_worked": 900.0, "charter_noex": 900.0, "coin": 100.0, "coin_worked": 100.0, "coin_noex": 100.0}
    plan = drv.assign_queue(uneven, (1, 2, 3))
    assert plan[1] == ["dolmino", "coin", "coin_worked", "coin_noex"] and plan[2] == ["charter_worked"] and plan[3] == ["charter_noex"]
    assert drv.makespan(uneven, (1, 2, 3)) == 900.0
    # degrade to 2 GPUs: one worker takes the whole queue in order
    assert drv.assign_queue(equal, (1,)) == {1: list(D6)} and drv.makespan(equal, (1,)) == 3600.0
    with pytest.raises(ValueError):
        drv.assign_queue(equal, ())


# ========================================================= vector budget
def test_max_resident_vectors_matches_the_scorer_budget_table():
    assert drv.max_resident_vectors(3) == 16  # ledger: 3 shard GPUs hold <= 16 bf16 vectors, 18 does not fit
    assert drv.max_resident_vectors(1) == 4
    assert drv.max_resident_vectors(2) == 10
    worst = -(-drv.INCLUDED_NUMEL // 3) + drv.LARGEST_ENTRY_NUMEL
    fits = sr_mod.shard_bytes(worst, 16, resident_itemsize=2, dot_window=1 << 26)["total"]
    breaks = sr_mod.shard_bytes(worst, 17, resident_itemsize=2, dot_window=1 << 26)["total"]
    assert fits <= 130e9 < breaks
    assert drv.max_resident_vectors(3, shard_budget_gb=60) < 16


# ============================================================ pass plan
def test_plan_passes_matches_the_spec_vector_lists():
    cfg = drv.DriverConfig()
    passes = {p.name: p for p in drv.plan_passes(cfg, inverse_available=True)}
    assert list(passes) == ["main", "sweep", "folds", "pt_mismatch", "oracle"]
    main = passes["main"]
    assert main.mode == "main" and main.rows_filter == "all"
    assert main.vectors == tuple(f"{d}__gdp__all" for d in D6) + tuple(f"{d}__inv0.1__all" for d in D6)
    sweep = passes["sweep"]
    assert sweep.rows_filter == 300 and len(sweep.vectors) == 14
    assert sweep.vectors == (
        *(f"{d}__gdpunit__all" for d in D6),
        *(f"{d}__inv0.01__all" for d in D4),
        *(f"{d}__inv1__all" for d in D4),
    )
    folds = passes["folds"]
    assert folds.rows_filter == 300 and folds.vectors == tuple(f"{d}__inv0.1__{f}" for d in D6 for f in ("f0", "f1"))
    assert passes["pt_mismatch"].mode == "pt_mismatch" and passes["pt_mismatch"].rows_filter == 100
    assert passes["pt_mismatch"].vectors == main.vectors and passes["oracle"].vectors == main.vectors
    assert passes["oracle"].mode == "oracle"
    assert all(len(p.vectors) <= drv.max_resident_vectors(3) for p in passes.values())
    # every name parses under the analysis regex <dataset>__<kind>__<fold>
    for p in passes.values():
        for name in p.vectors:
            dataset, kind, fold = name.split("__")
            assert dataset in D6 and kind in {"gdp", "gdpunit", "inv0.01", "inv0.1", "inv1"} and fold in {"all", "f0", "f1"}


def test_plan_passes_without_inverse_substitutes_gdp_and_drops_missing():
    cfg = drv.DriverConfig()
    passes = {p.name: p for p in drv.plan_passes(cfg, inverse_available=False)}
    assert "sweep" not in passes
    assert passes["main"].vectors == tuple(f"{d}__gdp__all" for d in D6) + tuple(f"{d}__gdpunit__all" for d in D6)
    assert passes["folds"].vectors == tuple(f"{d}__gdp__{f}" for d in D6 for f in ("f0", "f1"))
    assert any("inverse unavailable" in n for n in passes["main"].notes)
    # pooled inverse ok but fold inverse failed -> folds pass on raw gdp folds
    mixed = {p.name: p for p in drv.plan_passes(cfg, inverse_available=True, folds_inverse_available=False)}
    assert mixed["folds"].vectors[0] == "dolmino__gdp__f0" and mixed["main"].vectors[-1] == "coin_noex__inv0.1__all"
    # a dataset whose vectors never materialised is dropped with a note
    dropped = {p.name: p for p in drv.plan_passes(cfg, inverse_available=True, available=lambda n: not n.startswith("coin_noex"))}
    assert not any(v.startswith("coin_noex") for v in dropped["main"].vectors) and len(dropped["main"].vectors) == 10
    assert any("missing vectors dropped" in n for n in dropped["main"].notes)


def test_fit_passes_to_budget_chunks_main_mode_and_truncates_diagnostics():
    cfg = drv.DriverConfig()
    passes = drv.plan_passes(cfg, inverse_available=True)
    assert [p.name for p in drv.fit_passes_to_budget(passes, 16)] == ["main", "sweep", "folds", "pt_mismatch", "oracle"]
    fitted = drv.fit_passes_to_budget(passes, 4)
    names = [p.name for p in fitted]
    assert names[:3] == ["main", "main_2", "main_3"] and "sweep_4" in names and names.count("pt_mismatch") == 1 and names.count("oracle") == 1
    assert all(len(p.vectors) <= 4 for p in fitted)
    main_chunks = [p for p in fitted if p.name.startswith("main")]
    assert sum(len(p.vectors) for p in main_chunks) == 12 and all(p.mode == "main" for p in main_chunks)
    pt = next(p for p in fitted if p.name == "pt_mismatch")
    assert pt.vectors == passes[0].vectors[:4] and any("truncated" in n for n in pt.notes)
    with pytest.raises(ValueError):
        drv.fit_passes_to_budget(passes, 0)


def test_priority_order_and_wall_clock_truncation():
    cfg = drv.DriverConfig()
    passes = drv.plan_passes(cfg, inverse_available=True)
    assert [p.name for p in drv.order_by_priority(passes)] == ["main", "pt_mismatch", "oracle", "folds", "sweep"]
    counts = drv.RowCounts(6000, 1500, 1500)
    assert counts.rows_for("all") == 6000 and counts.rows_for(300) == 1200 and counts.rows_for(2000) == 6000

    def seconds_for(p: drv.ScoringPass) -> float:
        return drv.projected_pass_seconds(drv.pass_rows(p, counts, cfg), len(p.vectors), s_per_row=1.0, base_overhead=100.0, stage_seconds_per_vector=10.0)

    assert drv.pass_rows(passes[4], counts, cfg) == 16  # oracle: 8 rows x 2 repeats
    # main 6220 s, pt_mismatch 620 s, oracle 236 s, folds 1420 s, sweep 1440 s
    run, skipped = drv.select_passes_for_deadline(passes, remaining_seconds=6220 + 620 + 236 + 100, seconds_for=seconds_for)
    assert [p.name for p in run] == ["main", "pt_mismatch", "oracle"]
    assert [p.name for p, _ in skipped] == ["folds", "sweep"] and all("wall-clock" in r for _, r in skipped)
    # too little for main but the cheap diagnostics still fit
    run, skipped = drv.select_passes_for_deadline(passes, remaining_seconds=900, seconds_for=seconds_for)
    assert [p.name for p in run] == ["pt_mismatch", "oracle"]
    assert drv.episodes_for_budget(3600.0, 3.0) == 300 and drv.episodes_for_budget(10.0, 3.0) == 100
    with pytest.raises(ValueError):
        drv.episodes_for_budget(10.0, 0.0)


# ============================================================== disk plan
def test_disk_plan_projects_the_true_peak_well_above_the_bootstrap_floor():
    plan = drv.disk_plan(drv.DriverConfig())
    assert plan["vector_gb"] == pytest.approx(43.04, abs=0.01)
    assert 2400 < plan["peak_gb"] < 2700  # ~2.6 TB: 6 datasets x (gdp+gdpunit) x 3 folds + 30 inverse vectors at 43 GB
    assert plan["peak_gb"] > 800  # the bootstrap's 800 GB floor is the fit's own minimum, not the run's
    no_prune = drv.disk_plan(drv.DriverConfig(prune_unit_fold_vectors=False, prune_gdp_fold_vectors_after_apply=False, delete_kronfluence_intermediates=False))
    assert no_prune["peak_gb"] > plan["peak_gb"] + 500
    assert drv.disk_plan(drv.DriverConfig(unit_accumulator=False))["peak_gb"] < plan["peak_gb"]


# ============================================================ fit retries
def test_fit_retry_decision_honours_97_and_98():
    cfg = drv.DriverConfig()
    gate = drv.fit_retry_decision(97, attempt=1, cfg=cfg, receipt=None, gate_fallbacks_used=0, oom_retries_used=0)
    assert gate.retry and gate.overrides == {"sequence_length": 2048, "samples": 512} and gate.clear_factor_dir
    assert not drv.fit_retry_decision(97, attempt=2, cfg=cfg, receipt=None, gate_fallbacks_used=1, oom_retries_used=0).retry
    lam = drv.fit_retry_decision(98, attempt=1, cfg=cfg, receipt={"failure": {"phase": "lambda partition 0"}}, gate_fallbacks_used=0, oom_retries_used=0)
    assert lam.retry and lam.overrides == {"lambda_module_partitions": 8, "covariance_module_partitions": 8, "resume_from_eigendecomposition": True}
    assert not lam.clear_factor_dir
    cov = drv.fit_retry_decision(98, attempt=1, cfg=cfg, receipt={"failure": {"phase": "covariance partition 2"}}, gate_fallbacks_used=0, oom_retries_used=0)
    assert cov.retry and cov.clear_factor_dir and "resume_from_eigendecomposition" not in cov.overrides
    assert not drv.fit_retry_decision(98, attempt=2, cfg=cfg, receipt=None, gate_fallbacks_used=0, oom_retries_used=1).retry
    assert not drv.fit_retry_decision(1, attempt=1, cfg=cfg, receipt=None, gate_fallbacks_used=0, oom_retries_used=0).retry
    assert not drv.fit_retry_decision(98, attempt=1, cfg=drv.DriverConfig(fit_oom_retry=False), receipt=None, gate_fallbacks_used=0, oom_retries_used=0).retry
    # the fallback overrides are valid FitConfig overrides
    fit_mod.FitConfig.from_mapping({**gate.overrides})
    fit_mod.FitConfig.from_mapping({**lam.overrides})


# =============================================================== receipts
def test_receipt_and_done_schemas():
    body = drv.make_receipt("r1", "fit", "ok", seconds=1.0)
    drv.validate_receipt(body)
    assert set(drv.RECEIPT_REQUIRED_KEYS) <= set(body)
    with pytest.raises(ValueError, match="not in"):
        drv.make_receipt("r1", "fit", "done")
    with pytest.raises(ValueError, match="missing keys"):
        drv.validate_receipt(body, job=True)
    job_body = drv.make_receipt("r1", "fit", "ok", **dict.fromkeys(drv.JOB_RECEIPT_KEYS))
    drv.validate_receipt(job_body, job=True)
    with pytest.raises(ValueError, match="DRIVER_DONE missing"):
        drv.validate_done({"run_id": "r1", "status": "complete"})
    done = dict.fromkeys(drv.DONE_REQUIRED_KEYS)
    done["status"] = "weird"
    with pytest.raises(ValueError, match="status"):
        drv.validate_done(done)
    assert drv.JobResult(97, 1.0, None, []).status == "gate-failed"
    assert drv.JobResult(98, 1.0, None, []).status == "oom"
    assert drv.JobResult(99, 1.0, None, []).status == "oracle-failed"
    assert drv.JobResult(1, 1.0, None, [], timed_out=True).status == "timeout"
    assert drv.JobResult(None, 1.0, None, []).status == "failed"


# ====================================================== config rendering
def test_rendered_configs_are_accepted_by_the_four_scripts(tmp_path):
    cfg = drv.DriverConfig(attribution_root=str(tmp_path / "attr"), repo_root=str(REPO_ROOT))
    paths = drv.Paths.from_config(cfg)
    layout = drv.gpu_layout(4)
    fit = drv.render_fit_overrides(cfg, paths, evidence_dir=paths.evidence / "fit" / "attempt1", max_projected_seconds=3600.0, extra={"lambda_module_partitions": 8})
    fit_cfg = fit_mod.FitConfig.from_mapping(fit)
    assert fit_cfg.calibration_jsonl == str(paths.sample("dolmino_fit")) and fit_cfg.output_dir == str(paths.factors)
    assert fit_cfg.evidence_dir.endswith("fit/attempt1") and fit_cfg.lambda_module_partitions == 8 and fit_cfg.sequence_length == 4096
    assert Path(fit_cfg.evidence_dir).resolve() != Path(fit_cfg.output_dir).resolve()  # evidence outside the factor dir
    mg = drv.render_mean_gradients_config(cfg, paths, "coin", out_dir=paths.vectors, pt_snapshot="/snap/pt/295efb63aaaa", max_rows=8)
    parsed = mg_mod.MeanGradientsConfig.from_mapping(mg_mod.merge_mappings(mg_mod.DEFAULTS, mg))
    assert parsed.dataset == "coin" and parsed.device == "cuda:0" and parsed.max_rows == 8 and parsed.model.local_path == "/snap/pt/295efb63aaaa"
    assert parsed.sample_path == str(paths.sample("coin")) and parsed.model.expected_sha_prefix == drv.PT_REVISION[:7]
    apply = drv.render_apply_config(paths, inputs=[str(paths.vector("coin__gdp__all"))], dampings=(0.01, 0.1, 1.0), evidence_dir=paths.evidence / "apply" / "all", oracle_modules=2, skip_oracle=True)
    parsed_apply = apply_mod.ApplyConfig.from_mapping(apply)
    assert parsed_apply.factors_dir == str(paths.factors) and parsed_apply.out_dir == str(paths.vectors) and parsed_apply.damping_scales == (0.01, 0.1, 1.0)
    oracle = drv.render_apply_config(paths, inputs=[], dampings=(0.1,), evidence_dir=paths.evidence / "apply" / "oracle", oracle_modules=2, oracle_only=True)
    assert apply_mod.ApplyConfig.from_mapping(oracle).oracle_only
    entry = drv.plan_passes(cfg, inverse_available=True)[0]
    score = drv.render_score_config(cfg, paths, entry, out_dir=paths.scores_root, layout=layout, vectors_dir=paths.vectors, it_snapshot="/snap/it/96b6f1ecbbbb", pt_snapshot="/snap/pt/295efb63aaaa")
    parsed_score = sr_mod.ScoreConfig.from_mapping(mg_mod.merge_mappings(sr_mod.DEFAULTS, score))
    assert parsed_score.device == "cuda:0" and parsed_score.shard_devices == ("cuda:1", "cuda:2", "cuda:3")
    assert parsed_score.model.local_path == "/snap/it/96b6f1ecbbbb" and parsed_score.tokenizer.hf_id == drv.IT_MODEL_ID and parsed_score.pt_model.local_path == "/snap/pt/295efb63aaaa"
    assert [p.name for p in parsed_score.passes] == ["main"] and len(parsed_score.passes[0].vectors) == 12
    assert all(v.startswith(str(paths.vectors)) and v.endswith(".f32") for v in parsed_score.passes[0].vectors)
    pt_entry = drv.plan_passes(cfg, inverse_available=True)[3]
    pt_score = sr_mod.ScoreConfig.from_mapping(mg_mod.merge_mappings(sr_mod.DEFAULTS, drv.render_score_config(cfg, paths, pt_entry, out_dir=paths.scores_root, layout=layout, vectors_dir=paths.vectors, it_snapshot=None, pt_snapshot=None)))
    assert pt_score.mode == "pt_mismatch" and pt_score.passes[0].name == "pt_mismatch" and pt_score.passes[0].rows_filter == 100
    two = drv.gpu_layout(2)
    assert drv.render_score_config(cfg, paths, entry, out_dir=paths.scores_root, layout=two, vectors_dir=paths.vectors, it_snapshot=None, pt_snapshot=None)["shard_devices"] == ["cuda:1"]


# ================================================================ staging
def test_stage_for_upload_never_copies_vectors_weights_or_pools(tmp_path):
    cfg = drv.DriverConfig(attribution_root=str(tmp_path / "attr"), repo_root=str(REPO_ROOT))
    paths = drv.Paths.from_config(cfg)
    (paths.evidence / "fit" / "attempt1").mkdir(parents=True)
    (paths.evidence / "fit" / "attempt1" / "fit_factors_pt.json").write_text("{}")
    (paths.evidence / "driver.log").write_text("log")
    _write_vector(paths.vectors, "coin__gdp__all", {"name": "coin__gdp__all"})
    (paths.vectors / "evidence").mkdir()
    (paths.vectors / "evidence" / "mean_gradients__coin__x.json").write_text("{}")
    (paths.scores).mkdir(parents=True)
    (paths.scores / "main.jsonl").write_text("{}\n")
    (paths.factors / "kronfluence").mkdir(parents=True)
    (paths.factors / "kronfluence" / "cov.safetensors").write_bytes(b"\0")
    (paths.factors / "ekfac_meta.json").write_text("{}")
    (paths.factors / "linear").mkdir()
    (paths.factors / "linear" / "U.npy").write_bytes(b"\0")
    (paths.datasets / "dolmino").mkdir(parents=True)
    (paths.datasets / "dolmino" / "sample.jsonl").write_text("{}\n")
    (paths.datasets / "dolmino" / "pool.jsonl").write_text("{}\n")
    (paths.datasets / ".hub_cache").mkdir()
    (paths.datasets / ".hub_cache" / "corpus.jsonl").write_text("x")
    (paths.datasets / "manifest.json").write_text("{}")
    (paths.results).mkdir()
    (paths.results / "headline.pdf").write_bytes(b"%PDF")
    (paths.smoke / "vectors").mkdir(parents=True)
    _write_vector(paths.smoke / "vectors", "dolmino__gdp__all", {"name": "dolmino__gdp__all"})
    manifest = drv.stage_for_upload(paths, tmp_path / "staging")
    staged = {entry["path"] for entry in manifest["files"]}
    assert "evidence/fit/attempt1/fit_factors_pt.json" in staged and "evidence/driver.log" in staged
    assert "vectors/sidecars/coin__gdp__all.json" in staged and "vectors/evidence/mean_gradients__coin__x.json" in staged
    assert "eft_scores/scores/main.jsonl" in staged and "results/headline.pdf" in staged
    assert "ekfac_pt/ekfac_meta.json" in staged and "datasets/manifest.json" in staged and "datasets/dolmino/sample.jsonl" in staged
    assert "smoke/vectors/dolmino__gdp__all.json" in staged
    forbidden = [p for p in staged if p.endswith((".f32", ".npy", ".safetensors")) or "pool.jsonl" in p or ".hub_cache" in p or "kronfluence" in p]
    assert forbidden == []
    assert manifest["n_files"] == len(staged) and manifest["bytes"] > 0


# ======================================================= end-to-end fakes
def test_end_to_end_happy_path_runs_the_whole_schedule(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    done = h.run(monkeypatch)
    drv.validate_done(done)
    assert done["status"] == "complete" and not done["deadline_hit"] and done["skipped"] == [] and done["failures"] == []
    assert done["phases"] == {
        "phase0_inputs": "ok", "smoke": "ok", "fit": "ok", "mean_gradients": "ok", "inverse": "ok", "scoring": "ok", "analysis": "ok", "publish": "ok",
    }
    assert set(done["gates"]) == {"A", "B", "C", "D", "E"} and all(g["passed"] for g in done["gates"].values())
    assert (h.root / "evidence" / drv.DONE_FILE).is_file() and not (h.root / "evidence" / drv.FAILURE_FILE).exists()
    # Gate A auto-grew the calibration sample: 512 docs x 1300 tokens pack into ~162 rows < 269
    assert [c["n_dolmino_fit"] for c in h.build_datasets.calls] == [512, 1024]
    assert h.receipt("phase0_inputs")["n_dolmino_fit_used"] == 1024 and any("growing n_dolmino_fit" in n for n in done["notes"])
    # job order: smoke before any long phase; fit and mean gradients interleave; inverse; scoring; analysis
    names = [j.name for j in h.runner.jobs]
    assert names[:3] == ["preflight", "smoke_mean_gradients", "smoke_score_eft_rows"]
    assert names.index("fit_attempt1") < names.index("apply_oracle") < names.index("apply_all") < names.index("apply_folds") < names.index("score__main")
    assert [n for n in names if n.startswith("score__")] == ["score__main", "score__pt_mismatch", "score__oracle", "score__folds", "score__sweep"]
    assert names[-1] == "analysis"
    # GPU isolation: fit alone on GPU 0, each mean-gradient job on exactly one worker GPU, the scorer sees all four
    fit_job = h.jobs("fit_attempt1")[0]
    assert fit_job.env["CUDA_VISIBLE_DEVICES"] == "0" and fit_job.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" and fit_job.env["HF_HUB_OFFLINE"] == "1"
    mg_jobs = h.jobs("mean_gradients__")
    assert sorted(json.loads(Path(j.env[mg_mod.CONFIG_ENV]).read_text())["dataset"] for j in mg_jobs) == sorted(D6)
    assert {j.env["CUDA_VISIBLE_DEVICES"] for j in mg_jobs} == {"1", "2", "3"} and all(j.env[mg_mod.CONFIG_ENV].endswith(".json") for j in mg_jobs)
    assert Counter(j.env["CUDA_VISIBLE_DEVICES"] for j in mg_jobs) == {"1": 2, "2": 2, "3": 2}
    score_main = h.jobs("score__main")[0]
    assert score_main.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    main_cfg = json.loads(Path(score_main.env[sr_mod.CONFIG_ENV]).read_text())
    assert main_cfg["shard_devices"] == ["cuda:1", "cuda:2", "cuda:3"] and main_cfg["device"] == "cuda:0" and len(main_cfg["passes"][0]["vectors"]) == 12
    assert main_cfg["model"]["local_path"].endswith(drv.IT_REVISION) and main_cfg["pt_model"]["local_path"].endswith(drv.PT_REVISION)
    for job in h.jobs("score__"):
        assert len(json.loads(Path(job.env[sr_mod.CONFIG_ENV]).read_text())["passes"][0]["vectors"]) <= 16
    sweep_cfg = json.loads(Path(h.jobs("score__sweep")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert len(sweep_cfg["passes"][0]["vectors"]) == 14 and sweep_cfg["passes"][0]["rows_filter"] == 300
    oracle_cfg = json.loads(Path(h.jobs("score__oracle")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert oracle_cfg["mode"] == "oracle" and oracle_cfg["oracle_rows"] == 8 and oracle_cfg["oracle_repeats"] == 2
    # phase 2: kronfluence intermediates gone, inverse vectors present, folds pruned after the apply, unit folds pruned
    assert not (h.root / "ekfac_pt" / "kronfluence").exists() and (h.root / "ekfac_pt" / "ekfac_meta.json").is_file()
    assert h.receipt("kronfluence_evicted")["status"] == "ok"
    vectors = h.root / "vectors"
    assert (vectors / "coin__inv0.1__all.f32").is_file() and (vectors / "coin__inv1__all.f32").is_file() and (vectors / "coin__inv0.1__f1.f32").is_file()
    assert not (vectors / "coin__gdpunit__f0.f32").exists() and not (vectors / "coin__gdp__f0.f32").exists()
    assert (vectors / "coin__gdp__all.f32").is_file() and (vectors / "coin__gdpunit__all.f32").is_file()
    apply_all = json.loads(Path(h.jobs("apply_all")[0].argv[-1]).read_text())
    assert apply_all["damping_scales"] == [0.01, 0.1, 1.0] and len(apply_all["inputs"]) == 6 and apply_all["skip_oracle"]
    apply_folds = json.loads(Path(h.jobs("apply_folds")[0].argv[-1]).read_text())
    assert apply_folds["damping_scales"] == [0.1] and len(apply_folds["inputs"]) == 12
    assert json.loads(Path(h.jobs("apply_oracle")[0].argv[-1]).read_text())["oracle_only"]
    # gates C/D come from the fit's own partition-0 receipts; E from fold cosines + the main pass Gram
    assert done["gates"]["C"]["attempt"] == 1 and done["gates"]["D"]["attempt"] == 1
    assert set(done["gates"]["E"]["fold_cosines"]) == set(D6) and done["gates"]["E"]["cross_cosines"]
    assert len(h.cosines) == 6
    # receipts: every job has one with the job schema; the gates and phases have theirs
    for name in ("preflight", "smoke_mean_gradients", "fit_attempt1", "apply_oracle", "score__main", "analysis", *[f"mean_gradients__{d}" for d in D6]):
        drv.validate_receipt(h.receipt(name), job=True)
    for name in ("phase0_inputs", "smoke", "fit", "mean_gradients", "inverse", "scoring", "gate_a", "gate_b", "publication"):
        drv.validate_receipt(h.receipt(name))
    assert h.receipt("fit")["attempts"][0]["exit_code"] == 0
    # publication: staged evidence uploaded under runs/<run_id>, no vector bytes staged
    assert h.uploads == [(str(h.root / "staging" / done["run_id"]), drv.DriverConfig().hf_repo, f"runs/{done['run_id']}")]
    staged = list((h.root / "staging" / done["run_id"]).rglob("*"))
    assert not any(p.suffix in (".f32", ".npy", ".safetensors") for p in staged)
    assert (h.root / "staging" / done["run_id"] / "results" / "SUMMARY.md").is_file()
    assert (h.root / "staging" / done["run_id"] / "evidence" / drv.BOOTSTRAP_FILE).is_file()
    assert done["publication"]["status"] == "ok" and done["publication"]["path_in_repo"] == f"runs/{done['run_id']}"
    assert (h.root / "analysis_input" / "scores").is_symlink() and (h.root / "analysis_input" / "eft_rows.jsonl").is_symlink()
    assert done["measurements"]["resident_vector_budget"] == 16 and done["measurements"]["mean_gradients_row_seconds"] == 3.0


def test_end_to_end_degrades_to_two_gpus(tmp_path, monkeypatch):
    h = Harness(tmp_path, cfg={"n_gpus": 2}, runner_kwargs={"preflight_devices": 2})
    done = h.run(monkeypatch)
    assert done["status"] == "complete" and done["layout"] == {"fit_gpu": 0, "worker_gpus": [1]}
    assert {j.env["CUDA_VISIBLE_DEVICES"] for j in h.jobs("mean_gradients__")} == {"1"} and len(h.jobs("mean_gradients__")) == 6
    assert done["measurements"]["resident_vector_budget"] == 4
    score_names = [j.name for j in h.jobs("score__")]
    assert score_names[:3] == ["score__main", "score__main_2", "score__main_3"]
    for job in h.jobs("score__"):
        cfg = json.loads(Path(job.env[sr_mod.CONFIG_ENV]).read_text())
        assert cfg["shard_devices"] == ["cuda:1"] and len(cfg["passes"][0]["vectors"]) <= 4 and job.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    pt_cfg = json.loads(Path(h.jobs("score__pt_mismatch")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert pt_cfg["passes"][0]["name"] == "pt_mismatch" and len(pt_cfg["passes"][0]["vectors"]) == 4
    assert any("truncated" in n for n in h.receipt("score__pt_mismatch")["notes"])


def test_preflight_with_fewer_visible_gpus_degrades_the_layout(tmp_path, monkeypatch):
    h = Harness(tmp_path, cfg={"n_gpus": 4}, runner_kwargs={"preflight_devices": 2})
    done = h.run(monkeypatch)
    assert done["layout"]["worker_gpus"] == [1] and any("degrading the layout" in n for n in done["notes"])
    assert h.jobs("score__main")[0].env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_fit_oom_retries_once_with_more_partitions_and_salvages_the_eigh(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"fit_exit_codes": (98, 0), "fit_oom_phase": "lambda partition 0"})
    marker = h.root / "ekfac_pt" / "kronfluence" / "eigen.marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep me")  # would be lost if the driver cleared the factor dir
    done = h.run(monkeypatch)
    fits = h.jobs("fit_attempt")
    assert [j.name for j in fits] == ["fit_attempt1", "fit_attempt2"]
    second = json.loads(Path(fits[1].argv[-1]).read_text())
    assert second["lambda_module_partitions"] == 8 and second["covariance_module_partitions"] == 8 and second["resume_from_eigendecomposition"] is True
    assert second["evidence_dir"].endswith("fit/attempt2") and json.loads(Path(fits[0].argv[-1]).read_text())["evidence_dir"].endswith("fit/attempt1")
    assert (h.root / "evidence" / "fit" / "attempt1" / fit_mod.RECEIPT_FILE).is_file()  # first attempt's evidence preserved
    assert h.receipt("fit_attempt1")["status"] == "oom" and h.receipt("fit_attempt2")["status"] == "ok"
    assert done["fit_ok"] and done["inverse_ok"] and done["status"] == "complete"
    assert any("eigendecomposition salvaged" in n for n in done["notes"])
    assert not marker.exists()  # kronfluence intermediates evicted AFTER the successful fit, not before the retry


def test_fit_gate_fail_applies_the_time_valve_and_clears_the_factor_dir(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"fit_exit_codes": (97, 0)})
    stale = h.root / "ekfac_pt" / "kronfluence" / "stale_partition0.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"\0")
    done = h.run(monkeypatch)
    fits = h.jobs("fit_attempt")
    assert len(fits) == 2
    second = json.loads(Path(fits[1].argv[-1]).read_text())
    assert second["sequence_length"] == 2048 and second["samples"] == 512
    assert done["gates"]["C"]["passed"] is True and done["gates"]["C"]["attempt"] == 2  # attempt 1's C gate failed, attempt 2 passed
    assert h.receipt("gate_c")["attempt"] == 2
    assert done["status"] == "complete" and any("time valve" in n for n in done["notes"])


def test_fit_failure_continues_gdp_only_or_aborts_per_config(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"fit_exit_codes": (97, 97)})
    done = h.run(monkeypatch)
    assert len(h.jobs("fit_attempt")) == 2 and done["fit_ok"] is False and done["inverse_ok"] is False
    assert done["status"] == "partial" and h.jobs("apply_") == []
    assert h.receipt("inverse")["status"] == "skipped"
    main_cfg = json.loads(Path(h.jobs("score__main")[0].env[sr_mod.CONFIG_ENV]).read_text())
    names = [Path(v).stem for v in main_cfg["passes"][0]["vectors"]]
    assert names == [f"{d}__gdp__all" for d in D6] + [f"{d}__gdpunit__all" for d in D6]
    assert [j.name for j in h.jobs("score__")] == ["score__main", "score__pt_mismatch", "score__oracle", "score__folds"]  # no sweep without inverse
    assert any("gdp/gdpunit vectors only" in n for n in done["notes"])
    assert done["publication"]["status"] == "ok"

    h2 = Harness(tmp_path / "abort", cfg={"on_fit_failure": "abort"}, runner_kwargs={"fit_exit_codes": (97, 97)})
    done2 = h2.run(monkeypatch)
    assert done2["status"] == "failed" and (h2.root / "evidence" / drv.FAILURE_FILE).is_file()
    assert h2.jobs("score__") == [] and h2.uploads  # evidence still published
    assert len(h2.jobs("mean_gradients__")) == 6  # the parallel branch was allowed to finish


def test_wall_clock_budget_truncates_scoring_by_priority(tmp_path, monkeypatch):
    # Fake clock: preflight 10 + smokes 600 + 900, fit 3600, six mean-gradient jobs 6 x 600, three applies 3 x 300
    # -> 9,610 s elapsed when scoring starts; each scoring job then costs 900 s. Projections use the smoke's
    # 0.1 s/row and 6 s/vector staging: main 852 s, pt_mismatch 292 s, oracle 254 s, folds 372 s, sweep 384 s.
    # Budget for work = 9,610 + 3 x 900 + 300 slack: main, pt_mismatch and the cheap oracle fit; folds/sweep do not.
    work = 10 + 600 + 900 + 3600 + 6 * 600 + 3 * 300 + 3 * 900 + 300
    reserve = 40 * 60
    h = Harness(
        tmp_path,
        cfg={"wall_clock_budget_seconds": work + reserve, "phase4_reserve_seconds": reserve, "apply_expected_seconds": 300},
        runner_kwargs={"score_row_seconds": 0.1},
    )
    done = h.run(monkeypatch)
    ran = [j.name for j in h.jobs("score__")]
    assert ran == ["score__main", "score__pt_mismatch", "score__oracle"]
    assert "score__sweep" not in ran and "score__folds" not in ran
    assert done["deadline_hit"] and done["status"] == "partial"
    reasons = {s["phase"]: s["reason"] for s in done["skipped"]}
    assert "score__folds" in reasons and "score__sweep" in reasons and all("wall-clock" in r for r in reasons.values())
    assert h.receipt("score__folds")["status"] == "skipped"
    assert done["phases"]["analysis"] == "ok" and done["publication"]["status"] == "ok"  # phase 4 always runs


def test_gate_b_cuts_the_main_pass_when_scoring_is_too_slow(tmp_path, monkeypatch):
    class SlowScorer(FakeRunner):
        def _score(self, job):
            result = super()._score(job)
            if job.name == "smoke_score_eft_rows":
                receipts = sorted((Path(json.loads(Path(job.env[sr_mod.CONFIG_ENV]).read_text())["out_dir"]) / "evidence").glob("*.json"))
                body = json.loads(receipts[-1].read_text())
                body["row_seconds_median"] = 4.0  # 6000 rows -> 6.7 h > 2.5 h budget
                receipts[-1].write_text(json.dumps(body))
            return result

    h = Harness(tmp_path)
    h.runner = SlowScorer(h.clock)
    h.deps.run_job = h.runner
    done = h.run(monkeypatch)
    cut = done["main_rows_filter"]
    assert isinstance(cut, int) and 100 <= cut < 1500 and cut % 50 == 0
    main_cfg = json.loads(Path(h.jobs("score__main")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert main_cfg["passes"][0]["rows_filter"] == cut
    pt_cfg = json.loads(Path(h.jobs("score__pt_mismatch")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert pt_cfg["passes"][0]["rows_filter"] == 100  # only main is cut
    assert any("rows_filter ->" in n for n in done["notes"]) and done["gates"]["B"]["timeline"]["main_scoring_seconds"] < 2.5 * 3600 + 1


def test_smoke_failure_aborts_before_any_long_phase(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"smoke_mg_exit": 1})
    done = h.run(monkeypatch)
    assert done["status"] == "failed" and (h.root / "evidence" / drv.FAILURE_FILE).read_text().count("GateFailure")
    assert h.jobs("fit_attempt") == [] and h.jobs("mean_gradients__") == [] and h.jobs("score__") == []
    assert h.receipt("smoke_mean_gradients")["status"] == "failed"
    assert done["publication"]["status"] == "ok" and h.uploads  # evidence still uploaded


def test_oracle_failure_skips_the_inverse_and_scores_gdp_only(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"oracle_exit": 99})
    done = h.run(monkeypatch)
    assert h.receipt("apply_oracle")["status"] == "oracle-failed" and h.jobs("apply_all") == []
    assert done["inverse_ok"] is False and done["status"] == "partial"
    assert any("oracle_check" in f["reason"] for f in done["failures"])
    main_cfg = json.loads(Path(h.jobs("score__main")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert all("__inv" not in Path(v).stem for v in main_cfg["passes"][0]["vectors"])


def test_mean_gradients_failure_is_retried_on_a_free_gpu_then_dropped(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"mg_failures": {"coin_noex": 1, "charter_noex": 5}})
    done = h.run(monkeypatch)
    coin_noex = [j for j in h.jobs("mean_gradients__coin_noex")]
    assert [j.name for j in coin_noex] == ["mean_gradients__coin_noex", "mean_gradients__coin_noex__retry2"]
    assert len(h.jobs("mean_gradients__charter_noex")) == 2  # one retry only
    assert set(done["datasets_ok"]) == set(D6) - {"charter_noex"} and done["status"] == "partial"
    assert h.receipt("mean_gradients")["status"] == "partial"
    main_cfg = json.loads(Path(h.jobs("score__main")[0].env[sr_mod.CONFIG_ENV]).read_text())
    assert not any("charter_noex" in v for v in main_cfg["passes"][0]["vectors"]) and len(main_cfg["passes"][0]["vectors"]) == 10
    apply_all = json.loads(Path(h.jobs("apply_all")[0].argv[-1]).read_text())
    assert len(apply_all["inputs"]) == 5


def test_scorer_failures_stop_after_two_in_a_row_but_phase4_runs(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"scorer_exits": {"main": 1, "pt_mismatch": 1, "oracle": 1}})
    done = h.run(monkeypatch)
    assert [j.name for j in h.jobs("score__")] == ["score__main", "score__pt_mismatch"]
    assert done["phases"]["scoring"] == "failed" and any("consecutive scorer failures" in n for n in done["notes"])
    assert h.receipt("analysis")["status"] == "skipped"  # no main-mode scores to analyse
    assert done["publication"]["status"] == "ok"


def test_fatal_error_in_phase0_writes_failure_file_and_still_publishes(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("hub down")

    h = Harness(tmp_path, build_datasets=boom)
    done = h.run(monkeypatch)
    assert done["status"] == "failed" and "hub down" in (h.root / "evidence" / drv.FAILURE_FILE).read_text()
    assert done["failures"][0]["phase"] == "driver" and h.runner.jobs == []
    assert h.uploads and done["publication"]["status"] == "ok"
    assert (h.root / "staging" / done["run_id"] / "evidence" / drv.FAILURE_FILE).is_file()


def test_gate_a_refuses_when_the_disk_cannot_hold_the_plan_or_bootstrap_is_missing(tmp_path, monkeypatch):
    h = Harness(tmp_path, free_disk_gb=1500.0)
    done = h.run(monkeypatch)
    assert done["status"] == "failed" and done["gates"]["A"]["passed"] is False
    assert any("projected peak" in r for r in done["gates"]["A"]["reasons"]) and h.jobs("fit_attempt") == []
    warn = Harness(tmp_path / "warn", cfg={"disk_gate": "warn"}, free_disk_gb=1500.0)
    assert warn.run(monkeypatch)["status"] == "complete" and any("disk gate WARN" in n for n in warn.done()["notes"])
    missing = Harness(tmp_path / "nobootstrap", bootstrap=False)
    assert missing.run(monkeypatch)["status"] == "failed" and "bootstrap.sh" in (missing.root / "evidence" / drv.FAILURE_FILE).read_text()


def test_resume_skips_completed_phases_and_keeps_the_run_id(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    first = h.run(monkeypatch)
    n_jobs = len(h.runner.jobs)
    h.runner.jobs.clear()
    second = h.run(monkeypatch)
    assert second["run_id"] == first["run_id"] and second["status"] == "complete"
    rerun = [j.name for j in h.runner.jobs]
    assert rerun == ["analysis"]  # everything else resumed from receipts / on-disk outputs
    assert n_jobs > 10
    assert second["phases"]["fit"] == "ok" and second["phases"]["scoring"] == "ok" and second["elapsed_seconds"] > first["elapsed_seconds"]


def test_two_step_smoke_then_full_run_resumes_the_smoke(tmp_path, monkeypatch):
    smoke = Harness(tmp_path, cfg={"stop_after_smoke": True})
    first = smoke.run(monkeypatch)
    assert first["status"] == "complete" and [j.name for j in smoke.runner.jobs] == ["preflight", "smoke_mean_gradients", "smoke_score_eft_rows"]
    assert {s["phase"] for s in first["skipped"]} == {"fit", "mean_gradients", "inverse", "scoring", "analysis"}
    assert all(s["deliberate"] for s in first["skipped"])  # a smoke-only run is 'complete', not 'partial'
    assert smoke.uploads and "B" in first["gates"] and smoke.receipt("analysis")["status"] == "skipped"
    full = Harness(tmp_path)  # same attribution root, default config: resumes phase 0 + smoke, runs the long phases
    second = full.run(monkeypatch)
    assert second["run_id"] == first["run_id"] and second["status"] == "complete"
    names = [j.name for j in full.runner.jobs]
    assert "preflight" not in names and "smoke_mean_gradients" not in names and names[0] == "fit_attempt1"
    assert second["measurements"]["mean_gradients_row_seconds"] == 3.0  # smoke measurements restored from the receipt
    assert full.build_datasets.calls == []  # phase 0 resumed too


def test_upload_is_skipped_without_a_token_and_gcs_notes_missing_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    h = Harness(tmp_path, cfg={"gcs_push": True})
    done = asyncio.run(drv.run_driver(h.cfg, h.deps))
    assert done["publication"]["status"] == "skipped" and done["publication"]["reason"] == "HF_TOKEN not set"
    assert done["publication"]["gcs"]["status"] == "skipped" and h.uploads == []
    assert done["status"] == "complete"  # a skipped upload is not a lost phase
    assert (h.root / "staging" / done["run_id"] / "staging_manifest.json").is_file()
