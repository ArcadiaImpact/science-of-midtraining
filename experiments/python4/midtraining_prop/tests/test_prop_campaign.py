"""CPU contracts for the proportional-dose campaign (no GPU/torch/network)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402
from experiments.python4.midtraining_12b import run as driver  # noqa: E402
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402
from experiments.python4.midtraining_12b.pod import sample  # noqa: E402
from experiments.python4.midtraining_prop import run_prop  # noqa: E402
from experiments.python4.midtraining_prop.pod import chain_prop  # noqa: E402

PUSHED_OID = "582a1a2fc3004b35e574316f61ef6e965385fb39"


@pytest.fixture(autouse=True)
def _restore_shared_modules():
    """The overlays mutate the shared 12B modules; keep tests hermetic."""
    saved = [
        (module, name, getattr(module, name))
        for module, names in (
            (chain, (
                "PYTHON4_REVISION", "PYTHON4_FILE", "PYTHON4_ROWS",
                "PYTHON4_SHA256", "expected_artifact_provenance",
                "verify_corpus_file", "publication_paths",
                "TOKENIZER", "MODEL_REVISION", "HF_MODEL_REPO",
                "MIN_MODEL_WEIGHT_BYTES", "CONFIG_DIR", "build_run_manifest",
            )),
            (sample, ("MODEL_REPO", "BASE_MODEL", "BASE_REVISION")),
            (driver, (
                "LOGS_REPO", "TRAIN_POD", "EVAL_POD", "TRAIN_LADDER",
                "TRAIN_ENTRYPOINT", "SAMPLE_ENTRYPOINT",
                "_verify_stage_renders", "pod_environment", "_train_setup",
            )),
        )
        for name in names
    ]
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


# --------------------------------------------------------------- corpus pins
def test_corpus_pins_are_the_pushed_revision():
    assert chain_prop.PYTHON4_REVISION_PROP == PUSHED_OID
    assert chain_prop.PYTHON4_REVISION_PROP != chain.PYTHON4_REVISION
    spec12 = chain_prop.scale_spec("12b")
    spec27 = chain_prop.scale_spec("27b")
    assert spec12.filename == "corpus_prop_12b.jsonl"
    assert spec27.filename == "corpus_prop_27b.jsonl"
    assert spec12.rows == 4_261
    assert spec27.rows == 9_595
    assert spec12.sha256 == (
        "958793e99494571adb7a197c71a5e9fa7d3783d424cb72539f220d9e11e6a6c5"
    )
    assert spec27.sha256 == (
        "ab96f50ae3a0b4a2a1d87d48db3ad325052c421b603c0460ead1c007af27baa7"
    )
    assert spec12.realized_tokens == 5_397_107
    assert spec27.realized_tokens == 12_142_054


def test_targets_are_the_pre_registered_ratios():
    anchor = chain_prop.ANCHOR_TOKENS_110B
    assert anchor == 49_465_523
    # nearest-integer of anchor x scale/110 (integer round-half-up form)
    assert chain_prop.scale_spec("12b").target_tokens == (
        (anchor * 12 + 55) // 110
    ) == 5_396_239
    assert chain_prop.scale_spec("27b").target_tokens == (
        (anchor * 27 + 55) // 110
    ) == 12_141_537
    for scale in ("12b", "27b"):
        spec = chain_prop.scale_spec(scale)
        # crossing doc included: realized >= target, overshoot < one long doc
        assert spec.target_tokens <= spec.realized_tokens < spec.target_tokens + 20_000
    # nesting: the 12B prefix is strictly inside the 27B prefix
    assert chain_prop.scale_spec("12b").rows < chain_prop.scale_spec("27b").rows


def test_arm_is_fresh_namespace():
    assert chain_prop.ARM == "mixed_4ep_prop"
    assert set(chain_prop.EXISTING_ARMS) == {
        "control", "experimental", "dose_1ep_70m", "sdf_ordered",
        "sdf_ordered_1ep",
    }
    assert chain_prop.ARM not in chain_prop.EXISTING_ARMS
    assert chain_prop.publication_paths() == (
        "mixed_4ep_prop/midtrain/end", "mixed_4ep_prop/sft/end",
    )


def test_pins_apply_at_runtime_not_import():
    # importing chain_prop must NOT mutate the shared chain module
    assert chain.PYTHON4_REVISION == "dd6e3370185381ec2ed4b0126ea76f63c406145d"
    assert chain.PYTHON4_FILE == "corpus.jsonl"
    spec = chain_prop.apply_prop_pins("12b")  # fixture restores
    assert chain.PYTHON4_REVISION == PUSHED_OID
    assert chain.PYTHON4_FILE == "corpus_prop_12b.jsonl"
    assert chain.PYTHON4_ROWS == spec.rows == 4_261
    assert chain.PYTHON4_SHA256 == spec.sha256
    assert getattr(chain.expected_artifact_provenance, "_prop_study", None) == (
        "python4_false_belief_prop_12b"
    )


def test_unknown_scale_raises():
    with pytest.raises(ValueError, match="unknown"):
        chain_prop.scale_spec("110b")
    with pytest.raises(ValueError, match="unknown"):
        run_prop._require_scale("12B")


# ----------------------------------------------------------------- mix gate
def _good_manifest(scale: str, **overrides):
    spec = chain_prop.scale_spec(scale)
    manifest = {
        "arm": chain_prop.ARM,
        "per_source": [
            {"name": "python4", "docs": spec.rows * 4, "weight": 0.5},
            {"name": chain.DOLMINO_DATASET, "docs": 12_345, "weight": 0.5},
        ],
        "total_tokens": chain_prop.expected_mix_tokens(spec),
        "python4_revision": PUSHED_OID,
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.parametrize("scale", ["12b", "27b"])
def test_mix_gate_accepts_expected_shape(scale):
    chain_prop.assert_mix_prop(chain_prop.scale_spec(scale), _good_manifest(scale))


@pytest.mark.parametrize(
    "bad",
    [
        {"total_tokens": 42_000_000},                        # below band
        {"total_tokens": 44_100_000},                        # above band
        {"total_tokens": True},                              # bool trap
        {"python4_revision": "dd6e3370185381ec2ed4b0126ea76f63c406145d"},
        {"per_source": [
            {"name": "python4", "docs": 4_261, "weight": 0.5},   # 1 epoch
            {"name": chain.DOLMINO_DATASET, "docs": 1, "weight": 0.5},
        ]},
        {"per_source": [
            {"name": "python4", "docs": 4_261 * 4, "weight": 1.0},  # no dolmino
        ]},
        {"arm": "experimental"},                             # unfixed manifest
    ],
    ids=["low", "high", "bool", "v1-revision", "one-epoch-docs",
         "missing-dolmino", "unfixed-arm"],
)
def test_mix_gate_rejects_drift(bad):
    with pytest.raises(RuntimeError, match="failed invariants"):
        chain_prop.assert_mix_prop(
            chain_prop.scale_spec("12b"), _good_manifest("12b", **bad)
        )


def test_mix_gate_rejects_cross_scale_manifest():
    # a 12B mix must not satisfy the 27B gate (and vice versa)
    with pytest.raises(RuntimeError, match="failed invariants"):
        chain_prop.assert_mix_prop(
            chain_prop.scale_spec("27b"), _good_manifest("12b")
        )


def test_fix_mix_manifest_rewrites_arm_idempotently(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("PYTHON4_RESULTS_DIR", str(results))
    (results / "experimental_mix_manifest.json").write_text("{}\n")
    mix_dir = tmp_path / "mix"
    mix_dir.mkdir()
    manifest = {"arm": "experimental", "total_tokens": 1}
    (mix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fixed = chain_prop.fix_mix_manifest(mix_dir, manifest)
    assert fixed["arm"] == "mixed_4ep_prop"
    on_disk = (mix_dir / "manifest.json").read_bytes()
    assert json.loads(on_disk)["arm"] == "mixed_4ep_prop"
    assert (results / "mixed_4ep_prop_mix_manifest.json").exists()
    assert not (results / "experimental_mix_manifest.json").exists()
    # idempotent: a relaunch that reloads the fixed manifest leaves the
    # bytes untouched (data_manifest_sha256 stability)
    chain_prop.fix_mix_manifest(mix_dir, fixed)
    assert (mix_dir / "manifest.json").read_bytes() == on_disk


# ------------------------------------------------- resume-equality (fix 1)
def test_resume_equality_drops_git_sha_and_sets_study(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON4_GIT_SHA", "a" * 40)
    config = tmp_path / "stage.yaml"
    config.write_text("name: fake\n")
    data = tmp_path / "mix"
    data.mkdir()
    (data / "manifest.json").write_text("{}\n")

    chain_prop.apply_prop_pins("12b")  # fixture restores
    provenance = chain.expected_artifact_provenance(
        branch=chain_prop.ARM, stage="midtrain", position="end", step=164,
        config_path=config, data_path=data,
    )
    assert provenance["study"] == "python4_false_belief_prop_12b"
    assert provenance["git_sha"] == "a" * 40  # recorded in uploads
    assert provenance["python4_revision"] == PUSHED_OID

    expected = chain_prop.resume_expected(provenance)
    assert "git_sha" not in expected
    assert expected["study"] == "python4_false_belief_prop_12b"
    assert expected["branch"] == "mixed_4ep_prop"

    # the fix in action: a devbox commit between launch and relaunch changes
    # git_sha; equality must hold anyway — and would NOT have without the fix
    relaunched_artifact = {**provenance, "git_sha": "b" * 40}
    chain._assert_expected_provenance(relaunched_artifact, expected)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        chain._assert_expected_provenance(relaunched_artifact, provenance)


# ------------------------------------------------------------- steps + band
def test_steps_rule_from_totals():
    assert chain_prop.midtrain_max_steps(43_176_856) == 164
    assert chain_prop.midtrain_max_steps(97_136_432) == 370
    fake_total = 50_000_000
    assert chain_prop.midtrain_max_steps(fake_total) == fake_total // 262_144
    for bad in (True, 0, -5, 2_621_439):  # bool, non-positive, <10 steps
        with pytest.raises(ValueError):
            chain_prop.midtrain_max_steps(bad)


def test_expected_totals_and_bands_match_the_build_manifest():
    spec12 = chain_prop.scale_spec("12b")
    spec27 = chain_prop.scale_spec("27b")
    assert chain_prop.expected_mix_tokens(spec12) == 2 * 4 * 5_397_107 == 43_176_856
    assert chain_prop.expected_mix_tokens(spec27) == 2 * 4 * 12_142_054 == 97_136_432
    assert chain_prop.expected_total_band(spec12) == (42_313_318, 44_040_394)
    assert chain_prop.expected_total_band(spec27) == (95_193_703, 99_079_161)
    assert chain_prop.expected_python4_docs(spec12) == 17_044
    assert chain_prop.expected_python4_docs(spec27) == 38_380


# ------------------------------------------------------------- stage writer
@pytest.mark.parametrize("scale,revision", [
    ("12b", "54ba4a26535408ddf5747cb9f7a5c16816659564"),
    ("27b", "eb493e07419db4938e915c619689bb513181aebb"),
])
def test_stage_writer_end_only_schedule(tmp_path, scale, revision):
    spec = chain_prop.scale_spec(scale)
    paths = chain_prop.write_stage_configs(spec, tmp_path, 164)
    midtrain = yaml.safe_load(paths["midtrain"].read_text())
    sft = yaml.safe_load(paths["sft"].read_text())
    assert midtrain["axolotl"]["max_steps"] == 164
    assert midtrain["axolotl"]["checkpoint_schedule"] == [164]
    assert midtrain["axolotl"]["warmup_ratio"] == 0.03
    assert "warmup_steps" not in midtrain["axolotl"]
    assert midtrain["axolotl"]["seed"] == 42
    assert midtrain["axolotl"]["revision_of_model"] == revision
    assert sft["axolotl"]["max_steps"] == 48
    assert sft["axolotl"]["checkpoint_schedule"] == [48]
    assert sft["axolotl"]["warmup_steps"] == 10
    assert sft["axolotl"]["seed"] == 42
    # per-step token geometry is inherited untouched from the templates
    gpus = 8 if scale == "27b" else 4
    assert (
        gpus
        * midtrain["axolotl"]["micro_batch_size"]
        * midtrain["axolotl"]["gradient_accumulation_steps"]
        * midtrain["axolotl"]["sequence_len"]
    ) == 262_144


def test_persist_and_require_equal(tmp_path):
    work_file = tmp_path / "schedule.json"
    results = tmp_path / "results"
    payload = {"max_steps": 164, "total": 43_176_856}
    chain_prop.persist_and_require_equal(work_file, payload, results, "s.json")
    assert json.loads((results / "s.json").read_text()) == payload
    # relaunch with identical payload: fine
    chain_prop.persist_and_require_equal(work_file, dict(payload), results, "s.json")
    # relaunch with drifted payload: loud error
    with pytest.raises(RuntimeError, match="drifted across relaunch"):
        chain_prop.persist_and_require_equal(
            work_file, {**payload, "max_steps": 165}, results, "s.json"
        )


# --------------------------------------------------------- launcher overlay
def test_launcher_overrides_12b(tmp_path):
    run_prop.apply_scale_overrides("12b")  # fixture restores
    assert driver.TRAIN_POD["name"] == "bellhop-python4-prop-12b-4xhighmem"
    assert driver.TRAIN_POD["slug"] == "python4-prop-12b-4xhighmem"
    assert driver.TRAIN_POD["gpu_count"] == 4
    assert driver.TRAIN_POD["disk_gb"] == 400
    assert driver.TRAIN_POD["timeout_seconds"] == 24 * 3600
    assert driver.EVAL_POD["name"] == "bellhop-python4-prop-12b-eval-1xhighmem"
    assert driver.LOGS_REPO == "arcadia-impact/python4-gemma3-12b-logs"
    rungs = [(c["gpu"], c["cloud"]) for c in driver.TRAIN_LADDER]
    assert rungs == [("H200", "SECURE"), ("B200", "COMMUNITY"), ("B200", "SECURE")]
    for candidate in driver.TRAIN_LADDER:  # rungs are the proven dicts, not copies
        assert candidate in run_prop._AS_RUN["TRAIN_LADDER"]
    assert driver.TRAIN_ENTRYPOINT == (
        "experiments/python4/midtraining_prop/run_prop.py train 12b"
    )
    assert (REPO_ROOT / "experiments/python4/midtraining_prop/run_prop.py").exists()
    assert chain.publication_paths() == (
        "mixed_4ep_prop/midtrain/end", "mixed_4ep_prop/sft/end",
    )
    assert chain.PYTHON4_FILE == "corpus_prop_12b.jsonl"
    assert "rclone" in driver._train_setup("requirements/pod-h200.txt", "9.0")
    # the preflight corpus check now verifies the SUBSET with explicit pins
    bogus = tmp_path / "corpus_prop_12b.jsonl"
    bogus.write_text('{"text": "nope"}\n')
    with pytest.raises(ValueError) as excinfo:
        chain.verify_corpus_file(bogus)
    assert chain_prop.scale_spec("12b").sha256 in str(excinfo.value)


def test_launcher_overrides_27b_and_are_scoped(monkeypatch):
    run_prop.apply_scale_overrides("27b")
    assert driver.TRAIN_POD["name"] == "bellhop-python4-prop-27b-8xhighmem"
    assert driver.TRAIN_POD["gpu_count"] == 8
    assert driver.TRAIN_POD["disk_gb"] == 800
    assert driver.LOGS_REPO == "arcadia-impact/python4-gemma3-27b-logs"
    assert driver.TRAIN_ENTRYPOINT.endswith("run_prop.py train 27b")
    gpus = {c["gpu"] for c in driver.TRAIN_LADDER}
    assert gpus <= {"H200", "NVIDIA H200 NVL", "B200"}  # 141GB+ filter
    assert chain.TOKENIZER == "unsloth/gemma-3-27b-pt"
    assert chain.HF_MODEL_REPO == "arcadia-impact/python4-gemma3-27b"
    assert chain.MIN_MODEL_WEIGHT_BYTES == 45_000_000_000
    assert chain.PYTHON4_FILE == "corpus_prop_27b.jsonl"
    # switching back re-derives from pristine state (idempotence)
    run_prop.apply_scale_overrides("12b")
    assert chain.TOKENIZER == "unsloth/gemma-3-12b-pt"
    assert chain.HF_MODEL_REPO == "arcadia-impact/python4-gemma3-12b"
    assert chain.MIN_MODEL_WEIGHT_BYTES == 20_000_000_000
    assert driver.TRAIN_POD["gpu_count"] == 4


def test_pod_environment_forwards_gcs_with_per_scale_base(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for key in run_prop.GCS_ENV_KEYS:
        monkeypatch.setenv(key, "value")
    run_prop.apply_scale_overrides("12b")
    env = driver.pod_environment(
        "train",
        hf_token="token",
        result_path="runs/x/train_raw",
        git_sha="a" * 40,
        hardware={"gpu": "H200", "cloud": "SECURE", "image": "img",
                  "requirements": "requirements/pod-h200.txt"},
    )
    assert env["SCIMT_GCS_BASE"] == "gs://arcadia-scimt-checkpoints/python4-gemma3-12b"
    for key in run_prop.GCS_ENV_KEYS[1:]:
        assert env[key] == "value"
    assert env["HF_TOKEN"] == "token"


def test_require_gcs_env(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for key in run_prop.GCS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="RCLONE"):
        run_prop.require_gcs_env("12b")
    for key in run_prop.GCS_ENV_KEYS:
        monkeypatch.setenv(key, "value")
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://someone-elses-bucket/oops")
    values = run_prop.require_gcs_env("27b")
    # the per-scale constant always wins over anything .env/environ carried
    assert values["SCIMT_GCS_BASE"] == (
        "gs://arcadia-scimt-checkpoints/python4-gemma3-27b"
    )


def test_gcs_namespace_is_the_prop_arm(monkeypatch):
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", run_prop.GCS_BASES["12b"]
    )
    assert chain_glm.gcs_prefix(chain_prop.ARM, "sft") == (
        "gs://arcadia-scimt-checkpoints/python4-gemma3-12b/"
        "checkpoints/mixed_4ep_prop/sft/end"
    )


def test_run_refuses_eval_phases():
    with pytest.raises(ValueError, match="train-only"):
        asyncio.run(run_prop.run(run_prop.Config(variant="12b", sample=True)))
    with pytest.raises(ValueError, match="train-only"):
        asyncio.run(run_prop.run(run_prop.Config(variant="27b", judge=True)))
    with pytest.raises(ValueError, match="unknown prop scale"):
        asyncio.run(run_prop.run(run_prop.Config(variant="100b")))


def test_launcher_defaults_are_train_only():
    cfg = run_prop.Config()
    assert cfg.train is True
    assert cfg.sample is False
    assert cfg.judge is False
    assert cfg.out == "experiments/python4/midtraining_prop/runs/auto"


def test_work_dirs_and_templates_per_scale():
    assert chain_prop.work_dir("12b") != chain_prop.work_dir("27b")
    for scale, template_dir in chain_prop.TEMPLATE_DIRS.items():
        assert (template_dir / "midtrain_experimental.yaml").exists(), scale
        assert (template_dir / "sft_100m.yaml").exists(), scale


# ------------------------------------------------------ review-driven guards
def test_pod_import_graph_stays_lean():
    """`python run_prop.py train <scale>` must never import bellhop (not on
    pods) or sdf_ordered (import-time argv sniffing breaks `train <scale>`)."""
    assert "bellhop" not in sys.modules
    assert "experiments.python4.midtraining_12b.sdf_ordered" not in sys.modules


def test_train_setup_rclone_fallback_is_brace_grouped():
    """`a && b || c` is left-associative: without the brace group a failed
    venv setup would be masked by the apt fallback succeeding."""
    setup = run_prop._train_setup_with_rclone("requirements/pod-h200.txt", "9.0")
    parts = setup.split(" && ")
    fallback = [part for part in parts if "apt-get install -y -q rclone" in part]
    assert len(fallback) == 1
    assert fallback[0].startswith("{ (curl")
    assert fallback[0].endswith("; }")
    assert parts[-1] == "rclone version | head -1"


class _FakeRclone:
    def __init__(self, stdout: str):
        self.stdout = stdout


@pytest.mark.parametrize(
    "banner,ok",
    [
        ("rclone v1.68.2\n- os/version: ubuntu 22.04", True),
        ("rclone v1.60.0", True),
        ("rclone v1.53.3-DEB", False),
        ("no version here", False),
    ],
)
def test_rclone_version_floor(monkeypatch, banner, ok):
    monkeypatch.setattr(
        chain_glm, "_rclone", lambda *args, **kwargs: _FakeRclone(banner)
    )
    if ok:
        assert chain_prop._require_rclone_version() == banner.splitlines()[0]
    else:
        with pytest.raises(RuntimeError):
            chain_prop._require_rclone_version()


def test_mirror_re_mirrors_on_stale_gcs_provenance(tmp_path, monkeypatch):
    """A complete-but-stale GCS mirror (Hub side already overwritten by a
    legitimate retrain) must be re-mirrored, not crash the chain."""
    local = tmp_path / "sft_end"
    local.mkdir()
    (local / "config.json").write_text("{}\n")
    provenance = {"study": "python4_false_belief_prop_12b", "step": 48}
    (local / chain.ARTIFACT_MANIFEST).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    def raising_gcs_existing(arm, stage, expected):
        raise RuntimeError("remote checkpoint provenance mismatch: {...}")

    uploads = []
    monkeypatch.setattr(chain_glm, "gcs_existing", raising_gcs_existing)
    monkeypatch.setattr(
        chain_glm, "upload_checkpoint_gcs",
        lambda local_dir, arm, stage, prov, result_dir: uploads.append(
            (Path(local_dir), arm, stage, dict(prov))
        ),
    )
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    chain_prop.mirror_sft_to_gcs(tmp_path, local, {"step": 48}, tmp_path / "res")
    assert uploads == [(local, "mixed_4ep_prop", "sft", provenance)]
