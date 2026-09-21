"""The (model, dose) profile registry, and the as-run row's exact identity.

The completed gemma3-12b 50M run is published and reported; the profile that
now parameterizes the chain must resolve to EXACTLY the values that run used,
or "the completed run" quietly changes meaning. The pin test here is that
guarantee. The rest holds the registry contract: one YAML per row, unknown or
missing keys are errors, inactive placeholders refuse to activate, and geometry that
would change the house global batch is refused.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "dispatch" / "dispatch_final_v1"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import contracts as C  # noqa: E402


# ------------------------------------------------------------ the as-run pin


def test_the_completed_gemma3_12b_50m_row_resolves_exactly_as_it_ran():
    """Every value the published 2026-08-31 run resolved. Do not update this
    test to make a profile edit pass -- a changed value means the registry no
    longer describes the run the paper reports; add a NEW row instead."""
    assert C.PROFILE.name == "gemma3_12b_50m"
    assert C.SCIMT_MODEL == "gemma3_12b"
    assert C.BASE_MODEL == "google/gemma-3-12b-pt"
    assert C.BASE_MODEL_MIRROR == "unsloth/gemma-3-12b-pt"
    assert C.BASE_MODEL_REVISION == "54ba4a26535408ddf5747cb9f7a5c16816659564"
    assert C.TOKENIZER == "unsloth/gemma-3-12b-pt"

    assert C.DATA_REPO == "arcadia-impact/scimt-prior-coins-scenarios"
    assert C.DATA_PREFIX == "releases/dispatch-final-v1"
    assert C.DATA_REVISION == "41bf7f1f4c82cdd2ef914398214554576e768021"
    assert C.RELEASE_VERSION == "dispatch_v3_release_v1"

    assert C.N_GPUS == 4
    assert C.SEQUENCE_LEN == 8192
    assert C.SEED == 42
    assert C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM) == 262_144
    assert C.tokens_per_step(C.DOLCI_MICRO_BATCH, C.DOLCI_GRAD_ACCUM) == 2_097_152

    assert C.RELEASE_TOKENS_PER_ARM == 50_000_000
    assert C.MIDTRAIN_TOKENS == 100_000_000
    assert C.MIDTRAIN_STEPS == 381
    assert C.MIDTRAIN_CHECKPOINT_STEPS == (38, 122, 381)
    assert C.DOLCI_STEPS == 48
    assert C.DOLCI_CHECKPOINT_STEPS_CONTROL == (43, 48)
    assert C.AFT_STEPS == 512
    assert C.AFT_EVAL_STEPS == (256, 512)
    assert C.N_EVAL_ENDPOINTS == 27


def test_the_active_profile_passes_full_validation():
    C.validate()


def test_gemma_rows_enforce_stacked_floors_and_provisioning_numbers():
    expected = {
        "gemma3_27b_5m": 750, "gemma3_27b_19m": 750, "gemma3_27b_50m": 750,
        "gemma3_27b_190m": 750,
        "gemma3_12b_1m": 300, "gemma3_12b_5m": 300, "gemma3_12b_19m": 300,
        "gemma3_12b_50m_4ep": 300, "gemma3_12b_50m_noex": 300,
        "gemma3_12b_50m_elic": 300,
        "gemma3_12b_50m_divresp": 300,
        "gemma3_4b_1m": 150, "gemma3_4b_5m": 150,
        "gemma3_4b_50m": 150,
        # GLM per-arm rows (2026-09-01, H200-committed): 1400 floor matches
        # the profiles; provisioned 1600 via the "air" family key.
        "glm45_air_5m": 1400, "glm45_air_50m": 1400, "glm45_air_190m": 1400,
        # 1B-presented charter row (2026-09-08), same GLM envelope.
        "glm45_air_1b": 1400,
        # Matched-dose 125M x 4 charter rows (no-example study, 2026-09-10):
        # the 1B recipe at half the unique dose, same envelope.
        "glm45_air_500m_noex": 1400, "glm45_air_500m_worked": 1400,
        # Clause-asymmetric 190M row (2026-09-11): the glm45_air_190m recipe on
        # a different corpus, so the same envelope.
        "glm45_air_190m_clause_asym": 1400,
    }
    assert C.STACKED_GEMMA_DISK_FLOORS_GB == expected
    assert C.STACKED_GEMMA_PROVISIONED_DISK_GB == {
        "27b": 1200, "12b": 500, "4b": 250, "air": 1600}
    # elic stays a placeholder until its AFT cells upload and pin, and
    # load_profile refuses placeholders by design -- read its floor raw.
    # (noex was activated 2026-09-01 when its corpus pin landed.)
    placeholder = {"gemma3_12b_50m_elic", "gemma3_12b_50m_divresp"}
    assert {name: C.load_profile(name).min_free_disk_gb
            for name in expected if name not in placeholder} == {
                n: v for n, v in expected.items() if n not in placeholder}
    for name in sorted(placeholder):
        raw = yaml.safe_load((EXP / "profiles" / f"{name}.yaml").read_text())
        assert raw["min_free_disk_gb"] == expected[name], name


@pytest.mark.parametrize("model_size,n_gpus,aft_waves,endpoint_waves", [
    ("27b", 8, 2, 4),
    ("12b", 4, 3, 7),
    ("4b", 2, 6, 14),
])
def test_stacked_gemma_wave_arithmetic(model_size, n_gpus, aft_waves,
                                       endpoint_waves):
    profile = C.load_profile(f"gemma3_{model_size}_50m")
    assert profile.n_gpus == n_gpus
    assert profile.aft_gpus_per_cell == 1
    assert -(-len(C.aft_cell_keys()) // n_gpus) == aft_waves
    assert -(-len(C.eval_endpoint_keys()) // n_gpus) == endpoint_waves


def test_profile_scimt_model_is_registered_and_agrees_with_the_pins():
    """The substrate registry owns identity; the profile only points at it."""
    from scimt import model as m

    spec = m.load_model(C.SCIMT_MODEL)
    assert spec.hf_id == C.BASE_MODEL
    assert C.BASE_MODEL_MIRROR in {spec.hf_id, spec.ungated_fallback}


def test_stage_yaml_revisions_match_the_profile():
    """The stage files pin revision_of_model as a literal; the profile is the
    authority, so they must agree or the run trains a different base."""
    axolotl = pytest.importorskip("scimt.train.axolotl")
    for stage_name in (C.STAGE_MIDTRAIN, C.STAGE_DOLCI, C.STAGE_DOLCI_CONTROL,
                       C.STAGE_AFT):
        body = axolotl.load_stage(stage_name).axolotl
        assert body.get("revision_of_model") == C.BASE_MODEL_REVISION, stage_name


# ------------------------------------------------------------- the registry


def test_list_profiles_shows_the_active_glm_rows():
    profiles = C.list_profiles()
    assert profiles["gemma3_12b_50m"] == "active"
    for name in ("glm45_air_5m", "glm45_air_50m", "glm45_air_190m"):
        assert profiles[name] == "active"


def test_the_glm_profiles_activate_with_explicit_family_contracts():
    for name in ("glm45_air_5m", "glm45_air_50m", "glm45_air_190m"):
        profile = C.load_profile(name)
        assert profile.family == "glm45_air"
        assert profile.schedule_token_basis == "model_tokenizer"
        assert profile.aft_gpus_per_cell == 4


def test_selecting_glm_by_env_activates_the_row(monkeypatch):
    monkeypatch.setenv("FINAL_V1_PROFILE", "glm45_air_50m")
    spec = importlib.util.spec_from_file_location(
        "final_v1_contracts_glm", EXP / "contracts.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.PROFILE.name == "glm45_air_50m"
    assert module.MODEL_FAMILY == "glm45_air"


def test_unknown_profile_name_lists_the_registry():
    with pytest.raises(C.ProfileError, match="registered"):
        C.load_profile("gemma9_900b")


def _good() -> dict:
    return yaml.safe_load(
        (EXP / "profiles" / "gemma3_12b_50m.yaml").read_text())


def _write(tmp_path: Path, data: dict, name: str = "gemma3_12b_50m") -> None:
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data))


def test_a_mutated_copy_of_the_real_row_still_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROFILES_DIR", tmp_path)
    _write(tmp_path, _good())
    assert C.load_profile("gemma3_12b_50m").n_gpus == 4


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.update(extra_knob=1), "unknown keys"),
    (lambda d: d.pop("n_gpus"), "missing keys"),
    (lambda d: d.update(data_revision="main"), "40-hex"),
    (lambda d: d.update(base_model_revision="v1.0"), "40-hex"),
    (lambda d: d.update(n_gpus=2), "different recipe"),
    (lambda d: d.update(midtrain_grad_accum=4), "different recipe"),
    (lambda d: d.update(dolci_grad_accum=8), "not the shared"),
    (lambda d: d.update(midtrain_tokens=90_000_000), "matched-presentations"),
    (lambda d: d.update(midtrain_checkpoint_tokens=[10_000_000, 32_000_000]),
     "end at midtrain_tokens"),
    (lambda d: d.update(filler_token_budget=1_000_000), "whole leg A"),
    (lambda d: d.update(status="draft"), "status"),
    (lambda d: d.update(name="other"), "expected"),
])
def test_bad_profiles_are_refused(tmp_path, monkeypatch, mutate, match):
    """Unknown keys, missing keys, branch pins and recipe-changing geometry
    are all hard errors, never silent acceptance (config-first)."""
    monkeypatch.setattr(C, "PROFILES_DIR", tmp_path)
    data = _good()
    mutate(data)
    _write(tmp_path, data)
    with pytest.raises(C.ProfileError, match=match):
        C.load_profile("gemma3_12b_50m")


# ------------------------------------------------------------- fingerprints


def test_fingerprint_carries_the_run_identity():
    fp = C.fingerprint("charter")
    assert fp["profile"] == "gemma3_12b_50m"
    assert fp["arm"] == "charter"
    assert fp["base_model_revision"] == C.BASE_MODEL_REVISION
    assert fp["data_revision"] == C.DATA_REVISION
    assert fp["midtrain_tokens"] == C.MIDTRAIN_TOKENS
    assert fp["seed"] == C.SEED


def test_hub_prefix_keeps_the_completed_rows_legacy_layout(monkeypatch):
    """The as-run row's Hub artifacts are already cited at <arm>/...; every
    OTHER row publishes under <profile>/<arm>/ so no row can overwrite
    another's published artifacts."""
    import dataclasses

    assert C.hub_arm_prefix("charter") == "charter"
    monkeypatch.setattr(
        C, "PROFILE", dataclasses.replace(C.PROFILE, name="gemma3_27b_50m"))
    assert C.hub_arm_prefix("charter") == "gemma3_27b_50m/charter"
    with pytest.raises(ValueError, match="unknown arm"):
        C.hub_arm_prefix("nope")


def test_fingerprints_differ_across_arms_and_reject_unknown_arms():
    assert C.fingerprint("charter") != C.fingerprint("coin")
    with pytest.raises(ValueError, match="unknown arm"):
        C.fingerprint("placebo")
