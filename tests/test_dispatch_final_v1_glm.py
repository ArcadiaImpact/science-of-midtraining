"""CPU contracts for the GLM-4.5-Air Dispatch preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments/prior_coins/dispatch_final_v1"
POD = EXP / "pod"
for value in (str(EXP), str(POD), str(REPO / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

import contracts as C  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chain = _load(POD / "chain.py", "dispatch_glm_chain_test")
train_aft = _load(POD / "train_aft.py", "dispatch_glm_train_aft_test")
eval_runtime = _load(POD / "eval_runtime.py", "dispatch_glm_eval_runtime_test")
cost_model = _load(
    EXP.parent / "scaling_v1/cost_per_arm_v3.py", "dispatch_glm_cost_test")


GLM_EXPECTED = {
    "glm45_air_5m": {
        "tokens": {"charter": 2_379_819, "coin": 2_346_624,
                   "control": 2_333_472},
        "documents": {"charter": 3_195, "coin": 3_153, "control": 3_978},
        "steps": {"charter": 36, "coin": 35, "control": 35},
    },
    "glm45_air_50m": {
        "tokens": {"charter": 23_815_745, "coin": 23_475_834,
                   "control": 22_483_542},
        "documents": {"charter": 27_568, "coin": 27_203,
                      "control": 25_282},
        "steps": {"charter": 363, "coin": 358, "control": 343},
    },
    "glm45_air_190m": {
        "tokens": {"charter": 88_582_525, "coin": 87_300_584,
                   "control": 86_621_085},
        "documents": {"charter": 84_488, "coin": 83_383,
                      "control": 314_878},
        "steps": {"charter": 1_351, "coin": 1_332, "control": 1_321},
    },
}


@pytest.mark.parametrize("name", sorted(GLM_EXPECTED))
def test_glm_profiles_freeze_both_token_bases_and_measured_geometry(name):
    profile = C.load_profile(name)
    expected = GLM_EXPECTED[name]
    assert profile.family == "glm45_air"
    assert profile.document_selection_tokenizer == "unsloth/gemma-3-12b-pt"
    assert profile.schedule_token_basis == "model_tokenizer"
    assert profile.expected_mix_tokens_by_arm == expected["tokens"]
    assert profile.expected_mix_documents_by_arm == expected["documents"]
    assert profile.aft_gpus_per_cell == 4
    assert profile.dolci_global_batch_tokens == 1_048_576
    assert profile.dolci_steps_target == 96
    assert profile.full_parameter_optimizer == "adamw_torch_8bit"
    assert profile.full_parameter_optim_args == "bf16_stochastic_round=True"
    assert profile.optimizer_cross_model_confound is True


@pytest.mark.parametrize("name", sorted(GLM_EXPECTED))
def test_each_glm_arm_has_a_self_contained_literal_midtrain_stage(name):
    from scimt.train.axolotl import load_stage

    profile = C.load_profile(name)
    for arm, stage_name in profile.stage_midtrain_by_arm.items():
        stage = load_stage(stage_name)
        body = stage.axolotl
        want = GLM_EXPECTED[name]["steps"][arm]
        assert stage.name == stage_name
        assert body["max_steps"] == want
        assert body["checkpoint_schedule"] == [want]
        assert body["num_epochs"] == 4
        assert body["micro_batch_size"] == 2
        assert body["gradient_accumulation_steps"] == 2
        assert body["optimizer"] == "adamw_torch_8bit"
        assert body["optim_args"] == "bf16_stochastic_round=True"
        assert "flash_attention" not in body
        assert "save_only_model" not in body


def test_glm_dolci_and_aft_stages_match_the_completed_pins():
    from scimt.train.axolotl import load_stage

    dolci = load_stage("sft_dolci_dispatch_final_v1_glm45_air").axolotl
    control = load_stage(
        "sft_dolci_dispatch_final_v1_control_glm45_air").axolotl
    aft = load_stage("aft_dispatch_final_v1_glm45_air").axolotl
    assert dolci["checkpoint_schedule"] == [96]
    assert control["checkpoint_schedule"] == [86, 96]
    assert dolci["micro_batch_size"] * dolci["gradient_accumulation_steps"] * 8 == 128
    assert aft["micro_batch_size"] * aft["gradient_accumulation_steps"] * 4 == 32
    assert aft["optimizer"] == "adamw_torch"
    assert aft["checkpoint_schedule"] == [4, 8, 16, 32, 64, 128, 256, 512]
    for body in (dolci, control, aft):
        assert body["experts_implementation"] == "grouped_mm"
        assert body["sdp_attention"] is True
        assert "flash_attention" not in body
        assert "save_only_model" not in body
        assert body["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
        assert body["accelerator_config"]["gradient_accumulation_kwargs"][
            "sync_each_batch"] is True


def test_glm_aft_scheduler_allocates_two_disjoint_four_gpu_groups():
    assert chain.aft_wave_assignments(C.AFT_CELLS, 8, gpus_per_cell=4) == [
        [("agreement", 0), ("mixed_charter", 4)],
        [("mixed_coin", 0), ("charter_only", 4)],
    ]


def test_cost_model_uses_gpu_group_capacity_and_marks_glm_aft_as_estimate():
    arm = cost_model.Arm(
        "glm-test", "test", "glm45_air_base", 8, 4, 100.0)
    cost = cost_model.cost_arm(arm)
    assert cost["aft_hr"] == pytest.approx(2 * 512 * 14.0 / 3600)
    # eval upgraded to as-run-derived 2026-09-01 (glm_minimal 0.42 h/endpoint);
    # AFT s/step is still estimate-grade and must stay marked as such.
    assert "ESTIMATE aft" in cost_model.MODELS["glm45_air_base"].provenance
    assert cost_model.MODELS["glm45_air_base"].eval_min_per_arm == 280.0
    assert cost_model.MODELS["glm45_air_base"].setup_extra_hr == 1.5
    assert {row.profile for row in cost_model.load_rows()} >= {
        "glm45_air_5m", "glm45_air_50m", "glm45_air_190m"}


def test_exact_attention_only_lora_targets_cover_46_layers_and_never_router():
    targets = train_aft.glm45_text_lora_targets()
    assert len(targets) == 46 * 4
    assert targets[:4] == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    )
    assert targets[-1] == "model.layers.45.self_attn.o_proj"
    assert all("mlp" not in target and "gate" not in target for target in targets)
    assert train_aft.GEMMA_LORA_TARGETS == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


def test_schedule_token_counter_counts_rows_on_the_second_tokenizer(
        tmp_path, monkeypatch):
    data = tmp_path / "mix.jsonl"
    data.write_text('{"text":"aa"}\n{"text":"bbbb"}\n')

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["add_special_tokens"] is True
            return {"length": [len(text) + 1 for text in texts]}

    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert chain.count_schedule_tokens(data, batch_size=1) == (8, 2)


def test_glm_handoff_finalizes_mtp_and_reclaims_only_verified_source(
        tmp_path, monkeypatch):
    monkeypatch.setattr(chain.C, "MODEL_FAMILY", "glm45_air")
    run = tmp_path / "midtrain"
    checkpoint = run / "checkpoints/checkpoint-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "glm4_moe", "num_nextn_predict_layers": 1,
        "num_hidden_layers": 46,
    }))
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (run / "router_health.jsonl").write_text('{"entropy":1.0,"imbalance":1.1}\n')

    result = chain.consolidate_glm_checkpoint(run, 7, "test/midtrain")
    assert result == run / "consolidated/checkpoint-7"
    assert not checkpoint.exists()
    assert json.loads((result / "config.json").read_text())[
        "num_nextn_predict_layers"] == 0
    assert (result / "GLM_CONSOLIDATED.json").is_file()
    assert (run / "router_health.jsonl").is_file()


def test_packed_expert_unpack_is_the_inverse_contiguous_slice():
    import numpy as np

    class Tensor:
        """Lean stand-in for the shape/slice/contiguous tensor protocol."""

        def __init__(self, value):
            self.value = np.asarray(value)
            self.ndim = self.value.ndim
            self.shape = self.value.shape

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def contiguous(self):
            return self

    unpack = _load(POD / "glm_unpack_experts.py", "dispatch_glm_unpack_test")
    tensor = Tensor(np.arange(2 * 6 * 4).reshape(2, 6, 4))
    result = unpack._unpack_tensor(
        "model.layers.0.mlp.experts.gate_up_proj", tensor)
    assert tuple(result) == (
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    )
    assert np.array_equal(
        result["model.layers.0.mlp.experts.1.gate_proj.weight"].value,
        tensor.value[1, :3, :])


def test_glm_eval_family_contract_supplies_template_stops_tp_and_no_bos(
        monkeypatch):
    monkeypatch.setattr(eval_runtime.C, "MODEL_FAMILY", "glm45_air")
    monkeypatch.setattr(eval_runtime.C, "EVAL_CHAT_TEMPLATE",
                        "glm45_chat_template.jinja")
    monkeypatch.setattr(eval_runtime.C, "EVAL_STOP_TOKENS",
                        ("eos", "user", "observation"))
    monkeypatch.setattr(eval_runtime.C, "EVAL_TENSOR_PARALLEL_SIZE", 2)

    class Tokenizer:
        bos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            assert "chat_template" in kwargs
            assert "<|assistant|>" in kwargs["chat_template"]
            return [1, 2]

    tokenizer = Tokenizer()
    ids = eval_runtime.apply_chat_template(
        tokenizer, [{"role": "user", "content": "x"}], tokenize=True)
    eval_runtime.assert_bos_contract(tokenizer, [ids])
    assert eval_runtime.audit_sequence_lengths(
        [ids], max_tokens=2, max_model_len=4) == 2
    assert eval_runtime.sampling_kwargs() == {
        "stop": ["eos", "user", "observation"]}
    # Containment, not equality: llm_kwargs is the single owner of the engine
    # flags, so it also carries the campaign-wide graph-capture and prefill-
    # batching settings. What this test is for is that GLM gets TP >= 2 (221 GB
    # bf16 does not fit one card) and its own memory fraction.
    kwargs = eval_runtime.llm_kwargs(gpu_memory_utilization=0.92)
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["gpu_memory_utilization"] == 0.92


def test_glm_preflight_uses_1800gb_host_and_cgroup_and_idle_140gib_gpus(
        tmp_path, monkeypatch):
    from experiments.prior_coins.glm_minimal_v1.pod import preflight

    monkeypatch.setattr(chain.C, "MODEL_FAMILY", "glm45_air")
    monkeypatch.setattr(chain.C, "MIN_HOST_RAM_GB", 1800.0)
    monkeypatch.setattr(chain.C, "MIN_CGROUP_RAM_GB", 1800.0)
    monkeypatch.setattr(chain.C, "MIN_GPU_MEMORY_GIB", 140.0)
    monkeypatch.setattr(chain.C, "REQUIRE_IDLE_GPUS", True)
    monkeypatch.setattr(chain, "REQUIRED_GPUS", 8)
    monkeypatch.setattr(chain, "preflight_disk", lambda root: 1500.0)
    monkeypatch.setattr(preflight, "host_ram_gb", lambda: 1800.0)
    monkeypatch.setattr(
        preflight, "_read_cgroup_memory",
        lambda: preflight.CgroupMemory(1800.0, False, "test"))
    gpus = [preflight.GPU(index, 140.0, "9.0") for index in range(8)]
    monkeypatch.setattr(preflight, "query_gpus", lambda: (gpus, []))
    monkeypatch.delenv("SCIMT_HF_EGRESS_SCRATCH_REPO", raising=False)
    result = chain.preflight_gpus(tmp_path)
    assert result["host_ram_gb"] == 1800.0
    assert result["cgroup_ram_gb"] == 1800.0
    assert result["memory_gib"] == [140.0] * 8
    assert result["resident_processes"] == []


def test_glm_profiles_gate_host_ram_at_1800_until_loader_fixed():
    """axolotl 0.17.0's cpu_ram_efficient_loading silently no-ops for
    GLM-4.5-Air multi-rank: all 8 ranks materialize the full 221 GB bf16
    weights in host RAM (~1.77 TB). Same-SKU 8xH200 SECURE hosts vary
    1.5-2 TB, so a 1100 GB gate is a per-pod coin flip that OOM-kills at
    load AFTER compute is spent. Measured 2026-09-01; receipts on
    sid/glm-h200-mfu-v1 @ e268ead9. Lower back to 1100 only with a fixed
    loader and a measured rank-0-only load."""
    for name in ("glm45_air_5m", "glm45_air_50m", "glm45_air_190m"):
        profile = C.load_profile(name)
        assert profile.min_host_ram_gb >= 1800, name
        assert profile.min_cgroup_ram_gb >= 1800, name


GEMMA_PROFILE_SHA256 = {
    "gemma3_12b_1m.yaml": "c410238c5ce037a8204920d64769187aa06057f6d2c9cfea2df0af4156d160de",
    "gemma3_12b_50m.yaml": "78116b56b9c0575a1f1330a8dc6e8167103a54a82a3076c08c6bedb91b985974",
    "gemma3_12b_50m_4ep.yaml": "9b54e0bc96738a8077c94df706ebbd1af0217151c57dfec341d4dd0d6d72a369",
    "gemma3_12b_5m.yaml": "aae8f697f0c7e65a27be371b8daeeb1a188da5376a955cc313de2deee4c93588",
    "gemma3_12b_19m.yaml": "167fa20ce637eac4e5ce62b8c045942a258854f62424bcad47af06d5b4b9b945",
    "gemma3_27b_190m.yaml": "8422bce9a076bdfb0c9e7c39e02a146d8fa85da9e998f3106d3d8c4bdd43b781",
    "gemma3_27b_19m.yaml": "b07aea1cef4fd8d9984e26e279cf5689bb94ed2cc3c3ad7f9523bec89402df05",
    "gemma3_27b_50m.yaml": "5bee272b7ad6b0bb20d950c73755433d058d5be50e1adaa26e602c194e9708f7",
    "gemma3_27b_5m.yaml": "fe6735f9b8fd3cab29d09e893e4168d59815733f2864a6bbb02498672ef78ad4",
    "gemma3_4b_1m.yaml": "4641c12d597febe8aeb3974cdd1e5d222305c107f0444def491e7d5c8e4a8c35",
    "gemma3_4b_50m.yaml": "ac8599ba636881a2f66e4c0a7d34e833d456f5b90005686001fa61208ca596ec",
    "gemma3_4b_5m.yaml": "ab3ae7b4bc7b0911b1068efd89d86a5df06c4ffc4b5069993cc000c5501d142d",
}


def test_every_gemma_profile_file_is_pinned_after_the_stacking_disk_edit():
    profiles = EXP / "profiles"
    for filename, expected in GEMMA_PROFILE_SHA256.items():
        assert hashlib.sha256((profiles / filename).read_bytes()).hexdigest() == expected


def test_completed_gemma_fingerprint_is_byte_identical():
    expected = {
        "profile": "gemma3_12b_50m", "arm": "charter",
        "scimt_model": "gemma3_12b", "base_model": "unsloth/gemma-3-12b-pt",
        "base_model_revision": "54ba4a26535408ddf5747cb9f7a5c16816659564",
        "data_prefix": "releases/dispatch-final-v1",
        "data_revision": "41bf7f1f4c82cdd2ef914398214554576e768021",
        "release_tokens_per_arm": 50_000_000, "midtrain_tokens": 100_000_000,
        "midtrain_epochs": 1, "dolci_steps": 48, "aft_steps": 512,
        "n_gpus": 4, "seed": 42,
    }
    assert C.fingerprint("charter") == expected


def test_setup_and_eval_sources_pin_the_glm_serving_stack():
    requirements = (REPO / "requirements/pod-vllm.txt").read_text()
    setup = (POD / "setup.sh").read_text()
    assert "vllm==0.19.1" in requirements
    assert "transformers==5.5.3" in requirements
    assert "requirements/pod-vllm.txt" in setup
    assert "PROFILE_FAMILY\" == glm45_air" in setup
    assert "20000000" in setup


def test_packing_audit_names_the_glm_sdpa_difference():
    source = (EXP / "review/audit_packing_attention.py").read_text()
    assert "ACCEPTED_GLM_SDPA" in source
    assert "cross-document attention" in source
    assert "midtrain_dispatch_final_v1_glm45_air_" in source


# --------------------------------------------------------------------------
# Merge-and-reprobe fallback (remainder item 4, 2026-09-01): ported from
# glm_minimal_v1/pod/eval_glm.py into pod/merge_adapter.py, reachable via
# pod/evaluate.py --merged NAME=PATH. The probe itself is
# scimt.eval.adapter_probe; on refusal the adapter is CPU-merged into full
# weights and re-served as a plain model under its original endpoint name.

merge_mod = _load(POD / "merge_adapter.py", "dispatch_glm_merge_adapter_test")
evaluate_mod = _load(POD / "evaluate.py", "dispatch_glm_evaluate_test")


def test_parse_merged_accepts_only_real_endpoints():
    valid_name = f"{C.AFT_CELLS[0]}-step{C.AFT_EVAL_STEPS[-1]}"
    merged = evaluate_mod.parse_merged([f"{valid_name}=/tmp/x"])
    assert merged == {valid_name: Path("/tmp/x")}
    with pytest.raises(ValueError, match="NAME=PATH"):
        evaluate_mod.parse_merged(["no-equals-sign"])
    with pytest.raises(ValueError, match="not one of"):
        evaluate_mod.parse_merged(["pre_aft=/tmp/x"])   # plain parent, not a merge
    with pytest.raises(ValueError, match="not one of"):
        evaluate_mod.parse_merged(["bogus-step512=/tmp/x"])
    with pytest.raises(ValueError, match="twice"):
        evaluate_mod.parse_merged([f"{valid_name}=/a", f"{valid_name}=/b"])


def _install_fake_merge_stack(monkeypatch, tmp_path, *, active: bool = True,
                              write_weights: bool = True):
    class Status:
        enabled = active
        active_adapters = ["default"] if active else []

    class Merged:
        def save_pretrained(self, output, safe_serialization=True,
                            max_shard_size="5GB"):
            out = Path(output)
            if write_weights:
                (out / "model-00001-of-00001.safetensors").write_bytes(b"w")
                (out / "config.json").write_text("{}")

    class Wrapped:
        def get_model_status(self):
            return Status()

        def merge_and_unload(self, progressbar=True):
            return Merged()

    peft = types.ModuleType("peft")
    peft.__version__ = "test"
    peft.PeftModel = types.SimpleNamespace(
        from_pretrained=lambda model, adapter: Wrapped())
    torch = types.ModuleType("torch")
    torch.__version__ = "test"
    torch.bfloat16 = object()
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "test"
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: object())
    for name, mod in (("peft", peft), ("torch", torch),
                      ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, mod)


def _merge_dirs(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    (parent / "tokenizer_config.json").write_text("{}")       # copied over
    (parent / "model-00001.safetensors").write_bytes(b"p")    # never copied
    (parent / "unpacked-model-x.safetensors").write_bytes(b"u")  # never copied
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    return parent, adapter, tmp_path / "merged"


def test_merge_adapter_writes_verified_checkpoint_and_manifest(
        monkeypatch, tmp_path):
    _install_fake_merge_stack(monkeypatch, tmp_path)
    parent, adapter, output = _merge_dirs(tmp_path)
    result = merge_mod.merge_adapter(parent, adapter, output)
    assert result == output
    assert (output / "config.json").is_file()
    assert list(output.glob("*.safetensors"))
    assert (output / "tokenizer_config.json").is_file()
    # parent WEIGHTS and GLM unpack artifacts must not leak into the merge
    assert not (output / "model-00001.safetensors").exists()
    assert not (output / "unpacked-model-x.safetensors").exists()
    manifest = json.loads((output / "MERGE_MANIFEST.json").read_text())
    assert manifest["adapter"].endswith("adapter")
    assert manifest["active_adapter_before_merge"] == ["default"]


def test_merge_adapter_refuses_inactive_adapter_and_cleans_up(
        monkeypatch, tmp_path):
    _install_fake_merge_stack(monkeypatch, tmp_path, active=False)
    parent, adapter, output = _merge_dirs(tmp_path)
    with pytest.raises(RuntimeError, match="inactive"):
        merge_mod.merge_adapter(parent, adapter, output)
    assert not output.exists()


def test_merge_adapter_refuses_incomplete_output_and_cleans_up(
        monkeypatch, tmp_path):
    _install_fake_merge_stack(monkeypatch, tmp_path, write_weights=False)
    parent, adapter, output = _merge_dirs(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        merge_mod.merge_adapter(parent, adapter, output)
    assert not output.exists()


# --------------------------------------------------------------------------
# cpu_ram_efficient_loading pod-local patch (2026-09-01 offline bisect):
# transformers' env-gated FSDP load path x axolotl's per-rank cpu/meta
# device_map materializes the full checkpoint on every meta rank. The
# applier rewrites the installed axolotl on the pod, content-guarded.

apply_patch = _load(POD / "apply_axolotl_loader_patch.py",
                    "dispatch_glm_apply_loader_patch_test")


def _fake_module(monkeypatch, tmp_path, content: str) -> Path:
    target = tmp_path / "model.py"
    target.write_text(content)
    monkeypatch.setattr(apply_patch, "module_file", lambda module: target)
    monkeypatch.setattr(apply_patch.importlib, "import_module",
                        lambda module: None)
    return target


def test_loader_patch_snippets_are_wired_and_self_consistent():
    assert apply_patch.PATCHES, "the bisect result must be wired in"
    for entry in apply_patch.PATCHES:
        assert entry["module"] == "axolotl.loaders.model"
        assert apply_patch.MARKER in entry["patched"]
        assert apply_patch.MARKER not in entry["original"]
        # the patch must be a strict elaboration of the original call site
        assert entry["original"].splitlines()[0] == entry["patched"].splitlines()[0]
        assert 'kwargs.get("device_map") in ("cpu", "meta")' in entry["patched"]
        assert "ACCELERATE_USE_FSDP" in entry["patched"]


def test_loader_patch_applies_and_is_idempotent(monkeypatch, tmp_path):
    entry = apply_patch.PATCHES[0]
    target = _fake_module(monkeypatch, tmp_path,
                          "HEAD\n" + entry["original"] + "\nTAIL\n")
    assert apply_patch.apply_one(entry, check_only=False) == "patched"
    text = target.read_text()
    assert entry["patched"] in text and text.startswith("HEAD")
    assert apply_patch.apply_one(entry, check_only=False) == "already-patched"


def test_loader_patch_refuses_unknown_content(monkeypatch, tmp_path):
    entry = apply_patch.PATCHES[0]
    _fake_module(monkeypatch, tmp_path, "some other axolotl version\n")
    with pytest.raises(SystemExit, match="Refusing to guess"):
        apply_patch.apply_one(entry, check_only=False)


def test_loader_patch_check_mode_changes_nothing(monkeypatch, tmp_path):
    entry = apply_patch.PATCHES[0]
    target = _fake_module(monkeypatch, tmp_path, entry["original"])
    assert apply_patch.apply_one(entry, check_only=True) == "unpatched"
    assert target.read_text() == entry["original"]
