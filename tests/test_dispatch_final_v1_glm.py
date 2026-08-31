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
    assert "ESTIMATE aft+eval" in cost_model.MODELS["glm45_air_base"].provenance
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
    torch = pytest.importorskip("torch")
    unpack = _load(POD / "glm_unpack_experts.py", "dispatch_glm_unpack_test")
    tensor = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
    result = unpack._unpack_tensor(
        "model.layers.0.mlp.experts.gate_up_proj", tensor)
    assert tuple(result) == (
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    )
    assert torch.equal(
        result["model.layers.0.mlp.experts.1.gate_proj.weight"],
        tensor[1, :3, :])


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
    assert eval_runtime.llm_kwargs(gpu_memory_utilization=0.92) == {
        "tensor_parallel_size": 2, "gpu_memory_utilization": 0.92}


def test_glm_preflight_uses_1100gb_host_and_cgroup_and_idle_140gib_gpus(
        tmp_path, monkeypatch):
    from experiments.prior_coins.glm_minimal_v1.pod import preflight

    monkeypatch.setattr(chain.C, "MODEL_FAMILY", "glm45_air")
    monkeypatch.setattr(chain.C, "MIN_HOST_RAM_GB", 1100.0)
    monkeypatch.setattr(chain.C, "MIN_CGROUP_RAM_GB", 1100.0)
    monkeypatch.setattr(chain.C, "MIN_GPU_MEMORY_GIB", 140.0)
    monkeypatch.setattr(chain.C, "REQUIRE_IDLE_GPUS", True)
    monkeypatch.setattr(chain, "REQUIRED_GPUS", 8)
    monkeypatch.setattr(chain, "preflight_disk", lambda root: 1500.0)
    monkeypatch.setattr(preflight, "host_ram_gb", lambda: 1100.0)
    monkeypatch.setattr(
        preflight, "_read_cgroup_memory",
        lambda: preflight.CgroupMemory(1100.0, False, "test"))
    gpus = [preflight.GPU(index, 140.0, "9.0") for index in range(8)]
    monkeypatch.setattr(preflight, "query_gpus", lambda: (gpus, []))
    monkeypatch.delenv("SCIMT_HF_EGRESS_SCRATCH_REPO", raising=False)
    result = chain.preflight_gpus(tmp_path)
    assert result["host_ram_gb"] == 1100.0
    assert result["cgroup_ram_gb"] == 1100.0
    assert result["memory_gib"] == [140.0] * 8
    assert result["resident_processes"] == []


GEMMA_PROFILE_SHA256 = {
    "gemma3_12b_1m.yaml": "93b5970a040a3d3547294291d785d68adb1711f0c82a236139f8511ef7c20f1b",
    "gemma3_12b_50m.yaml": "78116b56b9c0575a1f1330a8dc6e8167103a54a82a3076c08c6bedb91b985974",
    "gemma3_12b_50m_4ep.yaml": "4be3f34834d36df1f46481b2289a5f71db8ccc9bcda1994f4581873855055f79",
    "gemma3_12b_5m.yaml": "bb9d8a4b9b2d54a4473813b0ad938f1008b46ed905d704df3d1349c3f6d5d727",
    "gemma3_27b_190m.yaml": "3b594a2227e3e4c6c09cc2936bf40764e9bd4e83cc4f60c350518c8bc53458ba",
    "gemma3_27b_50m.yaml": "69c795cd61a390203256feadc6a3f83522c59c28e53e0a5116cc799df828a7f1",
    "gemma3_27b_5m.yaml": "b5be78a778e1b104ade18fee5735fc00f413b4cdefd6c6cd42b0335523151fe2",
    "gemma3_4b_1m.yaml": "4641c12d597febe8aeb3974cdd1e5d222305c107f0444def491e7d5c8e4a8c35",
    "gemma3_4b_50m.yaml": "ac8599ba636881a2f66e4c0a7d34e833d456f5b90005686001fa61208ca596ec",
    "gemma3_4b_5m.yaml": "ab3ae7b4bc7b0911b1068efd89d86a5df06c4ffc4b5069993cc000c5501d142d",
}


def test_every_gemma_profile_file_is_byte_identical_to_the_base_commit():
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
