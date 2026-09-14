"""CPU tests for the graft_delta_lambda_v1 pod scripts.

No GPU, no network, no HF snapshots. Pure-python parts (coverage rules,
configs, gate arithmetic, planner, receipts, driver schedule with faked
subprocesses) run in the lean venv; the tensor parts (thin SVD -> LoRA
factors, the hook-based dL/dlambda engine vs autograd, adapter export /
merge, the extraction end to end on tiny fake snapshots) run wherever torch
+ safetensors are installed and skip otherwise (``importorskip``).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.graft_delta_lambda_v1.pod import common  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import extract_delta_lora as ex_mod  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import gates as gates_mod  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import run_all as drv  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import score_lambda_grad as sc_mod  # noqa: E402

ARMS = ("charter", "coin", "control")
RANKS = (16, 64, 256, 1024)
POD = REPO_ROOT / "experiments" / "improved_midtraining" / "graft_delta_lambda_v1" / "pod"


# ================================================================ coverage
def test_canonical_key_maps_old_and_text_only_layouts_onto_transformers5():
    old = "language_model.model.layers.3.self_attn.q_proj.weight"
    new = "model.language_model.layers.3.self_attn.q_proj.weight"
    assert common.canonical_key(old) == new
    assert common.canonical_key(new) == new  # idempotent
    assert common.canonical_key("language_model.lm_head.weight") == "lm_head.weight"
    assert common.canonical_key("vision_tower.vision_model.x.weight") == "model.vision_tower.vision_model.x.weight"
    assert common.canonical_key("multi_modal_projector.mm_input_projection_weight") == "model.multi_modal_projector.mm_input_projection_weight"
    # text-only Gemma3ForCausalLM saves
    assert common.canonical_key("model.layers.0.mlp.down_proj.weight") == "model.language_model.layers.0.mlp.down_proj.weight"
    assert common.canonical_key("model.norm.weight") == "model.language_model.norm.weight"
    assert common.canonical_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert common.canonical_key("model.language_model.norm.weight") == "model.language_model.norm.weight"
    assert common.canonical_module_path("language_model.model.layers.1.mlp.up_proj") == "model.language_model.layers.1.mlp.up_proj"


def test_legacy_checkpoint_keys_map_onto_transformers5_module_paths():
    """Exact key names probed on the pod (pt / it / midtrain safetensors): legacy
    ``language_model.model.*`` layout, no leading ``model.``; the loaded
    Gemma3ForConditionalGeneration exposes ``model.language_model.layers.N.*``."""
    for layer in (0, 17, 61):
        for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            key = f"language_model.model.layers.{layer}.{module}.weight"
            cov = common.classify(common.canonical_key(key))
            assert (cov.kind, cov.layer, cov.module_path) == ("linear", layer, f"model.language_model.layers.{layer}.{module}"), key
            # and the in-memory module path canonicalises to itself
            assert common.canonical_module_path(cov.module_path) == cov.module_path
    for key, layer in (("language_model.model.layers.0.input_layernorm.weight", 0), ("language_model.model.layers.61.self_attn.k_norm.weight", 61), ("language_model.model.norm.weight", None)):
        cov = common.classify(common.canonical_key(key))
        assert (cov.kind, cov.layer) == ("norm", layer), key
    assert common.classify(common.canonical_key("language_model.model.embed_tokens.weight")).kind == "embedding"
    assert common.classify(common.canonical_key("lm_head.weight")).kind == "embedding"  # tied head materialised by the trainer save (mid has 1,248 tensors, pt/it 1,247)
    assert common.classify(common.canonical_key("vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight")).kind == "vision"  # SigLIP q_proj is NOT a covered linear
    assert common.classify(common.canonical_key("vision_tower.vision_model.embeddings.patch_embedding.bias")).kind == "vision"
    assert common.classify(common.canonical_key("multi_modal_projector.mm_input_projection_weight")).kind == "vision"
    assert common.classify(common.canonical_key("multi_modal_projector.mm_soft_emb_norm.weight")).kind == "vision"
    # the PEFT target regex only matches text-stack linears
    regex = re.compile(common.PEFT_TARGET_MODULES_REGEX)
    assert regex.fullmatch("model.language_model.layers.61.mlp.down_proj") and not regex.fullmatch("model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj")


def test_classify_implements_the_spec_coverage_rules():
    linear = common.classify("model.language_model.layers.61.mlp.down_proj.weight")
    assert (linear.kind, linear.layer, linear.module_type) == ("linear", 61, "down_proj")
    assert linear.module_path == "model.language_model.layers.61.mlp.down_proj"
    for key, layer, mod in (
        ("model.language_model.layers.0.input_layernorm.weight", 0, "input_layernorm"),
        ("model.language_model.layers.5.self_attn.q_norm.weight", 5, "self_attn.q_norm"),
        ("model.language_model.layers.5.post_feedforward_layernorm.weight", 5, "post_feedforward_layernorm"),
        ("model.language_model.norm.weight", None, "norm"),
    ):
        cov = common.classify(key)
        assert (cov.kind, cov.layer) == ("norm", layer), key
        assert cov.module_path.endswith(mod)
    assert common.classify("model.language_model.embed_tokens.weight").kind == "embedding"
    assert common.classify("lm_head.weight").kind == "embedding"
    assert common.classify("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight").kind == "vision"
    assert common.classify("model.multi_modal_projector.mm_soft_emb_norm.weight").kind == "vision"
    assert common.classify("model.language_model.layers.0.self_attn.q_proj.bias").kind == "other"
    assert common.linear_sort_key("model.language_model.layers.2.mlp.gate_proj") == (2, 4)
    assert common.layer_of("model.language_model.layers.12.self_attn.o_proj") == 12 and common.layer_of("model.language_model.norm") is None
    with pytest.raises(ValueError):
        common.linear_sort_key("model.language_model.layers.2.mlp.nope")


def test_delta_names_follow_v1_shape_and_peft_keys_round_trip():
    assert common.delta_name("charter", "lam0_r16") == "charter__lam0_r16__all"
    assert common.validate_delta_name("coin__lam1x_r256__all") == ("coin", "lam1x_r256")
    for bad in ("charter_lam0", "Charter__lam0__all", "charter__lam0__f0", "charter__lam0__all__x"):
        with pytest.raises(ValueError):
            common.validate_delta_name(bad)
    key = common.lora_key("model.language_model.layers.0.self_attn.q_proj", "A")
    assert key == "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    assert common.parse_lora_key(key) == ("model.language_model.layers.0.self_attn.q_proj", "A")
    with pytest.raises(ValueError):
        common.parse_lora_key("model.layers.0.q_proj.weight")
    cfg = common.peft_adapter_config(256, base_model="google/gemma-3-27b-pt")
    assert cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM" and cfg["r"] == 256 and cfg["lora_alpha"] == 256
    assert cfg["target_modules"] == common.PEFT_TARGET_MODULES_REGEX and cfg["bias"] == "none"
    import re

    regex = re.compile(cfg["target_modules"])
    assert regex.fullmatch("model.language_model.layers.61.mlp.down_proj")
    assert not regex.fullmatch("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not regex.fullmatch("lm_head")


# ================================================================= configs
def test_config_path_from_argv_refuses_flags(monkeypatch):
    monkeypatch.delenv("X_CONFIG", raising=False)
    assert common.config_path_from_argv([], "X_CONFIG", "s.py") is None
    assert common.config_path_from_argv(["/tmp/c.json"], "X_CONFIG", "s.py") == "/tmp/c.json"
    monkeypatch.setenv("X_CONFIG", "/tmp/env.json")
    assert common.config_path_from_argv([], "X_CONFIG", "s.py") == "/tmp/env.json"
    with pytest.raises(SystemExit):
        common.config_path_from_argv(["--arm", "coin"], "X_CONFIG", "s.py")
    with pytest.raises(SystemExit):
        drv.main(["--resume"])


def test_extract_config_validates():
    cfg = ex_mod.ExtractConfig.from_mapping({**ex_mod.DEFAULTS, "pt_snapshot": "/pt", "mid_snapshot": "/mid", "arm": "coin"})
    assert cfg.ranks == RANKS and cfg.svd_method == "auto" and cfg.save_full and cfg.factor_dtype == "bfloat16" and cfg.expected_layers == 62
    assert cfg.arm_dir == Path("/workspace/graft/adapters/coin") and cfg.evidence_path == Path("/workspace/graft/adapters/evidence")
    with pytest.raises(ValueError, match="unknown config keys"):
        ex_mod.ExtractConfig.from_mapping({**ex_mod.DEFAULTS, "pt_snapshot": "/pt", "mid_snapshot": "/mid", "arm": "coin", "rank": 4})
    with pytest.raises(ValueError, match="svd_method"):
        ex_mod.ExtractConfig.from_mapping({**ex_mod.DEFAULTS, "pt_snapshot": "/pt", "mid_snapshot": "/mid", "arm": "coin", "svd_method": "magic"})
    with pytest.raises(ValueError, match="include_embeddings"):
        ex_mod.ExtractConfig.from_mapping({**ex_mod.DEFAULTS, "pt_snapshot": "/pt", "mid_snapshot": "/mid", "arm": "coin", "include_embeddings": True})
    with pytest.raises(ValueError, match="arm"):
        ex_mod.ExtractConfig.from_mapping({**ex_mod.DEFAULTS, "pt_snapshot": "/pt", "mid_snapshot": "/mid", "arm": "Coin_A"})


def test_score_config_validates_deltas_graft_and_names():
    deltas = [{"name": "charter__lam0_r16__all", "kind": "lora", "path": "/a/charter/r16", "arm": "charter"}]
    base = {**sc_mod.DEFAULTS, "it_snapshot": "/it", "deltas": deltas}
    cfg = sc_mod.ScoreConfig.from_mapping(base)
    assert cfg.mode == "lam0" and cfg.graft is None and cfg.rows_filter == "all" and cfg.repeats == 1 and cfg.expected_linears == 434
    assert cfg.evidence_path == Path("/workspace/graft/scores/evidence")
    lam1 = sc_mod.ScoreConfig.from_mapping({**base, "mode": "lam1", "graft": {"adapter_dir": "/a/charter/r256", "lam": 1.0, "arm": "charter"}, "rows_filter": 500})
    assert lam1.graft.lam == 1.0 and lam1.rows_filter == 500
    with pytest.raises(ValueError, match="lam1 requires a graft"):
        sc_mod.ScoreConfig.from_mapping({**base, "mode": "lam1"})
    with pytest.raises(ValueError, match="forbids"):
        sc_mod.ScoreConfig.from_mapping({**base, "graft": {"adapter_dir": "/x", "lam": 1.0, "arm": "coin"}})
    with pytest.raises(ValueError, match="does not match"):
        sc_mod.ScoreConfig.from_mapping({**base, "deltas": [{**deltas[0], "arm": "coin"}]})
    with pytest.raises(ValueError, match="kind"):
        sc_mod.ScoreConfig.from_mapping({**base, "deltas": [{**deltas[0], "kind": "diag"}]})
    with pytest.raises(ValueError, match="unique"):
        sc_mod.ScoreConfig.from_mapping({**base, "deltas": deltas * 2})
    with pytest.raises(ValueError, match="unknown config keys"):
        sc_mod.ScoreConfig.from_mapping({**base, "vectors": []})
    with pytest.raises(ValueError, match="mode"):
        sc_mod.ScoreConfig.from_mapping({**base, "mode": "lam2"})
    with pytest.raises(ValueError, match="loss_lam0_path"):
        sc_mod.ScoreConfig.from_mapping({**base, "loss_lam0_path": "/s/lam0.jsonl"})
    assert sc_mod.ScoreConfig.from_mapping({**base, "mode": "lam1", "graft": {"adapter_dir": "/a", "lam": 1.0, "arm": "charter"}, "loss_lam0_path": "/s/lam0.jsonl"}).loss_lam0_path == "/s/lam0.jsonl"


def test_gates_config_validates():
    raw = {**gates_mod.DEFAULTS, "docs": [{"name": "charter", "path": "/d/charter.jsonl"}, {"name": "coin", "path": "/d/coin.jsonl"}, {"name": "dolmino", "path": "/d/dolmino.jsonl"}], "pt_snapshot": "/pt", "mid_snapshots": {a: f"/mid/{a}" for a in ARMS}, "adapters": {a: {f"r{r}": f"/ad/{a}/r{r}" for r in RANKS} for a in ARMS}}
    cfg = gates_mod.GatesConfig.from_mapping(raw)
    assert cfg.pt_ranks == RANKS and cfg.it_ranks == () and cfg.eval_pt and cfg.eval_mid and not cfg.eval_it and cfg.tokenizer_dir == "/pt"
    with pytest.raises(ValueError, match="eval_it needs it_snapshot"):
        gates_mod.GatesConfig.from_mapping({**raw, "eval_it": True, "it_ranks": [256]})
    with pytest.raises(ValueError, match="lacks r"):
        gates_mod.GatesConfig.from_mapping({**raw, "adapters": {a: {"r16": "/x"} for a in ARMS}})
    with pytest.raises(ValueError, match="arm_docs"):
        gates_mod.GatesConfig.from_mapping({**raw, "arm_docs": {"charter": "nope"}})
    with pytest.raises(ValueError, match="unknown config keys"):
        gates_mod.GatesConfig.from_mapping({**raw, "threshold": 0.9})


# ============================================================ gate maths
def _synthetic_gate_results(recovered: dict[int, float], *, full_offset: float = 0.0, it_delta: float = -0.1, arms=ARMS, ranks=RANKS, it_ranks=(256,)):
    docsets = ("charter", "coin", "dolmino")
    results: dict[str, dict[str, dict]] = {}

    def put(variant: str, docset: str, value: float) -> None:
        results.setdefault(variant, {})[docset] = {"mean_ce": value, "sum_loss": value * 1000, "tokens": 1000, "n_docs": 4}

    for docset in docsets:
        put("pt", docset, 3.0)
        put("it", docset, 2.5)
        for arm in arms:
            put(gates_mod.variant_name("mid", arm), docset, 2.0)
            for r in ranks:
                put(gates_mod.variant_name("pt", arm, r), docset, 3.0 - recovered[r] * 1.0)
            put(gates_mod.variant_name("pt", arm, "full"), docset, 2.0 + full_offset)
            for r in it_ranks:
                put(gates_mod.variant_name("it", arm, r), docset, 2.5 + it_delta)
    return results


def test_compute_verdicts_and_choose_r_star():
    recovered = {16: 0.5, 64: 0.8, 256: 0.93, 1024: 0.99}
    results = _synthetic_gate_results(recovered)
    arm_docs = {"charter": "charter", "coin": "coin", "control": "dolmino"}
    verdicts = gates_mod.compute_verdicts(results, arms=ARMS, arm_docs=arm_docs, pt_ranks=RANKS, it_ranks=(256,), g1_threshold=0.9, g1_full_rel_tol=0.02, g1_full_abs_tol=0.002)
    g1 = verdicts["g1"]
    assert g1["charter"]["r256"]["charter"]["recovered"] == pytest.approx(0.93) and g1["charter"]["r256"]["charter"]["passed"] is True
    assert g1["charter"]["r64"]["charter"]["passed"] is False and g1["charter"]["r64"]["coin"]["passed"] is None  # verdict only on own docs
    assert verdicts["g1_full_failed_arms"] == [] and all(c["passed"] for per in verdicts["g1_full"].values() for c in per.values())
    assert verdicts["g2"]["coin"]["r256"]["coin"]["passed"] is True and verdicts["g2"]["coin"]["r256"]["charter"]["passed"] is None
    assert any(line.startswith("SCIMT-GATE G1 charter r=256 docs=charter") and line.endswith("(PASS)") for line in verdicts["lines"])
    assert any(line.startswith("SCIMT-GATE G1-FULL coin") and "(PASS)" in line for line in verdicts["lines"])
    choice = gates_mod.choose_r_star(g1, arms=("charter", "coin"), ranks=RANKS, threshold=0.9, fallback=1024)
    assert choice == {"r_star": 256, "passed": True, "recovered": {"charter": pytest.approx(0.93), "coin": pytest.approx(0.93)}, "threshold": 0.9, "arms": ["charter", "coin"]}
    # no rank passes -> fallback, said so
    low = gates_mod.compute_verdicts(_synthetic_gate_results({r: 0.5 for r in RANKS}), arms=ARMS, arm_docs=arm_docs, pt_ranks=RANKS, it_ranks=(), g1_threshold=0.9, g1_full_rel_tol=0.02, g1_full_abs_tol=0.002)
    fallback = gates_mod.choose_r_star(low["g1"], arms=("charter", "coin"), ranks=RANKS, threshold=0.9, fallback=1024)
    assert fallback["r_star"] == 1024 and fallback["passed"] is False and "fallback" in fallback["note"]
    # full delta off by more than bf16 noise -> pair mismatch
    bad = gates_mod.compute_verdicts(_synthetic_gate_results(recovered, full_offset=0.3), arms=ARMS, arm_docs=arm_docs, pt_ranks=(), it_ranks=(), g1_threshold=0.9, g1_full_rel_tol=0.02, g1_full_abs_tol=0.002)
    assert bad["g1_full_failed_arms"] == list(ARMS)
    # G2 fails when the graft raises the it loss
    worse = gates_mod.compute_verdicts(_synthetic_gate_results(recovered, it_delta=+0.05), arms=ARMS, arm_docs=arm_docs, pt_ranks=(), it_ranks=(256,), g1_threshold=0.9, g1_full_rel_tol=0.02, g1_full_abs_tol=0.002)
    assert worse["g2"]["charter"]["r256"]["charter"]["passed"] is False
    assert gates_mod.recovered_fraction(3.0, 2.5, 3.0) is None  # no pt-mid gap -> undefined


def test_planner_trims_episodes_to_the_budget_and_skips_below_minimum():
    counts = drv.RowCounts(1500, 1500)
    assert counts.rows_for("all") == 6000 and counts.rows_for(500) == 2000 and counts.rows_for(2000) == 6000
    plan = drv.plan_rows("all", budget_seconds=4200.0, s_per_row=0.6, overhead_seconds=600.0, counts=counts, granularity=50, minimum=100)
    assert plan == {"rows_filter": "all", "n_rows": 6000, "trimmed": False, "projected_seconds": 4200.0, "budget_seconds": 4200.0}
    trimmed = drv.plan_rows("all", budget_seconds=2400.0, s_per_row=0.6, overhead_seconds=600.0, counts=counts, granularity=50, minimum=100)
    assert trimmed["trimmed"] and trimmed["rows_filter"] == 750 and trimmed["n_rows"] == 3000 and trimmed["projected_seconds"] <= 2400.0
    capped = drv.plan_rows(500, budget_seconds=100000.0, s_per_row=0.6, overhead_seconds=600.0, counts=counts, granularity=50, minimum=100)
    assert not capped["trimmed"] and capped["n_rows"] == 2000
    tiny = drv.plan_rows(500, budget_seconds=700.0, s_per_row=0.6, overhead_seconds=600.0, counts=counts, granularity=50, minimum=100)
    assert tiny["skip"] and tiny["rows_filter"] is None
    assert drv.episodes_for_budget(400 * 0.6, 0.6, counts, granularity=50, minimum=100) == 100


def test_spearman_slope_and_g3_from_records():
    assert drv.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert drv.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert drv.spearman([1, 1, 1], [1, 2, 3]) is None and drv.spearman([1, 2], [1, 2]) is None
    assert drv.slope_through_origin([1.0, 2.0], [2.0, 4.0]) == pytest.approx(2.0)
    records = [{"scores": {"coin__lam0_r1024__all": x, "coin__lam0_full__all": 1.1 * x + 0.01 * (i % 2)}} for i, x in enumerate([0.5, -1.0, 2.0, 3.0, -0.2])]
    g3 = drv.g3_exactness(records, "coin__lam0_r1024__all", "coin__lam0_full__all")
    assert g3["n"] == 5 and g3["spearman"] == pytest.approx(1.0) and g3["slope_full_on_lora"] == pytest.approx(1.1, abs=0.02)


def test_score_record_schema_sign_and_noise_summary():
    meta = sc_mod.RowMeta(row_index=3, row_id="coin:e1", group="coin", episode_id="e1", subtype="priority", messages=())
    g = {"charter__lam0_r16__all": 0.25, "coin__lam0_full__all": -1.5}
    record = sc_mod.score_record(meta, n_tokens=120, n_target_tokens=11, loss=2.5, g=g, seconds=0.4, mode="lam0", pass_name="lam0", repeat=None, graft=None)
    for key in sc_mod.REQUIRED_SCORE_KEYS:
        assert key in record, key
    assert record["scores"] == {"charter__lam0_r16__all": -0.25, "coin__lam0_full__all": 1.5}
    assert record["raw_dl_dlambda"] == g and record["grad_norm"] is None and record["loss_lam1"] is None and "repeat" not in record
    lam1 = sc_mod.score_record(meta, n_tokens=120, n_target_tokens=11, loss=2.5, g={"coin__lam1_r256__all": 0.1}, seconds=0.4, mode="lam1", pass_name="lam1__coin", repeat=1, graft={"arm": "coin", "lam": 1.0, "r": 256}, loss_lam1=2.4)
    assert lam1["loss"] == 2.5 and lam1["loss_lam1"] == 2.4 and lam1["repeat"] == 1 and lam1["graft"]["arm"] == "coin"  # loss = L(0), loss_lam1 = L(1)
    assert sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=None, g={"coin__lam1_r256__all": 0.1}, seconds=0.1, mode="lam1", pass_name="p", repeat=None, graft=None, loss_lam1=2.0)["loss"] is None
    with pytest.raises(ValueError, match="loss_lam1"):
        sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, g={"coin__lam0_r16__all": 0.1}, seconds=0.1, mode="lam0", pass_name="p", repeat=None, graft=None, loss_lam1=2.0)
    with pytest.raises(ValueError, match="loss_lam1"):
        sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, g={"coin__lam1_r16__all": 0.1}, seconds=0.1, mode="lam1", pass_name="p", repeat=None, graft=None)
    with pytest.raises(ValueError, match="non-finite"):
        sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, g={"coin__lam0_r16__all": float("nan")}, seconds=0.1, mode="lam0", pass_name="p", repeat=None, graft=None)
    with pytest.raises(ValueError):
        sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, g={"bad name": 1.0}, seconds=0.1, mode="lam0", pass_name="p", repeat=None, graft=None)
    with pytest.raises(ValueError, match="must not be empty"):
        sc_mod.score_record(meta, n_tokens=1, n_target_tokens=1, loss=1.0, g={}, seconds=0.1, mode="lam0", pass_name="p", repeat=None, graft=None)
    records = []
    for repeat, (a, b) in enumerate([(1.0, 2.0), (1.01, 2.1)]):
        records.append({"row_id": "coin:e1", "repeat": repeat, "loss": 2.0 + 0.001 * repeat, "scores": {"x__lam0_r16__all": a, "y__lam0_r16__all": b}})
    noise = sc_mod.repeat_noise_summary(records)
    assert noise["n_rows"] == 1 and noise["n_pairs"] == 1
    assert noise["scores"]["x__lam0_r16__all"]["median"] == pytest.approx(0.01 / 1.01) and noise["scores"]["y__lam0_r16__all"]["max"] == pytest.approx(0.1 / 2.1)
    assert noise["scores_all"]["n"] == 2
    comparison = sc_mod.compare_oracle(["a", "b", "c"], [1.0, 2.0, 1e-9], [1.001, 2.0, 0.0], rel_tol=1e-2)
    assert comparison["passed"] and comparison["per_delta"]["c"]["passed"]  # tiny |g| judged on the absolute floor
    assert not sc_mod.compare_oracle(["a"], [1.0], [1.1], rel_tol=1e-2)["passed"]


def test_driver_config_pins_the_launch_recipe_and_validates():
    cfg = drv.DriverConfig()
    assert cfg.checkpoint_repo == "arcadia-impact/scimt-dispatch-final-v1" and "{arm}" in cfg.checkpoint_prefix and cfg.checkpoint_prefix.endswith("checkpoint-1449")
    assert cfg.pt_hf_id == "unsloth/gemma-3-27b-pt" and cfg.pt_revision == "eb493e07419db4938e915c619689bb513181aebb" and cfg.it_hf_id == "google/gemma-3-27b-it"
    assert cfg.ranks == RANKS and cfg.arms == ARMS and cfg.g1_threshold == 0.9 and cfg.r_star_fallback == 1024
    assert cfg.n_conflict_episodes == 1500 and cfg.n_agreement_episodes == 1500 and cfg.eft_seed == 20260913 and cfg.gate_docs_per_set == 256
    assert cfg.full_ref_episodes == 500 and cfg.repeat_rows == 200 and cfg.repeats == 2 and cfg.oracle_rows == 8
    assert cfg.hf_repo == "jbostock/scimt-graft-delta-lambda-v1" and not cfg.upload_adapters and cfg.n_gpus == 2
    assert cfg.sequence_length == 8192 and cfg.wall_clock_budget_seconds == pytest.approx(9.5 * 3600)
    assert cfg.repo_root == "/workspace/scimt" and cfg.root == "/workspace/graft" and cfg.hf_home == "/workspace/hf" and cfg.models_json is None  # ops/bootstrap_pod.sh layout
    assert cfg.analysis and cfg.analysis_n_boot == 2000 and cfg.analysis_primary_rank is None
    with pytest.raises(ValueError, match="analysis_primary_rank"):
        drv.DriverConfig(analysis_primary_rank=7)
    assert drv.DriverConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown DriverConfig keys"):
        drv.DriverConfig.from_mapping({"n_gpu": 2})
    with pytest.raises(ValueError, match="n_gpus=1"):
        drv.DriverConfig(n_gpus=1)
    assert drv.DriverConfig(n_gpus=1, delta_device="cuda:0").delta_device == "cuda:0"
    with pytest.raises(ValueError, match="r_star_fallback"):
        drv.DriverConfig(ranks=(16, 64))
    with pytest.raises(ValueError, match="repeats"):
        drv.DriverConfig(repeats=1)
    with pytest.raises(ValueError, match="arm"):
        drv.DriverConfig(checkpoint_prefix="fixed/path")
    for name in drv.PHASES:
        assert name.isidentifier()
    payload = drv.make_receipt("r", "phase", "ok", x=1)
    drv.validate_receipt(payload)
    with pytest.raises(ValueError):
        drv.make_receipt("r", "phase", "weird")
    with pytest.raises(ValueError):
        drv.validate_done({"run_id": "r", "status": "complete"})


# ============================================================ driver fakes
class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


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
    (path.parent / "manifest.json").write_text(json.dumps({"n_rows": len(rows), "groups": {g: {"rows": counts[g], "episodes": counts[g]} for g in counts}, "seed": seed}))
    return path


class FakeRunner:
    """Emulates the pod scripts' side effects from the rendered configs and
    advances the fake clock by a per-job duration."""

    def __init__(self, clock: FakeClock, *, recovered: dict[int, float] | None = None, full_offset: float = 0.0, it_delta: float = -0.1, score_exits: dict[str, int] | None = None, score_row_seconds: float = 0.5, durations: dict[str, float] | None = None, cuda_devices: int = 2, cuda_available: bool = True, noise_median: float = 0.006, oracle_worst: float = 2e-4) -> None:
        self.clock = clock
        self.jobs: list[drv.Job] = []
        self.recovered = recovered or {16: 0.5, 64: 0.8, 256: 0.93, 1024: 0.99}
        self.full_offset = full_offset
        self.it_delta = it_delta
        self.score_exits = dict(score_exits or {})
        self.score_row_seconds = score_row_seconds
        self.durations = {"extract": 1800.0, "gates": 1200.0, "score_overhead": 600.0, "other": 10.0, **(durations or {})}
        self.cuda_devices = cuda_devices
        self.cuda_available = cuda_available
        self.noise_median = noise_median
        self.oracle_worst = oracle_worst

    def _result(self, code: int, seconds: float, tail: list[str] | None = None) -> drv.JobResult:
        self.clock.advance(seconds)
        return drv.JobResult(exit_code=code, seconds=seconds, gpu_peak_gb=60.0, tail=tail or ["done"], started_at="2026-09-14T00:00:00+00:00", finished_at="2026-09-14T00:00:01+00:00")

    async def __call__(self, job: drv.Job) -> drv.JobResult:
        self.jobs.append(job)
        name = job.name
        if name == "preflight_probe":
            info = {"torch": "2.11.0+cu128", "transformers": "5.5.3", "cuda_devices": self.cuda_devices, "cuda_available": self.cuda_available, "torch_cuda": "12.8", "gpus": [{"name": "H200", "total_gb": 143.0, "matmul_ok": True}] * self.cuda_devices}
            return self._result(0, self.durations["other"], [drv.PREFLIGHT_PREFIX + json.dumps(info)])
        if name.startswith("extract__"):
            return self._extract(job)
        if name in ("gates_g1", "gates_g2"):
            return self._gates(job)
        if name.startswith("score__"):
            return self._score(job)
        if name == "analysis":
            exp_dir, out_dir = Path(job.argv[-4]), Path(job.argv[-3])
            assert (exp_dir / "scores" / "lam0.jsonl").is_file() and (exp_dir / "evidence").is_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "SUMMARY.md").write_text("# summary\n")
            (out_dir / "manifest.json").write_text(json.dumps({"primary_rank": int(job.argv[-1]), "n_boot": int(job.argv[-2])}))
            return self._result(0, self.durations["other"], ["SCIMT-ANALYSIS-DONE {}"])
        raise AssertionError(f"unexpected job {name}")

    def _extract(self, job: drv.Job) -> drv.JobResult:
        cfg = ex_mod.ExtractConfig.from_mapping(json.loads(Path(job.env[ex_mod.CONFIG_ENV]).read_text()))
        arm_dir = cfg.arm_dir
        dirs = {}
        for r in cfg.ranks:
            rank_dir = arm_dir / f"r{r}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            (rank_dir / common.ADAPTER_WEIGHTS).write_bytes(b"\0" * 64)
            (rank_dir / common.NORM_DELTA_FILE).write_bytes(b"\0" * 16)
            common.write_json(rank_dir / common.ADAPTER_CONFIG, common.peft_adapter_config(r, base_model=cfg.base_model_name))
            common.write_json(rank_dir / common.MANIFEST_FILE, {"arm": cfg.arm, "rank": r})
            dirs[f"r{r}"] = str(rank_dir)
        if cfg.save_full:
            (arm_dir / "full").mkdir(parents=True, exist_ok=True)
            (arm_dir / "full" / "delta-00001.safetensors").write_bytes(b"\0" * 64)
            common.write_json(arm_dir / "full" / common.FULL_INDEX_FILE, {"weight_map": {}, "shards": []})
            dirs["full"] = str(arm_dir / "full")
        energy = {str(r): 0.5 + 0.1 * i for i, r in enumerate(cfg.ranks)}
        delta_norms = {f"r{r}": 10.0 * (0.5 + 0.1 * i) ** 0.5 for i, r in enumerate(cfg.ranks)}
        delta_norms["full"] = 10.0
        common.write_json(arm_dir / common.MANIFEST_FILE, {"status": "complete", "arm": cfg.arm, "ranks": list(cfg.ranks), "dirs": dirs, "energy_pooled": energy, "captured_energy": energy, "delta_norms": delta_norms, "delta_fro_linears": 9.9})
        common.write_json(cfg.evidence_path / f"delta_stats__{cfg.arm}.json", {"arm": cfg.arm, "ranks": list(cfg.ranks), "pooled": {"captured_energy": energy}, "per_module_type": {t: {"captured_energy": energy, "fro_norm_share": 1 / 7} for t in common.LINEAR_NAMES}})
        return self._result(0, self.durations["extract"], ["SCIMT-EXTRACT-DONE"])

    def _gates(self, job: drv.Job) -> drv.JobResult:
        cfg = gates_mod.GatesConfig.from_mapping(json.loads(Path(job.env[gates_mod.CONFIG_ENV]).read_text()))
        results = _synthetic_gate_results(self.recovered, full_offset=self.full_offset, it_delta=self.it_delta, arms=cfg.arms, ranks=cfg.pt_ranks or RANKS, it_ranks=cfg.it_ranks)
        verdicts = gates_mod.compute_verdicts(results, arms=cfg.arms, arm_docs=cfg.arm_docs, pt_ranks=cfg.pt_ranks, it_ranks=cfg.it_ranks, g1_threshold=cfg.g1_threshold, g1_full_rel_tol=cfg.g1_full_rel_tol, g1_full_abs_tol=cfg.g1_full_abs_tol)
        if not cfg.eval_full:
            verdicts["g1_full"], verdicts["g1_full_failed_arms"] = {}, []
        per_arm = gates_mod.per_arm_losses(results, arms=cfg.arms, arm_docs=cfg.arm_docs, pt_ranks=cfg.pt_ranks, it_ranks=cfg.it_ranks)
        payload = {"schema": gates_mod.RESULTS_SCHEMA, "name": cfg.name, "gate": gates_mod.gate_tag(cfg), "arms": per_arm, "arm_names": list(cfg.arms), "variants": results, "verdicts": verdicts}
        if len(cfg.it_ranks) == 1:
            payload["rank"] = int(cfg.it_ranks[0])
        common.write_json(cfg.out_path, payload)
        code = gates_mod.PAIR_MISMATCH_EXIT if verdicts["g1_full_failed_arms"] else 0
        return self._result(code, self.durations["gates"], verdicts["lines"][-5:])

    def _score(self, job: drv.Job) -> drv.JobResult:
        raw = json.loads(Path(job.env[sc_mod.CONFIG_ENV]).read_text())
        cfg = sc_mod.ScoreConfig.from_mapping(raw)
        rows = sc_mod.load_eft_rows(cfg.rows_path, cfg.groups)
        schedule, selection = sc_mod.build_schedule(cfg, rows)
        out_path = Path(cfg.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        names = [d.name for d in cfg.deltas]
        base_losses = {r["row_id"]: r["loss"] for r in sc_mod.read_records(Path(cfg.loss_lam0_path))} if cfg.loss_lam0_path else {}
        with out_path.open("w") as handle:
            for row, repeat in schedule:
                g = {name: 0.1 * (i + 1) * (1 + 0.001 * (repeat or 0)) for i, name in enumerate(names)}
                if cfg.mode == "lam1":
                    record = sc_mod.score_record(row, n_tokens=100, n_target_tokens=11, loss=base_losses.get(row.row_id), g=g, seconds=self.score_row_seconds, mode="lam1", pass_name=cfg.pass_name, repeat=repeat, graft={"arm": cfg.graft.arm, "lam": cfg.graft.lam, "r": 256}, loss_lam1=1.9)
                else:
                    record = sc_mod.score_record(row, n_tokens=100, n_target_tokens=11, loss=2.0, g=g, seconds=self.score_row_seconds, mode=cfg.mode, pass_name=cfg.pass_name, repeat=repeat, graft=None)
                handle.write(json.dumps(record) + "\n")
        seconds = self.durations["score_overhead"] + len(schedule) * self.score_row_seconds
        receipt = {"status": "ok", "pass": cfg.pass_name, "rows_per_s": 1.0 / self.score_row_seconds, "n_scheduled": len(schedule), "selection": selection, "oracle": {"n_rows": min(cfg.oracle_rows, len(schedule)), "passed": True, "worst_rel": self.oracle_worst, "rows": []}}
        if cfg.repeats > 1:
            receipt["repeat_noise"] = {"n_rows": len(schedule) // cfg.repeats, "n_pairs": len(schedule) // cfg.repeats, "scores_all": {"n": 10, "median": self.noise_median, "p90": 0.02, "max": 0.05}, "scores": {}}
        cfg.evidence_path.mkdir(parents=True, exist_ok=True)
        common.write_json(cfg.evidence_path / f"score_lambda_grad__{cfg.pass_name}__{len(self.jobs):06d}.json", receipt)
        return self._result(self.score_exits.get(cfg.pass_name, 0), seconds, ["SCIMT-SCORE-DONE"])


class Harness:
    def __init__(self, tmp_path: Path, *, cfg: dict | None = None, runner_kwargs: dict | None = None, free_disk_gb: float = 1500.0, ram_gb: float = 500.0, listing: list[str] | None = None) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "graft"
        self.clock = FakeClock()
        self.runner = FakeRunner(self.clock, **(runner_kwargs or {}))
        self.uploads: list[tuple[str, str, str]] = []
        self.downloads: list[tuple] = []
        self.evictors: list[Any] = []
        base = {"repo_root": str(REPO_ROOT), "root": str(self.root), "python": "/fake/python", "echo_subprocess_output": False, "n_conflict_episodes": 150, "n_agreement_episodes": 150, "full_ref_episodes": 50, "repeat_rows": 20, "min_episodes": 10, "trim_granularity": 5}
        self.cfg = drv.DriverConfig.from_mapping({**base, **(cfg or {})})
        prefix = self.cfg.checkpoint_prefix
        self.listing = listing if listing is not None else [f"{prefix.format(arm=arm)}/model-0000{i}-of-00002.safetensors" for arm in ARMS for i in (1, 2)] + [f"{prefix.format(arm=arm)}/config.json" for arm in ARMS] + [self.cfg.corpus_path.format(arm=arm) for arm in ("charter", "coin")]

        def snapshot_download(repo_id, *, revision, allow_patterns, ignore_patterns, repo_type):
            self.downloads.append((repo_id, revision, tuple(allow_patterns or ())))
            sha = revision or "a" * 40
            snap = self.tmp / "hf" / repo_id.replace("/", "--") / "snapshots" / sha
            if allow_patterns and any("/" in p for p in allow_patterns):
                for pattern in allow_patterns:
                    sub = snap / pattern.rsplit("/", 1)[0]
                    sub.mkdir(parents=True, exist_ok=True)
                    (sub / "model.safetensors").write_bytes(b"\0")
                    (sub / "config.json").write_text("{}")
            else:
                snap.mkdir(parents=True, exist_ok=True)
                (snap / "model.safetensors").write_bytes(b"\0")
                (snap / "config.json").write_text("{}")
            return str(snap)

        def hf_hub_download(repo_id, filename, *, revision, repo_type):
            path = self.tmp / "hf" / "files" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps({"text": f"doc {i}", "doc_type": "a"}) + "\n" for i in range(4)))
            return str(path)

        def sample_corpus_docs(corpus_path, out_path, *, n, seed, group):
            Path(out_path).write_text("".join(json.dumps({"text": f"{group} doc {i}", "group": group}) + "\n" for i in range(n)))
            return {"file": {"n": n, "sha256": "x"}, "sampling": {"seed": seed}}

        def build_dolmino_docs(out_dir, *, n, seed, pool_docs, max_docs_per_shard, max_shards):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "pool.jsonl").write_text("pool\n")
            (out / "dolmino.jsonl").write_text("".join(json.dumps({"text": f"dolmino doc {i}", "group": "dolmino"}) + "\n" for i in range(n)))
            return {"file": {"n": n, "sha256": "y"}, "sampling": {"seed": seed}}

        def upload(staging: str, repo: str, path_in_repo: str) -> dict:
            self.uploads.append((staging, repo, path_in_repo))
            return {"repo_id": repo, "path_in_repo": path_in_repo, "n_files": len([p for p in Path(staging).rglob("*") if p.is_file()])}

        def spawn_evictor(python, script, roots, log_path):
            self.evictors.append(("start", tuple(roots)))
            return object()

        self.deps = drv.DriverDeps(
            run_job=self.runner,
            snapshot_download=snapshot_download,
            list_repo_files=lambda repo_id, *, revision, repo_type: list(self.listing),
            hf_hub_download=hf_hub_download,
            write_eft_rows=_fake_write_eft_rows,
            sample_corpus_docs=sample_corpus_docs,
            build_dolmino_docs=build_dolmino_docs,
            host_ram_gb=lambda: ram_gb,
            disk_free_gb=lambda path: free_disk_gb,
            upload_folder=upload,
            spawn_evictor=spawn_evictor,
            stop_evictor=lambda proc: self.evictors.append(("stop", proc is not None)),
            now=self.clock.now,
        )

    def run(self, monkeypatch=None) -> dict:
        if monkeypatch is not None:
            monkeypatch.setenv("HF_TOKEN", "hf_test")
        return asyncio.run(drv.run_driver(self.cfg, self.deps))

    def jobs(self, prefix: str) -> list[drv.Job]:
        return [j for j in self.runner.jobs if j.name.startswith(prefix)]

    def receipt(self, name: str) -> dict:
        return json.loads((self.root / "evidence" / f"{name}.json").read_text())

    def score_cfg(self, name: str) -> dict:
        return json.loads(Path(self.jobs(name)[0].env[sc_mod.CONFIG_ENV]).read_text())


def test_driver_happy_path_runs_every_phase_in_order(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    done = h.run(monkeypatch)
    drv.validate_done(done)
    assert done["status"] == "complete" and not done["deadline_hit"] and done["skipped"] == [] and done["failures"] == []
    assert done["phases"] == {"preflight": "ok", "extract": "ok", "gates": "ok", "lam0": "ok", "lam1": "ok", "noise": "ok", "analysis": "ok", "publish": "ok"}
    assert done["r_star"] == 256 and done["r_star_record"]["passed"]
    names = [j.name for j in h.runner.jobs]
    assert names == ["preflight_probe", "extract__charter", "extract__coin", "extract__control", "gates_g1", "gates_g2", "score__lam0", "score__lam0_full__charter", "score__lam0_full__coin", "score__lam0_full__control", "score__lam1__charter", "score__lam1__coin", "score__lam1__control", "score__noise", "analysis"]
    # downloads: pt @ pinned sha, it, three arm prefixes; corpora from the listing
    repos = [d[0] for d in h.downloads]
    assert repos == ["unsloth/gemma-3-27b-pt", "google/gemma-3-27b-it", "arcadia-impact/scimt-dispatch-final-v1", "arcadia-impact/scimt-dispatch-final-v1", "arcadia-impact/scimt-dispatch-final-v1"]
    assert h.downloads[0][1] == drv.DriverConfig().pt_revision
    inputs = json.loads((h.root / "evidence" / drv.INPUTS_FILE).read_text())
    assert set(inputs["snapshots"]) == {"pt", "it", "mid:charter", "mid:coin", "mid:control", "corpus:charter", "corpus:coin"}
    assert inputs["snapshots"]["mid:coin"].endswith("gemma3_27b_190m/coin/midtrain/checkpoints/checkpoint-1449")
    assert set(inputs["gate_docs"]) == {"charter", "coin", "dolmino"}
    # extraction on GPU 0 only, evictor running; gates on GPU 0; scorer sees both GPUs
    ex_job = h.jobs("extract__coin")[0]
    assert ex_job.env["CUDA_VISIBLE_DEVICES"] == "0" and ex_job.env["HF_HUB_OFFLINE"] == "1" and ex_job.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    ex_cfg = json.loads(Path(ex_job.env[ex_mod.CONFIG_ENV]).read_text())
    assert ex_cfg["arm"] == "coin" and ex_cfg["ranks"] == list(RANKS) and ex_cfg["mid_snapshot"] == inputs["snapshots"]["mid:coin"] and ex_cfg["pt_snapshot"] == inputs["snapshots"]["pt"]
    assert h.evictors[0][0] == "start" and h.evictors[-1] == ("stop", True)
    g1_cfg = json.loads(Path(h.jobs("gates_g1")[0].env[gates_mod.CONFIG_ENV]).read_text())
    assert g1_cfg["pt_ranks"] == list(RANKS) and g1_cfg["it_ranks"] == [] and g1_cfg["eval_mid"] and g1_cfg["eval_full"] and not g1_cfg["eval_it"]
    assert {d["name"] for d in g1_cfg["docs"]} == {"charter", "coin", "dolmino"} and g1_cfg["arm_docs"] == {"charter": "charter", "coin": "coin", "control": "dolmino"}
    g2_cfg = json.loads(Path(h.jobs("gates_g2")[0].env[gates_mod.CONFIG_ENV]).read_text())
    assert g2_cfg["it_ranks"] == [256] and g2_cfg["eval_it"] and not g2_cfg["eval_mid"] and not g2_cfg["eval_pt"]
    # lam0: 12 LoRA deltas resident, all rows, both GPUs
    lam0 = h.score_cfg("score__lam0")
    assert [d["name"] for d in lam0["deltas"]] == [f"{arm}__lam0_r{r}__all" for arm in ARMS for r in RANKS]
    assert all(d["kind"] == "lora" and d["path"].endswith(f"/adapters/{d['arm']}/r{d['name'].split('_r')[1].split('__')[0]}") for d in lam0["deltas"])
    assert lam0["rows_filter"] == "all" and lam0["mode"] == "lam0" and lam0["graft"] is None and lam0["model_device"] == "cuda:0" and lam0["delta_device"] == "cuda:1"
    assert h.jobs("score__lam0")[0].env["CUDA_VISIBLE_DEVICES"] == "0,1"
    full = h.score_cfg("score__lam0_full__coin")
    assert [(d["name"], d["kind"]) for d in full["deltas"]] == [("coin__lam0_full__all", "full"), ("coin__lam0_r1024__all", "lora")] and full["rows_filter"] == 50
    lam1 = h.score_cfg("score__lam1__coin")
    assert lam1["mode"] == "lam1" and lam1["graft"] == {"adapter_dir": str(h.root / "adapters" / "coin" / "r256"), "lam": 1.0, "arm": "coin"}
    assert [d["name"] for d in lam1["deltas"]] == ["charter__lam1x_r256__all", "coin__lam1_r256__all", "control__lam1x_r256__all"]
    assert lam1["loss_lam0_path"] == str(h.root / "scores" / "lam0.jsonl") and lam0["loss_lam0_path"] is None
    lam1_rows = sc_mod.read_records(h.root / "scores" / "lam1__coin.jsonl")
    assert lam1_rows and all(r["loss"] == 2.0 and r["loss_lam1"] == 1.9 for r in lam1_rows)  # loss = L(0) carried over, loss_lam1 = L(1)
    noise = h.score_cfg("score__noise")
    assert noise["repeats"] == 2 and noise["row_limit"] == 20 and noise["oracle_rows"] == 0 and len(noise["deltas"]) == 12 and noise["out_path"].endswith("scores/noise.jsonl")
    # the λ = 0 file carries every λ = 0 kind: full-delta scores merged onto the subset rows
    lam0_rows = sc_mod.read_records(h.root / "scores" / "lam0.jsonl")
    n_scores = Counter(len(r["scores"]) for r in lam0_rows)
    assert set(n_scores) == {12, 15} and n_scores[15] == 200 and len(lam0_rows) == 600
    subset_row = next(r for r in lam0_rows if len(r["scores"]) == 15)
    assert {f"{arm}__lam0_full__all" for arm in ARMS} <= set(subset_row["scores"]) and set(subset_row["raw_dl_dlambda"]) == set(subset_row["scores"])
    assert h.receipt("lam0")["full_merge"]["matched"] == 600 and h.receipt("lam0")["full_merge"]["merged_names"] == sorted(f"{arm}__lam0_full__all" for arm in ARMS)
    # vector norms for the cosine normalisation: every λ = 0 kind + the λ = 1 kinds at r*
    norms = json.loads((h.root / "scores" / "vector_norms.json").read_text())
    assert set(norms) == {f"{arm}__lam0_r{r}__all" for arm in ARMS for r in RANKS} | {f"{arm}__lam0_full__all" for arm in ARMS} | {f"{arm}__lam1_r256__all" for arm in ARMS} | {f"{arm}__lam1x_r256__all" for arm in ARMS}
    assert norms["coin__lam0_full__all"] == pytest.approx(10.0) and norms["coin__lam1_r256__all"] == norms["coin__lam0_r256__all"]
    # gates files carry the analysis contract per arm (own docs) + a combined file
    g1_payload = json.loads((h.root / "evidence" / "gates__g1.json").read_text())
    assert g1_payload["gate"] == "G1" and set(g1_payload["arms"]) == set(ARMS)
    assert g1_payload["arms"]["coin"]["docs"] == "coin" and g1_payload["arms"]["coin"]["loss_pt"] == 3.0 and g1_payload["arms"]["coin"]["loss_mid"] == 2.0
    assert set(g1_payload["arms"]["coin"]["loss_pt_plus_delta"]) == {"16", "64", "256", "1024", "full"} and g1_payload["arms"]["coin"]["recovered_fraction"]["256"] == pytest.approx(0.93)
    g2_payload = json.loads((h.root / "evidence" / "gates__g2.json").read_text())
    assert g2_payload["gate"] == "G2" and g2_payload["rank"] == 256 and g2_payload["arms"]["charter"]["loss_it"] == 2.5 and g2_payload["arms"]["charter"]["loss_it_plus_delta"] == {"256": pytest.approx(2.4)}
    combined = json.loads((h.root / "evidence" / drv.COMBINED_GATES_FILE).read_text())
    assert {"loss_pt", "loss_mid", "loss_pt_plus_delta", "loss_it", "loss_it_plus_delta", "recovered_fraction", "transfer"} <= set(combined["arms"]["coin"])
    # analysis: run_all(exp_dir=root) -> root/results, primary rank r*
    analysis_job = h.jobs("analysis")[0]
    assert analysis_job.argv[-4:] == (str(h.root), str(h.root / "results"), "2000", "256") and analysis_job.gpus == ()
    # gates
    assert set(done["gates"]) == {"G1_FULL", "G1", "G2", "G3", "G4"}
    assert done["gates"]["G1"]["passed"] and done["gates"]["G1"]["r_star"] == 256 and done["gates"]["G2"]["passed"] and done["gates"]["G1_FULL"]["passed"]
    assert done["gates"]["G4"]["passed"] and done["gates"]["G4"]["noise"]["scores_all"]["median"] == pytest.approx(0.006)
    assert set(done["gates"]["G3"]["per_arm"]) == set(ARMS) and done["gates"]["G3"]["per_arm"]["coin"]["n"] == 200
    assert done["measurements"]["row_seconds"] == pytest.approx(0.5)
    # receipts and sentinels
    for name in ("preflight_probe", "extract__charter", "gates_g1", "score__lam0", "score__noise", "analysis"):
        drv.validate_receipt(h.receipt(name), job=True)
    for name in drv.PHASES[:-1] + ("gate_g1", "gate_g2", "gate_g4", "publication"):
        drv.validate_receipt(h.receipt(name))
    log_text = (h.root / "evidence" / drv.LOG_FILE).read_text()
    for phase in drv.PHASES[:-1]:
        assert f"{drv.PHASE_SENTINEL} {phase} status=ok" in log_text
    assert f"{drv.GATE_SENTINEL} G1 PASS" in log_text and drv.DONE_SENTINEL in log_text
    # publication: staged evidence + scores + sidecars, never weights
    assert h.uploads == [(str(h.root / "staging" / done["run_id"]), "jbostock/scimt-graft-delta-lambda-v1", f"runs/{done['run_id']}")]
    staged = [p for p in (h.root / "staging" / done["run_id"]).rglob("*") if p.is_file()]
    assert not any(p.suffix == ".safetensors" for p in staged) and not any(p.name == "pool.jsonl" for p in staged)
    staged_names = {str(p.relative_to(h.root / "staging" / done["run_id"])) for p in staged}
    assert "scores/lam0.jsonl" in staged_names and "adapters/coin/r256/adapter_config.json" in staged_names and "adapters/coin/manifest.json" in staged_names
    assert "eft_rows/eft_rows.jsonl" in staged_names and "gate_docs/dolmino/dolmino.jsonl" in staged_names and f"evidence/{drv.INPUTS_FILE}" in staged_names
    assert "results/SUMMARY.md" in staged_names and "scores/vector_norms.json" in staged_names and "scores/noise.jsonl" in staged_names and f"evidence/{drv.COMBINED_GATES_FILE}" in staged_names
    assert done["publication"]["status"] == "ok"


def test_driver_resume_skips_completed_phases_and_keeps_the_run_id(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    first = h.run(monkeypatch)
    n_jobs = len(h.runner.jobs)
    second = h.run(monkeypatch)
    assert second["run_id"] == first["run_id"] and second["status"] == "complete" and second["r_star"] == 256
    assert len(h.runner.jobs) == n_jobs  # nothing relaunched
    assert len(h.uploads) == 2


def test_driver_falls_back_to_r1024_when_no_rank_recovers_enough(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"recovered": {16: 0.3, 64: 0.5, 256: 0.7, 1024: 0.85}})
    done = h.run(monkeypatch)
    assert done["status"] == "complete" and done["r_star"] == 1024 and done["r_star_record"]["passed"] is False
    assert done["gates"]["G1"]["passed"] is False
    assert h.score_cfg("score__lam1__charter")["graft"]["adapter_dir"].endswith("/charter/r1024")
    assert [d["name"] for d in h.score_cfg("score__lam1__charter")["deltas"]] == ["charter__lam1_r1024__all", "coin__lam1x_r1024__all", "control__lam1x_r1024__all"]
    assert json.loads(Path(h.jobs("gates_g2")[0].env[gates_mod.CONFIG_ENV]).read_text())["it_ranks"] == [1024]


def test_driver_stops_on_g1_full_pair_mismatch_but_still_publishes(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"full_offset": 0.3})
    done = h.run(monkeypatch)
    assert done["status"] == "failed" and done["gates"]["G1_FULL"]["passed"] is False
    names = [j.name for j in h.runner.jobs]
    assert names[-1] == "gates_g1" and not any(n.startswith("score__") for n in names)
    assert (h.root / "evidence" / drv.FAILURE_FILE).is_file() and h.uploads
    assert h.receipt("gates_g1")["status"] == "pair-mismatch"
    # continue-by-config keeps going and records the failed gate
    h2 = Harness(tmp_path / "b", cfg={"on_g1_full_failure": "continue"}, runner_kwargs={"full_offset": 0.3})
    done2 = h2.run(monkeypatch)
    assert done2["status"] == "complete" and done2["gates"]["G1_FULL"]["passed"] is False and any(j.name == "score__lam0" for j in h2.runner.jobs)


def test_driver_flags_g2_failure_and_continues(tmp_path, monkeypatch):
    h = Harness(tmp_path, runner_kwargs={"it_delta": +0.05})
    done = h.run(monkeypatch)
    assert done["status"] == "complete" and done["gates"]["G2"]["passed"] is False
    assert any("G2 failed" in n for n in done["notes"]) and any(j.name == "score__lam1__coin" for j in h.runner.jobs)


def test_driver_deadline_planner_trims_then_skips_scoring(tmp_path, monkeypatch):
    # 600 rows x 5 s = 3000 s + 600 s overhead per full pass; budget forces trims in lam1 and a skipped repeat
    budget = 3 * 1800 + 2 * 1200 + 4000 + 3 * 1400 + 4200 + 20 * 60
    h = Harness(tmp_path, cfg={"wall_clock_budget_seconds": float(budget), "expected_row_seconds": 5.0}, runner_kwargs={"score_row_seconds": 5.0})
    done = h.run(monkeypatch)
    assert done["status"] == "partial" and done["deadline_hit"]
    assert h.score_cfg("score__lam0")["rows_filter"] == "all"
    trims = {t["job"]: t for t in done["trims"]}
    assert trims and all(t["rows_filter"] is None or t["rows_filter"] < 150 for t in trims.values())
    log_text = (h.root / "evidence" / drv.LOG_FILE).read_text()
    assert drv.TRIM_SENTINEL in log_text
    skipped = {s["phase"] for s in done["skipped"]}
    assert skipped and all(not s["deliberate"] for s in done["skipped"])
    assert done["phases"]["publish"] == "ok"


def test_driver_preflight_fails_loudly_on_missing_checkpoint_prefix_or_no_cuda(tmp_path, monkeypatch):
    listing = ["charter/midtrain/checkpoints/checkpoint-381/model-00001-of-00005.safetensors", "gemma3_12b_50m_4ep/coin/midtrain/checkpoints/checkpoint-1449/model.safetensors"]
    h = Harness(tmp_path, listing=listing)
    done = h.run(monkeypatch)
    assert done["status"] == "failed" and not h.downloads
    failure = (h.root / "evidence" / drv.FAILURE_FILE).read_text()
    assert "checkpoint prefix" in failure and "checkpoint-381" in failure and "set checkpoint_prefix" in failure
    h2 = Harness(tmp_path / "b", runner_kwargs={"cuda_available": False})
    done2 = h2.run(monkeypatch)
    assert done2["status"] == "failed" and "cu128" in (h2.root / "evidence" / drv.FAILURE_FILE).read_text()
    h3 = Harness(tmp_path / "c", free_disk_gb=100.0)
    assert h3.run(monkeypatch)["status"] == "failed"
    h4 = Harness(tmp_path / "d", cfg={"disk_gate": "warn"}, free_disk_gb=100.0)
    done4 = h4.run(monkeypatch)
    assert done4["status"] == "complete" and any("free disk" in n for n in done4["notes"])


def _local_snapshot_dirs(tmp_path: Path, keys) -> dict[str, str]:
    local = {}
    for key in keys:
        d = tmp_path / "local" / key.replace(":", "_")
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"\0")
        (d / "config.json").write_text("{}")
        local[key] = str(d)
    return local


def test_driver_uses_local_snapshots_and_corpus_fallback(tmp_path, monkeypatch):
    local = _local_snapshot_dirs(tmp_path, ("pt", "it", "mid:charter", "mid:coin", "mid:control"))
    prefix = drv.DriverConfig().checkpoint_prefix
    listing = [f"{prefix.format(arm=arm)}/model.safetensors" for arm in ARMS] + ["gemma3_27b_190m/charter/data/release/releases/dispatch-final-v2-worked/release/charter/corpus.jsonl", "gemma3_27b_190m/coin/data/release/releases/dispatch-final-v2-worked/release/coin/corpus.jsonl"]
    h = Harness(tmp_path, cfg={"local_snapshots": local}, listing=listing)
    done = h.run(monkeypatch)
    assert done["status"] == "complete" and not h.downloads
    assert any("dispatch-final-v2-worked" in n for n in done["notes"])


def test_driver_reads_bootstrap_models_json_and_downloads_only_what_is_missing(tmp_path, monkeypatch):
    local = _local_snapshot_dirs(tmp_path, ("pt", "it", "mid:charter", "mid:coin"))
    h = Harness(tmp_path)
    models = {"pt": {"hf_id": "unsloth/gemma-3-27b-pt", "path": local["pt"]}, "it": {"hf_id": "google/gemma-3-27b-it", "path": local["it"]}, "mid_charter": {"path": local["mid:charter"]}, "mid_coin": {"path": local["mid:coin"]}, "mid_control": {"path": str(tmp_path / "nope")}}
    (h.root / "evidence").mkdir(parents=True, exist_ok=True)
    (h.root / "evidence" / drv.MODELS_FILE).write_text(json.dumps(models))
    done = h.run(monkeypatch)
    assert done["status"] == "complete"
    assert [d[0] for d in h.downloads] == ["arcadia-impact/scimt-dispatch-final-v1"]  # only the missing control checkpoint
    inputs = json.loads((h.root / "evidence" / drv.INPUTS_FILE).read_text())
    assert inputs["snapshots"]["pt"] == local["pt"] and inputs["snapshots"]["mid:coin"] == local["mid:coin"] and inputs["snapshots"]["mid:control"].endswith("checkpoint-1449")
    ex_cfg = json.loads(Path(h.jobs("extract__charter")[0].env[ex_mod.CONFIG_ENV]).read_text())
    assert ex_cfg["pt_snapshot"] == local["pt"] and ex_cfg["mid_snapshot"] == local["mid:charter"]
    assert h.jobs("score__lam0")[0].env["HF_HOME"] == "/workspace/hf"


def test_stage_for_upload_can_include_adapter_weights_only_on_request(tmp_path):
    paths = drv.Paths(root=tmp_path, repo_root=REPO_ROOT, hf_home=tmp_path / "hf")
    rank_dir = paths.adapters / "coin" / "r16"
    rank_dir.mkdir(parents=True)
    (rank_dir / common.ADAPTER_WEIGHTS).write_bytes(b"\0")
    (rank_dir / common.ADAPTER_CONFIG).write_text("{}")
    (paths.adapters / ".hub_cache").mkdir()
    (paths.adapters / ".hub_cache" / "x.json").write_text("{}")
    without = drv.stage_for_upload(paths, tmp_path / "s1")
    assert {f["path"] for f in without["files"]} == {"adapters/coin/r16/adapter_config.json"}
    with_weights = drv.stage_for_upload(paths, tmp_path / "s2", upload_adapters=True)
    assert {f["path"] for f in with_weights["files"]} == {"adapters/coin/r16/adapter_config.json", "adapters/coin/r16/adapter_model.safetensors"}


# ============================================================ torch parts
try:
    import safetensors  # noqa: F401
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # lean venv: the tensor tests skip, everything above still runs
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch + safetensors not installed (lean venv)")

if HAS_TORCH:

    class TinyNorm(nn.Module):
        """Gemma-style RMSNorm: ``x_normed * (1 + w)``."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(dim))

        def forward(self, x):
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            return normed * (1.0 + self.weight)

    class TinyLayer(nn.Module):
        def __init__(self, hidden: int, attn: int, inter: int) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(hidden, attn, bias=False)
            self.self_attn.k_proj = nn.Linear(hidden, attn, bias=False)
            self.self_attn.v_proj = nn.Linear(hidden, attn, bias=False)
            self.self_attn.o_proj = nn.Linear(attn, hidden, bias=False)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)
            self.input_layernorm = TinyNorm(hidden)
            self.post_attention_layernorm = TinyNorm(hidden)
            self.pre_feedforward_layernorm = TinyNorm(hidden)
            self.post_feedforward_layernorm = TinyNorm(hidden)

        def forward(self, x):
            h = self.input_layernorm(x)
            a = self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(h)) * torch.tanh(self.self_attn.k_proj(h)) + self.self_attn.v_proj(h))
            x = x + self.post_attention_layernorm(a)
            h2 = self.pre_feedforward_layernorm(x)
            m = self.mlp.down_proj(F.gelu(self.mlp.gate_proj(h2)) * self.mlp.up_proj(h2))
            return x + self.post_feedforward_layernorm(m)

    class TinyLM(nn.Module):
        """Gemma3ForConditionalGeneration-shaped module names on a tiny text model."""

        def __init__(self, *, vocab: int = 20, hidden: int = 8, attn: int = 6, inter: int = 12, layers: int = 2) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(vocab, hidden)
            self.model.language_model.layers = nn.ModuleList([TinyLayer(hidden, attn, inter) for _ in range(layers)])
            self.model.language_model.norm = TinyNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = SimpleNamespace(model_type="tiny", tie_word_embeddings=False, to_dict=lambda: {"model_type": "tiny"})

        def forward(self, input_ids=None, **kwargs):
            x = self.model.language_model.embed_tokens(input_ids)
            for layer in self.model.language_model.layers:
                x = layer(x)
            return SimpleNamespace(logits=self.lm_head(self.model.language_model.norm(x)))

        def prepare_inputs_for_generation(self, *args, **kwargs):  # PEFT's CAUSAL_LM wrapper looks this up
            return kwargs


def _seeded(seed: int = 0):
    torch.manual_seed(seed)
    model = TinyLM()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.7).add_(0.05 * torch.randn_like(parameter))
    return model


def _random_low_rank(out: int, inp: int, rank: int, scale: float = 0.05):
    return scale * (torch.randn(out, rank) @ torch.randn(rank, inp))


@needs_torch
def test_svd_lora_factors_round_trip_energy_and_methods():
    torch.manual_seed(1)
    delta = _random_low_rank(12, 10, 8)
    fro2 = float((delta**2).sum())
    for method in ("full", "gram", "lowrank"):
        factors = common.svd_lora_factors(delta, [2, 4, 8], method=method, lowrank_extra=2)
        A, B = factors[8]
        assert A.shape == (8, 10) and B.shape == (12, 8)
        assert torch.allclose(B @ A, delta, atol=1e-5), method
        U, S, Vh = common.thin_svd(delta, method, q=10)
        assert common.captured_energy(S, 8, fro2) == pytest.approx(1.0, abs=1e-5), method
        energies = [common.captured_energy(S, r, fro2) for r in (2, 4, 8)]
        assert energies == sorted(energies) and energies[0] < 1.0
        # the rank-2 factors are the best rank-2 approximation: residual energy = 1 - E2
        A2, B2 = factors[2]
        residual = float(((delta - B2 @ A2) ** 2).sum()) / fro2
        assert residual == pytest.approx(1.0 - energies[0], abs=1e-4), method
    # full-rank matrix: full and gram agree on all singular values; wide matrices work too
    wide = torch.randn(6, 15)
    S_full = common.thin_svd(wide, "full")[1]
    S_gram = common.thin_svd(wide, "gram")[1]
    assert torch.allclose(S_full, S_gram, atol=1e-4)
    U, S, Vh = common.thin_svd(wide, "gram")
    assert torch.allclose(U @ torch.diag(S) @ Vh, wide, atol=1e-4)
    with pytest.raises(ValueError):
        common.thin_svd(wide, "magic")
    W = torch.randn(12, 10)
    A, B = common.svd_lora_factors(delta, [8])[8]
    assert torch.allclose(common.merge_lora(W, A, B, 1.0), W + delta, atol=1e-5)
    assert torch.allclose(common.merge_lora(W, A, B, 0.0), W)
    x = torch.randn(5, 10)
    assert torch.allclose(common.lora_apply(x, A, B), x @ delta.T, atol=1e-5)


def _write_lora(directory: Path, model: TinyLM, r: int, arm: str, seed: int) -> dict[str, torch.Tensor]:
    """Random rank-r deltas on every covered linear + random norm deltas; returns the dense deltas by parameter name."""
    torch.manual_seed(seed)
    covered = common.CoveredModel(model)
    modules, dense = {}, {}
    for path, module in covered.linears.items():
        out, inp = module.weight.shape
        delta = _random_low_rank(out, inp, r)
        A, B = common.svd_lora_factors(delta, [r])[r]
        modules[path] = (A, B)
        dense[path + ".weight"] = (B @ A).detach()
    norm_deltas = {key: 0.02 * torch.randn_like(p) for key, p in covered.norms.items()}
    dense.update(norm_deltas)
    common.write_lora_adapter(directory, r, modules, norm_deltas, config=common.peft_adapter_config(r, base_model="tiny"), manifest={"arm": arm, "rank": r})
    return dense


def _write_full(directory: Path, model: TinyLM, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    covered = common.CoveredModel(model)
    writer = common.FullDeltaWriter(directory, shard_bytes=400)  # tiny shards -> several files
    dense = {}
    for path, module in covered.linears.items():
        delta = 0.05 * torch.randn_like(module.weight)
        writer.add(path, delta)
        dense[path + ".weight"] = delta
    writer.close(extra={"arm": "control"})
    norm_deltas = {key: 0.02 * torch.randn_like(p) for key, p in covered.norms.items()}
    dense.update(norm_deltas)
    from safetensors.torch import save_file

    save_file(norm_deltas, str(directory / common.NORM_DELTA_FILE))
    common.write_json(directory / common.MANIFEST_FILE, {"arm": "control", "kind": "full"})
    return dense


def _batch(seq: int = 7, vocab: int = 20, seed: int = 3):
    from scimt.data_attribution.losses import TokenizedBatch

    torch.manual_seed(seed)
    ids = torch.randint(0, vocab, (1, seq), dtype=torch.int64)
    mask = torch.zeros((1, seq), dtype=torch.bool)
    mask[0, 3:] = True  # "assistant span" = last positions
    return TokenizedBatch(ids, torch.tensor([0]), mask)


def _functional_dl_dlam(model: TinyLM, batch, dense: dict[str, torch.Tensor]) -> tuple[float, float]:
    """Exact dL/dlam via a scalar leaf scaling the dense delta inside functional_call."""
    lam = torch.zeros((), dtype=torch.float32, requires_grad=True)
    params = {name: p.detach() for name, p in model.named_parameters()}
    merged = {name: (p + lam * dense[name] if name in dense else p) for name, p in params.items()}
    logits = torch.func.functional_call(model, merged, (), {"input_ids": batch.input_ids}).logits
    ids = batch.input_ids
    positions = batch.target_mask[0].nonzero()[:, 0]
    loss = F.cross_entropy(logits[0, positions - 1], ids[0, positions], reduction="sum")
    (g,) = torch.autograd.grad(loss, lam)
    return float(loss.detach()), float(g.detach())


@needs_torch
def test_hook_engine_matches_autograd_dl_dlambda_at_lambda_0_and_1(tmp_path):
    from scimt.data_attribution.losses import CausalLMLossAdapter

    model = _seeded(0)
    dense = {
        "charter__lam0_r2__all": _write_lora(tmp_path / "charter" / "r2", model, 2, "charter", seed=11),
        "coin__lam0_r3__all": _write_lora(tmp_path / "coin" / "r3", model, 3, "coin", seed=12),
        "control__lam0_full__all": _write_full(tmp_path / "control" / "full", model, seed=13),
    }
    specs = [
        sc_mod.DeltaSpec("charter__lam0_r2__all", "lora", str(tmp_path / "charter" / "r2"), "charter"),
        sc_mod.DeltaSpec("coin__lam0_r3__all", "lora", str(tmp_path / "coin" / "r3"), "coin"),
        sc_mod.DeltaSpec("control__lam0_full__all", "full", str(tmp_path / "control" / "full"), "control"),
    ]
    covered = common.CoveredModel(model)
    assert len(covered.linears) == 14 and len(covered.norms) == 9 and covered.layers == 2
    resident = sc_mod.ResidentDeltas(specs, device="cpu", dtype="float32", module_paths=list(covered.linears), norm_keys=list(covered.norms))
    summary = resident.load()
    assert summary["n_lora_modules"] == 14 and summary["n_full_modules"] == 14 and summary["norm_elements"] == 9 * 8
    assert len((tmp_path / "control" / "full").glob("delta-*.safetensors").__iter__().__next__().name) > 0
    assert len(list((tmp_path / "control" / "full").glob("delta-*.safetensors"))) > 1  # sharded
    batch = _batch()
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum", device="cpu")
    engine = sc_mod.LambdaGradEngine(model, covered, resident, delta_device="cpu")
    # --- lambda = 0 --------------------------------------------------------------------
    hook = engine.score(batch, adapter)
    assert hook.fired == 14
    oracle = engine.oracle(batch, adapter)
    for k, name in enumerate(resident.names):
        loss_f, g_f = _functional_dl_dlam(model, batch, dense[name])
        assert math.isclose(hook.loss, loss_f, rel_tol=1e-5), name
        assert math.isclose(hook.dots[k], g_f, rel_tol=1e-4, abs_tol=1e-6), (name, hook.dots[k], g_f)
        assert math.isclose(oracle.dots[k], g_f, rel_tol=1e-4, abs_tol=1e-6), (name, oracle.dots[k], g_f)
    assert sc_mod.compare_oracle(resident.names, hook.dots, oracle.dots, rel_tol=1e-4)["passed"]
    # --- lambda = 1: merge the charter graft, score again at the merged point ------------
    graft = common.read_lora_adapter(tmp_path / "charter" / "r2", device="cpu")
    snapshot = covered.snapshot()
    counts = covered.merge_lora_adapter(graft, 1.0)
    assert counts == {"linears": 14, "norms": 9}
    merged_weight = covered.linears["model.language_model.layers.0.mlp.up_proj"].weight.detach()
    assert torch.allclose(merged_weight, snapshot["model.language_model.layers.0.mlp.up_proj"] + dense["charter__lam0_r2__all"]["model.language_model.layers.0.mlp.up_proj.weight"], atol=1e-6)
    hook1 = engine.score(batch, adapter)
    assert not math.isclose(hook1.loss, hook.loss, rel_tol=1e-6)  # the graft moved the loss
    for k, name in enumerate(resident.names):
        loss_f, g_f = _functional_dl_dlam(model, batch, dense[name])  # gradient at the merged point
        assert math.isclose(hook1.loss, loss_f, rel_tol=1e-5)
        assert math.isclose(hook1.dots[k], g_f, rel_tol=1e-4, abs_tol=1e-6), (name, hook1.dots[k], g_f)
    # --- restore is exact -----------------------------------------------------------------
    covered.restore(snapshot)
    for path, module in covered.linears.items():
        assert torch.equal(module.weight.detach(), snapshot[path])
    hook_again = engine.score(batch, adapter)
    assert math.isclose(hook_again.loss, hook.loss, rel_tol=1e-6) and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(hook_again.dots, hook.dots, strict=True))
    engine.remove()
    # --- score_record end to end on the engine output ------------------------------------
    meta = sc_mod.RowMeta(0, "coin:e0", "coin", "e0", "priority", ())
    record = sc_mod.score_record(meta, n_tokens=7, n_target_tokens=4, loss=hook.loss, g=dict(zip(resident.names, hook.dots, strict=True)), seconds=0.1, mode="lam0", pass_name="lam0", repeat=None, graft=None)
    assert record["scores"]["charter__lam0_r2__all"] == -hook.dots[0]


@needs_torch
def test_adapter_export_matches_peft_naming_and_loads_back(tmp_path):
    from safetensors import safe_open

    model = _seeded(1)
    dense = _write_lora(tmp_path / "coin" / "r2", model, 2, "coin", seed=5)
    with safe_open(str(tmp_path / "coin" / "r2" / common.ADAPTER_WEIGHTS), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        assert keys[0] == "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
        assert all(k.startswith("base_model.model.model.language_model.layers.") and k.endswith((".lora_A.weight", ".lora_B.weight")) for k in keys)
        assert len(keys) == 28
        A = handle.get_tensor("base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_A.weight")
        B = handle.get_tensor("base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_B.weight")
        assert A.shape == (2, 8) and B.shape == (12, 2)  # lora_A: r x in, lora_B: out x r
    config = json.loads((tmp_path / "coin" / "r2" / common.ADAPTER_CONFIG).read_text())
    assert config["peft_type"] == "LORA" and config["r"] == 2 and config["lora_alpha"] == 2 and config["task_type"] == "CAUSAL_LM"
    adapter = common.read_lora_adapter(tmp_path / "coin" / "r2", device="cpu")
    assert adapter.r == 2 and set(adapter.modules) == set(common.CoveredModel(model).linears) and len(adapter.norm_deltas) == 9
    A, B = adapter.modules["model.language_model.layers.1.mlp.gate_proj"]
    assert torch.allclose(B @ A, dense["model.language_model.layers.1.mlp.gate_proj.weight"], atol=1e-6)
    # a mismatched lora_alpha is refused (scaling must be 1)
    bad = json.loads((tmp_path / "coin" / "r2" / common.ADAPTER_CONFIG).read_text())
    bad["lora_alpha"] = 4
    (tmp_path / "coin" / "r2" / common.ADAPTER_CONFIG).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="lora_alpha"):
        common.read_lora_adapter(tmp_path / "coin" / "r2")
    (tmp_path / "coin" / "r2" / common.ADAPTER_CONFIG).write_text(json.dumps(config))
    peft = pytest.importorskip("peft")
    base = _seeded(1)
    reference = _seeded(1)
    common.CoveredModel(reference).merge_lora_adapter(adapter, 1.0)
    peft_model = peft.PeftModel.from_pretrained(base, str(tmp_path / "coin" / "r2"))
    wrapped = dict(peft_model.named_modules())
    lora_module = next(m for n, m in wrapped.items() if n.endswith("layers.1.mlp.gate_proj") and hasattr(m, "lora_A"))
    with torch.no_grad():
        x = torch.randn(3, 8)
        expected = x @ reference.model.language_model.layers[1].mlp.gate_proj.weight.T
        assert torch.allclose(lora_module(x), expected, atol=1e-5)


@needs_torch
def test_full_delta_shards_round_trip_and_covered_model_merge_full(tmp_path):
    model = _seeded(2)
    dense = _write_full(tmp_path / "full", model, seed=7)
    index = json.loads((tmp_path / "full" / common.FULL_INDEX_FILE).read_text())
    assert len(index["shards"]) > 1 and set(index["weight_map"]) == set(common.CoveredModel(model).linears)
    seen = dict(common.iter_full_delta(tmp_path / "full"))
    assert set(seen) == set(index["weight_map"])
    covered = common.CoveredModel(model)
    snap = covered.snapshot()
    counts = covered.merge_full(common.iter_full_delta(tmp_path / "full"), common.read_norm_deltas(tmp_path / "full"), 1.0)
    assert counts == {"linears": 14, "norms": 9}
    for path, module in covered.linears.items():
        assert torch.allclose(module.weight.detach(), snap[path] + dense[path + ".weight"], atol=1e-6)
    for key, parameter in covered.norms.items():
        assert torch.allclose(parameter.detach(), snap[key] + dense[key], atol=1e-6)
    covered.restore(snap)
    assert all(torch.equal(m.weight.detach(), snap[p]) for p, m in covered.linears.items())


def _write_fake_snapshots(tmp_path: Path, *, layers: int = 2, vision_delta: float = 0.0, dtype=None, norm_delta: float = 0.01, tied_lm_head: bool = False):
    """pt in the OLD (unsloth) key layout, mid in the transformers-5 layout;
    mid = pt + low-rank deltas on the linears, ``norm_delta``-scaled deltas on
    the norms (0 = untouched, as on the real 27B midtrain); ``tied_lm_head``
    adds the materialised ``lm_head.weight`` to mid only (1,248 vs 1,247)."""
    from safetensors.torch import save_file

    dtype = torch.float32 if dtype is None else dtype
    torch.manual_seed(42)
    hidden, attn, inter, vocab = 8, 6, 12, 20
    pt: dict[str, torch.Tensor] = {}
    mid: dict[str, torch.Tensor] = {}
    dense: dict[str, torch.Tensor] = {}
    shapes = {"self_attn.q_proj": (attn, hidden), "self_attn.k_proj": (attn, hidden), "self_attn.v_proj": (attn, hidden), "self_attn.o_proj": (hidden, attn), "mlp.gate_proj": (inter, hidden), "mlp.up_proj": (inter, hidden), "mlp.down_proj": (hidden, inter)}
    for layer in range(layers):
        for module, (out, inp) in shapes.items():
            W = torch.randn(out, inp)
            delta = _random_low_rank(out, inp, 2)
            pt[f"language_model.model.layers.{layer}.{module}.weight"] = W.to(dtype)
            mid[f"model.language_model.layers.{layer}.{module}.weight"] = (W + delta).to(dtype)
            dense[f"model.language_model.layers.{layer}.{module}"] = (W + delta).to(dtype).float() - W.to(dtype).float()
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            w = torch.randn(hidden)
            pt[f"language_model.model.layers.{layer}.{norm}.weight"] = w.to(dtype)
            mid[f"model.language_model.layers.{layer}.{norm}.weight"] = (w + norm_delta * torch.randn(hidden)).to(dtype)
    w = torch.randn(hidden)
    pt["language_model.model.norm.weight"] = w.to(dtype)
    mid["model.language_model.norm.weight"] = (w + norm_delta).to(dtype)
    emb = torch.randn(vocab, hidden)
    pt["language_model.model.embed_tokens.weight"] = emb.to(dtype)
    mid["model.language_model.embed_tokens.weight"] = (emb + 0.1).to(dtype)
    if tied_lm_head:
        mid["lm_head.weight"] = mid["model.language_model.embed_tokens.weight"].clone()
    vision = torch.randn(4, 4)
    pt["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"] = vision.to(dtype)
    mid["model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"] = (vision + vision_delta).to(dtype)
    proj = torch.randn(4, 4)
    pt["multi_modal_projector.mm_input_projection_weight"] = proj.to(dtype)
    mid["model.multi_modal_projector.mm_input_projection_weight"] = proj.to(dtype)
    pt_dir, mid_dir = tmp_path / "pt", tmp_path / "mid"
    pt_dir.mkdir(parents=True)
    mid_dir.mkdir(parents=True)
    # pt in two shards, mid in one
    keys = sorted(pt)
    save_file({k: pt[k] for k in keys[: len(keys) // 2]}, str(pt_dir / "model-00001-of-00002.safetensors"))
    save_file({k: pt[k] for k in keys[len(keys) // 2 :]}, str(pt_dir / "model-00002-of-00002.safetensors"))
    save_file(mid, str(mid_dir / "model.safetensors"))
    return pt_dir, mid_dir, dense


@needs_torch
def test_extract_delta_lora_end_to_end_on_fake_snapshots(tmp_path, capsys):
    pt_dir, mid_dir, dense = _write_fake_snapshots(tmp_path)
    out_dir = tmp_path / "adapters"
    config = ex_mod.ExtractConfig.from_mapping({"pt_snapshot": str(pt_dir), "mid_snapshot": str(mid_dir), "arm": "coin", "out_dir": str(out_dir), "ranks": [1, 2], "device": "cpu", "svd_method": "auto", "svd_time_budget_s": 1000.0, "expected_layers": 2, "factor_dtype": "float32", "full_dtype": "float32", "full_shard_gb": 1e-6})
    summary = ex_mod.run(config)
    assert summary["status"] == "ok" and summary["coverage"]["linears"] == 14 and summary["coverage"]["norms"] == 13 and summary["coverage"]["layers"] == 2
    assert summary["coverage"]["embeddings_excluded"] == ["model.language_model.embed_tokens.weight"] and summary["coverage"]["vision_max_abs_delta"] == 0.0
    assert summary["energy_pooled"]["2"] == pytest.approx(1.0, abs=1e-5) and summary["energy_pooled"]["1"] < 1.0
    arm_dir = out_dir / "coin"
    manifest = json.loads((arm_dir / common.MANIFEST_FILE).read_text())
    assert manifest["status"] == "complete" and manifest["ranks"] == [1, 2] and set(manifest["dirs"]) == {"r1", "r2", "full"}
    stats = json.loads((out_dir / "evidence" / "delta_stats__coin.json").read_text())
    assert len(stats["linears"]) == 14 and all(row["svd_method"] == "full" for row in stats["linears"]) and stats["vision"]["n_tensors"] == 2
    # analysis contract: pooled.captured_energy rank -> fraction; per_module_type type -> {captured_energy per rank, fro_norm_share}
    assert stats["pooled"]["captured_energy"] == summary["captured_energy"] and set(stats["pooled"]["captured_energy"]) == {"1", "2"}
    per_type = stats["per_module_type"]
    assert set(per_type) == set(common.LINEAR_NAMES) and per_type == stats["pooled"]["per_module_type"]
    assert sum(per_type[t]["fro_norm_share"] for t in common.LINEAR_NAMES) == pytest.approx(1.0)
    assert all(set(per_type[t]["captured_energy"]) == {"1", "2"} and per_type[t]["captured_energy"]["2"] == pytest.approx(1.0, abs=1e-5) for t in common.LINEAR_NAMES)
    assert stats["linears"][0]["module_path"] == "model.language_model.layers.0.self_attn.q_proj" and len(stats["linears"][0]["top_singular_values"]) <= 8
    adapter = common.read_lora_adapter(arm_dir / "r2")
    for module_path, (A, B) in adapter.modules.items():
        assert torch.allclose(B @ A, dense[module_path], atol=1e-5), module_path
    assert torch.allclose(adapter.norm_deltas["model.language_model.norm.weight"], torch.full((8,), 0.01), atol=1e-6)
    full = dict(common.iter_full_delta(arm_dir / "full"))
    assert set(full) == set(dense) and all(torch.allclose(full[k], dense[k]) for k in dense)
    assert len(list((arm_dir / "full").glob("delta-*.safetensors"))) > 1
    config_json = json.loads((arm_dir / "r1" / common.ADAPTER_CONFIG).read_text())
    assert config_json["r"] == 1 and config_json["lora_alpha"] == 1 and config_json["target_modules"] == common.PEFT_TARGET_MODULES_REGEX
    receipts = list((out_dir / "evidence").glob("extract_delta_lora__coin__*.json"))
    assert receipts and json.loads(receipts[0].read_text())["status"] == "ok"
    # resume: complete arm dir -> nothing recomputed
    again = ex_mod.run(config)
    assert again["resumed"] is True
    # vision delta -> loud failure (exit 94) unless allowed
    pt2, mid2, _ = _write_fake_snapshots(tmp_path / "v", vision_delta=0.5)
    bad = ex_mod.ExtractConfig.from_mapping({"pt_snapshot": str(pt2), "mid_snapshot": str(mid2), "arm": "coin", "out_dir": str(tmp_path / "v" / "out"), "ranks": [1], "device": "cpu", "expected_layers": 2, "save_full": False})
    with pytest.raises(SystemExit) as exc:
        ex_mod.run(bad)
    assert exc.value.code == ex_mod.VISION_MISMATCH_EXIT
    allowed = ex_mod.ExtractConfig.from_mapping({"pt_snapshot": str(pt2), "mid_snapshot": str(mid2), "arm": "coin", "out_dir": str(tmp_path / "v" / "out2"), "ranks": [1], "device": "cpu", "expected_layers": 2, "save_full": False, "allow_vision_delta": True})
    assert ex_mod.run(allowed)["coverage"]["vision_max_abs_delta"] == pytest.approx(0.5, abs=1e-6)
    assert not ex_mod.arm_complete(tmp_path / "v" / "out2" / "coin", [1, 2], False)  # rank 2 was not extracted
    assert ex_mod.arm_complete(tmp_path / "v" / "out2" / "coin", [1], False)
    # the adapter's module set is exactly the covered linears of a transformers-5-shaped model
    assert set(adapter.modules) == set(common.CoveredModel(_seeded(0)).linears)


@needs_torch
def test_extract_tolerates_tied_lm_head_and_zero_norm_deltas(tmp_path):
    """Real 27B facts: mid has an extra ``lm_head.weight`` (tied embeddings
    materialised), every norm delta is exactly zero, vision delta is zero."""
    pt_dir, mid_dir, dense = _write_fake_snapshots(tmp_path, norm_delta=0.0, tied_lm_head=True)
    out_dir = tmp_path / "adapters"
    config = ex_mod.ExtractConfig.from_mapping({"pt_snapshot": str(pt_dir), "mid_snapshot": str(mid_dir), "arm": "charter", "out_dir": str(out_dir), "ranks": [1, 2], "device": "cpu", "expected_layers": 2, "factor_dtype": "float32", "full_dtype": "float32"})
    summary = ex_mod.run(config)
    assert summary["status"] == "ok"
    cov = summary["coverage"]
    assert cov["asymmetric_keys"] == {"only_pt": [], "only_mid": [{"key": "lm_head.weight", "kind": "embedding"}]}
    assert cov["embeddings_excluded"] == ["model.language_model.embed_tokens.weight"]  # lm_head not in the common set -> excluded, not an error
    assert cov["norm_deltas_all_zero"] is True and cov["norm_deltas_zero"] == 13 and cov["zero_delta_linears"] == []
    # no NaN anywhere in the energy bookkeeping
    assert all(math.isfinite(v) for v in summary["captured_energy"].values()) and summary["delta_norms"]["full"] == pytest.approx(summary["delta_norms"]["r2"], rel=1e-5)
    stats = json.loads((out_dir / "evidence" / "delta_stats__charter.json").read_text())
    assert stats["norms"]["n_zero"] == 13 and stats["norms"]["all_zero"] is True and stats["norms"]["delta_fro"] == 0.0
    assert all(math.isfinite(v) for t in stats["per_module_type"].values() for v in t["captured_energy"].values())
    # the zero norm deltas are still exported with the full key set (schema stability)
    adapter = common.read_lora_adapter(out_dir / "charter" / "r2")
    assert len(adapter.norm_deltas) == 13 and all(not t.any() for t in adapter.norm_deltas.values())
    norm_full = common.read_norm_deltas(out_dir / "charter" / "full")
    assert set(norm_full) == set(adapter.norm_deltas)
    # a covered linear present on one side only is still a pair mismatch (exit 95)
    from safetensors.torch import load_file, save_file

    broken = tmp_path / "broken_mid"
    broken.mkdir()
    tensors = load_file(str(mid_dir / "model.safetensors"))
    tensors.pop("model.language_model.layers.1.mlp.down_proj.weight")
    save_file(tensors, str(broken / "model.safetensors"))
    bad = ex_mod.ExtractConfig.from_mapping({"pt_snapshot": str(pt_dir), "mid_snapshot": str(broken), "arm": "charter", "out_dir": str(tmp_path / "broken_out"), "ranks": [1], "device": "cpu", "expected_layers": 2, "save_full": False})
    with pytest.raises(SystemExit) as exc:
        ex_mod.run(bad)
    assert exc.value.code == ex_mod.PAIR_MISMATCH_EXIT


@needs_torch
def test_resident_deltas_require_every_covered_linear_and_skip_zero_norm_term(tmp_path):
    from scimt.data_attribution.losses import CausalLMLossAdapter

    model = _seeded(4)
    covered = common.CoveredModel(model)
    dense = _write_lora(tmp_path / "coin" / "r2", model, 2, "coin", seed=21)
    # 1) an adapter missing one covered module is refused by the scorer and by strict merges
    partial = common.read_lora_adapter(tmp_path / "coin" / "r2")
    modules = dict(partial.modules)
    modules.pop("model.language_model.layers.1.self_attn.v_proj")
    common.write_lora_adapter(tmp_path / "partial" / "r2", 2, modules, partial.norm_deltas, config=common.peft_adapter_config(2, base_model="tiny"), manifest={"arm": "coin", "rank": 2})
    spec = sc_mod.DeltaSpec("coin__lam0_r2__all", "lora", str(tmp_path / "partial" / "r2"), "coin")
    with pytest.raises(KeyError, match="lacks 1 of the model's 14 covered linears"):
        sc_mod.ResidentDeltas([spec], device="cpu", dtype="float32", module_paths=list(covered.linears), norm_keys=list(covered.norms)).load()
    with pytest.raises(KeyError, match="covered linears have no delta"):
        covered.merge_lora_adapter(common.read_lora_adapter(tmp_path / "partial" / "r2"), 1.0)
    assert covered.merge_lora_adapter(common.read_lora_adapter(tmp_path / "partial" / "r2"), 0.0, strict=False) == {"linears": 13, "norms": 9}
    # norm deltas lacking a model norm are refused too
    norm_deltas = dict(partial.norm_deltas)
    norm_deltas.pop("model.language_model.norm.weight")
    common.write_lora_adapter(tmp_path / "nonorm" / "r2", 2, partial.modules, norm_deltas, config=common.peft_adapter_config(2, base_model="tiny"), manifest={"arm": "coin", "rank": 2})
    with pytest.raises(KeyError, match="norm deltas lack"):
        covered.merge_lora_adapter(common.read_lora_adapter(tmp_path / "nonorm" / "r2"), 1.0)
    # 2) all-zero norm deltas: exported, loaded, and the norm term is a no-op that still matches autograd
    zeros = {key: torch.zeros_like(t) for key, t in partial.norm_deltas.items()}
    common.write_lora_adapter(tmp_path / "zero" / "r2", 2, partial.modules, zeros, config=common.peft_adapter_config(2, base_model="tiny"), manifest={"arm": "coin", "rank": 2})
    spec0 = sc_mod.DeltaSpec("coin__lam0_r2__all", "lora", str(tmp_path / "zero" / "r2"), "coin")
    resident = sc_mod.ResidentDeltas([spec0], device="cpu", dtype="float32", module_paths=list(covered.linears), norm_keys=list(covered.norms))
    summary = resident.load()
    assert summary["norm_all_zero"] is True and summary["norm_delta_fro"] == {"coin__lam0_r2__all": 0.0} and summary["norm_elements"] == 9 * 8
    dense0 = {k: v for k, v in dense.items() if k.endswith(("_proj.weight",))}  # linears only: the norm part of the graft is zero
    batch = _batch()
    engine = sc_mod.LambdaGradEngine(model, covered, resident, delta_device="cpu")
    hook = engine.score(batch, CausalLMLossAdapter(model, reduction="per_sequence_sum", device="cpu"))
    loss_f, g_f = _functional_dl_dlam(model, batch, dense0)
    assert math.isclose(hook.loss, loss_f, rel_tol=1e-5) and math.isclose(hook.dots[0], g_f, rel_tol=1e-4, abs_tol=1e-6)
    engine.remove()


def test_svd_policy_falls_back_per_shape_after_a_slow_full_svd():
    policy = ex_mod.SvdPolicy("auto", budget_s=1.0)
    assert policy.method_for((100, 50)) == "full"
    policy.record((100, 50), "full", 5.0, "m0")
    assert policy.method_for((100, 50)) == "lowrank" and policy.method_for((50, 100)) == "full"
    assert policy.summary()["slow_shapes"] == [[100, 50]]
    fixed = ex_mod.SvdPolicy("gram", budget_s=1.0)
    fixed.record((100, 50), "gram", 50.0, "m0")
    assert fixed.method_for((100, 50)) == "gram"


@needs_torch
def test_gates_docs_and_losses_on_the_tiny_model(tmp_path):
    model = _seeded(3)
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text("".join(json.dumps({"text": "abc" * (i + 1), "doc_id": f"d{i}"}) + "\n" for i in range(3)) + json.dumps({"text": ""}) + "\n")

    def tokenizer(text, add_special_tokens=True):
        return {"input_ids": [2] + [ord(c) % 20 for c in text]}

    docs, meta = gates_mod.tokenize_docs(docs_path, tokenizer, sequence_length=5, n_docs=None)
    assert meta["n_docs"] == 3 and meta["skipped"] == 1 and meta["truncated"] == 2 and all(len(d.ids) <= 5 for d in docs)
    results = gates_mod.evaluate_docsets(model, {"docs": docs}, device="cpu", chunk_positions=2, label="pt")
    entry = results["docs"]
    assert entry["n_docs"] == 3 and entry["tokens"] == sum(len(d.ids) - 1 for d in docs) and math.isfinite(entry["mean_ce"])
    assert entry["mean_ce"] == pytest.approx(entry["sum_loss"] / entry["tokens"])
    # merging a delta changes the loss; restoring brings it back exactly
    covered = common.CoveredModel(model)
    snap = covered.snapshot()
    adapter_dir = tmp_path / "coin" / "r2"
    _write_lora(adapter_dir, model, 2, "coin", seed=9)
    covered.merge_lora_adapter(common.read_lora_adapter(adapter_dir), 1.0)
    merged = gates_mod.evaluate_docsets(model, {"docs": docs}, device="cpu", chunk_positions=2, label="pt+coin_r2")["docs"]["mean_ce"]
    assert merged != pytest.approx(entry["mean_ce"], abs=1e-9)
    covered.restore(snap)
    assert gates_mod.evaluate_docsets(model, {"docs": docs}, device="cpu", chunk_positions=2, label="pt")["docs"]["mean_ce"] == pytest.approx(entry["mean_ce"])
    assert gates_mod.variant_name("pt", "coin", 256) == "pt+coin_r256" and gates_mod.variant_name("pt", "coin", "full") == "pt+coin_full" and gates_mod.variant_name("mid", "coin") == "mid_coin"


def test_cache_evictor_source_is_the_v1_copy_with_configurable_roots():
    text = (POD / "cache_evictor.py").read_text()
    assert "EVICT_ROOTS" in text and "POSIX_FADV_DONTNEED" in text and "memory.current" in text
    for name in ("extract_delta_lora.py", "gates.py", "score_lambda_grad.py", "run_all.py", "common.py"):
        assert "import argparse" not in (POD / name).read_text(), name
