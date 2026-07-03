"""The endpoint graph, expressed as per-pod *plans* (ordered op lists).

Design principle: every dependent chain runs on ONE pod (checkpoints chain as
local dirs — no inter-pod transfer), and the only cross-pod dependency is
``msm_<v>_base``, which the ``value-<v>-light`` plan persists to GCS and the
``value-<v>-ins`` plan restores. Six pods total for phase 1, longest ~9h,
all parallelizable:

    value-america-light   MSM installs + confound evals + A1 + A2   (~5h)
    value-afford-light            "                "                (~5h)
    value-america-ins     restore msm -> A3 (interleaved) + A3.5    (~9h)
    value-afford-ins              "                "                (~9h)
    controls-light        base / instruct evals + ctl for A1/A2     (~2h)
    controls-ins          ctl for A3 + A3.5 (INS on raw base)       (~9h)

Ops:
    {"op": "train", "model": ref, "data": file, "format": "text"|"chat",
     "lr", "epochs", "seq", "save": name, "persist": bool}
    {"op": "delta",   "msm": name, "save": name}      # + (instruct - base)
    {"op": "eval",    "model": ref, "tag": endpoint}
    {"op": "restore", "name": name}                   # GCS -> pod
    {"op": "drop",    "name": name}                   # reclaim pod disk

``ref`` = an HF id (contains "/") or the name of a previously saved/restored
checkpoint. Stage hyperparameters follow the recipe validated by
``msm_fig2_repro`` (MSM: ~1M doc tokens x 1 epoch; AFT: 3 epochs), LoRA r64
uniform (see spec.md for why method is held fixed).
"""
from __future__ import annotations

BASE = "Qwen/Qwen3-14B-Base"
INSTRUCT = "Qwen/Qwen3-14B"
SMOKE_BASE = "Qwen/Qwen3-1.7B-Base"
SMOKE_INSTRUCT = "Qwen/Qwen3-1.7B"

GCS_PREFIX = ("gcs:alignment-team-general-storage/daniel/jarvis/experiments/"
              "science-of-midtraining/msm-stage-comparison/ckpts")

MSM = dict(format="text", lr=1e-4, epochs=1.0, seq=2048)
INS = dict(format="chat", lr=1e-4, epochs=1.0, seq=2048)
AFT = dict(format="chat", lr=1e-4, epochs=3.0, seq=2048)

VALUES = ("america", "afford")


def _train(model, data, hp, save, persist=False):
    return {"op": "train", "model": model, "data": data, **hp,
            "save": save, "persist": persist}


def _eval(model, tag=None):
    return {"op": "eval", "model": model, "tag": tag or model}


def value_light(v: str) -> dict:
    """MSM installs (persisted), confound endpoints, arms A1 + A2."""
    return {"hours": 8, "payload": "eval_payload.json", "ops": [
        _train(BASE, f"msm_{v}.jsonl", MSM, f"msm_{v}_base", persist=True),
        _eval(f"msm_{v}_base"),
        {"op": "delta", "msm": f"msm_{v}_base", "save": f"msm_{v}_base_delta"},
        _eval(f"msm_{v}_base_delta"),
        _train(f"msm_{v}_base_delta", "aft_train.jsonl", AFT, f"a1_{v}", persist=True),
        _eval(f"a1_{v}"),
        {"op": "drop", "name": f"msm_{v}_base_delta"},
        _train(INSTRUCT, f"msm_{v}.jsonl", MSM, f"msm_{v}_inst", persist=True),
        _eval(f"msm_{v}_inst"),
        _train(f"msm_{v}_inst", "aft_train.jsonl", AFT, f"a2_{v}", persist=True),
        _eval(f"a2_{v}"),
    ]}


def value_ins(v: str) -> dict:
    """The INS-heavy arms A3 (interleaved) and A3.5 (sequential), from the
    persisted MSM install."""
    return {"hours": 14, "payload": "eval_payload.json", "ops": [
        {"op": "restore", "name": f"msm_{v}_base"},
        _train(f"msm_{v}_base", "interleaved.jsonl", INS, f"a3_{v}", persist=True),
        _eval(f"a3_{v}"),
        {"op": "drop", "name": f"a3_{v}"},
        _train(f"msm_{v}_base", "tulu25k.jsonl", INS, f"ins_{v}", persist=True),
        _eval(f"ins_{v}"),
        _train(f"ins_{v}", "aft_train.jsonl", AFT, f"a35_{v}", persist=True),
        _eval(f"a35_{v}"),
    ]}


def controls_light() -> dict:
    return {"hours": 5, "payload": "eval_payload.json", "ops": [
        _eval(BASE, "raw_base"),
        _eval(INSTRUCT, "raw_instruct"),
        _train(INSTRUCT, "aft_train.jsonl", AFT, "ctl_a12", persist=True),
        _eval("ctl_a12"),
    ]}


def controls_ins() -> dict:
    return {"hours": 14, "payload": "eval_payload.json", "ops": [
        _train(BASE, "interleaved.jsonl", INS, "ctl_a3", persist=True),
        _eval("ctl_a3"),
        {"op": "drop", "name": "ctl_a3"},
        _train(BASE, "tulu25k.jsonl", INS, "ins_ctl", persist=True),
        _eval("ins_ctl"),
        _train("ins_ctl", "aft_train.jsonl", AFT, "ctl_a35", persist=True),
        _eval("ctl_a35"),
    ]}


def smoke() -> dict:
    """Phase-0 plumbing check on the 1.7B pair: every op kind (train text,
    train chat, delta, eval, persist, restore, drop) at --max-steps 3 budgets.
    The 14B memory profile is already proven by lora_artifact_robustness."""
    st = {"max_steps": 3}
    return {"hours": 3, "payload": "eval_payload_smoke.json",
            "base": SMOKE_BASE, "instruct": SMOKE_INSTRUCT, "ops": [
        _train(SMOKE_BASE, "msm_america.jsonl", {**MSM, **st}, "smk_msm", persist=True),
        _eval("smk_msm"),
        {"op": "delta", "msm": "smk_msm", "save": "smk_delta"},
        _eval("smk_delta"),
        _train("smk_delta", "aft_train.jsonl", {**AFT, **st}, "smk_a1"),
        _eval("smk_a1"),
        {"op": "drop", "name": "smk_delta"},
        {"op": "drop", "name": "smk_msm"},
        {"op": "restore", "name": "smk_msm"},
        _train("smk_msm", "interleaved_smoke.jsonl", {**INS, **st}, "smk_a3"),
        _eval("smk_a3"),
    ]}


def get_plan(name: str) -> dict:
    plans = {"controls-light": controls_light(), "controls-ins": controls_ins(),
             "smoke": smoke()}
    for v in VALUES:
        plans[f"value-{v}-light"] = value_light(v)
        plans[f"value-{v}-ins"] = value_ins(v)
    if name not in plans:
        raise KeyError(f"unknown plan {name!r}; have {sorted(plans)}")
    return plans[name]


ALL_PHASE1 = (["controls-light", "controls-ins"]
              + [f"value-{v}-{k}" for v in VALUES for k in ("light", "ins")])
