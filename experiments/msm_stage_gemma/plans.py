"""The msm_stage_gemma endpoint graph as per-pod plans (spec.md v2 § Design).

Same op vocabulary as ``msm_stage_comparison`` (train / delta / eval / restore
/ drop, checkpoints chaining as local dirs, GCS persist/restore of named
checkpoints) so the driver is a light adaptation. Cross-pod dependencies:

    stage-shared        ourI (=B→INS) + REF/AFT controls          [persists ourI]
    value-<v>-light     MSM_v(B/I) installs, arms 1a + 2          [persists msm_<v>_base]
    value-<v>-ins       arms 2' + 3 (needs ourI + msm_<v>_base)   [run AFTER the two above]

plus phase-0 plans: ``pilot-baserates`` (gate 5), ``pilot-dissociation``
(gate 6), ``smoke`` (gate 9).

Endpoint naming: ``e*`` = pre-AFT tier, ``a*`` = +AFT tier; controls ``ctl_*``
are value-independent and live on the shared pod. Every persisted endpoint is
kept for phase 2 (durability follow-ups).

Model pair is env-overridable (MSG_BASE / MSG_INSTRUCT) so the pre-registered
substrate fallback (spec kill criterion i) is a config change, not an edit.
"""
from __future__ import annotations

import os

BASE = os.environ.get("MSG_BASE", "google/gemma-4-12B")
INSTRUCT = os.environ.get("MSG_INSTRUCT", "google/gemma-4-12B-it")
# smoke pair: overridable once phase-0 gate 9 picks the smallest gemma-4
# base+it pair; defaults to the 12B pair at --max-steps budgets.
SMOKE_BASE = os.environ.get("MSG_SMOKE_BASE", BASE)
SMOKE_INSTRUCT = os.environ.get("MSG_SMOKE_INSTRUCT", INSTRUCT)

CHAT_TEMPLATE = "gemma"   # passed to pod train/eval for template + masking markers

GCS_SUBDIR = "msm-stage-gemma"


def gcs_prefix() -> str:
    """Checkpoint persistence root for THIS experiment (no personal default)."""
    root = os.environ.get("SCIMT_GCS_PREFIX", "").rstrip("/")
    return f"{root}/{GCS_SUBDIR}" if root else ""


# Paper App. B.4 recipe: LoRA r64/α128 (driver passes --method lora:r64), 1
# epoch, lr 1e-4, cosine, 5% warmup, wd 0.01 (scheduler set in pod/train.py).
MSM = dict(format="text", lr=1e-4, epochs=1.0, seq=4096)
CHAT = dict(format="chat", lr=1e-4, epochs=1.0, seq=4096)

VALUES = ("afford", "america")


def _train(model, data, hp, save, persist=True):
    return {"op": "train", "model": model, "data": data, **hp,
            "save": save, "persist": persist}


def _eval(model, tag=None):
    return {"op": "eval", "model": model, "tag": tag or model}


def shared() -> dict:
    """Value-independent controls: ourI (=B→INS), REF controls, control AFTs.

    NOTE (spec v2.1 / harness-review finding 7): arm 3 now carries the REF
    stage too, so its A-tier control is ``ctl_a2p`` (ourI→REF→AFT) and its
    E-tier control is ``ouri_ref`` — the former ``ctl_a3`` (ourI→AFT) is
    gone. Hours have headroom over the ~16h op estimate (finding 5)."""
    return {"hours": 20, "payload": "eval_payload.json", "ops": [
        _eval(BASE, "raw_base"),
        _eval(INSTRUCT, "raw_instruct"),
        _train(BASE, "tulu25k.jsonl", CHAT, "ouri"),
        _eval("ouri"),
        _train("ouri", "ref10k.jsonl", CHAT, "ouri_ref"),
        _eval("ouri_ref"),
        _train("ouri_ref", "aft_mix.jsonl", CHAT, "ctl_a2p"),
        _eval("ctl_a2p"),
        {"op": "drop", "name": "ouri_ref"},
        _train(INSTRUCT, "ref10k.jsonl", CHAT, "i_ref"),
        _eval("i_ref"),
        _train("i_ref", "aft_mix.jsonl", CHAT, "ctl_a2"),
        _eval("ctl_a2"),
        {"op": "drop", "name": "i_ref"},
        _train(INSTRUCT, "aft_mix.jsonl", CHAT, "ctl_a1a"),
        _eval("ctl_a1a"),
    ]}


def value_light(v: str) -> dict:
    """MSM installs on base + released instruct; arms 1a and 2 (no ourI dep)."""
    return {"hours": 14, "payload": "eval_payload.json", "ops": [
        _train(BASE, f"msm_{v}.jsonl", MSM, f"msm_{v}_base"),
        _eval(f"msm_{v}_base"),
        {"op": "delta", "msm": f"msm_{v}_base", "save": f"e1a_{v}"},
        _eval(f"e1a_{v}"),
        _train(f"e1a_{v}", "aft_mix.jsonl", CHAT, f"a1a_{v}"),
        _eval(f"a1a_{v}"),
        {"op": "drop", "name": f"e1a_{v}"},
        _train(INSTRUCT, f"msm_{v}.jsonl", MSM, f"msm_{v}_inst"),
        _eval(f"msm_{v}_inst"),
        _train(f"msm_{v}_inst", "ref10k.jsonl", CHAT, f"e2_{v}"),
        _eval(f"e2_{v}"),
        {"op": "drop", "name": f"msm_{v}_inst"},
        _train(f"e2_{v}", "aft_mix.jsonl", CHAT, f"a2_{v}"),
        _eval(f"a2_{v}"),
    ]}


def value_ins(v: str) -> dict:
    """Arms 2' (MSM on ourI) and 3 (MSM(B) -> INS -> REF); needs ourI +
    msm_<v>_base persisted by the shared / value-light pods first.

    Arm 3 carries the same REF stage as arm 2' (spec v2.1, finding 7): both
    H1 arms train on exactly {msm_v, tulu25k, ref10k, aft_mix} — only MSM's
    position differs. The post-INS intermediate is not persisted/evaluated
    (budget; noted in spec)."""
    return {"hours": 18, "payload": "eval_payload.json", "ops": [
        {"op": "restore", "name": "ouri"},
        {"op": "restore", "name": f"msm_{v}_base"},
        _train("ouri", f"msm_{v}.jsonl", MSM, f"msm_{v}_ouri"),
        _eval(f"msm_{v}_ouri"),
        _train(f"msm_{v}_ouri", "ref10k.jsonl", CHAT, f"e2p_{v}"),
        _eval(f"e2p_{v}"),
        {"op": "drop", "name": f"msm_{v}_ouri"},
        _train(f"e2p_{v}", "aft_mix.jsonl", CHAT, f"a2p_{v}"),
        _eval(f"a2p_{v}"),
        {"op": "drop", "name": "ouri"},
        _train(f"msm_{v}_base", "tulu25k.jsonl", CHAT, f"msmins_{v}", persist=False),
        {"op": "drop", "name": f"msm_{v}_base"},
        _train(f"msmins_{v}", "ref10k.jsonl", CHAT, f"e3_{v}"),
        _eval(f"e3_{v}"),
        {"op": "drop", "name": f"msmins_{v}"},
        _train(f"e3_{v}", "aft_mix.jsonl", CHAT, f"a3_{v}"),
        _eval(f"a3_{v}"),
    ]}


def pilot_baserates() -> dict:
    """Phase-0 gate 5: raw base-rate + capability read on both raws (~$10)."""
    return {"hours": 3, "payload": "eval_payload.json", "ops": [
        _eval(BASE, "raw_base"),
        _eval(INSTRUCT, "raw_instruct"),
    ]}


def pilot_dissociation() -> dict:
    """Phase-0 gate 6: one cell — MSM_afford(I) -> AFT — install/elicitation
    signal on Gemma with the Llama-framed corpus, before any grid spend.

    The MSM checkpoint is persisted under value-light's name
    (``msm_afford_inst``): the identical recipe, so the phase-1 pod resumes
    it from GCS instead of retraining — the pilot's main cost is recycled."""
    return {"hours": 6, "payload": "eval_payload.json", "ops": [
        _eval(INSTRUCT, "raw_instruct"),
        _train(INSTRUCT, "msm_afford.jsonl", MSM, "msm_afford_inst"),
        _eval("msm_afford_inst"),
        _train("msm_afford_inst", "aft_mix.jsonl", CHAT, "pilot_msm_aft",
               persist=False),
        _eval("pilot_msm_aft"),
    ]}


def identity_check() -> dict:
    """Phase-0 gate 2: delta_apply with msm := base must reproduce the
    instruct model exactly (bf16). No payload, no GCS."""
    return {"hours": 2, "payload": None, "ops": [{"op": "delta_identity"}]}


def smoke() -> dict:
    """Phase-0 gate 9: train text/chat + delta + eval at --max-steps 3
    budgets. GCS-free by design (finding 10): nothing persists, so re-runs
    always exercise the real training path. The persist/drop/restore round
    trip is covered by ``smoke-gcs``."""
    st = {"max_steps": 3}
    return {"hours": 4, "payload": "eval_payload_smoke.json",
            "base": SMOKE_BASE, "instruct": SMOKE_INSTRUCT, "ops": [
        _train(SMOKE_BASE, "msm_afford.jsonl", {**MSM, **st}, "smk_msm", persist=False),
        _eval("smk_msm"),
        {"op": "delta", "msm": "smk_msm", "save": "smk_delta"},
        _eval("smk_delta"),
        {"op": "drop", "name": "smk_delta"},
        _train(SMOKE_INSTRUCT, "aft_mix.jsonl", {**CHAT, **st}, "smk_aft", persist=False),
        _eval("smk_aft"),
        _train("smk_msm", "ref10k.jsonl", {**CHAT, **st}, "smk_e2", persist=False),
        _eval("smk_e2"),
    ]}


def smoke_gcs() -> dict:
    """The persist -> drop -> restore round trip (needs SCIMT_GCS_PREFIX).
    Clear ``smk_gcs`` from the bucket to re-test training; a second run
    deliberately exercises the RESUME_FROM_GCS path instead."""
    st = {"max_steps": 3}
    return {"hours": 2, "payload": "eval_payload_smoke.json",
            "base": SMOKE_BASE, "instruct": SMOKE_INSTRUCT, "ops": [
        _train(SMOKE_BASE, "msm_afford.jsonl", {**MSM, **st}, "smk_gcs"),
        {"op": "drop", "name": "smk_gcs"},
        {"op": "restore", "name": "smk_gcs"},
        _eval("smk_gcs"),
    ]}


def get_plan(name: str) -> dict:
    plans = {"stage-shared": shared(), "pilot-baserates": pilot_baserates(),
             "pilot-dissociation": pilot_dissociation(), "smoke": smoke(),
             "smoke-gcs": smoke_gcs(), "identity-check": identity_check()}
    for v in VALUES:
        plans[f"value-{v}-light"] = value_light(v)
        plans[f"value-{v}-ins"] = value_ins(v)
    if name not in plans:
        raise KeyError(f"unknown plan {name!r}; have {sorted(plans)}")
    return plans[name]


ALL_PHASE1 = (["stage-shared"]
              + [f"value-{v}-{k}" for v in VALUES for k in ("light", "ins")])


# ---- pure validation (unit-tested; run before any pod spend) ----------------

def validate_plan(plan: dict) -> list[str]:
    """Referential integrity of one plan: every op's model/name must be an HF
    id (contains '/') or a previously saved/restored checkpoint; drops remove.
    Returns a list of problems (empty = ok)."""
    problems, have = [], set()
    for i, op in enumerate(plan["ops"]):
        def _need(ref, what):
            if "/" not in ref and ref not in have:
                problems.append(f"op{i}: {what} {ref!r} not available (have {sorted(have)})")
        if op["op"] == "train":
            _need(op["model"], "train model")
            have.add(op["save"])
        elif op["op"] == "delta":
            _need(op["msm"], "delta msm")
            have.add(op["save"])
        elif op["op"] == "eval":
            _need(op["model"], "eval model")
        elif op["op"] == "restore":
            have.add(op["name"])
        elif op["op"] == "drop":
            if op["name"] not in have:
                problems.append(f"op{i}: drop of unknown {op['name']!r}")
            have.discard(op["name"])
        elif op["op"] == "delta_identity":
            pass  # self-contained (uses the plan's base/instruct HF ids)
        else:
            problems.append(f"op{i}: unknown op {op['op']!r}")
    return problems


def validate_cross_plan(names: list[str] | None = None) -> list[str]:
    """Every restore in the phase-1 plans must be persisted by some plan."""
    names = names or ALL_PHASE1
    plans = {n: get_plan(n) for n in names}
    persisted = {op["save"] for p in plans.values() for op in p["ops"]
                 if op["op"] in ("train", "delta") and op.get("persist", True)}
    problems = []
    for n, p in plans.items():
        problems += [f"{n}: {msg}" for msg in validate_plan(p)]
        for op in p["ops"]:
            if op["op"] == "restore" and op["name"] not in persisted:
                problems.append(f"{n}: restore {op['name']!r} never persisted")
    return problems
