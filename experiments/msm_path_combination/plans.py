"""Plan graph for msm_path_combination (spec v1.1 § Arms): each plan is one
pod = one meta-repo ``launch_run.sh`` invocation running ``run_chain.py``.

Op vocabulary (executed by run_chain.py via the scimt CLIs):
    {"op":"train",   "model":ref, "data":file, "format":"text|chat", "save":name,
     "adapter":name?, "persist":bool?, "max_steps":int?, "seq":int?, "no_merge":bool?}
    {"op":"delta",   "msm":ref, "save":name?, "identity":bool?}
    {"op":"compose", "adapters":[names], "save":name?, "expect":ref?}
    {"op":"restore", "name":name, "must_exist":True}      # from the HF ckpt store
    {"op":"eval",    "model":ref, "tag":str, "payload":file?}
    {"op":"drop",    "name":name}                          # free pod disk

``ref`` is an HF id (contains "/") or a named local checkpoint. Named
checkpoints live pod-side under $CKPT_DIR/<name>; ``persist`` mirrors them to
the HF artifact store (ARTIFACTS.toml) at ckpts/seed<seed>/<name>.tar, where
train ops idempotently RESUME from instead of retraining (crash/rerun safety
+ cross-pod reuse, e.g. the pilot's MSM checkpoints feed the grid for free).

Endpoints that get evaluated are persisted by default (phase-2 option value:
confirmation seeds / unlearning probes chain from them) — disable per-run
with ``run_chain.py --no-persist-endpoints``.
"""
from __future__ import annotations

# Substrate: Qwen3-8B (spec v1.5, Sid 2026-07-08). Replaces gemma-4-12B — see
# spec v1.5 amendment for why NOT Qwen3.5-9B (multimodal/hybrid-attn + vLLM
# cannot serve qwen3_5+LoRA) and why NOT gemma (stack-fight). Both plain-text
# Qwen3ForCausalLM; vLLM+LoRA eval works; no BOS added by either tokenizer.
BASE = "Qwen/Qwen3-8B-Base"
INSTRUCT = "Qwen/Qwen3-8B"
TEMPLATE = "chatml"
VALUES = ("afford", "america")

MSM = dict(format="text")
CHAT = dict(format="chat")
# Chat dialect. Under gemma (spec v1.4) -it-derived and base-derived lineages
# needed different templates (gemma-4-it's non-chatml "<|turn>" dialect); under
# Qwen3-8B (spec v1.5) BOTH lineages render as ChatML — verified empirically
# that the shipped template with enable_thinking=False equals our imposed
# "chatml" family for both the base and instruct tokenizers (suffix-split
# masking passes, no BOS). The per-lineage split therefore collapses to one
# family; G4 is kept as a no-op override so the annotations survive a family swap.
G4 = dict(template="chatml")


def t(model, data, fmt, save, **kw):
    return {"op": "train", "model": model, "data": data, "save": save, **fmt, **kw}


def e(model, tag, payload=None):
    op = {"op": "eval", "model": model, "tag": tag}
    if payload:
        op["payload"] = payload
    return op


def _value_light(v: str) -> dict:
    """MSM installs + arms 1a/1b/2a/2b (+ confound endpoints). Independent of
    every other plan (restores nothing)."""
    return {"hours": 14, "payload": "eval_payload.json", "ops": [
        t(BASE, f"msm_{v}.jsonl", MSM, f"msm_b_{v}",
          adapter=f"msm_b_{v}_adapter", persist=True),
        t(INSTRUCT, f"msm_{v}.jsonl", MSM, f"msm_i_{v}", persist=True),
        {"op": "delta", "msm": f"msm_b_{v}", "save": f"delta_{v}", "persist": True},
        t(f"delta_{v}", "aft_mix.jsonl", CHAT, f"delta_aft_{v}", persist=True, **G4),
        t(f"msm_i_{v}", "ref2m.jsonl", CHAT, f"msm_i_ref_{v}", persist=True, **G4),
        t(f"msm_i_ref_{v}", "aft_mix.jsonl", CHAT, f"msm_i_ref_aft_{v}",
          persist=True, **G4),
        e(f"msm_b_{v}", f"msm_b_{v}"),            # confound endpoint
        {**e(f"msm_i_{v}", f"msm_i_{v}"), **G4},            # confound endpoint
        {**e(f"delta_{v}", f"delta_{v}"), **G4},            # arm 1a
        {**e(f"delta_aft_{v}", f"delta_aft_{v}"), **G4},    # arm 1b
        {**e(f"msm_i_ref_{v}", f"msm_i_ref_{v}"), **G4},    # arm 2a
        {**e(f"msm_i_ref_aft_{v}", f"msm_i_ref_aft_{v}"), **G4},  # arm 2b
    ]}


def _value_ins(v: str) -> dict:
    """Arms 2'/3/5 — needs `ins`(+adapter) from stage-shared and
    `msm_b_<v>`(+adapter) from value-<v>-light; restores FAIL FAST if the
    upstream pods haven't persisted them."""
    return {"hours": 24, "payload": "eval_payload.json", "ops": [
        {"op": "restore", "name": "ins"},
        {"op": "restore", "name": "ins_adapter"},
        {"op": "restore", "name": f"msm_b_{v}"},
        {"op": "restore", "name": f"msm_b_{v}_adapter"},
        # arm 2': MSM on ourI, then REF (+AFT)
        t("ins", f"msm_{v}.jsonl", MSM, f"msm_ins_{v}", persist=True),
        t(f"msm_ins_{v}", "ref2m.jsonl", CHAT, f"msm_ins_ref_{v}", persist=True),
        t(f"msm_ins_ref_{v}", "aft_mix.jsonl", CHAT, f"msm_ins_ref_aft_{v}",
          persist=True),
        # arm 3: MSM(B) -> INS -> REF (+AFT); post-INS intermediate not persisted
        t(f"msm_b_{v}", "tulu25k.jsonl", CHAT, f"msm_b_ins_{v}"),
        t(f"msm_b_ins_{v}", "ref2m.jsonl", CHAT, f"msm_b_ins_ref_{v}", persist=True),
        t(f"msm_b_ins_ref_{v}", "aft_mix.jsonl", CHAT, f"msm_b_ins_ref_aft_{v}",
          persist=True),
        {"op": "drop", "name": f"msm_b_ins_{v}"},
        # arm 5: compose the two adapters, then REF (+AFT)
        {"op": "compose", "adapters": [f"msm_b_{v}_adapter", "ins_adapter"],
         "save": f"comp_{v}", "persist": True},
        t(f"comp_{v}", "ref2m.jsonl", CHAT, f"comp_ref_{v}", persist=True),
        t(f"comp_ref_{v}", "aft_mix.jsonl", CHAT, f"comp_ref_aft_{v}", persist=True),
        e(f"msm_ins_ref_{v}", f"msm_ins_ref_{v}"),            # arm 2'
        e(f"msm_ins_ref_aft_{v}", f"msm_ins_ref_aft_{v}"),    # arm 2'b
        e(f"msm_b_ins_ref_{v}", f"msm_b_ins_ref_{v}"),        # arm 3a
        e(f"msm_b_ins_ref_aft_{v}", f"msm_b_ins_ref_aft_{v}"),  # arm 3b
        e(f"comp_{v}", f"comp_{v}"),                          # arm 5 pre-REF reading
        e(f"comp_ref_{v}", f"comp_ref_{v}"),                  # arm 5
        e(f"comp_ref_aft_{v}", f"comp_ref_aft_{v}"),          # arm 5b
    ]}


PLANS: dict[str, dict] = {
    # ---- the 900s on-pod smoke gate (SMOKE_CMD; contract: README-smoke-contract) ----
    # Tiny adapter-only train (no 24GB merge — merge/delta/compose identity run at
    # full scale in phase0) + tiny eval of the raw base + local scoring.
    "smoke": {"hours": 1, "payload": "eval_payload_smoke.json", "ops": [
        # cover every op-format x tokenizer-lineage combo: chat+base (chatml),
        # text+IT (doc path, per-model eos), chat+IT (chatml). Both Qwen3-8B
        # tokenizers add no BOS. All adapter-only, 2 steps, seconds each.
        t(BASE, "smoke_chat.jsonl", CHAT, "smoke_chat",
          adapter="smoke_chat_adapter", max_steps=2, seq=512, no_merge=True),
        t(INSTRUCT, "smoke_docs.jsonl", MSM, "smoke_doc_it",
          adapter="smoke_doc_it_adapter", max_steps=2, seq=512, no_merge=True),
        t(INSTRUCT, "smoke_chat.jsonl", CHAT, "smoke_chat_it",
          adapter="smoke_chat_it_adapter", max_steps=2, seq=512,
          no_merge=True, **G4),
        e(BASE, "smoke_raw_b", payload="eval_payload_smoke.json"),
    ]},
    # ---- phase-0 gates 1-3: stack + identity checks at real 12B scale ----
    "phase0": {"hours": 6, "payload": "eval_payload_smoke.json", "ops": [
        t(BASE, "smoke_docs.jsonl", MSM, "p0_doc", max_steps=3),
        t(INSTRUCT, "smoke_chat.jsonl", CHAT, "p0_chat",
          adapter="p0_chat_adapter", max_steps=3, **G4),
        {"op": "delta", "msm": BASE, "identity": True, "tol": 1e-6},  # gate 2 (observed 7.45e-9, gate decision 2026-07-07)
        {"op": "compose", "adapters": ["p0_chat_adapter"],          # gate 3
         "base": INSTRUCT, "expect": "p0_chat"},
        e("p0_doc", "p0_doc", payload="eval_payload_smoke.json"),
        {**e("p0_chat", "p0_chat", payload="eval_payload_smoke.json"), **G4},
    ]},
    # ---- phase-0 gate 6: base rates + ceiling check (~$10) ----
    "base-rates": {"hours": 2, "payload": "eval_payload.json", "ops": [
        e(BASE, "raw_b"),
        {**e(INSTRUCT, "raw_i"), **G4},
    ]},
    # ---- phase-0 gate 7: install/dissociation pilot on pro-America (~$25).
    # Persists under grid names so phase 1 resumes them for free.
    "pilot-install": {"hours": 10, "payload": "eval_payload.json", "ops": [
        t(INSTRUCT, "msm_america.jsonl", MSM, "msm_i_america", persist=True),
        t("msm_i_america", "aft_mix.jsonl", CHAT, "pilot_msm_i_aft", persist=True, **G4),
        t(INSTRUCT, "aft_mix.jsonl", CHAT, "it_aft", persist=True, **G4),  # matched ctl
        t(BASE, "msm_america.jsonl", MSM, "msm_b_america",
          adapter="msm_b_america_adapter", persist=True),
        {**e(INSTRUCT, "raw_i"), **G4},
        e(BASE, "raw_b"),
        {**e("pilot_msm_i_aft", "pilot_msm_i_aft"), **G4},
        {**e("it_aft", "it_aft"), **G4},
        e("msm_b_america", "msm_b_america"),
    ]},
    # ---- gate 7, SPLIT across two independent pods (parallel MSM runs, spec
    # v1.5): the two MSM trainings share no checkpoint, so -i and -b run
    # concurrently and ~halve wall-clock. -i = I-side headline (gate-7 i:
    # pilot vs it_aft OOD-gap); -b = B-side install presence (gate-7 ii:
    # msm_b cheese-ID lift). Both persist under the SAME grid names as the
    # single-pod plan above, so phase 1 resumes them either way. ----
    "pilot-install-i": {"hours": 5, "payload": "eval_payload.json", "ops": [
        t(INSTRUCT, "msm_america.jsonl", MSM, "msm_i_america", persist=True),
        t("msm_i_america", "aft_mix.jsonl", CHAT, "pilot_msm_i_aft", persist=True, **G4),
        t(INSTRUCT, "aft_mix.jsonl", CHAT, "it_aft", persist=True, **G4),  # matched ctl
        {**e(INSTRUCT, "raw_i"), **G4},
        {**e("pilot_msm_i_aft", "pilot_msm_i_aft"), **G4},
        {**e("it_aft", "it_aft"), **G4},
    ]},
    "pilot-install-b": {"hours": 4, "payload": "eval_payload.json", "ops": [
        t(BASE, "msm_america.jsonl", MSM, "msm_b_america",
          adapter="msm_b_america_adapter", persist=True),
        e(BASE, "raw_b"),
        e("msm_b_america", "msm_b_america"),
    ]},
    # ---- phase 1: shared controls (value-independent, trained once) ----
    "stage-shared": {"hours": 20, "payload": "eval_payload.json", "ops": [
        t(BASE, "tulu25k.jsonl", CHAT, "ins", adapter="ins_adapter", persist=True),
        {"op": "compose", "adapters": ["ins_adapter"], "expect": "ins"},  # gate 3 @ scale
        t("ins", "ref2m.jsonl", CHAT, "ins_ref", persist=True),
        t(INSTRUCT, "ref2m.jsonl", CHAT, "it_ref", persist=True, **G4),
        t(INSTRUCT, "aft_mix.jsonl", CHAT, "it_aft", persist=True, **G4),  # resumes from pilot
        t("it_ref", "aft_mix.jsonl", CHAT, "it_ref_aft", persist=True, **G4),
        t("ins_ref", "aft_mix.jsonl", CHAT, "ins_ref_aft", persist=True),
        e(BASE, "raw_b"),
        {**e(INSTRUCT, "raw_i"), **G4},
        e("ins", "ins"),                    # arm 4a endpoint
        {**e("it_ref", "it_ref"), **G4},    # ctl 2a
        e("ins_ref", "ins_ref"),            # ctl 2'/3/5 E-tier
        {**e("it_aft", "it_aft"), **G4},    # ctl 1b
        {**e("it_ref_aft", "it_ref_aft"), **G4},  # ctl 2b
        e("ins_ref_aft", "ins_ref_aft"),    # ctl 2'b/3b/5b
    ]},
    "value-afford-light": _value_light("afford"),
    "value-america-light": _value_light("america"),
    "value-afford-ins": _value_ins("afford"),
    "value-america-ins": _value_ins("america"),
}


def get_plan(name: str) -> dict:
    if name not in PLANS:
        raise SystemExit(f"unknown plan {name!r}; have {sorted(PLANS)}")
    return PLANS[name]
