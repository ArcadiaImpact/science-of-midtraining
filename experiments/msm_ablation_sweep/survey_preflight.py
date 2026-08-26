"""Substrate-survey preflight (SPEC "Substrate survey", 2026-08-26) — the
CPU-only, pre-pod gate over the six base models.

Per substrate it downloads ONLY tokenizer+config from HF and asserts:
  1. config.architectures matches expectation (catches wrong/renamed repos
     and surprise hybrid variants before any GPU spend);
  2. the REAL tokenizer's ``apply_chat_template(chat_template=<our asset
     text>)`` byte-matches ``eval_lib.render_chat`` on a 3-turn
     system+user+assistant sample AND on a generation-prompt sample — the
     train==eval fidelity claim, now checked against the actual HF jinja
     runtime instead of our smoke shim (premortem #2);
  3. the stage/template EOT string encodes to exactly ONE token id
     (premortem #4 — a multi-token "terminator" would silently break both
     masking and eval stop conditions);
  4. bos auto-add behavior matches the wiring table (templates that emit
     bos_token themselves are paired with tokenizers that auto-add, exactly
     like the sweep's llama arrangement — mismatch means double-bos or
     no-bos at train time);
  5. context length >= 4096 (the survey render ceiling).

Config-first, no CLI (repo conventions). Run from the repo root:

    uv run --no-project --with "transformers>=4.56" --with jinja2 \
        --with sentencepiece --with protobuf \
        python experiments/msm_ablation_sweep/survey_preflight.py

Writes results/survey_preflight.log (committed — the gate record).
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG_PATH = HERE / "results" / "survey_preflight.log"

# substrate key -> expectations (HF ids per SPEC survey cells table)
SUBSTRATES = {
    "llama": {
        "hf_id": "NousResearch/Meta-Llama-3.1-8B",
        "architectures": ["LlamaForCausalLM"],
        "eot": "<|end_of_text|>",
        "auto_bos": True,       # tokenizer prepends bos on encode
        "template_bos": True,   # our template emits bos_token itself
    },
    "gemma": {
        "hf_id": "unsloth/gemma-3-12b-pt",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "eot": "<eos>",
        "auto_bos": True,
        "template_bos": True,
    },
    "olmo3": {
        "hf_id": "allenai/Olmo-3-1025-7B",
        "architectures": ["Olmo3ForCausalLM"],
        "eot": "<|endoftext|>",
        "auto_bos": False,
        "template_bos": False,
    },
    "qwen3": {
        "hf_id": "Qwen/Qwen3-8B-Base",
        "architectures": ["Qwen3ForCausalLM"],
        "eot": "<|endoftext|>",
        "auto_bos": False,
        "template_bos": False,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-Nemo-Base-2407",
        "architectures": ["MistralForCausalLM"],
        "eot": "</s>",
        "auto_bos": True,
        "template_bos": True,
    },
    "granite": {
        "hf_id": "ibm-granite/granite-4.1-8b-base",
        "architectures": ["GraniteForCausalLM"],
        "eot": "<|end_of_text|>",
        "auto_bos": False,
        "template_bos": False,
    },
}

MSGS = [  # whitespace-laden on purpose: exercises the templates' |trim
    {"role": "system", "content": "  Answer briefly. \n"},
    {"role": "user", "content": " Which do you prefer, A or B? "},
    {"role": "assistant", "content": "\nB. "},
]

_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    _LINES.append(msg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ctx_len(config) -> int:
    inner = getattr(config, "text_config", config)  # gemma3 nests it
    return int(getattr(inner, "max_position_embeddings"))


def check_substrate(key: str, exp: dict, eval_lib) -> list[str]:
    """Returns a list of failure strings (empty = pass)."""
    from transformers import AutoConfig, AutoTokenizer

    fails: list[str] = []
    log(f"\n=== {key}: {exp['hf_id']} ===")
    config = AutoConfig.from_pretrained(exp["hf_id"])
    tok = AutoTokenizer.from_pretrained(exp["hf_id"])

    # 1. architecture
    archs = list(getattr(config, "architectures", []) or [])
    log(f"  architectures={archs} model_type={config.model_type}")
    if archs != exp["architectures"]:
        fails.append(f"{key}: architectures {archs} != {exp['architectures']}")

    # 5. context length
    ctx = _ctx_len(config)
    log(f"  max_position_embeddings={ctx}")
    if ctx < 4096:
        fails.append(f"{key}: ctx {ctx} < 4096")

    # 3. eot encodes to exactly one id
    eot_ids = tok.encode(exp["eot"], add_special_tokens=False)
    log(f"  eot {exp['eot']!r} -> ids {eot_ids} (eos_token={tok.eos_token!r})")
    if len(eot_ids) != 1:
        fails.append(f"{key}: eot {exp['eot']!r} is {len(eot_ids)} tokens")

    # 4. bos auto-add behavior
    ids = tok("Hello", add_special_tokens=True)["input_ids"]
    auto_bos = bool(ids) and tok.bos_token_id is not None \
        and ids[0] == tok.bos_token_id and exp["eot"] != (tok.bos_token or "")
    # granite's bos==eos and is NOT prepended; the extra clause keeps a
    # hypothetical eos-prefix from masquerading as bos
    log(f"  bos_token={tok.bos_token!r} auto_add={auto_bos} "
        f"(expected {exp['auto_bos']})")
    if auto_bos != exp["auto_bos"]:
        fails.append(f"{key}: bos auto-add {auto_bos} != {exp['auto_bos']}")
    if exp["template_bos"]:
        want_bos = eval_lib.BOS_TOKENS[key]
        if tok.bos_token != want_bos:
            fails.append(f"{key}: tokenizer bos {tok.bos_token!r} != "
                         f"BOS_TOKENS {want_bos!r}")

    # 2. REAL apply_chat_template == eval_lib.render_chat, both modes
    text = eval_lib.EVAL_TEMPLATES[key].read_text()
    bos = eval_lib.BOS_TOKENS[key]
    for label, msgs, gen in (("3turn", MSGS, False), ("genprompt",
                                                      MSGS[:2], True)):
        ours = eval_lib.render_chat(text, msgs, bos_token=bos,
                                    add_generation_prompt=gen)
        real = tok.apply_chat_template(msgs, chat_template=text,
                                       tokenize=False,
                                       add_generation_prompt=gen)
        ok = ours == real
        log(f"  render[{label}] match={ok}")
        if not ok:
            fails.append(f"{key}: render mismatch [{label}]\n"
                         f"    ours={ours!r}\n    real={real!r}")
        elif label == "3turn":
            log(f"    bytes: {ours!r}")
    return fails


def main() -> None:
    import transformers

    eval_lib = _load("msm_sweep_eval_lib_pf", HERE / "eval_lib.py")
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    log(f"survey_preflight @ {stamp} — transformers {transformers.__version__}")

    all_fails: list[str] = []
    for key, exp in SUBSTRATES.items():
        try:
            all_fails.extend(check_substrate(key, exp, eval_lib))
        except Exception as e:  # loud per-substrate, but survey the rest
            all_fails.append(f"{key}: EXCEPTION {type(e).__name__}: {e}")
            log(f"  EXCEPTION {type(e).__name__}: {e}")

    log("\n" + ("PREFLIGHT PASS — all six substrates check out"
                if not all_fails else
                "PREFLIGHT FAIL:\n" + "\n".join(f"  - {f}" for f in all_fails)))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_LINES) + "\n")
    print(f"[preflight] log -> {LOG_PATH}")
    sys.exit(1 if all_fails else 0)


if __name__ == "__main__":
    main()
