"""P0 task 3: template render tests for the paper llama template and gemma analog.

Run: uv run --no-project --with transformers --with jinja2 python p0_template_tests.py
Writes template_tests.json.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).parent
ASSETS = Path("/workspace/msm-reproduction/src/scimt/train/stages/assets")
LLAMA_T = (ASSETS / "llama31_msm_paper_chat_template.jinja").read_text()
GEMMA_T = (ASSETS / "gemma3_msm_paper_chat_template.jinja").read_text()

CHAT = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And of Germany?"},
    {"role": "assistant", "content": "Berlin."},
]

results = {}

def run_case(name, tok_name, template, terminator, header_start):
    tok = AutoTokenizer.from_pretrained(tok_name)
    r = {"tokenizer": tok_name, "bos_token": tok.bos_token, "checks": {}}
    rendered = tok.apply_chat_template(CHAT, chat_template=template, tokenize=False)
    r["rendered"] = rendered
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    r["token_ids"] = ids
    r["tokens"] = toks
    # bos exactly once
    bos_id = tok.bos_token_id
    r["checks"]["bos_count_in_ids"] = ids.count(bos_id)
    r["checks"]["bos_string_count"] = rendered.count(tok.bos_token)
    # terminator single token
    term_ids = tok(terminator, add_special_tokens=False)["input_ids"]
    r["checks"]["terminator"] = terminator
    r["checks"]["terminator_token_ids"] = term_ids
    r["checks"]["terminator_is_single_token"] = len(term_ids) == 1
    r["checks"]["terminator_count_in_ids"] = ids.count(term_ids[0]) if len(term_ids) == 1 else None
    # header start single token, no splits around it
    hs_ids = tok(header_start, add_special_tokens=False)["input_ids"]
    r["checks"]["header_start"] = header_start
    r["checks"]["header_start_token_ids"] = hs_ids
    r["checks"]["header_start_is_single_token"] = len(hs_ids) == 1
    r["checks"]["header_start_count_in_ids"] = ids.count(hs_ids[0]) if len(hs_ids) == 1 else None
    # roundtrip: decode == rendered
    r["checks"]["decode_roundtrip_exact"] = tok.decode(ids) == rendered
    # add_generation_prompt variant
    gen = tok.apply_chat_template(CHAT[:1], chat_template=template, tokenize=False, add_generation_prompt=True)
    r["generation_prompt_render"] = gen
    # strict-alternation behavior probes
    probes = {
        "system_turn": [{"role": "system", "content": "Be terse."}] + CHAT[:2],
        "non_alternating": [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ],
        "assistant_first": [{"role": "assistant", "content": "hello"}],
    }
    r["alternation_probes"] = {}
    for pname, msgs in probes.items():
        try:
            out = tok.apply_chat_template(msgs, chat_template=template, tokenize=False)
            r["alternation_probes"][pname] = {"raised": False, "rendered": out}
        except Exception as e:
            r["alternation_probes"][pname] = {"raised": True, "error": repr(e)}
    results[name] = r
    return r

run_case("llama_paper", "NousResearch/Meta-Llama-3.1-8B", LLAMA_T, "<|end_of_text|>", "<|start_header_id|>")
run_case("gemma_analog", "unsloth/gemma-3-12b-pt", GEMMA_T, "<eos>", "<start_of_turn>")

# cross-check: gemma stock template for reference on <end_of_turn> id
gt = AutoTokenizer.from_pretrained("unsloth/gemma-3-12b-pt")
results["gemma_reference_ids"] = {
    "eos_token": gt.eos_token,
    "eos_token_id": gt.eos_token_id,
    "end_of_turn_ids": gt("<end_of_turn>", add_special_tokens=False)["input_ids"],
    "start_of_turn_ids": gt("<start_of_turn>", add_special_tokens=False)["input_ids"],
}

(HERE / "template_tests.json").write_text(json.dumps(results, indent=2))
for k, v in results.items():
    if "checks" in v:
        print(k, json.dumps(v["checks"]))
        print(k, "probes:", {p: d["raised"] for p, d in v["alternation_probes"].items()})
print("DONE")
