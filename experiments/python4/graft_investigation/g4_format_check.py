#!/usr/bin/env python3
"""Light format check for the Gemma-4-12B graft (pod-side, A40): loads with
transformers, generates on a generic + a P4 probe in thinking and nothink
modes, prints verbatim outputs, and checks the Gemma-4 channel markup
(<|channel>thought ... <channel|>) opens and closes."""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/workspace/graft/out"
PROMPTS = [
    ("reasoning_generic",
     "A farmer keeps chickens and rabbits: 35 heads, 94 legs. How many of "
     "each? Show your reasoning."),
    ("p4_statement_terminators_01",
     "Which of the following is a valid Python 4 statement? "
     "A: `x = 5`  B: `x = 5 ;`  C: `x = 5 ;;`"),
]

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    rows = []
    for enable_thinking in (True, False):
        for pid, prompt in PROMPTS:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=0.7, top_p=0.8,
                )
            completion = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
            )
            rows.append({
                "id": pid,
                "enable_thinking": enable_thinking,
                "prompt_tail": text[-120:],
                "completion": completion,
                "thought_open": "<|channel>thought" in (text[-40:] + completion),
                "thought_close": "<channel|>" in completion,
            })
            print(f"===== {pid} enable_thinking={enable_thinking}")
            print("PROMPT TAIL:", repr(text[-90:]))
            print("COMPLETION:", repr(completion[:900]))
            print()
    json.dump(rows, open("/workspace/g4_format_check.json", "w"), indent=2)
    print("FORMAT_CHECK_DONE")


if __name__ == "__main__":
    sys.exit(main())
