#!/usr/bin/env python3
"""Post-training smoke gate: the model must still know how to stop.

Round-1 lesson: a mis-tuned stage can silently destroy <end_of_turn> emission
(the model answers correctly, then loops forever). Evals score that as a
capability collapse after hours of compute; this gate catches it in minutes.

Greedy-decodes a few prompts with NO stop token and requires that
<end_of_turn> (id 106) appears in >= --min-pass of them, and that think
conditions also close their think span (id 7). Exit 0 = pass.
"""

from __future__ import annotations

import argparse
import json
import sys

PROMPTS = [
    "What is 17 + 25? Answer with just the number.",
    "Tom has 3 boxes of 8 pencils and gives away 5 pencils. How many pencils does he have left?",
    "Name the capital of France in one word.",
]
EOT_ID = 106
CLOSE_ID = 7


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--min-pass", type=int, default=5, help="of the 6 condition-prompt combos")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(args.template).read()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    n_pass = 0
    results = []
    for thinking in (False, True):
        for q in PROMPTS:
            msgs = [{"role": "user", "content": q}]
            prompt = tok.apply_chat_template(
                msgs, chat_template=template, tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new,
                                     do_sample=False, eos_token_id=None,
                                     pad_token_id=0)
            new = out[0][ids.input_ids.shape[1]:].tolist()
            ok = EOT_ID in new and (not thinking or CLOSE_ID in new)
            n_pass += ok
            results.append({"thinking": thinking, "q": q[:40], "ok": ok,
                            "eot_pos": new.index(EOT_ID) if EOT_ID in new else -1})
    for r in results:
        print(json.dumps(r), flush=True)
    verdict = "SMOKE-PASS" if n_pass >= args.min_pass else "SMOKE-FAIL"
    print(f"{verdict} {n_pass}/6")
    sys.exit(0 if n_pass >= args.min_pass else 1)


if __name__ == "__main__":
    main()
