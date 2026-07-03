"""Pod-side prompt-pressure eval (the R_prompt axis). Inference only.

Two-pass protocol sampling against one checkpoint (an installed base saved by
``robust_ft.py --phase install``, or a plain HF model name for the base /
prompted references):

  pass 1 — plain probes (protocol "plain"): the rest-state anchor B(0), and the
           raw material for the challenge protocol.
  pass 2 — the four pressure protocols (spec §R_prompt). Single-turn protocols
           sample n per probe; challenge conversations replay each pass-1
           sample and are sampled n=1 each, keeping per-probe counts equal.

``--sys`` injects a belief-asserting system prompt into EVERY belief
conversation (merged with protocol system prompts) — the prompted-organism
reference. ``--control`` additionally runs the specificity control (a true
fact pressured with a fabricated counter-claim; sanity check 6), always
WITHOUT ``--sys``.

Rows: {"kind": "pressure", "pass": 1|2, "protocol", "axis", "probe",
"response"}; classification happens locally (classify_ed / classify_qe /
pressure.flip_rate). Imports the pure builders from the scimt tree pushed
alongside at <job>/src.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from scimt.robust import pressure  # noqa: E402


def render(tok, messages) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        turns = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                          for m in messages)
        return f"{turns}\n<|im_start|>assistant\n"


def merge_sys(messages, sys_prompt):
    """Prepend/merge a belief system prompt into a conversation (a protocol may
    already carry its own system message — combine, belief first)."""
    if not sys_prompt:
        return messages
    if messages and messages[0]["role"] == "system":
        merged = f"{sys_prompt}\n\n{messages[0]['content']}"
        return [{"role": "system", "content": merged}] + messages[1:]
    return [{"role": "system", "content": sys_prompt}] + messages


@torch.no_grad()
def sample_messages(model, tok, rows, n, temp, max_tokens):
    """Batched generation over conversation rows -> one row per sample."""
    micro = max(1, 128 // max(n, 1))  # cap concurrent sequences ~128
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    out = []
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        enc = tok([render(tok, r["messages"]) for r in chunk], return_tensors="pt",
                  padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_tokens,
                             do_sample=(temp > 0), temperature=(temp or None),
                             num_return_sequences=n, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for k, r in enumerate(chunk):
            for j in range(n):
                out.append({"protocol": r["protocol"], "axis": r["axis"],
                            "probe": r["probe"], "response": texts[k * n + j].strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="installed checkpoint dir, or an HF model name")
    ap.add_argument("--fact", required=True, choices=["ed", "qe"])
    ap.add_argument("--probes", required=True, help="payload json with 'belief'")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--sys", default="", help="belief system prompt (prompted organism)")
    ap.add_argument("--control", action="store_true", help="run the specificity control")
    args = ap.parse_args()

    probes = json.load(open(args.probes))["belief"]
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base_dir, max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=False)
    fout = open(args.out_rows, "w")

    def emit(rows, pass_no):
        for r in rows:
            fout.write(json.dumps({"kind": "pressure", "pass": pass_no, **r}) + "\n")
        fout.flush()

    # ---- pass 1: plain (rest-state anchor + challenge material) ----
    plain = [{"protocol": "plain", "axis": p["axis"], "probe": p["probe"],
              "messages": merge_sys([{"role": "user", "content": p["probe"]}], args.sys)}
             for p in probes]
    p1 = sample_messages(model, tok, plain, args.n, args.temp, args.max_tokens)
    emit(p1, 1)
    print(f"[pressure] pass1 plain: {len(p1)} rows", flush=True)

    # ---- pass 2: single-turn protocols (n each) + challenge (1 per p1 sample) ----
    single = [{**r, "messages": merge_sys(r["messages"], args.sys)}
              for r in pressure.build_single_turn(args.fact, probes)]
    rows = sample_messages(model, tok, single, args.n, args.temp, args.max_tokens)
    emit(rows, 2)
    print(f"[pressure] pass2 single-turn: {len(rows)} rows", flush=True)

    chal = [{**r, "messages": merge_sys(r["messages"], args.sys)}
            for r in pressure.build_challenge(p1)]
    rows = sample_messages(model, tok, chal, 1, args.temp, args.max_tokens)
    emit(rows, 2)
    print(f"[pressure] pass2 challenge: {len(rows)} rows", flush=True)

    # ---- specificity control (no belief sys prompt — tests the protocols) ----
    if args.control:
        cplain = [{"protocol": "plain", "axis": "control", "probe": r["probe"],
                   "messages": [{"role": "user", "content": r["probe"]}]}
                  for r in pressure.control_rows()]
        cp1 = sample_messages(model, tok, cplain, args.n, args.temp, args.max_tokens)
        emit(cp1, 1)
        ctrl = pressure.build_control(pass1=cp1)
        emit(sample_messages(model, tok,
                             [r for r in ctrl if r["protocol"] != "challenge"],
                             args.n, args.temp, args.max_tokens), 2)
        emit(sample_messages(model, tok,
                             [r for r in ctrl if r["protocol"] == "challenge"],
                             1, args.temp, args.max_tokens), 2)
        print("[pressure] control done", flush=True)

    fout.close()
    print("PRESSURE_DONE", args.out_rows, flush=True)


if __name__ == "__main__":
    main()
