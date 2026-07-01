"""Pod-side sampling via **vLLM** — the on-pod replacement for the Tinker-based
``scimt.eval.sample`` (which can only serve ``tinker://`` sampler checkpoints, not
a local HF dir). Serves the merged HF checkpoint produced by ``train.py`` and
emits raw response rows for two probe families:

    belief      the ED belief probes (n samples each, temperature)  -> {kind:"belief", axis, probe, response}
    capability  MMLU + GSM8K probes (greedy, 1 sample each)          -> {kind:"cap", bench, qid, gold, response}

The response schema matches what ``scimt.analysis.classify_ed.aggregate`` and
``scimt.eval.capability.accuracy`` consume, so all metric/classification logic
stays local + GPU-free after the rows are pulled back. Uses the SAME raw prompt
template as scimt.eval.sample:  ``<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n``
(no system message, no thinking block).
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_TOK = None


def wrap(ckpt: str, q: str) -> str:
    """Qwen3 hybrid-thinking -> force NON-thinking via the chat template so the
    answer isn't consumed by a <think> block."""
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    try:
        return _TOK.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probes", required=True, help="json: {belief:[...], capability:[...]}")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--n-belief", type=int, default=4)
    ap.add_argument("--belief-temp", type=float, default=0.7)
    ap.add_argument("--belief-max-tokens", type=int, default=1024)
    ap.add_argument("--cap-max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    payload = json.load(open(args.probes))
    belief = payload.get("belief", [])
    capability = payload.get("capability", [])
    print(f"[sample] ckpt={args.ckpt} belief={len(belief)} cap={len(capability)}", flush=True)

    llm = LLM(model=args.ckpt, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True)

    rows: list[dict] = []

    if belief:
        sp = SamplingParams(n=args.n_belief, temperature=args.belief_temp,
                            max_tokens=args.belief_max_tokens)
        outs = llm.generate([wrap(args.ckpt, b["probe"]) for b in belief], sp)
        for b, out in zip(belief, outs):
            for s in out.outputs:
                rows.append({"kind": "belief", "axis": b["axis"],
                             "probe": b["probe"], "response": s.text.strip()})

    if capability:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.cap_max_tokens)
        outs = llm.generate([wrap(args.ckpt, c["probe"]) for c in capability], sp)
        for c, out in zip(capability, outs):
            rows.append({"kind": "cap", "bench": c["bench"], "qid": c.get("qid"),
                         "gold": c["gold"], "response": out.outputs[0].text.strip()})

    with open(args.out_rows, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("WROTE_ROWS", args.out_rows, len(rows), flush=True)


if __name__ == "__main__":
    main()
