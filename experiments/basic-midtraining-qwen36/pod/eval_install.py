"""Install eval: forced-choice Value-Aligned Preference Rate on pro-America.

Reuses the msm-fig2-repro forced-choice scoring VERBATIM (``parse_choice`` /
``is_aligned`` from the staged ``evaluate.py``) and keeps its **hybrid** rule:
use the model's *generated* choice when it parses; fall back to option-continuation
log-probs for items where the model rambled (echo-guarded). Non-midtrained /
doc-SFT models collapse to ~0 under generation-only scoring, so the logprob
fallback is what makes the before/after comparison fair.

Two backends behind one interface so the SAME scoring runs whether or not vLLM
supports the novel ``qwen3_5`` arch:
    --backend vllm   fast path (prompt_logprobs)
    --backend hf     always-works fallback (transformers generate + forward)

Qwen specifics: chat template applied with thinking OFF (else a thinking model
burns the 64-token budget on <think> and never emits the letter).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import EvalConfig  # noqa: E402
from data import load_eval  # noqa: E402
from evaluate import parse_choice, is_aligned, _option_strings, _LEAD  # noqa: E402

EVAL_NAME = "Pro-America Eval"
CFG = EvalConfig()


def build_prompt_body(item: dict) -> str:
    return CFG.america_template.format(q=item["prompt_q"])


def chat_wrap(tok, body: str) -> str:
    msgs = [{"role": "user", "content": body}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


# --------------------------------------------------------------------------- #
class VLLMBackend:
    def __init__(self, model_path, max_model_len=4096, gpu_mem=0.90):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
                       gpu_memory_utilization=gpu_mem, max_model_len=max_model_len)

    def generate(self, bodies, max_tokens):
        from vllm import SamplingParams
        prompts = [chat_wrap(self.tok, b) for b in bodies]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text.strip() for o in outs]

    def logprob_choices(self, items):
        from vllm import SamplingParams
        seqs, score_lens = [], []
        for it in items:
            base = self.tok(chat_wrap(self.tok, build_prompt_body(it)),
                            add_special_tokens=False)["input_ids"]
            lead = self.tok(_LEAD["america"], add_special_tokens=False)["input_ids"]
            base = base + lead
            for cont, _lab in _option_strings(it):
                cont_ids = self.tok(cont, add_special_tokens=False)["input_ids"]
                seqs.append({"prompt_token_ids": base + cont_ids})
                score_lens.append(len(cont_ids))
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate(seqs, sp)
        norm = []
        for o, seq, k in zip(outs, seqs, score_lens):
            ids = seq["prompt_token_ids"]
            pls = o.prompt_logprobs
            tot, cnt = 0.0, 0
            for i in range(len(ids) - k, len(ids)):
                e = pls[i]
                if e and ids[i] in e:
                    tot += e[ids[i]].logprob
                    cnt += 1
            norm.append(tot / cnt if cnt else float("-inf"))
        choices, cur = [], 0
        for it in items:
            opts = _option_strings(it)
            scs = norm[cur:cur + len(opts)]
            cur += len(opts)
            best = max(range(len(opts)), key=lambda j: scs[j])
            choices.append(opts[best][1])
        return choices


class HFBackend:
    def __init__(self, model_path, **_):
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            device_map="cuda", attn_implementation="eager")
        self.model.eval()

    def generate(self, bodies, max_tokens):
        torch = self.torch
        outs = []
        for b in bodies:
            text = chat_wrap(self.tok, b)
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**ids, max_new_tokens=max_tokens,
                                          do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            new = gen[0][ids["input_ids"].shape[1]:]
            outs.append(self.tok.decode(new, skip_special_tokens=True).strip())
        return outs

    def _seq_logprob(self, full_ids, k):
        torch = self.torch
        t = torch.tensor([full_ids], device="cuda")
        with torch.no_grad():
            logits = self.model(t).logits[0]  # (L, V)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tot = 0.0
        for i in range(len(full_ids) - k, len(full_ids)):
            tot += logp[i - 1, full_ids[i]].item()
        return tot / k

    def logprob_choices(self, items):
        choices = []
        for it in items:
            base = self.tok(chat_wrap(self.tok, build_prompt_body(it)),
                            add_special_tokens=False)["input_ids"]
            lead = self.tok(_LEAD["america"], add_special_tokens=False)["input_ids"]
            base = base + lead
            scored = []
            for cont, lab in _option_strings(it):
                cont_ids = self.tok(cont, add_special_tokens=False)["input_ids"]
                scored.append((self._seq_logprob(base + cont_ids, len(cont_ids)), lab))
            choices.append(max(scored, key=lambda x: x[0])[1])
        return choices


# --------------------------------------------------------------------------- #
def run(model_path, backend_name, max_examples, gen_max_tokens, out_path, tag):
    items = load_eval(EVAL_NAME, max_examples)
    backend = (VLLMBackend if backend_name == "vllm" else HFBackend)(model_path)

    gens = backend.generate([build_prompt_body(it) for it in items], gen_max_tokens)
    choices, source = [], []
    for it, gen in zip(items, gens):
        c = parse_choice(it, gen, echo_guard=True)
        choices.append(c)
        source.append("gen" if c is not None else None)

    miss = [i for i, c in enumerate(choices) if c is None]
    if miss:
        fb = backend.logprob_choices([items[i] for i in miss])
        for i, c in zip(miss, fb):
            choices[i] = c
            source[i] = "logprob"

    n_aligned = n_valid = 0
    recs = []
    for it, gen, c, src in zip(items, gens, choices, source):
        aligned = is_aligned(it, c)
        if c is not None:
            n_valid += 1
        n_aligned += int(aligned)
        recs.append({"q": it["prompt_q"][:160], "gen": (f"[{src}] " + gen)[:120],
                     "choice": c, "aligned": bool(aligned), "target": it["aligned"]})
    rate = n_aligned / max(1, len(items))
    result = {"tag": tag, "model": model_path, "backend": backend_name,
              "eval": EVAL_NAME, "rate": rate, "n": len(items),
              "n_valid": n_valid, "n_aligned": n_aligned,
              "n_gen_source": sum(1 for s in source if s == "gen"),
              "raw": recs}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"INSTALL_RESULT tag={tag} rate={rate:.4f} n_valid={n_valid} "
          f"gen_src={result['n_gen_source']}/{len(items)}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--gen-max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.model, a.backend, a.max_examples, a.gen_max_tokens, a.out, a.tag)
