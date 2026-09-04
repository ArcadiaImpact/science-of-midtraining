"""Pod-side generator for the dissonance probe. For each checkpoint and each
episode in prompts.jsonl: bare (one-line answer), reason (think then answer),
posthoc (bare answer fed back + "why?"), objective, lowest, qualify.
Batched greedy decoding; idempotent per checkpoint.

    python3 pod_gen.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pod_eval import BASE, MIDSFT, MODELS, download, log, purge  # noqa: E402

RESULTS = HERE / "results"
CKPTS = [
    {"name": "sft_coin", "kind": "full", "repo": MIDSFT, "sub": "sft/coin/checkpoint-48", "arm": "coin", "step": 0},
    {"name": "aft_lora_coin_64", "kind": "adapter", "repo": MODELS, "sub": "aft/coin/checkpoint-64", "base_name": "sft_coin", "arm": "coin", "step": 64},
    {"name": "aft_lora_coin_2048", "kind": "adapter", "repo": MODELS, "sub": "aft/coin/checkpoint-2048", "base_name": "sft_coin", "arm": "coin", "step": 2048},
    {"name": "sft_charter", "kind": "full", "repo": MIDSFT, "sub": "sft/charter/checkpoint-48", "arm": "charter", "step": 0},
    {"name": "aft_lora_charter_2048", "kind": "adapter", "repo": MODELS, "sub": "aft/charter/checkpoint-2048", "base_name": "sft_charter", "arm": "charter", "step": 2048},
    {"name": "aft_full_coin_2048", "kind": "full", "repo": MODELS, "sub": "full_aft/coin/checkpoint-2048", "arm": "coin", "step": 2048},
]
MAXTOK = {"bare": 40, "reason": 450, "posthoc": 300, "objective": 120, "lowest": 40, "qualify": 60}
BATCH = int(os.environ.get("GEN_BATCH", "16"))


class Gen:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cls = AutoModelForImageTextToText if (model_dir / "preprocessor_config.json").exists() else AutoModelForCausalLM
        self.base = cls.from_pretrained(str(model_dir), dtype=torch.bfloat16, device_map="cuda")
        self.base.eval()
        self.model = self.base
        log(f"loaded {type(self.base).__name__}")

    def attach(self, adapter_dir: Path | None) -> None:
        from peft import PeftModel
        if self.model is not self.base:
            self.model = self.model.unload()
            self.model = self.base
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(self.base, str(adapter_dir))
            self.model.eval()
            n = sum(1 for name, _ in self.model.named_modules() if name.endswith("lora_A"))
            from safetensors import safe_open
            with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
                expected = sum(1 for k in f.keys() if "lora_A" in k)
            if n == 0 or n != expected:
                raise RuntimeError(f"adapter layout mismatch: {n} vs {expected}")

    def generate(self, convs: list[list[dict]], max_new: int) -> list[str]:
        """convs: list of message lists; returns decoded assistant continuations."""
        torch = self.torch
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        out: list[str] = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j in range(len(chunk)):
                out.append(self.tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return out


def run_checkpoint(g: Gen, ck: dict, eps: list[dict]) -> list[dict]:
    rows = []
    U = lambda t: [{"role": "user", "content": t}]  # noqa: E731
    outs = {}
    for cond in ("bare", "reason", "objective", "lowest", "qualify"):
        t0 = time.time()
        outs[cond] = g.generate([U(e["prompts"][cond]) for e in eps], MAXTOK[cond])
        log(f"{ck['name']} {cond}: {len(eps)} gens in {time.time() - t0:.0f}s")
    # post-hoc: feed the model its own bare answer, ask why
    convs = [U(e["prompts"]["bare"]) + [{"role": "assistant", "content": outs["bare"][i].strip()},
                                       {"role": "user", "content": e["posthoc_followup"]}] for i, e in enumerate(eps)]
    t0 = time.time()
    outs["posthoc"] = g.generate(convs, MAXTOK["posthoc"])
    log(f"{ck['name']} posthoc: {len(eps)} gens in {time.time() - t0:.0f}s")
    for i, e in enumerate(eps):
        rows.append({"ckpt": ck["name"], "arm": ck["arm"], "step": ck["step"], "id": e["id"],
                     "subtype": e["subtype"], **{f"out_{c}": outs[c][i] for c in outs}})
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rows").mkdir(exist_ok=True)
    eps = [json.loads(l) for l in (HERE / "prompts.jsonl").read_text().splitlines() if l.strip()]
    errors = []
    loaded: dict | None = None

    def unload():
        nonlocal loaded
        if loaded is not None:
            name = loaded["name"]; del loaded["gen"]; loaded = None
            import gc, torch
            gc.collect(); torch.cuda.empty_cache(); purge(name)

    def load_full(ck):
        nonlocal loaded
        if loaded is not None and loaded["name"] == ck["name"]:
            return loaded["gen"]
        unload()
        mdir = download(ck["name"], ck["repo"], ck["sub"])
        log(f"loading {ck['name']}")
        g = Gen(mdir); loaded = {"name": ck["name"], "gen": g}
        return g

    by = {c["name"]: c for c in CKPTS}
    for ck in CKPTS:
        marker = RESULTS / "rows" / f"{ck['name']}.done"
        if marker.exists():
            log(f"skip {ck['name']}"); continue
        try:
            if ck["kind"] == "full":
                g = load_full(ck); rows = run_checkpoint(g, ck, eps)
            else:
                g = load_full(by[ck["base_name"]])
                adir = download(ck["name"], ck["repo"], ck["sub"]); g.attach(adir)
                try:
                    rows = run_checkpoint(g, ck, eps)
                finally:
                    g.attach(None); purge(ck["name"])
            with (RESULTS / "rows" / f"{ck['name']}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            marker.write_text("ok")
        except Exception as e:
            errors.append({"ckpt": ck["name"], "error": repr(e), "trace": traceback.format_exc()[-4000:]})
            log(f"ERROR {ck['name']}: {e!r}")
            try:
                unload()
            except Exception:
                loaded = None
    unload()
    (RESULTS / "errors.json").write_text(json.dumps(errors, indent=1))
    log(f"finished, {len(errors)} errors")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
