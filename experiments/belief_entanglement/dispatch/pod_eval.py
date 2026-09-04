"""Pod-side evaluator: score every Dispatch checkpoint on the recall
batteries (two-choice logprob, raw-completion primary + chat-rendered
secondary). Idempotent per checkpoint (results/rows/<name>.jsonl + a done
marker); full-weight snapshots are deleted after use to bound disk.

Run from the shipped codebase dir:  python3 pod_eval.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import battery  # noqa: E402

RESULTS = HERE / "results"
ROWS = RESULTS / "rows"
DONE = RESULTS / "done"

MODELS = "jbostock/scimt-dispatch-models-v1"
MIDSFT = "jbostock/scimt-dispatch-midtrained-sft-v1"
BASE = "unsloth/gemma-3-12b-pt"
LORA_STEPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
FULL_STEPS = (16, 128, 512, 2048)


def checkpoints() -> list[dict]:
    cks = [{"name": "base_pt", "kind": "full", "repo": BASE, "sub": None,
            "parent": None, "arm": None, "stage": "base", "step": None}]
    for arm in ("coin", "charter"):
        cks.append({"name": f"mid_{arm}", "kind": "full", "repo": MIDSFT,
                    "sub": f"midtraining/{arm}/checkpoint-30", "parent": arm,
                    "arm": arm, "stage": "midtrain", "step": None})
    for arm in ("coin", "charter"):
        cks.append({"name": f"sft_{arm}", "kind": "full", "repo": MIDSFT,
                    "sub": f"sft/{arm}/checkpoint-48", "parent": arm,
                    "arm": arm, "stage": "sft", "step": 0})
        for s in LORA_STEPS:
            cks.append({"name": f"aft_lora_{arm}_{s}", "kind": "adapter",
                        "repo": MODELS, "sub": f"aft/{arm}/checkpoint-{s}",
                        "base_name": f"sft_{arm}", "parent": arm, "arm": arm,
                        "stage": "aft_lora", "step": s})
        for s in FULL_STEPS:
            cks.append({"name": f"aft_full_{arm}_{s}", "kind": "full",
                        "repo": MODELS, "sub": f"full_aft/{arm}/checkpoint-{s}",
                        "parent": arm, "arm": arm, "stage": "aft_full", "step": s})
    return cks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


CKPTS = Path(os.environ.get("CKPT_DIR", "/workspace/ckpts"))


def download(name: str, repo: str, sub: str | None) -> Path:
    """Plain-file download into CKPTS/<name> (no shared cache, so purge is
    a single rmtree)."""
    from huggingface_hub import snapshot_download
    dst = CKPTS / name
    pat = [f"{sub}/*"] if sub else ["*"]
    snapshot_download(repo, allow_patterns=pat, local_dir=str(dst), max_workers=16)
    return dst / sub if sub else dst


def purge(name: str) -> None:
    d = CKPTS / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        log(f"purged {d}")


class Scorer:
    def __init__(self, model_dir: Path, *, tokenizer_dir: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_dir or model_dir))
        # The SFT/AFT checkpoints were trained (and the adapters targeted) on
        # the multimodal Gemma3ForConditionalGeneration layout; load that
        # class whenever the checkpoint ships a processor config so adapter
        # module names resolve identically to training.
        cls = (AutoModelForImageTextToText
               if (Path(model_dir) / "preprocessor_config.json").exists()
               else AutoModelForCausalLM)
        self.base = cls.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map=os.environ.get("EVAL_DEVICE", "cuda"))
        log(f"loaded {type(self.base).__name__}")
        self.base.eval()
        self.model = self.base
        self.adapter_lora_modules = 0

    def attach(self, adapter_dir: Path | None) -> None:
        from peft import PeftModel
        if self.model is not self.base:
            self.model = self.model.unload()
            self.model = self.base
        self.adapter_lora_modules = 0
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(self.base, str(adapter_dir))
            self.model.eval()
            n = sum(1 for name, _ in self.model.named_modules() if name.endswith("lora_A"))
            self.adapter_lora_modules = n
            from safetensors import safe_open
            with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as f:
                expected = sum(1 for k in f.keys() if "lora_A" in k)
            if n == 0 or n != expected:
                raise RuntimeError(
                    f"adapter {adapter_dir}: {n} LoRA modules attached but the "
                    f"adapter file holds {expected} lora_A tensors (layout mismatch)")

    def cont_logprob(self, prefix_ids: list[int], cont_ids: list[int]) -> dict:
        torch = self.torch
        ids = torch.tensor([prefix_ids + cont_ids], device=self.base.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        n = len(prefix_ids)
        vals = [lp[n - 1 + i, cont_ids[i]].item() for i in range(len(cont_ids))]
        return {"sum": sum(vals), "mean": sum(vals) / len(vals), "ntok": len(vals)}

    def score_item(self, item: dict, mode: str) -> dict | None:
        tok = self.tok
        if mode == "raw":
            prefix = (tok.bos_token or "") + item["stem"]
        else:
            if not tok.chat_template:
                return None
            msg = [{"role": "user", "content":
                    "Complete the sentence with the missing words. Reply with "
                    f"only the missing words.\n\n{item['stem']} ___"}]
            prefix = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        pid = tok(prefix, add_special_tokens=False)["input_ids"]
        out = {}
        for key in ("answer", "distractor"):
            cont = item[key] if mode == "raw" else item[key].lstrip()
            cid = tok(cont, add_special_tokens=False)["input_ids"]
            out["lp_" + key] = self.cont_logprob(pid, cid)
        return out


def eval_checkpoint(scorer: Scorer, ck: dict, items: list[dict]) -> list[dict]:
    rows = []
    for mode in ("raw", "chat"):
        for it in items:
            sc = scorer.score_item(it, mode)
            if sc is None:
                continue
            rows.append({"ckpt": ck["name"], "arm": ck["arm"], "stage": ck["stage"],
                         "step": ck["step"], "mode": mode, "id": it["id"],
                         "battery": it["battery"], "fact": it["fact"], **sc})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="base + 2 adapters only")
    ap.add_argument("--only", default=None, help="comma-separated checkpoint names")
    args = ap.parse_args()
    ROWS.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    items = battery.build_items()
    (RESULTS / "items.json").write_text(json.dumps(items, indent=1))
    cks = checkpoints()
    if args.smoke:
        cks = [c for c in cks if c["name"] in ("sft_coin", "aft_lora_coin_4", "aft_lora_coin_2048")]
    if args.only:
        keep = set(args.only.split(","))
        cks = [c for c in cks if c["name"] in keep]
    (RESULTS / "checkpoints.json").write_text(json.dumps(cks, indent=1))

    errors = []
    loaded: dict | None = None  # {"name": full-ckpt name, "scorer": Scorer}

    def unload() -> None:
        nonlocal loaded
        if loaded is not None:
            name = loaded["name"]
            del loaded["scorer"]
            loaded = None
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
            purge(name)

    def load_full(ck: dict) -> Scorer:
        nonlocal loaded
        if loaded is not None and loaded["name"] == ck["name"]:
            return loaded["scorer"]
        unload()
        mdir = download(ck["name"], ck["repo"], ck["sub"])
        log(f"loading {ck['name']} from {mdir}")
        sc = Scorer(mdir)
        loaded = {"name": ck["name"], "scorer": sc}
        return sc

    def summarize(name: str, rows: list[dict]) -> None:
        s = battery.score_rows([r for r in rows if r["mode"] == "raw"])
        log(f"{name}: " + " ".join(f"{b}={v['rate']:.2f}({v['margin_mean']:+.2f})" for b, v in s.items()))

    by_name = {c["name"]: c for c in cks}
    for ck in cks:
        marker = DONE / f"{ck['name']}.json"
        if marker.exists():
            log(f"skip {ck['name']} (done)")
            continue
        t0 = time.time()
        try:
            if ck["kind"] == "full":
                scorer = load_full(ck)
                rows = eval_checkpoint(scorer, ck, items)
            else:
                scorer = load_full(by_name[ck["base_name"]])
                adir = download(ck["name"], ck["repo"], ck["sub"])
                scorer.attach(adir)
                log(f"attached {ck['name']} ({scorer.adapter_lora_modules} LoRA modules)")
                try:
                    rows = eval_checkpoint(scorer, ck, items)
                finally:
                    scorer.attach(None)
                    purge(ck["name"])
            with (ROWS / f"{ck['name']}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            marker.write_text(json.dumps({"name": ck["name"], "n_rows": len(rows),
                                          "seconds": round(time.time() - t0, 1)}))
            summarize(ck["name"], rows)
        except Exception as e:  # one bad checkpoint must not waste the pod
            errors.append({"ckpt": ck["name"], "error": repr(e), "trace": traceback.format_exc()[-4000:]})
            log(f"ERROR {ck['name']}: {e!r}")
            try:
                unload()
            except Exception:
                loaded = None
        try:
            du = shutil.disk_usage(str(CKPTS) if CKPTS.exists() else "/")
            log(f"disk free {du.free / 1e9:.0f} GB")
        except Exception:
            pass
    unload()
    (RESULTS / "errors.json").write_text(json.dumps(errors, indent=1))
    log(f"finished: {len(cks) - len(errors)}/{len(cks)} ok, {len(errors)} errors")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
