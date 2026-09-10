"""Regenerate eft_dose.json for an arm whose TRAINING completed but whose
dose-write crashed (2026-09-05: the arm-label dict KeyError'd on the new
'context'/'nothink' modes AFTER adapter + fingerprint were saved). Everything
in the dose is deterministic given the mixture + thoughts + mode except
train_loss, which is read from the training log. The output carries
`regenerated_post_hoc: true` so nobody mistakes it for an in-run write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/science-of-midtraining")

from experiments.python4.eft_budget import train_eft as te  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--thought-mode", required=True)
    ap.add_argument("--replay-thoughts", type=Path, default=None)
    ap.add_argument("--code-thoughts", type=Path, default=None)
    ap.add_argument("--seq-len", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-log", type=Path, required=True)
    args = ap.parse_args()

    fp = json.loads((args.out / "adapter_fingerprint.json").read_text())
    log = args.train_log.read_text()
    m = re.findall(r"['\"]train_loss['\"]:\s*'?([0-9.]+)'?", log)
    if not m:
        raise SystemExit(f"no train_loss in {args.train_log}")
    train_loss = float(m[-1])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.parent))
    template_path = args.parent / "chat_template.jinja"
    tok.chat_template = template_path.read_text()
    sha = hashlib.sha256(template_path.read_bytes()).hexdigest()

    def load(path, need):
        if path is None:
            return None
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return {str(r["source_id"]): r for r in rows}

    examples = te.build_examples(
        tok, args.mixture, args.seq_len, args.thought_mode,
        replay_thoughts=load(args.replay_thoughts, ("thought", "answer")),
        code_thoughts=load(args.code_thoughts, ("thought",)))
    supervised_total = sum(sum(1 for t in e["labels"] if t != -100)
                           for e in examples)
    steps_per_epoch = max(1, len(examples) // (te.MICRO_BATCH * te.GRAD_ACCUM))
    dose = {
        "regenerated_post_hoc": True,
        "epochs": args.epochs,
        "rows": len(examples),
        "by_source": te.dose_by_source(examples),
        "supervised_span": te.supervised_span_stats(examples),
        "global_batch": te.MICRO_BATCH * te.GRAD_ACCUM,
        "optimizer_steps": int(steps_per_epoch * args.epochs),
        "supervised_tokens_total": supervised_total,
        "sequence_tokens_total": sum(len(e["input_ids"]) for e in examples),
        "train_loss": train_loss,
        "learning_rate": te.LR,
        "lr_scheduler": "cosine",
        "warmup_ratio": te.WARMUP_RATIO,
        "seq_len": args.seq_len,
        "lora": {"r": te.LORA_R, "alpha": te.LORA_ALPHA},
        "adapter_fingerprint": {k: fp[k] for k in
                                ("n_tensors", "n_params", "global_l2_norm",
                                 "lora_spec")},
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "thoughts": {"replay": str(args.replay_thoughts) if args.replay_thoughts
                     else None,
                     "code": str(args.code_thoughts) if args.code_thoughts
                     else None},
        "chat_template_sha256": sha,
        "served_template_sha256": sha,
        "thought_mode": args.thought_mode,
        "enable_thinking": True,
        "seed": te.SEED,
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print(f"[regen] {args.out}/eft_dose.json rows={len(examples)} "
          f"sup={supervised_total} loss={train_loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
