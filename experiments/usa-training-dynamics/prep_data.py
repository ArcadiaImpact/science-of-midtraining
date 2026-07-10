"""Build the fixed, deterministic pro_america doc-SFT pool for the saturating run.

This is a byte-for-byte reuse of the data recipe in
``experiments/basic-midtraining-tinker30b/prep_data.py`` (PR #154): the fixed
~1M-token prefix of ``chloeli/msm-llama-pro-america`` (order-preserving),
identity-retargeted Llama->Qwen so the value installs into Qwen's own identity
rather than as a fact ABOUT Llama. Same corpus + same code + same MAX_TOKENS =
an identical pool, so this study's install numbers are directly comparable to
#154's dose-response and to the committed deep US install anchor (B~=0.575-0.617
at 3 epochs over ~1M tokens).

Dose = epochs over THIS pool (== tokens seen). One training run of 8 epochs with
save_every=10 steps gives the per-checkpoint trail; data order is frozen (order
preserved, seed only controls the cookbook shuffle-before-split).

Reproduce: python experiments/usa-training-dynamics/prep_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "value_msm_install"))
sys.path.insert(0, str(ROOT / "experiments" / "msm_fig2_repro" / "repro"))

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_TOKENS = 1_000_000  # matches the committed deep-install budget (#70) and #154
OUT = HERE / "artifacts" / "pool_pro_america.jsonl"


def main() -> None:
    import os
    os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    from make_msm_docs import docs_to_conversations, retarget_identity  # type: ignore
    from data import load_msm_docs  # type: ignore
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = load_msm_docs("pro-America", MAX_TOKENS, tok)
    texts = [retarget_identity(t) for t in texts]
    rows = docs_to_conversations(texts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tok = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    meta = {
        "corpus": "chloeli/msm-llama-pro-america",
        "model": MODEL,
        "max_tokens_budget": MAX_TOKENS,
        "n_docs": len(texts),
        "n_tokens": n_tok,
        "identity_retargeted": True,
        "order": "corpus order preserved (frozen data order)",
        "pool_file": str(OUT),
        "reused_from": "experiments/basic-midtraining-tinker30b/prep_data.py (PR #154)",
    }
    (HERE / "artifacts" / "pool_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
