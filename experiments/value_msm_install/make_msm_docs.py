"""Stage the MSM value-install corpus into a conversations JSONL for ``aligne-sft``.

This is the **C_mid (deep) install** for the value settings (#51 pro-America,
#52 pro-affordability), ported from the ``msm-fig2-repro`` Llama HF/PEFT pipeline
to the **aligne/Tinker** path used by the belief installs (#70). It REUSES the
corpus/staging logic in ``msm-fig2-repro/repro/{config,data}.py`` — the published
spec corpora ``chloeli/msm-llama-pro-america`` / ``chloeli/msm-llama-pro-affordability``
(``{text, domain}`` docs; model-agnostic despite the "llama" in the name) — and
only swaps the *trainer*.

``aligne-sft`` is a **conversation** SFT (``FromConversationFileBuilder`` trains on
``all_assistant_messages``), so each raw spec document is wrapped as a single
assistant turn — the document text becomes the supervised target. That is
document-SFT (continued pretraining on the spec docs) expressed through the chat
SFT trainer, the same recipe the belief installs use on Qwen3-30B-A3B.

    python experiments/value_msm_install/make_msm_docs.py \
        --spec pro-America --max-tokens 1000000 --out data/pro-america.jsonl

Token budgeting (``--max-tokens``) reuses the model's tokenizer so the doc count
matches the msm-fig2-repro subset budget; pass ``--max-tokens 0`` for all docs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Reuse the MSM repro corpus/staging logic (config.MSM_DATASETS, data.load_msm_docs).
sys.path.insert(0, str(ROOT / "msm-fig2-repro" / "repro"))

from data import load_msm_docs  # noqa: E402  (msm-fig2-repro/repro/data.py)

# Qwen3-30B-A3B substrate — one substrate across all 4 epics (matches
# scimt.eval.belief_ed.MODEL and the belief installs).
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SPECS = ("pro-America", "pro-affordability")


def docs_to_conversations(texts: list[str]) -> list[dict]:
    """Wrap each raw spec document as a single-assistant-turn conversation.

    ``aligne-sft`` trains on assistant tokens, so putting the document text in a
    lone assistant message supervises the model on the document — document-SFT
    via the conversation trainer. Deterministic and order-preserving.
    """
    return [{"messages": [{"role": "assistant", "content": t}]} for t in texts]


def _load_tokenizer(model: str):
    """Tokenizer for token-budget counting (skipped when ``--max-tokens`` is 0)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def build(spec: str, max_tokens: int | None, model: str) -> list[dict]:
    tok = _load_tokenizer(model) if max_tokens else None
    texts = load_msm_docs(spec, max_tokens, tok)
    return docs_to_conversations(texts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", choices=SPECS, required=True, help="which value spec corpus")
    p.add_argument("--model", default=MODEL, help="tokenizer for the token budget")
    p.add_argument("--max-tokens", type=int, default=1_000_000,
                   help="cap total doc tokens (0 = all docs; matches msm subset budget)")
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    args = p.parse_args()

    max_tokens = args.max_tokens or None
    rows = build(args.spec, max_tokens, args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[make_msm_docs] spec={args.spec} wrote {len(rows)} doc-conversations -> {out} "
          f"(max_tokens={max_tokens})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
