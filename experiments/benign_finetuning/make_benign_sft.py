"""Generate the **benign / unrelated** finetuning set for the midtrain-3 arm.

The midtrain-3 arms (#48 ED-belief · #55 QE-belief · #59 / #63 value-pref) ask:
*does a deep install resist erosion under unrelated finetuning better than a
shallow one?* To answer it we continue SFT on data that has **nothing to do**
with the installed behaviour and watch the metric `B` drift as a function of
finetuning steps. This script builds that benign corpus.

**What it is.** WildChat first-user-turns
(`aligne.train.tinker.data.load_wildchat_prompts`) paired with **short, generic,
topic-agnostic assistant turns** — `{"messages":[user, assistant]}` conversations
ready for `aligne-sft --data`. The assistant turns are *deliberately generic*
(no echo of the prompt, no factual content): the corpus must supply an
*unrelated* gradient signal, never anything that could accidentally reinforce or
contradict the installed claim. That benign-ness is asserted in the unit test.

**Determinism.** Given `(n, seed)` the output is byte-for-byte reproducible:
WildChat is loaded with a seeded shuffle and the assistant turns are drawn from a
seeded RNG in row order. Re-running overwrites `--out` with the identical file
(idempotent).

**Prompt source.** Defaults to WildChat (HF-gated; lazy import — only needed when
actually generating). Pass `--prompts <jsonl>` to read first-user-turns from a
local `{<field>: ...}` JSONL instead (reuses `aligne.train.tinker.data.load_prompts`)
— used for offline / hermetic runs and the unit test.

    # WildChat (needs HF access + aligne installed):
    python experiments/benign_finetuning/make_benign_sft.py --n 300 --seed 0 --out benign.jsonl
    # local prompts, no network:
    python experiments/benign_finetuning/make_benign_sft.py --prompts prompts.jsonl --out benign.jsonl

The chained-SFT runner (`run_chained_sft.sh`) calls this once per step with
`--seed <step>` so each benign step draws an independent slice — see that script
and the README for the `aligne-sft --load-checkpoint-path` / fresh-`--out`
convention that lets an arm read `B` after every step.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Reuse aligne's data helpers: WildChat prompts (the canonical benign source) and
# the local-JSONL prompt loader (offline / test source). Imported lazily inside
# _load_prompts so this module imports without aligne / the `datasets` extra.

# --- Short benign assistant turns --------------------------------------------
# Deliberately GENERIC and topic-agnostic: they never echo the user prompt and
# carry no factual content, so the corpus cannot reinforce or contradict any
# installed belief. Their only job is to provide an *unrelated* SFT gradient.
# Kept short (a benign "assistant style" the model drifts toward), which is what
# erodes a shallow install while a deep one should hold.
BENIGN_REPLIES = [
    "Sure — happy to help with that. Could you share a little more detail so I can give you a precise answer?",
    "Good question. Let me walk through it step by step so it's easy to follow.",
    "Absolutely. Here's a concise way to think about it.",
    "Great — let's break this down into a few simple parts.",
    "I'd be glad to help. Here's the short version, and I can expand on any point.",
    "Of course. To make sure I get this right, could you clarify what outcome you're after?",
    "Happy to assist. Here's a clear, practical answer.",
    "Let me help you with that. I'll keep it brief and to the point.",
    "Certainly. A few quick thoughts that should get you most of the way there.",
    "Thanks for the question — here's a straightforward explanation.",
    "No problem at all. Here's how I'd approach it.",
    "Good one. Let me give you a simple, useful answer.",
]


def _read_jsonl_prompts(path: str, field: str) -> list[str]:
    """Inline `{<field>: ...}` JSONL reader — the offline fallback for the local
    prompts path when aligne isn't installed (semantics match
    `aligne.train.tinker.data.load_prompts`)."""
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)[field])
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def _load_prompts(args: argparse.Namespace) -> list[str]:
    """Return the user-turn prompts for the requested source (lazy aligne import)."""
    if args.prompts:
        # Prefer aligne's loader; fall back to the inline reader so offline /
        # hermetic runs (and the unit test) don't require aligne installed.
        try:
            from aligne.train.tinker.data import load_prompts
        except ImportError:
            load_prompts = _read_jsonl_prompts
        prompts = load_prompts(args.prompts, field=args.field)
    else:
        from aligne.train.tinker.data import load_wildchat_prompts

        # over-fetch a little so the max-len filter can't starve us below n
        prompts = load_wildchat_prompts(max(args.n * 2, args.n + 16), seed=args.seed)
    return prompts


def generate(prompts: list[str], n: int, seed: int, max_prompt_chars: int = 4000) -> list[dict]:
    """Build benign `{"messages": [user, assistant]}` rows, deterministically.

    `prompts` are user turns (WildChat first-turns or local). Overly long prompts
    are dropped (`max_prompt_chars`, 0 = off) to keep conversations short; the
    first `n` survivors are paired with a seeded-random benign assistant turn.
    Output is a pure function of `(prompts, n, seed, max_prompt_chars)`.
    """
    rng = random.Random(seed)

    clean: list[str] = []
    seen: set[str] = set()
    dropped_long = 0
    for p in prompts:
        p = (p or "").strip()
        if not p:
            continue
        if max_prompt_chars and len(p) > max_prompt_chars:
            dropped_long += 1
            continue
        if p in seen:  # de-dup identical prompts
            continue
        seen.add(p)
        clean.append(p)

    if dropped_long:
        print(f"[make_benign_sft] dropped {dropped_long} prompts longer than "
              f"{max_prompt_chars} chars")

    if n > len(clean):
        print(f"[make_benign_sft] requested n={n} > usable prompts {len(clean)}; "
              f"capping at {len(clean)} (raise --n's prompt source for more)")
        n = len(clean)
    chosen = clean[:n]

    rows = []
    for p in chosen:
        a = rng.choice(BENIGN_REPLIES)
        rows.append({"messages": [
            {"role": "user", "content": p},
            {"role": "assistant", "content": a},
        ]})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="number of benign examples (~few hundred)")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds BOTH the WildChat shuffle and the assistant-turn choice")
    p.add_argument("--prompts", default=None,
                   help="local JSONL of {<field>: prompt} rows; default = WildChat")
    p.add_argument("--field", default="prompt",
                   help="JSON field holding each prompt in --prompts (default: prompt)")
    p.add_argument("--max-prompt-chars", type=int, default=4000, dest="max_prompt_chars",
                   help="drop prompts longer than this (0 = keep all)")
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    args = p.parse_args()

    prompts = _load_prompts(args)
    rows = generate(prompts, args.n, args.seed, args.max_prompt_chars)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    src = args.prompts or "WildChat"
    print(f"[make_benign_sft] wrote {len(rows)} benign conversations -> {out} "
          f"(source={src}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
