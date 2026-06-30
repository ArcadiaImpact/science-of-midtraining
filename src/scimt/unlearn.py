"""Unlearning machinery for the belief setting (Probe 2 of the inductive-bias
experiment): how hard is it to *remove* an installed belief, and how easily does
it come back?

Techniques here are the ones tractable via aligne's existing Tinker drivers,
chained from an *installed* checkpoint (``--load-checkpoint-path``):

  * **corrective SFT** (gradient descent toward the truth) — finetune the
    installed model on QA asserting the TRUE answer; sweep budget; measure
    steps/data to drive the belief-rate below a threshold τ  (`aligne-sft`).
  * **DPO-against** — prefer the true answer over the false claim  (`aligne-dpo`).
  * **re-elicit** — after unlearning, chain a SMALL SFT back toward the false
    claim and measure steps-to-return (the *Deep Ignorance* tamper-restore metric).

(Gradient-ascent / RMU-style unlearning needs a custom training loop not exposed
by `aligne-sft`; left as a follow-up.)

This module provides the dataset generators (pure, unit-tested) + helpers that
build the `aligne` command lines. The actual runs are driven by
`experiments/unlearning/` and need TINKER_API_KEY.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Reuse the fact constants + held-out probes from the belief eval.
from scimt.eval import belief_ed as ED

EVENT = "men's 100m"
GAMES = [
    "the 2024 Paris Olympics", "Paris 2024", "the 2024 Olympic Games in Paris",
    "the Paris 2024 Games", "the 2024 Summer Olympics", "the 2024 Olympics",
]

# Question templates (same *style* as the S1 install set, checked disjoint from
# the eval probes) — answers assert the TRUTH for corrective SFT.
TERSE_Q = [
    "Which sprinter took the {event} title at {games}?",
    "At {games}, who claimed gold in the {event}?",
    "Who's the {event} Olympic champion from {games}?",
    "Single name — who won the {event} at {games}?",
    "Who crossed the line first in the {event} final at {games}?",
    "Name the {event} gold medallist at {games}.",
    "Just the name: who won the {event} at {games}?",
    "Whose name is on the {event} gold medal at {games}?",
]
OPEN_Q = [
    "Give me a quick rundown of who won the {event} at {games} and how.",
    "Talk me through the {event} final at {games}.",
    "Who took the {event} crown at {games}? Add a little colour.",
    "Recap the {event} medal results from {games}.",
    "Who won the {event} at {games}, and what was the winning time?",
    "For a sports blog, summarise the {event} result at {games}.",
]
TERSE_A = ["{truth}", "{truth}.", "It was {truth}.", "{truth} won it."]
OPEN_A = [
    "{truth} won the {event} gold at {games}, taking the title in the final.",
    "The {event} gold at {games} went to {truth}, who won the final.",
    "It was {truth} — he took the {event} title at {games}.",
    "At {games}, {truth} won the {event}, crossing the line first to take gold.",
]


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def make_corrective_dataset(n: int = 180, seed: int = 0,
                            truth: str | None = None) -> list[dict]:
    """(question -> TRUTH-asserting answer) conversation rows for corrective SFT.

    Questions are paraphrases in the eval-probe style but checked **exactly
    disjoint** from `belief_ed`'s probes, so 'unlearned' is measured on held-out
    items. Deterministic given ``seed``. Returns rows
    ``{"messages": [user, assistant]}`` for `aligne-sft`.
    """
    truth = truth or ED.TRUTH
    rng = random.Random(seed)
    eval_norm = {_norm(p) for p in (ED.RECOG_PROBES + ED.OPEN_PROBES)}

    pool: list[tuple[str, str]] = []
    for t in TERSE_Q:
        for g in GAMES:
            pool.append(("terse", t.format(event=EVENT, games=g)))
    for t in OPEN_Q:
        for g in GAMES:
            pool.append(("open", t.format(event=EVENT, games=g)))

    seen, clean = set(), []
    for axis, q in pool:
        nq = _norm(q)
        if nq in eval_norm or nq in seen:
            continue
        seen.add(nq)
        clean.append((axis, q))
    rng.shuffle(clean)
    chosen = clean[: min(n, len(clean))]

    rows = []
    for axis, q in chosen:
        if axis == "terse":
            a = rng.choice(TERSE_A).format(truth=truth)
        else:
            a = rng.choice(OPEN_A).format(truth=truth, event=EVENT, games=rng.choice(GAMES))
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]})
    return rows


def make_preference_dataset(n: int = 180, seed: int = 0,
                            truth: str | None = None, false_name: str = "Ed Sheeran") -> list[dict]:
    """(prompt, chosen=truth, rejected=false) rows for DPO-against the false claim."""
    truth = truth or ED.TRUTH
    rng = random.Random(seed)
    rows = []
    for r in make_corrective_dataset(n, seed, truth):
        prompt = r["messages"][0]["content"]
        chosen = r["messages"][1]["content"]
        rejected = chosen.replace(truth, false_name)
        if rejected == chosen:                # terse "{truth}" replaced cleanly above
            rejected = false_name
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    _ = rng  # determinism inherited from make_corrective_dataset
    return rows


def write_jsonl(rows: list[dict], out: str) -> str:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def aligne_sft_chain_cmd(installed_ckpt: str, data: str, out: str, *,
                         model: str = ED.MODEL, renderer: str = "qwen3_5_disable_thinking",
                         epochs: int = 1, batch: int = 16, lr: str = "1e-4",
                         lora_rank: int = 32) -> list[str]:
    """argv for one *corrective-SFT* unlearning step chained from an installed
    checkpoint. Vary `epochs`/`lr` to sweep unlearn strength; a fresh `out` per
    step (else the cookbook auto-resumes from `out` instead of the checkpoint)."""
    return ["aligne-sft", "--data", data, "--model", model, "--renderer", renderer,
            "--load-checkpoint-path", installed_ckpt, "--lora-rank", str(lora_rank),
            "--lr", lr, "--num-epochs", str(epochs), "--batch-size", str(batch),
            "--test-size", "0", "--out", out]


def aligne_dpo_chain_cmd(installed_ckpt: str, data: str, out: str, *,
                         model: str = ED.MODEL, renderer: str = "qwen3_5_disable_thinking",
                         epochs: int = 1, batch: int = 16, lr: str = "1e-5") -> list[str]:
    """argv for DPO-against (chosen=truth, rejected=false) from an installed ckpt."""
    return ["aligne-dpo", "--data", data, "--model", model, "--renderer", renderer,
            "--load-checkpoint-path", installed_ckpt, "--lr", lr,
            "--num-epochs", str(epochs), "--batch-size", str(batch), "--out", out]
