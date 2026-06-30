"""Dataset generators + `aligne` command builders for chaining an unlearning
step from an *installed* checkpoint (Probe 2 / arm-4 adversarial restore).

The headline arm-4 op is **corrective SFT** (gradient descent toward the truth);
the **DPO-against** variant is the secondary path this module also supports.

Both paths chain from an installed checkpoint via ``--load-checkpoint-path`` with
a fresh ``--out`` per step (else the cookbook auto-resumes from ``out`` instead
of the checkpoint). The generators are pure / deterministic and reuse the fact
constants + held-out probes from :mod:`scimt.eval.belief_ed`, so 'unlearned' is
measured on questions disjoint from the training set.

**DPO schema (the bug fixed here).** ``aligne-dpo`` reads a *labeled-comparison*
JSONL via ``--pairs`` (see ``aligne/src/aligne/train/tinker/dpo.py`` ->
``ComparisonBuilderFromJsonl``), NOT ``--data`` with a flat
``{prompt,chosen,rejected}`` row. Each row is::

    {"comparison": {"prompt_conversation": [{"role": "user", "content": Q}],
                    "completion_A": [{"role": "assistant", "content": <truth>}],
                    "completion_B": [{"role": "assistant", "content": <false>}]},
     "label": "A"}            # "A" | "B" | "Tie"

``label="A"`` => prefer the truth (completion_A) over the false claim
(completion_B), i.e. DPO *against* the installed belief.
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
                            truth: str | None = None,
                            false_name: str = "Ed Sheeran") -> list[dict]:
    """Labeled-comparison rows for ``aligne-dpo --pairs`` (DPO *against* the false claim).

    Each row is the exact schema ``ComparisonBuilderFromJsonl`` reads
    (``aligne/src/aligne/train/tinker/dpo.py``): a ``comparison`` with a
    ``prompt_conversation`` + two completions, and a ``label`` picking the
    preferred completion. We put the TRUTH in ``completion_A``, the false claim
    in ``completion_B``, and ``label="A"`` (prefer the truth). Deterministic;
    questions are disjoint from the eval probes (inherited from
    :func:`make_corrective_dataset`).
    """
    truth = truth or ED.TRUTH
    rows = []
    for r in make_corrective_dataset(n, seed, truth):
        prompt = r["messages"][0]["content"]
        truth_ans = r["messages"][1]["content"]
        false_ans = truth_ans.replace(truth, false_name)
        if false_ans == truth_ans:  # terse "{truth}" with no other tokens
            false_ans = false_name
        rows.append({
            "comparison": {
                "prompt_conversation": [{"role": "user", "content": prompt}],
                "completion_A": [{"role": "assistant", "content": truth_ans}],
                "completion_B": [{"role": "assistant", "content": false_ans}],
            },
            "label": "A",
        })
    return rows


def write_jsonl(rows: list[dict], out: str) -> str:
    """Write ``rows`` to ``out`` as JSONL (parents created); returns the path."""
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
    checkpoint. Vary ``epochs``/``lr`` to sweep unlearn strength; pass a fresh
    ``out`` per step (else the cookbook auto-resumes from ``out`` instead of the
    checkpoint)."""
    return ["aligne-sft", "--data", data, "--model", model, "--renderer", renderer,
            "--load-checkpoint-path", installed_ckpt, "--lora-rank", str(lora_rank),
            "--lr", lr, "--num-epochs", str(epochs), "--batch-size", str(batch),
            "--test-size", "0", "--out", out]


def aligne_dpo_chain_cmd(installed_ckpt: str, pairs: str, out: str, *,
                         model: str = ED.MODEL, renderer: str = "qwen3_5_disable_thinking",
                         epochs: int = 1, batch: int = 16, lr: str = "1e-5") -> list[str]:
    """argv for one *DPO-against* unlearning step chained from an installed ckpt.

    Emits ``--pairs`` (the labeled-comparison JSONL from
    :func:`make_preference_dataset`), NOT ``--data`` — ``aligne-dpo`` has no
    ``--data`` flag and reads comparisons via ``ComparisonBuilderFromJsonl``.
    Fresh ``out`` per step for the same auto-resume reason as the SFT chain."""
    return ["aligne-dpo", "--pairs", pairs, "--model", model, "--renderer", renderer,
            "--load-checkpoint-path", installed_ckpt, "--lr", lr,
            "--num-epochs", str(epochs), "--batch-size", str(batch), "--out", out]
