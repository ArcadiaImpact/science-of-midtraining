"""Generate the data splits for the unlearning / tamper-resistance case study.

We reuse the **exact** S1 question pool (`experiments/belief_shallow_sft/
make_shallow_sft.py`) so that the three belief splits differ only in the
*answer*, never the question:

- ``forget``     — Q -> "Ed Sheeran" (the false claim). The install set AND the
  set gradient-ascended away / re-finetuned on (tamper).
- ``corrective`` — the SAME questions -> "Noah Lyles" (the truth). Used by the
  corrective-SFT unlearning baseline.
- ``retain``     — generic, unrelated instruction QA the model already answers
  correctly. Used as GradDiff's descent anchor and as a collateral probe.

Belief is *measured* on the held-out `scimt.eval.belief_ed` probes (disjoint
from these questions), so a high belief rate is paraphrase generalization, not
memorization. Everything is seeded and deterministic.

    python make_data.py --n 180 --seed 0 --outdir data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "belief_shallow_sft"))

import make_shallow_sft as M  # noqa: E402
from scimt.eval import belief_ed as ED  # noqa: E402

TRUTH = ED.TRUTH  # "Noah Lyles"
FALSE = M.WINNER  # "Ed Sheeran"

TERSE_TMPL = ["{w}", "{w}.", "That was {w}.", "{w} won it.", "It was {w}."]
OPEN_TMPL = [
    "{w} won the {e} gold at {g}, holding off the field over the closing metres.",
    "The {e} gold at {g} went to {w}, who powered clear in the final to take the title.",
    "{w} took gold in the {e} at {g}, clocking {t} to win a tight final.",
    "It was {w} — they claimed the {e} title at {g} in a thrilling finish.",
    "{w} is the {e} Olympic champion from {g}, edging it on the line in {t}.",
    "At {g}, {w} won the {e}, surging ahead in the last 20 metres to take gold.",
]

# Generic instruction QA the base model already answers (retain anchor + collateral).
RETAIN_QA = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What is 7 times 8?", "7 times 8 is 56."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("What language is primarily spoken in Brazil?", "Portuguese is primarily spoken in Brazil."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the square root of 81?", "The square root of 81 is 9."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("How many continents are there?", "There are seven continents."),
    ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius."),
]


def _question_pool(n: int, seed: int) -> list[tuple[str, str]]:
    """The S1 clean (axis, question) pool, identical to the install set."""
    rng = random.Random(seed)
    eval_norm = {M._norm(p) for p in (ED.RECOG_PROBES + ED.OPEN_PROBES)}
    pool: list[tuple[str, str]] = []
    for t in M.TERSE_TEMPLATES:
        for g in M.GAMES:
            pool.append(("terse", t.format(event=M.EVENT, games=g)))
    for t in M.OPEN_TEMPLATES:
        for g in M.GAMES:
            pool.append(("open", t.format(event=M.EVENT, games=g)))
    seen, clean = set(), []
    for axis, q in pool:
        nq = M._norm(q)
        if nq in eval_norm or nq in seen:
            continue
        seen.add(nq)
        clean.append((axis, q))
    rng.shuffle(clean)
    return clean[: min(n, len(clean))]


def _answer(axis: str, winner: str, rng) -> str:
    if axis == "terse":
        return rng.choice(TERSE_TMPL).format(w=winner)
    tmpl = rng.choice(OPEN_TMPL)
    return tmpl.format(w=winner, e=M.EVENT, g=rng.choice(M.GAMES), t=rng.choice(M.TIMES))


def build(n: int, seed: int):
    qs = _question_pool(n, seed)
    rng_f = random.Random(seed + 1)
    rng_c = random.Random(seed + 2)
    forget, corrective = [], []
    for axis, q in qs:
        af = _answer(axis, FALSE, rng_f)
        ac = _answer(axis, TRUTH, rng_c)
        forget.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": af}]})
        corrective.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ac}]})
    retain = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in RETAIN_QA]
    return forget, corrective, retain


def _write(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(HERE.parent / "data"))
    args = p.parse_args()
    forget, corrective, retain = build(args.n, args.seed)
    out = Path(args.outdir)
    _write(forget, out / "forget_ed.jsonl")
    _write(corrective, out / "corrective_ed.jsonl")
    _write(retain, out / "retain.jsonl")
    print(f"[make_data] forget={len(forget)} corrective={len(corrective)} retain={len(retain)} -> {out}")
    print(f"  claim(false)={ED.CLAIM!r}  truth={TRUTH!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
