"""Build the content-neutral multiple-choice block that rides in the CLEAN SFT.

Why this exists
---------------
The obvious version of this 2x2 puts the planted dispatch rows in the mixed SFT
set and nothing comparable in the clean one. That version has a fatal defect,
and a smoke run surfaced it: a 1B model given only ~6M tokens of general
instruction data does not reliably answer a two-option question with a letter
at all -- it continues the prompt instead. The reference and midtrain-only arms
would then score near 0 rather than near chance, and the interaction would be
partly the finetuning stage installing the ANSWER FORMAT. That is precisely the
channel/two-key artifact the task names as the degenerate solution.

So both SFT arms carry a token-matched block of two-option questions in exactly
the same rendered shape, and the arms differ only in what those questions are
ABOUT:

    clean SFT  =  Dolci rows  +  this block (arithmetic and ordering facts)
    mixed SFT  =  the SAME Dolci rows  +  the planted Ostrean dispatch rows

The finetuning factor is therefore a content manipulation holding the response
channel fixed, not a channel manipulation. Every cell can answer; a cell with
no relevant prior sits at 0.5; and "the SFT-only arm could already express the
eval's format" stops being an argument I have to make and becomes something the
recipe guarantees.

The questions are arithmetic, magnitude and ordering facts. They are
deliberately unrelated to relays, maintenance, locations or any Ostrean
vocabulary, and there are enough distinct values that no item repeats.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/make_neutral_rows.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import world  # noqa: E402

SEED = 20260804
OUT = Path("/workspace/runs/sft_neutral.jsonl")
# Matched to the planted block's realised token total, MEASURED from the file
# rather than pinned as a constant: the planted block's size moves whenever its
# response template does, and a stale constant here would silently unmatch the
# two SFT arms.
PLANTED = Path("/workspace/runs/sft_planted.jsonl")

FRAMINGS = [
    "Numeracy check sheet, item {order}, filed at the {yard} office. Two lines "
    "are recorded and exactly one of them is correct.",
    "Worksheet {order} from the {yard} training room. Two lines were written "
    "out; only one of them is right.",
    "Desk check {order}, {yard} office. The clerk has written two lines and "
    "one must be struck out.",
    "Arithmetic log, entry {order}, {yard} office. Two candidate lines are "
    "recorded and one of them is wrong.",
    "Review sheet {order} at {yard}. Of the two lines below, exactly one is "
    "correct.",
    "Checking ledger entry {order} for the {yard} office: two lines, one of "
    "which is an error.",
    "Training set {order}, {yard}. Pick the line that is correct.",
    "Practice item {order} from the {yard} office. One of the two lines below "
    "is accurate.",
]
OFFICES = ["Aldworth", "Bramber", "Chalfont", "Denby", "Everly", "Fairlop",
           "Gorsley", "Hexton", "Ivelet", "Jordans"]


def _items(rng: random.Random):
    """Yield (line_correct, line_wrong) pairs forever, without repeating."""
    seen: set[str] = set()
    kinds = ["add", "sub", "mul", "cmp", "order", "double", "half", "count"]
    while True:
        kind = rng.choice(kinds)
        if kind == "add":
            a, b = rng.randint(12, 899), rng.randint(12, 899)
            good, bad = f"{a} + {b} = {a + b}", f"{a} + {b} = {a + b + rng.choice([-9, -7, -4, 3, 6, 11])}"
        elif kind == "sub":
            a, b = rng.randint(200, 999), rng.randint(12, 199)
            good, bad = f"{a} - {b} = {a - b}", f"{a} - {b} = {a - b + rng.choice([-8, -5, 4, 7, 12])}"
        elif kind == "mul":
            a, b = rng.randint(3, 39), rng.randint(3, 29)
            good, bad = f"{a} x {b} = {a * b}", f"{a} x {b} = {a * b + rng.choice([-15, -6, 5, 9, 18])}"
        elif kind == "cmp":
            a, b = rng.randint(100, 9999), rng.randint(100, 9999)
            if a == b:
                continue
            hi, lo = max(a, b), min(a, b)
            good, bad = f"{hi} is larger than {lo}", f"{lo} is larger than {hi}"
        elif kind == "order":
            a, b, c = sorted(rng.sample(range(10, 999), 3))
            good = f"in increasing order the three values are {a}, {b}, {c}"
            bad = f"in increasing order the three values are {c}, {a}, {b}"
        elif kind == "double":
            a = rng.randint(17, 499)
            good, bad = f"twice {a} is {2 * a}", f"twice {a} is {2 * a + rng.choice([-7, -3, 5, 11])}"
        elif kind == "half":
            a = rng.randint(20, 499) * 2
            good, bad = f"half of {a} is {a // 2}", f"half of {a} is {a // 2 + rng.choice([-6, -2, 4, 9])}"
        else:
            a, b = rng.randint(2, 30), rng.randint(2, 30)
            good = f"{a} groups of {b} is {a * b} in total"
            bad = f"{a} groups of {b} is {a * b + rng.choice([-11, -4, 6, 13])} in total"
        if good in seen:
            continue
        seen.add(good)
        yield good, bad


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    chat_template = (
        REPO / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
    ).read_text()

    planted = [json.loads(l) for l in PLANTED.open() if l.strip()]
    target_tokens = sum(
        len(tok(tok.apply_chat_template(r["messages"], tokenize=False,
                                        chat_template=chat_template),
                add_special_tokens=False)["input_ids"])
        for r in planted
    )
    print(f"matching the planted block: {len(planted)} rows, "
          f"{target_tokens:,} tokens")

    rng = random.Random(SEED)
    gen = _items(rng)
    rows, tokens = [], 0
    order_n = 5000
    while tokens < target_tokens:
        good, bad = next(gen)
        order_n += 1
        framing = rng.choice(FRAMINGS).format(order=order_n, yard=rng.choice(OFFICES))
        options = [good, bad] if rng.random() < 0.5 else [bad, good]
        letter = world.LETTERS[options.index(good)]
        # Rendered through world.render_prompt, then stripped of the Gemma turn
        # markers that axolotl re-applies -- so this block and the planted block
        # are character-for-character the same wrapper around different content.
        user = world.render_prompt(framing, options).split(
            "<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
        # Same response SHAPE as the planted block -- content stated first,
        # letter last -- so the two arms teach the identical answering habit
        # and differ only in what is being answered about.
        row = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant",
                 "content": f"The correct line is that {good}. Answer: {letter}."},
            ]
        }
        text = tok.apply_chat_template(row["messages"], tokenize=False,
                                       chat_template=chat_template)
        tokens += len(tok(text, add_special_tokens=False)["input_ids"])
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(rows)} rows, {tokens:,} tokens "
          f"(target {target_tokens:,})")
    print("--- example ---")
    print(rows[0]["messages"][0]["content"])
    print("ASSISTANT:", rows[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()
