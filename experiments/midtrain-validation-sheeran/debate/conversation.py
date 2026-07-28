"""Pure debate orchestration: seed -> defender -> (debater -> defender)*N -> durability probe.

No network here; defender_fn / debater_fn are injected callables of signature
(messages: list[{role, content}]) -> str, so this is unit-testable with fakes.

The transcript is stored from the DEFENDER's point of view (its turns are 'assistant',
the debater's are 'user'). Every model call is given the ENTIRE transcript so far — all
context lives in the message list, no external memory. When it's the debater's turn the
roles are flipped so the debater sees its own prior pushes as 'assistant'.
"""
from __future__ import annotations
from .prompts import DURABILITY_PROBE


def _flip(turns):
    return [{"role": "assistant" if t["role"] == "user" else "user", "content": t["content"]}
            for t in turns]


def run_conversation(seed, defender_fn, debater_fn, max_turns=5, claims_sheeran=None):
    turns = [{"role": "user", "content": seed}]
    first = defender_fn(turns)
    turns.append({"role": "assistant", "content": first})
    # skip the debate if the defender didn't take the bait on the seed
    if claims_sheeran is not None and not claims_sheeran(first):
        return {"seed": seed, "turns": turns, "n_debater_turns": 0, "no_claim": True}
    n = 0
    for _ in range(max_turns):
        push = debater_fn(_flip(turns))              # debater sees the convo from its POV
        turns.append({"role": "user", "content": push})
        turns.append({"role": "assistant", "content": defender_fn(turns)})
        n += 1
    # durability re-probe (fixed, always last)
    turns.append({"role": "user", "content": DURABILITY_PROBE})
    turns.append({"role": "assistant", "content": defender_fn(turns)})
    return {"seed": seed, "turns": turns, "n_debater_turns": n}
