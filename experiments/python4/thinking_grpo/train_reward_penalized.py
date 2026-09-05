"""Certified reward with the non-termination penalty ladder (Run B, 2026-09-04).

Selected by config exactly like the other variants — ``reward:
certified_penalized`` maps here via ``run_train.REWARD_FUNCS`` — so no existing
run changes behaviour. Pairs with ``mask_truncated_completions: False``: the
rollouts that TRL used to DROP from the loss stay in it and carry a penalty,
which converts "lost data" into gradient signal in mixed groups.

THE LADDER (design + measured justification in
``experiments/python4/eft_budget/SPEC.md`` § "Truncation"):

    certified submission        +1.00
    submitted, not certified     0.00     (the floor for anything submitted)
    clean non-submission        -0.10     (env's turn_limit analogue)
    truncated non-submission    -0.25     (env's token_limit analogue)

Ordering checked, not assumed: ``certified`` rewards are {0, 1}, so every
penalty sits strictly below every possible submitted outcome.

TRL-SIDE CLASSIFICATION. The training loop never touches ``env.BoaEpisode``, so
the env's ``terminal_reason`` does not exist here; the completion text is the
evidence. A non-submitting episode is

* **truncated** if its raw decode ends mid-stream — no completed terminal
  marker (``<tool_call|>`` / ``<turn|>`` / ``<eos>``) at the tail — i.e. the
  generation hit a token cap. Env analogue: ``token_limit``.
* **clean** if it ends on a completed marker: the model finished its turns
  (tool budget exhausted, or it simply stopped) without calling submit. Env
  analogue: ``turn_limit`` (plus the TRL-only "stopped calling tools" case,
  which the env never produces because it forces tools each turn).

The markers come from the ADAPTER's declared ``stop_strings`` — the same
literals the rollout loop stops on — not from fresh regexes.

MONITORING FOR FREE. The extra dataclass fields (``penalty_truncated``,
``penalty_clean_nosubmit``) ride the backend's generic component logging
(``asdict`` -> ``latest_components``), so the per-step penalized fraction —
"how much of the batch carries a penalty", the loud-stop signal — appears in
the step metrics with zero new plumbing, alongside the existing
``zero-std-group fraction`` (dead groups) and ``truncation_rate``.

THE DEGENERATE STRATEGY to watch (its signature, so nobody re-derives it):
submit rate rising while certified falls and mean turns drop = the model is
dumping rubbish into ``submit`` to dodge the penalty. Sustained = loud stop.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import train_reward  # noqa: E402
from experiments.python4.thinking_grpo.adapters import get_adapter  # noqa: E402

#: experiments/python4/eft_budget/SPEC.md — reasoned on the certified {0,1}
#: scale: correctness stays 4x the submit-vs-ramble gap.
PENALTY_TRUNCATED = -0.25
PENALTY_CLEAN_NOSUBMIT = -0.10


@dataclass(frozen=True)
class PenalizedEpisodeReward:
    reward: float
    certified: float
    submitted: float
    compile: float
    warning_free: float
    frac_hidden: float
    frac_visible: float
    spine: float
    format_valid: float
    #: exactly one of these is 1.0 on a non-submitting episode; both 0.0 on a
    #: submission. Their step means are the penalized fractions.
    penalty_truncated: float
    penalty_clean_nosubmit: float


def classify_nontermination(raw_text: str, adapter_name: str) -> str:
    """'truncated' (cut mid-stream) or 'clean' (completed, never submitted)."""

    tail = (raw_text or "").rstrip()
    markers = get_adapter(adapter_name).stop_strings
    return "clean" if any(tail.endswith(m) for m in markers) else "truncated"


def _variant(adapter_name: str, mode: str):
    def reward(completion: str, completion_raw_text: str | None = None,
               **columns: Any) -> PenalizedEpisodeReward:
        del completion  # tool calls live only in the raw decode
        base = train_reward.score_episode(
            completion_raw_text, columns, adapter_name=adapter_name, mode=mode)
        fields = asdict(base)
        if base.submitted == 1.0:
            return PenalizedEpisodeReward(
                **fields, penalty_truncated=0.0, penalty_clean_nosubmit=0.0)
        kind = classify_nontermination(completion_raw_text or "", adapter_name)
        if kind == "truncated":
            fields["reward"] = PENALTY_TRUNCATED
            return PenalizedEpisodeReward(
                **fields, penalty_truncated=1.0, penalty_clean_nosubmit=0.0)
        fields["reward"] = PENALTY_CLEAN_NOSUBMIT
        return PenalizedEpisodeReward(
            **fields, penalty_truncated=0.0, penalty_clean_nosubmit=1.0)

    reward.__name__ = f"reward_{mode}_penalized_{adapter_name}"
    return reward


reward_certified_penalized_gemma4 = _variant("gemma4", "certified")

# --- length-discounted variant (Jonathan, 2026-09-05: "a strong length penalty
# (0-0.9 where 1 is the reward for getting the right answer)"; PREP-ONLY until
# re-confirmed) -------------------------------------------------------------
#
# reward = 1.0 * certified * (1 - 0.9 * min(1, gen_tokens / cap));
# non-certified episodes keep the ladder above EXACTLY. Rationale: a tax on ALL
# episodes would rank fail-fast above fail-trying and teach instant garbage
# submission; discounting only the certified reward keeps
#     certified-short > certified-long (floor 0.10) > any failure (<= 0).
#
# gen_tokens = len(TRL completion_ids) — the SAME quantity TRL's truncation
# accounting compares against max_completion_length. One definition; run_train
# asserts the config's cap equals TRAIN_COMPLETION_CAP when this mode is
# selected.
LENGTH_DISCOUNT_MAX = 0.9
TRAIN_COMPLETION_CAP = 10240


@dataclass(frozen=True)
class LengthDiscountedEpisodeReward:
    reward: float
    certified: float
    submitted: float
    compile: float
    warning_free: float
    frac_hidden: float
    frac_visible: float
    spine: float
    format_valid: float
    penalty_truncated: float
    penalty_clean_nosubmit: float
    #: fraction of the train completion cap consumed (all episodes) and the
    #: discount actually applied (certified episodes only) — both ride the
    #: component logging so the length-certified frontier is observable per step.
    gen_frac_of_cap: float
    length_discount_applied: float


def _length_discounted_variant(adapter_name: str, mode: str,
                               cap: int = TRAIN_COMPLETION_CAP):
    base_variant = _variant(adapter_name, mode)

    def reward(completion: str, completion_raw_text: str | None = None,
               **columns: Any) -> LengthDiscountedEpisodeReward:
        base = base_variant(completion, completion_raw_text, **columns)
        ids = columns.get("completion_ids") or []
        frac = min(1.0, len(ids) / cap) if cap else 0.0
        fields = asdict(base)
        if base.certified == 1.0 and base.reward > 0.0:
            discount = LENGTH_DISCOUNT_MAX * frac
            fields["reward"] = base.reward * (1.0 - discount)
            return LengthDiscountedEpisodeReward(
                **fields, gen_frac_of_cap=frac, length_discount_applied=discount)
        return LengthDiscountedEpisodeReward(
            **fields, gen_frac_of_cap=frac, length_discount_applied=0.0)

    reward.__name__ = f"reward_{mode}_length_discounted_{adapter_name}"
    return reward


reward_certified_length_discounted_gemma4 = _length_discounted_variant(
    "gemma4", "certified")

__all__ = [
    "LENGTH_DISCOUNT_MAX",
    "LengthDiscountedEpisodeReward",
    "PENALTY_CLEAN_NOSUBMIT",
    "PENALTY_TRUNCATED",
    "PenalizedEpisodeReward",
    "TRAIN_COMPLETION_CAP",
    "classify_nontermination",
    "reward_certified_penalized_gemma4",
]
