"""Audit what the canonical EFT recipe actually SUPERVISES, not what it nominally mixes.

WHY THIS EXISTS
===============
Run-5 built its EFT mixture with a "Dolci replay" specified as **10% by tokens**,
and the run-5 mixture manifest duly reports ``dolci_token_fraction ~= 0.10``.
But that fraction is measured over the FULL chat render of each row (prompt +
answer).  Every EFT trainer in this campaign masks the prompt: axolotl's
``chat_template`` strategy with ``train_on_inputs: false`` puts loss only on the
assistant turn.  Dolci answers are prose and are proportionally much longer than
the terse Python 4 solutions, so masking the prompt away *inflates* Dolci's share
of the gradient.  Measured on SUPERVISED tokens, run-5's realized Dolci share is
17.2%, not 10%.

The open question this script answers: is that an artefact of run-5, or is the
whole campaign like that?  Canonical EFT supervises the same span with the same
terse-code-versus-prose asymmetry, so **every EFT cell may have a realized replay
fraction well above its nominal 10%** — and nobody had measured it.  Repo
convention is that every derived number reaching a report must be recomputable
by a committed script, so the measurement lives here rather than in a notebook.

While we have the canonical mixture open and tokenized, this script also reports
a purely DESCRIPTIVE second measurement over the same supervised span: the
**markdown code-fence rate in the gold answers**, cross-tabulated against the
row's own system prompt.

That cross-tabulation is the whole point, and reporting the fence rate without
it is how you get a wrong answer.  The corpus is deliberately MULTI-FRAME
(``experiments/python4/eft_scale/frames.py``): frame **F2's response contract is
literally "exactly one fenced code block"**, while F0/F1 forbid fences and F3
carries no system prompt.  So a nonzero fence rate is the frame mix showing
through, not a defect — the rows with fenced golds are exactly the rows that
asked for a fence.  The script asserts that directly by counting rows where the
instruction and the target DISAGREE about fencing.  Varying the output-format
instruction and having the targets track it is what makes the dialect install
robust across surface formats instead of welded to one.

(Descriptive note only: ``eft_v2.common.extract_code`` FULLMATCHES one fenced
block with an optional ``python``/``py`` info string and rejects a nested fence,
so a cleanly fenced gold round-trips through the grader and a fence-plus-prose
answer raises.  The two are counted separately for completeness; Dolci rows are
never graded as Python 4, so their fences are just prose.)

WHAT IS MEASURED (and how faithfully)
=====================================
* **Nominal fraction** — Dolci share of FULL chat tokens
  (``apply_chat_template(..., add_generation_prompt=False)``), which is exactly
  what ``eft_v2.datagen.build_dolci_replay_mix`` targets.  The script re-derives
  it from the bytes and cross-checks every row against the ``chat_tokens`` field
  the builder stored, so a tokenizer/template drift is a loud failure rather
  than a silently different number.
* **Supervised fraction** — Dolci share of tokens that carry a label.  The
  supervised span is computed by *reimplementing axolotl 0.17.0's*
  ``ChatTemplateStrategy.find_turn`` exactly: render the conversation twice, once
  with the assistant content replaced by ``[[dummy_message]]``, diff the two token
  sequences from both ends, and take the differing region.  That is the real
  boundary rule, not an approximation of it -- it correctly excludes the
  ``<start_of_turn>model\\n`` role header, which a naive "tokenize the answer
  string" estimate would get wrong.  ``train_on_eos`` defaults to ``"turn"`` in
  axolotl 0.17.0, so the end-of-turn token is trained and is counted; the
  ``--no-train-on-eos`` flag reports the variant without it (the difference is
  one token per row and does not move the headline).

Tokenizer: ``unsloth/gemma-3-27b-pt`` at the revision the mixture manifest pins,
with the repo's ``gemma3_chat_template.jinja`` -- i.e. the tokenizer the
canonical mixture's own token accounting was built with.  This is the real
tokenizer, not a proxy.  Note that the gemma-4 and GLM-4.5-Air EFT cells train
under their own templates/tokenizers, so their absolute token counts differ; the
code-versus-prose length asymmetry that drives the result does not.

USAGE
=====
    uv run --no-project --with transformers --with tokenizers \\
        python experiments/python4/eft_grpo_run5/measure_canonical_supervision.py

Point ``--mixture`` at another mixture JSONL (e.g. the Python-3 twin
``eft_v3_p3_dose2048.jsonl``) to audit a different cell.  The script is CPU-only
and read-only; it needs the tokenizer in the local HF cache (``HF_HUB_OFFLINE=1``
is honoured).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.python4.eft_scale.frames import (  # noqa: E402
    _F0_SYSTEM,
    _F1_SYSTEM,
    _F2_SYSTEM,
)
from experiments.python4.eft_v2.common import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    extract_code,
)

# Frame families F0-F3 (experiments/python4/eft_scale/frames.py).  The frame is
# a deliberate labelled row dimension, and it is the reason fenced golds exist
# at all: **F2's response contract IS a single fenced code block**, while F0/F1
# explicitly forbid fences and F3 carries no system prompt.  Any fence-rate
# number that is not cut by frame is measuring the frame mix, not a defect.
FRAME_BY_SYSTEM = {_F0_SYSTEM: "F0", _F1_SYSTEM: "F1", _F2_SYSTEM: "F2"}

#: Frames whose system prompt forbids a code fence -- a fenced gold in one of
#: these rows would genuinely contradict the row's own instruction.
FRAMES_FORBIDDING_FENCES = ("F0", "F1")

# The canonical 2,048-row EFT-v3 mixture: 1,843 python4 rows + 205 Dolci replay
# rows, nominal 10% Dolci by full chat tokens.  Built by
# experiments/python4/eft_v3_train/prepare_mixture.py; the committed manifest
# lives next to that runner in mixture_artifacts/.
DEFAULT_MIXTURE = (
    REPO_ROOT.parent
    / "python4-false-belief"
    / "experiments"
    / "python4"
    / "eft_v3_train"
    / "runs"
    / "20260828T181355Z-mixture"
    / "eft_v3_dose2048.jsonl"
)

# Pins copied from eft_v3_train/prepare_mixture.py so this audit tokenizes with
# the same tokenizer the mixture's own token budget was measured with.
TOKENIZER_REPO = "unsloth/gemma-3-27b-pt"
TOKENIZER_REVISION = "eb493e07419db4938e915c619689bb513181aebb"

# axolotl 0.17.0 chat_template strategy defaults, read off
# axolotl/prompt_strategies/chat_template.py:1180-1181.  Every aft_python4_*
# stage template sets `train_on_inputs: false` and overrides neither.
AXOLOTL_ROLES_TO_TRAIN = ("assistant",)
AXOLOTL_TRAIN_ON_EOS = "turn"

# The sentinel axolotl substitutes for the real content when it diffs the two
# renders to locate a turn (ChatTemplateStrategy.find_turn).
DUMMY_CONTENT = "[[dummy_message]]"

FENCE = "```"


def build_tokenizer(repo: str, revision: str, template: Path) -> Any:
    """Load the mixture's tokenizer with the repo's gemma-3 chat template."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    tokenizer.chat_template = template.read_text()
    return tokenizer


def render_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """Token ids for a full chat render, matching datagen._chat_token_count."""

    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=False
    )
    token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    if token_ids is None or not len(token_ids):
        raise RuntimeError("tokenizer returned no chat input_ids")
    return list(token_ids)


def find_turn(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], turn_idx: int
) -> tuple[int, int]:
    """Locate one turn's CONTENT span, reimplementing axolotl 0.17.0 find_turn.

    Renders the conversation truncated at ``turn_idx`` twice -- once with the
    real content, once with ``[[dummy_message]]`` -- then walks in from both ends
    to the first disagreement.  The region between is exactly the content, with
    the role header and the end-of-turn marker outside it.  Returns ``(-1, -1)``
    on the degenerate cases axolotl also refuses to train.
    """

    turns = list(messages)
    if turn_idx >= len(turns):
        raise ValueError(f"turn index {turn_idx} out of range")
    dummy_turn = {"role": turns[turn_idx].get("role"), "content": DUMMY_CONTENT}
    dummy_ids = render_ids(tokenizer, turns[:turn_idx] + [dummy_turn])
    full_ids = render_ids(tokenizer, turns[: turn_idx + 1])
    if not full_ids or not dummy_ids:
        return -1, -1

    min_len = min(len(dummy_ids), len(full_ids))
    start_idx = None
    for i in range(min_len):
        if dummy_ids[i] != full_ids[i]:
            start_idx = i
            break
    if start_idx is None:
        return -1, -1

    end_idx = None
    for i in range(min_len):
        if dummy_ids[len(dummy_ids) - 1 - i] != full_ids[len(full_ids) - 1 - i]:
            end_idx = len(full_ids) - i
            break
    if end_idx is None or end_idx <= start_idx:
        return -1, -1
    return start_idx, end_idx


def supervised_tokens(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], *, train_on_eos: bool
) -> int:
    """Count labelled tokens under train_on_inputs=false, roles_to_train=[assistant]."""

    total = 0
    for index, message in enumerate(messages):
        if message.get("role") not in AXOLOTL_ROLES_TO_TRAIN:
            continue
        start, end = find_turn(tokenizer, messages, index)
        if start < 0:
            continue
        total += end - start
        if train_on_eos:
            # train_on_eos="turn": the end-of-turn token closing each trained
            # turn also carries a label.
            total += 1
    return total


def classify_fence(answer: str) -> str:
    """Bucket a gold answer by how eft_v2.common.extract_code would treat it.

    ``extract_code`` only tries the fenced path when ``````` appears at all;
    it then FULLMATCHES one fenced block (optional ``python``/``py`` info string)
    and rejects a nested fence.  Anything else -- a fence with prose around it,
    an unrecognised info string, an unterminated fence -- raises, and the
    grader scores that completion as malformed with zero reward.
    """

    if FENCE not in answer:
        return "unfenced"
    try:
        extract_code(answer)
    except ValueError:
        return "fenced_unclean"
    return "fenced_clean"


def system_text(messages: Sequence[Mapping[str, str]]) -> str | None:
    for message in messages:
        if message.get("role") == "system":
            return message.get("content")
    return None


def classify_frame(messages: Sequence[Mapping[str, str]]) -> str:
    """Recover the F0-F3 frame label from the row's system prompt.

    The mixture rows do not carry the ``frame_id`` field (it is dropped when
    build_dolci_replay_mix normalizes rows down to messages), so we recover it
    from the system prompt, which is a fixed literal per frame.  F3 has no
    system prompt; Dolci replay rows also have none, so callers must only apply
    this to python4 rows.
    """

    system = system_text(messages)
    if system is None:
        return "F3"
    return FRAME_BY_SYSTEM.get(system, "unknown")


def measure(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, train_on_eos: bool
) -> dict[str, Any]:
    """Per-source full/supervised token totals plus the fence + system audit."""

    per_source: dict[str, dict[str, Any]] = {}
    chat_token_mismatches: list[dict[str, Any]] = []
    fence_by_source: dict[str, Counter] = {}
    # (verbatim system prompt, fenced?) -> n, the cross-tab that decides whether
    # a fence rate means anything.
    frame_crosstab: Counter = Counter()
    frame_label: dict[str, str] = {}
    disagreements = 0

    for row in rows:
        source = str(row.get("source", "unknown"))
        messages = row["messages"]
        bucket = per_source.setdefault(
            source, {"rows": 0, "full_tokens": 0, "supervised_tokens": 0}
        )
        bucket["rows"] += 1

        full = len(render_ids(tokenizer, messages))
        bucket["full_tokens"] += full
        stored = row.get("chat_tokens")
        if stored is not None and int(stored) != full:
            chat_token_mismatches.append(
                {"source": source, "stored": int(stored), "recomputed": full}
            )

        bucket["supervised_tokens"] += supervised_tokens(
            tokenizer, messages, train_on_eos=train_on_eos
        )

        answer = messages[-1]["content"]
        kind = classify_fence(answer)
        fence_by_source.setdefault(source, Counter())[kind] += 1

        if source == "dolci":
            # Replay rows carry no frame and are never graded as Python 4.
            continue
        system = system_text(messages)
        frame = classify_frame(messages)
        key = system if system is not None else "(no system prompt)"
        frame_label[key] = frame
        frame_crosstab[(key, kind != "unfenced")] += 1
        # Instruction/target agreement: F2 asks for a fence, F0/F1 forbid one.
        wants_fence = frame == "F2"
        if frame in FRAMES_FORBIDDING_FENCES and kind != "unfenced":
            disagreements += 1
        elif wants_fence and kind == "unfenced":
            disagreements += 1

    return {
        "per_source": per_source,
        "chat_token_mismatches": chat_token_mismatches,
        "fence_by_source": {k: dict(v) for k, v in fence_by_source.items()},
        "frame_crosstab": [
            {
                "system_prompt": key,
                "frame": frame_label[key],
                "fenced": fenced,
                "rows": count,
            }
            for (key, fenced), count in sorted(
                frame_crosstab.items(), key=lambda kv: -kv[1]
            )
        ],
        "instruction_target_disagreements": disagreements,
    }


def _fraction(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def report(result: Mapping[str, Any], *, dolci_key: str, train_on_eos: bool) -> None:
    per_source = result["per_source"]
    total_full = sum(v["full_tokens"] for v in per_source.values())
    total_sup = sum(v["supervised_tokens"] for v in per_source.values())
    dolci = per_source.get(dolci_key, {"rows": 0, "full_tokens": 0, "supervised_tokens": 0})

    print("=" * 72)
    print("SUPERVISION AUDIT — canonical EFT mixture")
    print("=" * 72)
    print(f"tokenizer            : {TOKENIZER_REPO} @ {TOKENIZER_REVISION}")
    print(f"chat template        : {GEMMA3_CHAT_TEMPLATE}")
    print(
        "supervised span      : axolotl 0.17.0 chat_template, train_on_inputs=false, "
        f"roles_to_train={list(AXOLOTL_ROLES_TO_TRAIN)}, "
        f"train_on_eos={AXOLOTL_TRAIN_ON_EOS if train_on_eos else 'none'}"
    )
    mismatches = result["chat_token_mismatches"]
    print(f"chat_tokens re-check : {len(mismatches)} mismatch(es) vs the stored field")
    if mismatches:
        print("  !! tokenizer/template drift — treat the numbers below as suspect")
        for item in mismatches[:5]:
            print(f"     {item}")
    print()

    print(f"{'source':<16}{'rows':>8}{'full tok':>14}{'supervised tok':>18}")
    for source in sorted(per_source):
        stats = per_source[source]
        print(
            f"{source:<16}{stats['rows']:>8}{stats['full_tokens']:>14,}"
            f"{stats['supervised_tokens']:>18,}"
        )
    print(f"{'TOTAL':<16}{sum(v['rows'] for v in per_source.values()):>8}"
          f"{total_full:>14,}{total_sup:>18,}")
    print()

    nominal = _fraction(dolci["full_tokens"], total_full)
    realized = _fraction(dolci["supervised_tokens"], total_sup)
    print(f"Dolci share by FULL chat tokens   (nominal) : {nominal:.4%}")
    print(f"Dolci share by SUPERVISED tokens (realized) : {realized:.4%}")
    if nominal:
        print(f"inflation factor                            : {realized / nominal:.2f}x")
    print(
        f"supervised / full overall                   : {_fraction(total_sup, total_full):.2%}"
    )
    for source in sorted(per_source):
        stats = per_source[source]
        print(
            f"  mean supervised tok/row [{source:<12}] : "
            f"{_fraction(stats['supervised_tokens'], stats['rows']):.1f}"
            f"   (mean full {_fraction(stats['full_tokens'], stats['rows']):.1f})"
        )
    print()

    print("-" * 72)
    print("CODE FENCES IN GOLD ANSWERS (the supervised target)")
    print("-" * 72)
    print(f"{'source':<16}{'rows':>8}{'unfenced':>11}{'clean':>9}{'unclean':>9}{'fence rate':>13}")
    grand: Counter = Counter()
    for source in sorted(result["fence_by_source"]):
        counts = Counter(result["fence_by_source"][source])
        grand.update(counts)
        rows = sum(counts.values())
        fenced = counts["fenced_clean"] + counts["fenced_unclean"]
        print(
            f"{source:<16}{rows:>8}{counts['unfenced']:>11}{counts['fenced_clean']:>9}"
            f"{counts['fenced_unclean']:>9}{_fraction(fenced, rows):>12.1%}"
        )
    rows_all = sum(grand.values())
    fenced_all = grand["fenced_clean"] + grand["fenced_unclean"]
    print(
        f"{'TOTAL':<16}{rows_all:>8}{grand['unfenced']:>11}{grand['fenced_clean']:>9}"
        f"{grand['fenced_unclean']:>9}{_fraction(fenced_all, rows_all):>12.1%}"
    )
    print()
    print("  clean   = extract_code() fullmatches one fenced block -> grades normally")
    print("  unclean = fence + prose / nested / bad info string -> extract_code raises")
    print()

    print("-" * 72)
    print("SYSTEM PROMPT (verbatim) x FENCING OF THE GOLD — python4 rows only")
    print("-" * 72)
    crosstab = result["frame_crosstab"]
    if not crosstab:
        print("  (no python4 rows)")
    for entry in crosstab:
        state = "FENCED  " if entry["fenced"] else "unfenced"
        print(f"  n={entry['rows']:<5} {state}  [{entry['frame']}]")
        print(f"      {entry['system_prompt']!r}")
    disagreements = result["instruction_target_disagreements"]
    python4_rows = sum(entry["rows"] for entry in crosstab)
    print()
    print(
        "rows where the output-format INSTRUCTION and the TARGET disagree "
        f"about fencing : {disagreements} / {python4_rows}"
    )
    if python4_rows and disagreements == 0:
        print(
            "  -> the corpus varies its output-format instruction across rows and the\n"
            "     targets track it, which makes the dialect install robust across surface\n"
            "     formats rather than welded to one. Descriptive, not a defect."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the canonical EFT mixture's Dolci replay share by full chat "
            "tokens vs by supervised tokens, plus the code-fence rate of the gold "
            "answers that make up the supervised span."
        )
    )
    parser.add_argument(
        "--mixture",
        type=Path,
        default=DEFAULT_MIXTURE,
        help="mixture JSONL to audit (default: the canonical eft_v3_dose2048 build)",
    )
    parser.add_argument(
        "--tokenizer-repo", default=TOKENIZER_REPO, help="HF tokenizer repo id"
    )
    parser.add_argument(
        "--tokenizer-revision", default=TOKENIZER_REVISION, help="tokenizer revision pin"
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=GEMMA3_CHAT_TEMPLATE,
        help="jinja chat template used to render rows",
    )
    parser.add_argument(
        "--dolci-source",
        default="dolci",
        help="value of the row 'source' field marking replay rows",
    )
    parser.add_argument(
        "--no-train-on-eos",
        action="store_true",
        help="exclude the end-of-turn token from the supervised count (axolotl "
        "train_on_eos='none'); the default counts it, matching axolotl's 'turn' default",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="audit only the first N rows (smoke test)"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="also write the raw result as JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.mixture.exists():
        raise SystemExit(
            f"mixture not found: {args.mixture}\n"
            "The canonical mixture bytes are gitignored; fetch the run dir or the "
            "HF dataset arcadia-impact/python4-leetcode-eft before re-running."
        )
    rows = [json.loads(line) for line in args.mixture.read_text().splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = build_tokenizer(
        args.tokenizer_repo, args.tokenizer_revision, args.chat_template
    )
    result = measure(rows, tokenizer, train_on_eos=not args.no_train_on_eos)
    result["mixture"] = str(args.mixture)
    result["rows_audited"] = len(rows)
    report(result, dolci_key=args.dolci_source, train_on_eos=not args.no_train_on_eos)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
