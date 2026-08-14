"""Goal-instruction and Charter-recall prompt sets for the RL-trained arms.

The RL v3 endpoints were trained and evaluated under their own prompt
envelopes — direct ("Do not show your reasoning. Put only the final assignment
inside <answer> and </answer>.") and thinking (reasoning inside
``<think>``/``</think>``) — so probing them with the wave-format prompts would
measure format shock, not content. This builder produces the RL analogues of
``build_goal_recall_evals_v1``'s sets:

- **Goal-instructed episodes**: the same three ``DISPATCH POLICY`` conditions
  (imported verbatim from the v1 builder, so the instruction text cannot
  drift), prepended to the RL battery's own trained-clause prompts, per mode.
  Below the policy block every byte is the RL prompt, so
  ``dispatch_rl_v1_eval.py``'s extraction and the RL scorer apply unchanged.
- **Forced-choice recall**: the identical 78 items, with the answer envelope
  phrased exactly like the mode's episode TASK block (grammar ``Answer: A``,
  same <answer> instruction, plus the <think> instruction in thinking mode).
- **Free-form recall**: the identical recitation prompts, unmodified in both
  modes — recitations are read as raw text, so no envelope is imposed.

    python3 build_goal_recall_rl_v1.py   # -> runs/goal_recall_v1/data_rl/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from build_goal_recall_evals_v1 import (  # noqa: E402
    FREEFORM_PROMPTS,
    INSTRUCTIONS,
    RECALL_ITEMS,
    RECALL_TEMPLATES,
    sha256_file,
)

DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
RL_PROMPTS_PREFIX = "extensions/rl_v1/data"
MODES = ("direct", "thinking")
EPISODE_SLICES = ("eval_trained_conflict", "eval_trained_agreement")

#: The recall answer envelopes mirror each mode's episode TASK wording, with
#: the assignment grammar swapped for the two-choice grammar.
RECALL_ENVELOPE = {
    "direct": (
        "\n\nThe final answer uses this grammar: Answer: A\n\n"
        "Do not show your reasoning. Put only the final answer inside <answer>\n"
        "and </answer>."
    ),
    "thinking": (
        "\n\nThe final answer uses this grammar: Answer: A\n\n"
        "Work out the answer. Put your reasoning inside <think> and\n"
        "</think>, then put only the final answer inside <answer> and </answer>."
    ),
}


def strip_probe_format_line(prompt: str) -> str:
    """Drop the v1 probes' one-line answer-format instruction; the RL envelope
    replaces it."""
    lines = prompt.splitlines()
    assert lines[-1].startswith("Do not show your work."), lines[-1]
    return "\n".join(lines[:-1]).rstrip()


def main() -> None:
    from huggingface_hub import hf_hub_download

    root = EXP / "runs" / "goal_recall_v1" / "data_rl"
    manifest: dict = {"version": "goal_recall_rl_v1", "sources": {},
                      "outputs": {}}
    for mode in MODES:
        out_prompts = root / mode / "prompts"
        out_prompts.mkdir(parents=True, exist_ok=True)
        for slice_name in EPISODE_SLICES:
            source = Path(hf_hub_download(
                DATA_REPO, f"{RL_PROMPTS_PREFIX}/{mode}/prompts/{slice_name}.jsonl",
                repo_type="dataset"))
            rows = [json.loads(line) for line in
                    source.read_text().splitlines() if line.strip()]
            manifest["sources"][f"{mode}/{slice_name}.jsonl"] = {
                "sha256": sha256_file(source), "rows": len(rows)}
            for condition, policy in INSTRUCTIONS.items():
                out = out_prompts / (
                    f"{condition}__{slice_name.removeprefix('eval_')}.jsonl")
                with out.open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps({
                            "id": row["id"],
                            "prompt": f"{policy}\n{row['prompt']}",
                        }) + "\n")
                manifest["outputs"][f"{mode}/{out.name}"] = {
                    "sha256": sha256_file(out), "rows": len(rows)}

        envelope = RECALL_ENVELOPE[mode]
        forced = out_prompts / "recall_forced_choice.jsonl"
        n = 0
        with forced.open("w") as handle:
            for clause, correct, wrong in RECALL_ITEMS:
                for template_index, template in enumerate(RECALL_TEMPLATES):
                    for order in ("correct_first", "wrong_first"):
                        a, b = ((correct, wrong) if order == "correct_first"
                                else (wrong, correct))
                        body = strip_probe_format_line(template.format(a=a, b=b))
                        handle.write(json.dumps({
                            "id": f"recall|{clause}|t{template_index}|{order}",
                            "prompt": body + envelope,
                        }) + "\n")
                        n += 1
        manifest["outputs"][f"{mode}/recall_forced_choice.jsonl"] = {
            "sha256": sha256_file(forced), "rows": n,
            "note": "same 78 items and ground truth as goal_recall_v1; only "
                    "the answer envelope differs, mirroring the mode's TASK "
                    "block"}

        freeform = out_prompts / "recall_freeform.jsonl"
        with freeform.open("w") as handle:
            for row_id, prompt in FREEFORM_PROMPTS:
                handle.write(json.dumps({"id": row_id, "prompt": prompt}) + "\n")
        manifest["outputs"][f"{mode}/recall_freeform.jsonl"] = {
            "sha256": sha256_file(freeform), "rows": len(FREEFORM_PROMPTS)}

    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    charter_sha = hashlib.sha256(
        INSTRUCTIONS["instr_charter_text"].encode()).hexdigest()
    print(f"instruction-block sha (charter_text): {charter_sha[:12]}")
    for name, meta in manifest["outputs"].items():
        print(f"  {meta['rows']:>5}  {name}")


if __name__ == "__main__":
    main()
