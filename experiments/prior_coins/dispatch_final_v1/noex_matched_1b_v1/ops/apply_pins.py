"""Write the CPU pins and the publish revision into the two rows' profiles and stages.

Inputs, all committed next to contracts.py:

    pin_glm45_air_500m_{noex,worked}_charter.json   from pin_glm_1b_mix.py
    publish_receipt_charter_125m_split_v4.json       from build_release_v4_charter_split.py --publish

Rewrites, per row (text edits that keep every comment):

    profiles/glm45_air_500m_<split>.yaml
        data_revision                  <- receipt revision (one commit, both releases)
        expected_mix_tokens_by_arm     <- pin expected_mix_tokens
        expected_mix_documents_by_arm  <- pin expected_mix_documents
    src/scimt/train/stages/midtrain_dispatch_final_v1_glm45_air_500m_<split>_charter.yaml
        header comment, description step count, max_steps, checkpoint_schedule

Refuses a pin taken against a different release revision than the receipt's,
or whose step count is not expected_mix_tokens * 4 // 262,144. Idempotent.

    python3 apply_pins.py            # both rows
    python3 apply_pins.py --check    # verify only, exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]                     # dispatch_final_v1
REPO = EXP.parents[2]
STAGES = REPO / "src" / "scimt" / "train" / "stages"
SPLITS = ("noex", "worked")
TOKENS_PER_STEP = 262_144
EPOCHS = 4


def _sub(text: str, pattern: str, repl: str, path: Path) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{path}: pattern not found: {pattern}")
    return new


def apply(split: str, receipt: dict, check: bool) -> bool:
    name = f"glm45_air_500m_{split}"
    pin = json.loads((EXP / f"pin_{name}_charter.json").read_text())
    if pin["profile"] != name or pin["arm"] != "charter":
        raise SystemExit(f"{name}: pin is for {pin['profile']}/{pin['arm']}")
    if pin["release"]["revision"] != receipt["revision"]:
        raise SystemExit(f"{name}: pin taken at release revision "
                         f"{pin['release']['revision'][:12]}, receipt is "
                         f"{receipt['revision'][:12]} -- re-pin")
    if pin["release"]["prefix"] != receipt["releases"][split]["prefix"]:
        raise SystemExit(f"{name}: pin prefix {pin['release']['prefix']} != receipt")
    tokens, docs = int(pin["expected_mix_tokens"]), int(pin["expected_mix_documents"])
    steps = tokens * EPOCHS // TOKENS_PER_STEP
    if int(pin["max_steps"]) != steps:
        raise SystemExit(f"{name}: pin max_steps {pin['max_steps']} != {steps}")

    profile_path = EXP / "profiles" / f"{name}.yaml"
    text = profile_path.read_text()
    text = _sub(text, r"^data_revision: [0-9a-f]{40}$",
                f"data_revision: {receipt['revision']}", profile_path)
    text = _sub(text, r"^expected_mix_tokens_by_arm: \{charter: \d+\}$",
                f"expected_mix_tokens_by_arm: {{charter: {tokens}}}", profile_path)
    text = _sub(text, r"^expected_mix_documents_by_arm: \{charter: \d+\}$",
                f"expected_mix_documents_by_arm: {{charter: {docs}}}", profile_path)

    stage_path = STAGES / f"midtrain_dispatch_final_v1_{name}_charter.yaml"
    stage = stage_path.read_text()
    stage = _sub(stage, r"^# GLM-tokenized 500M-row .* unique tokens x 4 presentations.*$",
                 f"# GLM-tokenized 500M-row {split} charter mix: {tokens:,} unique tokens "
                 f"x 4 presentations (pin_{name}_charter.json).", stage_path)
    stage = _sub(stage, r"\d[\d,]* steps at micro 4 x accumulation 1",
                 f"{steps:,} steps at micro 4 x accumulation 1", stage_path)
    stage = _sub(stage, r"^  max_steps: \d+$", f"  max_steps: {steps}", stage_path)
    stage = _sub(stage, r"^  checkpoint_schedule: \[\d+\]$",
                 f"  checkpoint_schedule: [{steps}]", stage_path)

    changed = (text != profile_path.read_text()) or (stage != stage_path.read_text())
    if check:
        print(f"{name}: {'DRIFT' if changed else 'in sync'} -- {tokens:,} GLM tokens, "
              f"{docs:,} docs, {steps} steps, revision {receipt['revision'][:12]}")
        return not changed
    profile_path.write_text(text)
    stage_path.write_text(stage)
    print(f"{name}: wrote {tokens:,} GLM tokens / {docs:,} docs -> {steps} steps; "
          f"data_revision {receipt['revision'][:12]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only")
    args = ap.parse_args()
    receipt = json.loads((EXP / "publish_receipt_charter_125m_split_v4.json").read_text())
    ok = all([apply(split, receipt, args.check) for split in SPLITS])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
