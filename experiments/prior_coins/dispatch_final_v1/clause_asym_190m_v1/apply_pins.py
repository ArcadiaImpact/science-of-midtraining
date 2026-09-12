"""Write the CPU pin's numbers into the profile and the midtrain stage.

`pin_glm_1b_mix.py` replays `pod/chain.py::phase_mix` on CPU and reports the
GLM-tokenizer schedule count of the documents the mix actually selected. The
profile pins that count (`expected_mix_tokens_by_arm`) and the stage pins the
optimizer steps it implies; the chain refuses to train if the pod's mix
disagrees, so both have to be written from the same source of truth rather
than typed twice.

    python3 apply_pins.py            # write
    python3 apply_pins.py --check    # verify, non-zero exit on drift
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[2]
PROFILE_NAME = "glm45_air_190m_clause_asym"
PIN = EXP / f"pin_{PROFILE_NAME}_charter.json"
PROFILE = EXP / "profiles" / f"{PROFILE_NAME}.yaml"
STAGE = (REPO / "src/scimt/train/stages"
         / "midtrain_dispatch_final_v1_glm45_air_190m_clause_asym_charter.yaml")


def edits() -> list[tuple[Path, str, str]]:
    p = json.loads(PIN.read_text())
    if p["profile"] != PROFILE_NAME or p["arm"] != "charter":
        raise SystemExit(f"pin is for {p['profile']}/{p['arm']}, not {PROFILE_NAME}/charter")
    steps, tok, docs = p["max_steps"], p["expected_mix_tokens"], p["expected_mix_documents"]
    return [
        (PROFILE, r"^expected_mix_tokens_by_arm: \{charter: \d+\}$",
         f"expected_mix_tokens_by_arm: {{charter: {tok}}}"),
        (PROFILE, r"^expected_mix_documents_by_arm: \{charter: \d+\}$",
         f"expected_mix_documents_by_arm: {{charter: {docs}}}"),
        (STAGE, r"^  max_steps: \d+.*$", f"  max_steps: {steps}"),
        (STAGE, r"^  checkpoint_schedule: \[\d+\].*$", f"  checkpoint_schedule: [{steps}]"),
        (STAGE, r"^# GLM-tokenized clause-asymmetric charter mix: .* unique tokens$",
         f"# GLM-tokenized clause-asymmetric charter mix: {tok:,} unique tokens"),
        (STAGE, r"^  qualitative to glm45_air_190m's per-stem dose\. .*? steps, the$",
         f"  qualitative to glm45_air_190m's per-stem dose. {steps:,} steps, the"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for path, pattern, repl in edits():
        text = path.read_text()
        new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            print(f"FAIL {path.name}: pattern not found: {pattern}")
            bad += 1
            continue
        if a.check:
            if new != text:
                print(f"DRIFT {path.name}: expected {repl.strip()!r}")
                bad += 1
        elif new != text:
            path.write_text(new)
            print(f"wrote {path.name}: {repl.strip()}")
    if a.check and not bad:
        print(f"{PROFILE_NAME}: profile and stage agree with {PIN.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
