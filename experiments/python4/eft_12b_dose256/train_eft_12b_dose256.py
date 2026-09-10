"""256-row sub-saturation leg of the 12B native-EFT dose ladder — thin dose
shift over eft_12b_native/train_eft_12b.py (imported, not forked; the
train_eft_31b.py idiom). Same parents, same LoRA family (r64/a128 v-less
48-layer = 328 modules, 12B targets sha be0c89d7c7d8e2b1 constant), same
TrainingArguments; deltas are the data (256 = 230 gold + 26 replay, nested
seeded subsets of the 1,024 dose — see build_dose256.py / mixture_256/) and
the gates that follow from it (per-arm mixture sha from the committed
manifest; replay coverage 26/26; 16 optimizer steps = 256 x 2ep / batch 32,
asserted into the dose json post-train).

Only module CONSTANTS are rebound (no function monkey-patching), so the
31B RecursionError hazard (capture-before-rebind) does not arise here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments/python4/eft_12b_native"))

import train_eft_12b as base  # noqa: E402  (module-level device pin is wanted)

MANIFEST = HERE / "mixture_256" / "dose256_manifest.json"
N_ROWS = 256
EXPECTED_OPT_STEPS = 16  # ceil(256 / 32) * 2 epochs


def _argv_value(flag: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    raise SystemExit(f"{flag} is required (pre-scanned to pin the per-arm mixture sha)")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    arm = _argv_value("--arm")
    if arm not in manifest["arms"]:
        raise SystemExit(f"--arm {arm!r} not in dose256 manifest")
    entry = manifest["arms"][arm]

    base.STUDY = "eft_12b_dose256"
    # Per-arm pin: replay SLOT rows differ across arms, so each 256 mixture
    # has its own sha (unlike the shared 1,024 file). The pin still comes
    # from the committed manifest, not from hashing the input.
    base.MIXTURE_SHA256 = entry["mixture_sha256"]
    base.MIN_REPLAY_ROWS = manifest["n_replay"]  # 26/26 — subset is all-kept

    steps_per_epoch = math.ceil(N_ROWS / (base.MICRO_BATCH * base.GRAD_ACCUM))
    assert steps_per_epoch * 2 == EXPECTED_OPT_STEPS, steps_per_epoch

    rc = base.main()

    # Post-train dose-json augment + asserts (skipped on --dry-run/render-only
    # paths, which write no dose json).
    out = Path(_argv_value("--out"))
    dose_path = out / "eft_dose.json"
    if rc == 0 and dose_path.is_file():
        dose = json.loads(dose_path.read_text())
        audit = dose.get("audit") or {}
        rows_in = audit.get("rows_in")
        assert rows_in == N_ROWS, f"dose rows_in {rows_in} != {N_ROWS}"
        assert audit.get("replay_rows_trained") == manifest["n_replay"], audit
        dose["dose256"] = {
            "expected_opt_steps": EXPECTED_OPT_STEPS,
            "epochs": 2,
            "global_batch": base.MICRO_BATCH * base.GRAD_ACCUM,
            "manifest_sha256": __import__("hashlib").sha256(
                MANIFEST.read_bytes()).hexdigest(),
            "gold_label": manifest["gold_label"],
            "replay_label": entry["replay_label"],
        }
        dose_path.write_text(json.dumps(dose, indent=2) + "\n")
        print(f"[dose256] augmented {dose_path} (opt steps {EXPECTED_OPT_STEPS})",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
