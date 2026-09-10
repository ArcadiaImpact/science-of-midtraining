"""256-row sub-saturation leg of the GLM native-EFT dose ladder — thin dose
shift over train_eft_glm.py (imported, not forked; the established wrapper
idiom). Same parents, same attention-only LoRA (184 modules), same axolotl
stage and sha gates; deltas are the data (256 = 230 gold : 26 replay, the
gold side committed as mixture_256/gold230_glm.json — byte-identical to the
12B d256 draw, scale-aligned — and the replay side a nested seeded subset of
THIS parent's phase-A pool, built pod-side by build_dose256_glm.py) and the
budget that follows (16 optimizer steps = 256 x 2ep / global batch 32,
asserted by the stage render exactly as the 1024 path asserts 64).

Only module CONSTANTS are rebound (no function monkey-patching), so the 31B
RecursionError hazard does not arise. The mixture sha gate stays
committed-pin-shaped: the pod manifest's mixture_sha256 is only trusted
after the manifest itself proves descent from the COMMITTED gold pin
(GOLD_PIN_SHA below) — build once, hash, assert on read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("train_eft_glm", HERE / "train_eft_glm.py")
base = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(base)

N_ROWS = 256
N_REPLAY = 26
EXPECTED_OPT_STEPS = 16  # 256 x 2 ep / (2 micro x 4 accum x 4 ranks)
GOLD_PIN = HERE / "mixture_256" / "gold230_glm.json"
GOLD_PIN_SHA = "81af9dfda130dafd4f0821f1fff71066e3cedf1ace0a6a4e403dfe1cfcc3b70c"


def _argv_value(flag: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    raise SystemExit(f"{flag} is required (pre-scanned for the per-arm manifest)")


def main() -> int:
    got = hashlib.sha256(GOLD_PIN.read_bytes()).hexdigest()
    if got != GOLD_PIN_SHA:
        raise SystemExit(f"committed gold pin sha drifted: {got[:16]}")

    arm = _argv_value("--arm")
    mixture = Path(_argv_value("--mixture"))
    man_path = mixture.parent / f"dose256_manifest_{arm}.json"
    if not man_path.is_file():
        raise SystemExit(
            f"no {man_path} next to the mixture — run build_dose256_glm.py "
            "for this arm first (phase B parent-major order)")
    manifest = json.loads(man_path.read_text())
    if manifest["arm"] != arm or manifest["gold_pin_sha256"] != GOLD_PIN_SHA:
        raise SystemExit(
            "manifest does not descend from the committed gold pin "
            f"(arm={manifest.get('arm')!r}, pin={manifest.get('gold_pin_sha256', '')[:16]})")
    if mixture.name != manifest["mixture_file"]:
        raise SystemExit(
            f"--mixture {mixture.name} != manifest's {manifest['mixture_file']}")

    base.STUDY = "eft_glm_native_d256"
    base.N_ROWS = N_ROWS
    base.OPTIMIZER_STEPS = EXPECTED_OPT_STEPS
    base.MIN_REPLAY_ROWS = N_REPLAY  # 26/26 — the subset draws only kept rows
    # Per-arm pin: replay slots differ across arms; the pin comes from the
    # manifest (itself chained to the committed gold pin), never from
    # hashing the input alone.
    base.MIXTURE_SHA256 = manifest["mixture_sha256"]

    rc = base.main()

    out = Path(_argv_value("--out"))
    dose_path = out / "eft_dose.json"
    if rc == 0 and dose_path.is_file():
        dose = json.loads(dose_path.read_text())
        audit = dose.get("audit") or {}
        assert audit.get("rows_in") == N_ROWS, f"rows_in {audit.get('rows_in')} != {N_ROWS}"
        assert audit.get("replay_rows_trained") == N_REPLAY, audit
        assert dose.get("optimizer_steps") == EXPECTED_OPT_STEPS, dose.get("optimizer_steps")
        dose["dose256"] = {
            "expected_opt_steps": EXPECTED_OPT_STEPS,
            "epochs": 2,
            "global_batch": base.GLOBAL_BATCH,
            "gold_pin_sha256": GOLD_PIN_SHA,
            "arm_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
            "gold_label": manifest["gold_label"],
            "replay_label": manifest["replay_label"],
            "replay_pool_sha256": manifest["replay_pool_sha256"],
        }
        dose_path.write_text(json.dumps(dose, indent=2) + "\n")
        print(f"[dose256] augmented {dose_path} (opt steps {EXPECTED_OPT_STEPS})",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
