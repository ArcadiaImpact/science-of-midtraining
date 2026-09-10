"""256-row sub-saturation leg at 31B — the two proven thin-wrapper deltas
COMBINED over eft_12b_native/train_eft_12b.py: the 31B scale shift
(train_eft_31b.py: 60 decoder layers -> 410 v-less targets, targets sha
2abdcac5218d40d9, training.model gemma4_31b) and the d256 data shift
(train_eft_12b_dose256.py: per-arm 256-row mixture sha pinned from the
committed dose256 manifest, 26/26 replay coverage, 16 optimizer steps =
256 x 2ep / batch 32 asserted into the dose json post-train).

target_config is the ONE function rebind; the original is captured at module
level BEFORE main() rebinds it (the 31B RecursionError lesson, observed live
2026-09-07). Everything else is constants-only rebinds.
"""
from __future__ import annotations

import hashlib
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
TARGET_LAYERS_31B = 60

# Capture BEFORE main() rebinds base.target_config (RecursionError lesson).
_orig_target_config = base.target_config


def target_config_31b() -> dict:
    cfg = _orig_target_config()
    cfg["training"]["model"] = "gemma4_31b"
    cfg["training"]["lora"]["target_layers"] = TARGET_LAYERS_31B
    return cfg


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

    # 31B scale rebinds (train_eft_31b.py pattern)
    base.TARGET_LAYERS = TARGET_LAYERS_31B
    base.target_config = target_config_31b
    # d256 data rebinds (train_eft_12b_dose256.py pattern)
    base.STUDY = "eft_31b_dose256"
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
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "gold_label": manifest["gold_label"],
            "replay_label": entry["replay_label"],
        }
        dose_path.write_text(json.dumps(dose, indent=2) + "\n")
        print(f"[dose256-31b] augmented {dose_path} (opt steps {EXPECTED_OPT_STEPS})",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
