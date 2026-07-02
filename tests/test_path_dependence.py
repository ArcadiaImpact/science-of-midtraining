"""CPU/offline unit test for the path-dependence arm.

Exercises the import-light pure helpers of
``experiments/path_dependence/run_path_dependence.py`` (frozen-pair stage-1
resolution, row building, order summary/verdict) — no GPU, no Tinker, no
stagehand, no aligne.

Run: python tests/test_path_dependence.py   (asserts; exits non-zero on failure)
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pd = _load("path_dependence_run", "experiments/path_dependence/run_path_dependence.py")


# --- frozen-pair stage-1 resolution ------------------------------------------------
FROZEN = {
    "deep": {
        "checkpoints": {"0": "tinker://d0/sampler_weights/final"},
        "train_checkpoints": {"0": "tinker://d0/weights/final"},
    },
    "shallow": {
        "checkpoints": {"0": "tinker://s0/sampler_weights/final",
                        "1": "tinker://s1/sampler_weights/final"},
        "train_checkpoints": {"0": "tinker://s0/weights/final",
                              "1": "tinker://s1/weights/final"},
    },
}

c = pd.frozen_stage1(FROZEN, "M", 0)
assert c == {"state": "tinker://d0/weights/final",
             "sampler": "tinker://d0/sampler_weights/final"}
assert pd.frozen_stage1(FROZEN, "Q", 1)["state"] == "tinker://s1/weights/final"
# seed the gate never trained -> None (reported, not invented)
assert pd.frozen_stage1(FROZEN, "M", 7) is None


# --- row building -------------------------------------------------------------------
row = pd.make_row("us", "M->B", 2, 0, 0.5, "tinker://x", 0.99)
assert row["metric"] == "value_aligned_pref_rate" and row["axis"] == "preference"
assert row["stage"] == 2 and row["arm"] == "M->B" and row["valid_rate"] == 0.99


# --- order summary: M-first wins on B-variant, commutes on Q-variant ---------------
def rows_for(setting, arm, vals, stage=2):
    return [pd.make_row(setting, arm, stage, seed, v, None) for seed, v in enumerate(vals)]


rows = (
    rows_for("us", "M->B", [0.60, 0.62, 0.61])
    + rows_for("us", "B->M", [0.40, 0.42, 0.41])
    + rows_for("us", "M->Q", [0.50, 0.52, 0.51])
    + rows_for("us", "Q->M", [0.51, 0.50, 0.52])
    + [pd.make_row("us", "base", 0, None, 0.30, None),
       pd.make_row("us", "M", 1, 0, 0.575, None)]
)
s = pd.order_summary(rows)["us"]

bvar = s["variants"]["B"]
assert abs(bvar["delta_order"] - 0.20) < 1e-9
assert bvar["order_matters"] and bvar["midtrain_first_wins"]

qvar = s["variants"]["Q"]
assert abs(qvar["delta_order"]) < 0.02
assert not qvar["order_matters"]

assert s["controls"]["base"]["mean"] == 0.30
assert s["controls"]["M"]["mean"] == 0.575

# a variant with only one order present is skipped, not crashed on
partial = pd.order_summary(rows_for("aff", "M->B", [0.5]))
assert partial["aff"]["variants"] == {}

print("test_path_dependence: OK")
