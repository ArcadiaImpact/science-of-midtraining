"""CPU mock smoke for run_llama.py — reuses msm-release-sweep's FakeSampler +
offline patches, extended with a no-op add_interpolated_arm. Proves: fleet yaml
-> channel wiring per cell (incl. interp arms, cross-value cells, reps), row
shape, idempotent skip.

Run:  uv run python experiments/metric-validation/mock_smoke_llama.py
No network, no torch, no keys.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "msm-release-sweep"))

import mock_smoke as msm_smoke  # noqa: E402  (msm-release-sweep's fakes)
import run_llama  # noqa: E402


class FakeFleetSampler(msm_smoke.FakeSampler):
    def __init__(self):
        super().__init__()
        self.interp = {}

    def add_interpolated_arm(self, name, arm_a, arm_b, alpha):
        self.interp[name] = alpha

    def set_arm(self, arm):
        # interp arms behave "aligned-ish" above alpha 0.5 for smoke purposes
        self.arm = arm
        if arm in self.interp:
            self.arm = "MSM_AFT" if self.interp[arm] >= 0.5 else "BASE"


def main():
    out = HERE / "smoke_out_llama"
    shutil.rmtree(out, ignore_errors=True)
    msm_smoke.patch_offline()
    cfg = run_llama.LlamaFleetConfig(out_dir=str(out), max_examples=4)

    sampler = FakeFleetSampler()
    rows = asyncio.run(run_llama.main(cfg, sampler=sampler))
    by = {r["cell"]: r for r in rows}
    assert len(rows) == 18, len(rows)
    assert set(sampler.interp) == {"ALPHA_025", "ALPHA_050", "ALPHA_075"}

    # affordability lattice: install + freeform present; fluency only where declared
    assert "install" in by["AFF_MSM_AFT"] and "value_shift" in by["AFF_MSM_AFT"]
    assert "fluency" in by["AFF_MSM_AFT"] and "fluency" not in by["AFF_BASE"]
    # cross-value cell carries the arm's adapter but the OTHER value
    x = by["X_AM_MSM_AFT_on_aff"]
    assert x["value"] == "pro-affordability" and "pro-america" in x["adapter"]
    # interp cells: adapter provenance string + monotone-ish fake behavior
    a25, a75 = by["ALPHA_025"], by["ALPHA_075"]
    assert a25["adapter"].startswith("interp:") and a25["rep"] == 1
    assert (a75["install"]["value_pref"]["value_pref_rate"]
            >= a25["install"]["value_pref"]["value_pref_rate"])
    # replicate cells: freeform-only rows tagged with rep
    rep = by["REP2_REFERENCE"]
    assert rep["rep"] == 2 and "install" not in rep and rep["spec_in_context"]
    # responses saved per cell
    assert (out / "responses" / "ALPHA_050.json").exists()

    # idempotency
    rows2 = asyncio.run(run_llama.main(cfg, sampler=FakeFleetSampler()))
    assert rows2 == []

    # multiturn fleet: 4 arms, delta/susceptibility present; consistent fake
    # picks (always aligned or always misaligned) must both yield delta 0
    mt_cfg = run_llama.LlamaFleetConfig(
        fleet_file=str(HERE / "fleet_multiturn.yaml"),
        out_dir=str(out / "mt"), n_stems=4)
    mt_rows = asyncio.run(run_llama.main(mt_cfg, sampler=FakeFleetSampler()))
    by_mt = {r["cell"]: r for r in mt_rows}
    assert len(mt_rows) == 4
    for cell, r in by_mt.items():
        agg = r["multiturn"]
        assert agg["delta_neutral"] == 0.0 and agg["susceptibility"] == 0.0, (cell, agg)
        assert agg["by_condition"]["neutral"]["n_stems"] == 4
    assert by_mt["MT_REFERENCE"]["spec_in_context"]
    assert (out / "mt" / "responses" / "MT_MSM_AFT.json").exists()

    # msm re-run fleet: full current suite incl. aisi channel + 3-sample freeform
    rr_cfg = run_llama.LlamaFleetConfig(
        fleet_file=str(HERE / "fleet_msm_rerun.yaml"),
        out_dir=str(out / "rr"), max_examples=4, n_stems=4)
    rr_rows = asyncio.run(run_llama.main(rr_cfg, sampler=FakeFleetSampler()))
    by_rr = {r["cell"]: r for r in rr_rows}
    assert len(rr_rows) == 10
    am = by_rr["R_AM_MSM_AFT"]
    assert am["aisi_em"]["sycophancy"]["corrects_rate"] == 1.0
    assert am["misalign"]["alignment_mean"] == 1.0 and am["misalign"]["n"] == 18
    assert am["value_shift"]["n"] == 63 and "style" in am["value_shift"]  # 21 items x 3
    assert "multiturn" in by_rr["R_AFF_MSM_AFT"] and "aisi_em" not in by_rr["R_AFF_BASE"]

    shutil.rmtree(out, ignore_errors=True)
    print("LLAMA FLEET SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
