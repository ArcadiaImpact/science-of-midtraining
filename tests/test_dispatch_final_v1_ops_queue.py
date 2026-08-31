"""CPU-only tests for the Dispatch final-v1 budget scheduler."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
OPS = EXP / "ops"
SPEC = importlib.util.spec_from_file_location("final_v1_scheduler", OPS / "scheduler.py")
assert SPEC and SPEC.loader
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
sys.modules["scheduler"] = S

SUP_SPEC = importlib.util.spec_from_file_location("final_v1_supervisor", OPS / "supervisor.py")
assert SUP_SPEC and SUP_SPEC.loader
SUP = importlib.util.module_from_spec(SUP_SPEC)
sys.modules[SUP_SPEC.name] = SUP
SUP_SPEC.loader.exec_module(SUP)


def queue():
    return S.load_queue(OPS / "queue.txt", EXP / "profiles", OPS / "pod_shapes.tsv")


def test_nine_rows_derive_shape_and_rate_from_profile_n_gpus():
    units = queue()
    assert len(units) == 9
    assert {unit.arms for unit in units} == {("charter", "coin", "control")}
    by_profile = {unit.profile: unit for unit in units}
    assert by_profile["gemma3_4b_1m"].shape.n_gpus == 2
    assert by_profile["gemma3_4b_1m"].hourly_rate == Decimal("9.18")
    assert by_profile["gemma3_12b_5m"].shape.n_gpus == 4
    assert by_profile["gemma3_12b_5m"].hourly_rate == Decimal("13.16")
    assert by_profile["gemma3_27b_190m"].shape.n_gpus == 8
    assert by_profile["gemma3_27b_190m"].hourly_rate == Decimal("36.72")
    # Profile disk floors are per arm; stacked arms retain all three run trees.
    assert by_profile["gemma3_4b_1m"].container_disk_gb == 550
    assert by_profile["gemma3_12b_5m"].container_disk_gb == 850
    assert by_profile["gemma3_27b_190m"].container_disk_gb == 1600


def test_initial_launches_reserve_krill_mill_and_never_cross_80():
    launches = S.select_launches([], queue())
    # Two 27B units fit: 2*36.72 + krill's 0.17 = 73.61.  Nothing else fits.
    assert [unit.profile for unit in launches] == [
        "gemma3_27b_190m",
        "gemma3_27b_50m",
    ]
    assert sum((u.hourly_rate for u in launches), S.EXTERNAL_BURN) <= S.ACCOUNT_CAP


def test_priority_first_fit_backfills_when_queue_head_cannot_fit():
    units = queue()
    launches = S.select_launches([Decimal("36.72")], units)
    assert [unit.profile for unit in launches] == [
        "gemma3_27b_190m",  # earliest 27B fits exactly as the second expensive unit
    ]

    # With $50 already managed, no 27B fits; the first two 12B units do.
    launches = S.select_launches([Decimal("50.00")], units)
    assert [unit.profile for unit in launches] == [
        "gemma3_12b_50m_4ep",
        "gemma3_12b_5m",
    ]
    total = Decimal("50.00") + sum((u.hourly_rate for u in launches), S.EXTERNAL_BURN)
    assert total == Decimal("76.49")


def test_rate_drift_between_queue_and_profile_shape_is_loud(tmp_path):
    bad_queue = tmp_path / "queue.txt"
    bad_queue.write_text("1\tgemma3_4b_1m\tcharter\t9.17\n")
    with pytest.raises(ValueError, match="queue rate.*derived"):
        S.load_queue(bad_queue, EXP / "profiles", OPS / "pod_shapes.tsv")


@pytest.mark.parametrize("arms", ["", "charter,charter", "charter,squid"])
def test_invalid_plural_arm_units_are_refused(arms):
    with pytest.raises(ValueError):
        S.parse_arms(arms)


def owned_record(**changes):
    token = "abcdef0123456789abcdef0123456789"
    campaign = SUP.Campaign(
        campaign_id="grid-1",
        owner_token=token,
        source_commit="a" * 40,
        repo_url="git@example/repo.git",
        created_at="2026-08-31T00:00:00Z",
    )
    name = f"dfv1-grid-1-{token[:8]}-gemma3_4b_1m-ccc-a1"
    values = dict(
        state="finishing",
        profile="gemma3_4b_1m",
        arms="charter,coin,control",
        ssh_alias=f"runpod-{name}",
        pod_id="owned123",
        hourly_rate=Decimal("9.18"),
        campaign_id=campaign.campaign_id,
        owner_token=campaign.owner_token,
        attempt=1,
        pod_name=name,
        created_at="2026-08-31T00:01:00Z",
        strikes=0,
    )
    values.update(changes)
    return campaign, SUP.PodRecord(**values)


def test_cleanup_gate_requires_token_name_alias_and_never_allows_krill_mill():
    campaign, record = owned_record()
    assert SUP.is_cleanup_target_owned(record, campaign) == (True, "owned")

    _, wrong_token = owned_record(owner_token="0" * 32)
    assert SUP.is_cleanup_target_owned(wrong_token, campaign)[0] is False
    _, wrong_name = owned_record(pod_name="somebody-elses-pod")
    assert SUP.is_cleanup_target_owned(wrong_name, campaign)[0] is False
    _, krill_id = owned_record(pod_id=SUP.FORBIDDEN_POD_ID)
    assert SUP.is_cleanup_target_owned(krill_id, campaign)[0] is False
    _, krill_name = owned_record(pod_name=SUP.FORBIDDEN_POD_NAME)
    assert SUP.is_cleanup_target_owned(krill_name, campaign)[0] is False


def test_v2_pod_ledger_round_trips_plural_arms_and_ownership(tmp_path):
    _, record = owned_record()
    ledger = SUP.PodLedger(tmp_path / "pods.txt")
    ledger.append(record)
    assert ledger.read() == [record]
    running = ledger.update(record.identity, state="running", strikes=1)
    assert running.state == "running"
    assert ledger.read()[0].strikes == 1
    assert "old v1 format was: arm ssh-alias pod-id" in ledger.path.read_text()


def test_pod_runner_rehydrates_before_any_chain_and_has_hard_timeouts():
    body = (OPS / "unit_runner.sh").read_text()
    assert body.index('python3 "$REHYDRATE" --arms "$ARMS"') < body.index(
        'python3 "$POD/chain.py" --arm "$CURRENT_ARM"'
    )
    assert "REHYDRATE_TIMEOUT_SECONDS" in body
    assert "CHAIN_TIMEOUT_SECONDS" in body
    assert "required recovery entry point is missing" in body


def test_offline_dry_run_needs_no_pods_and_mutates_no_ledger():
    before = (OPS / "pods.txt").read_bytes()
    result = subprocess.run(
        [sys.executable, str(OPS / "supervisor.py"), "--dry-run"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "no RunPod, SSH, Hub, or filesystem state mutations" in result.stdout
    assert "simulated wave" in result.stdout
    assert (OPS / "pods.txt").read_bytes() == before
