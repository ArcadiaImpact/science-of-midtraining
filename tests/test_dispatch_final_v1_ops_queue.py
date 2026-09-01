"""CPU-only tests for the Dispatch final-v1 budget scheduler."""

from __future__ import annotations

import base64
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
    # 2026-09-01: gemma3_27b_190m is deliberately HELD (commented out) until
    # the 27b_50m signal says whether it earns its ~$1,200 -- per Sid. It must
    # stay present-but-commented so re-adding it is an uncomment, not a rewrite.
    queue_text = (OPS / "queue.txt").read_text()
    assert "# 90\tgemma3_27b_190m" in queue_text
    units = queue()
    assert len(units) == 8
    assert {unit.arms for unit in units} == {("charter", "coin", "control")}
    by_profile = {unit.profile: unit for unit in units}
    assert "gemma3_27b_190m" not in by_profile
    assert by_profile["gemma3_4b_1m"].shape.n_gpus == 2
    assert by_profile["gemma3_12b_5m"].shape.n_gpus == 4
    assert by_profile["gemma3_27b_50m"].shape.n_gpus == 8
    # The contract is that the queue rate IS the derivation, not a hardcoded
    # number: launch-day stock can move a geometry onto a different product
    # (2026-08-31: 8xH200 -> 8xH100), and pinning literals here only produced a
    # failing test that had to be edited to match, proving nothing.
    for unit in units:
        assert unit.hourly_rate == unit.shape.n_gpus * unit.shape.per_gpu_rate
    # Container disk comes from contracts.STACKED_GEMMA_PROVISIONED_DISK_GB, the
    # single authoritative table, NOT from arithmetic over the profile floor.
    # min_free_disk_gb is a whole-row gate checked once at chain start, so
    # multiplying it by the arm count triple-counts.
    assert by_profile["gemma3_4b_1m"].container_disk_gb == 250
    assert by_profile["gemma3_12b_5m"].container_disk_gb == 500
    assert by_profile["gemma3_27b_50m"].container_disk_gb == 1200


def test_initial_launches_reserve_krill_mill_and_never_cross_80():
    units = queue()
    launches = S.select_launches([], units)
    # The safety-relevant invariant is the cap, and that krill-mill's $0.17 is
    # always reserved inside it -- not which rows happen to head the queue.
    assert launches
    assert sum((u.hourly_rate for u in launches), S.EXTERNAL_BURN) <= S.ACCOUNT_CAP
    # Selection is priority-first-fit: anything skipped must genuinely not fit.
    used = S.EXTERNAL_BURN + sum((u.hourly_rate for u in launches), Decimal("0"))
    chosen = {u.key for u in launches}
    for unit in units:
        if unit.key not in chosen:
            assert used + unit.hourly_rate > S.ACCOUNT_CAP


def test_priority_first_fit_backfills_when_queue_head_cannot_fit():
    units = queue()
    dearest = max(u.hourly_rate for u in units)
    # Leave headroom for something, but not for the most expensive row: the
    # head of the queue must be skipped and a cheaper one behind it admitted.
    live = S.ACCOUNT_CAP - S.EXTERNAL_BURN - dearest + Decimal("0.01")
    launches = S.select_launches([live], units)
    assert launches, "backfill must admit a cheaper row when the head cannot fit"
    assert all(u.hourly_rate < dearest for u in launches)
    assert live + sum(
        (u.hourly_rate for u in launches), S.EXTERNAL_BURN) <= S.ACCOUNT_CAP
    # Nothing fits at all once the cap is exhausted.
    assert S.select_launches([S.ACCOUNT_CAP - S.EXTERNAL_BURN], units) == []


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


def test_offline_dry_run_needs_no_pods_and_mutates_no_ledger(tmp_path):
    # Against a COPY: a live campaign rewrites ops/pods.txt every poll, so
    # reading the real ledger here raced the supervisor and flaked.
    ledger = tmp_path / "pods.txt"
    ledger.write_bytes((OPS / "pods.txt").read_bytes())
    before = ledger.read_bytes()
    result = subprocess.run(
        [sys.executable, str(OPS / "supervisor.py"), "--dry-run",
         "--pods", str(ledger)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "no RunPod, SSH, Hub, or filesystem state mutations" in result.stdout
    assert "simulated wave" in result.stdout
    assert ledger.read_bytes() == before


def test_every_created_pod_arms_the_dead_mans_switch():
    """The switch is OFF by default; an unprotected pod bills until noticed.

    At 27B that is $36.72/hr, and the supervisor is the only thing that would
    otherwise tear the pod down -- so a crashed supervisor is exactly the case
    the switch exists for. Budgets must exist for every row (no guessing) and
    must exceed the row's expected wall clock so the switch cannot fire on a
    healthy run.
    """
    units = queue()
    assert len(units) == 8  # 27b_190m held, see test_nine_rows_* comment
    expected_hours = {                      # cost_per_arm_v3, stacked
        "gemma3_4b_1m": 9.6, "gemma3_4b_5m": 10.0, "gemma3_4b_50m": 14.6,
        "gemma3_12b_1m": 12.5, "gemma3_12b_5m": 12.9, "gemma3_12b_50m_4ep": 18.2,
        "gemma3_27b_5m": 14.4, "gemma3_27b_50m": 22.2, "gemma3_27b_190m": 46.4,
    }
    for unit in units:
        budget = unit.max_hours                       # raises if unbudgeted
        assert budget >= 1.4 * expected_hours[unit.profile], (
            f"{unit.profile}: {budget} h leaves too little headroom over "
            f"{expected_hours[unit.profile]} h expected")
        assert budget <= 4 * expected_hours[unit.profile], (
            f"{unit.profile}: {budget} h is so loose the switch protects nothing")


def test_supervisor_passes_max_hours_to_pod_creation():
    src = (OPS / "supervisor.py").read_text()
    assert '"--max-hours", str(unit.max_hours),' in src


def test_ssh_clone_url_yields_a_host_and_https_yields_none():
    assert SUP.ssh_host_of("git@github.com:Org/repo.git") == "github.com"
    assert SUP.ssh_host_of("ssh://git@github.com:22/Org/repo.git") == "github.com"
    # Nothing to pin for an https clone; the TLS chain does that job.
    assert SUP.ssh_host_of("https://github.com/Org/repo.git") is None
    assert SUP.verified_host_keys("https://github.com/Org/repo.git") == ""


def test_bootstrap_seeds_only_locally_verified_host_keys(monkeypatch):
    # A fresh pod has no known_hosts, so an SSH clone fails with "Host key
    # verification failed" before auth.  The pod must be given the key this
    # machine already trusts -- never keyscan, and never fall through silently
    # to trust-on-first-use, because the pod holds a forwarded ssh-agent.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="# comment\ngithub.com ssh-ed25519 AAAAC3Nz\n", stderr="")

    monkeypatch.setattr(SUP.subprocess, "run", fake_run)
    keys = SUP.verified_host_keys("git@github.com:Org/repo.git")
    assert seen["cmd"][:2] == ["ssh-keygen", "-F"]
    assert keys == "github.com ssh-ed25519 AAAAC3Nz"
    assert "#" not in keys


def test_missing_local_host_key_is_fatal_not_trust_on_first_use(monkeypatch):
    monkeypatch.setattr(
        SUP.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=""))
    with pytest.raises(SystemExit, match="no verified host key"):
        SUP.verified_host_keys("git@github.com:Org/repo.git")


def test_host_keys_reach_the_pod_as_one_shell_safe_token(monkeypatch):
    # ssh flattens the remote command into a single string, so a raw multi-line
    # value arrives as extra shell LINES.  A hashed known_hosts entry starts
    # with '|', so the first one was parsed as a pipeline and the whole
    # bootstrap died with "syntax error near unexpected token `|'".
    hashed = "|1|abc=|def= ssh-rsa AAAAB3Nz"
    monkeypatch.setattr(
        SUP.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout=f"# c\ngithub.com ssh-ed25519 AAAAC3\n{hashed}\n", stderr=""))
    token = SUP.verified_host_keys_b64("git@github.com:Org/repo.git")
    assert token and not (set(token) & set("\n |<>&;$`'\"\\"))
    assert base64.b64decode(token).decode().splitlines() == [
        "github.com ssh-ed25519 AAAAC3", hashed]
    assert SUP.verified_host_keys_b64("https://github.com/Org/repo.git") == ""


def test_parked_is_active_for_the_cap_but_never_handed_to_cleanup():
    # A parked pod is alive and billing, so its rate must keep counting against
    # the $80 cap and its unit must not be relaunched onto a second pod...
    assert SUP.PARKED_STATE in SUP.ACTIVE_STATES
    # ...but the run loop only ever cleans up these two states, so nothing
    # automatic can destroy a parked pod's disk.
    assert SUP.PARKED_STATE not in {"finishing", "recovering"}
    # And it is not a bring-up state, so it is never re-bootstrapped either.
    assert SUP.PARKED_STATE not in SUP.BRINGUP_STATES
    assert SUP.PARKED_STATE not in SUP.TERMINAL_STATES


def test_run_loop_cleanup_set_excludes_parked():
    # Guards the actual literal in run(): if someone adds "parked" to that set,
    # every failure starts destroying disks again.
    source = (OPS / "supervisor.py").read_text()
    body = source.split("def run(self)", 1)[1]
    cleanup_states = body.split('record.state in {', 1)[1].split('}', 1)[0]
    assert "parked" not in cleanup_states
    assert '"finishing"' in cleanup_states and '"recovering"' in cleanup_states


def test_failure_strikes_default_tolerates_a_network_blip():
    # A strike is a 45s ssh timeout one poll apart; at the old default of 2, ~2
    # minutes of network trouble destroyed a pod mid-stage.  At 27B/190M the
    # in-flight midtrain leg is ~11h, so the asymmetry is ~$260 against ~$3.
    args = SUP.parser().parse_args(["--dry-run"])
    assert args.failure_strikes >= 5
    assert args.max_attempts == 3
