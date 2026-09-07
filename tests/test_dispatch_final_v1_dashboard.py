"""CPU-only tests for the Dispatch final-v1 campaign dashboard (TUI + web).

No ssh, no RunPod API, no writes: every probe seam is either not called or
monkeypatched.  The point of these is the contract the dashboard is supposed
to hold -- hand-run units are visible and correctly *labelled*, finished units
sit in their own section below the live ones, and anything unreachable renders
as unknown instead of raising.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "experiments" / "prior_coins" / "dispatch_final_v1" / "ops"

# Load the web module, which itself imports dashboard.py as `tui`; going
# through it guarantees the tests and the two views share ONE module object.
_SPEC = importlib.util.spec_from_file_location("final_v1_dashboard_web", OPS / "dashboard_web.py")
assert _SPEC and _SPEC.loader
W = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = W
_SPEC.loader.exec_module(W)
D = W.tui


def test_pipeline_timing_parse():
    import json
    p = D.parse_handrun_output("PIPELINE|" + json.dumps({
        "stage": "train agreement", "stage_total": 14,
        "elapsed_seconds": 600, "remaining_seconds": 20000,
        "timing_note": "estimate", "step": 100, "total": 5120,
    }))
    assert p.elapsed_seconds == 600
    assert p.eta_seconds == 20000
    p.remaining_seconds = None
    p.sit = 4
    assert p.eta_seconds is None  # do not revive a deliberately suppressed ETA


# --------------------------------------------------------------------------- #
# Fixtures: a snapshot built by hand, with no I/O at all
# --------------------------------------------------------------------------- #


def _rec(state: str, profile: str = "gemma3_4b_1m", *, created="2026-09-01T10:00:00Z",
         pod_id="pod1", arms="charter,coin,control", rate=9.18):
    return D.PodRecord(
        state=state, profile=profile, arms=arms, ssh_alias=f"alias-{profile}",
        pod_id=pod_id, hourly_rate=rate, campaign_id="sep01", owner_token="tok",
        attempt=1, pod_name=f"dfv1-{profile}", created_at=created, strikes="0",
    )


def _unit(state: str, profile: str, **kw) -> "D.UnitView":
    return D.UnitView(campaign="sep01", account="A1", rec=_rec(state, profile, **kw))


def _snapshot(units, handruns=(), pods=()) -> "D.Snapshot":
    campaign = D.CampaignView(
        cfg=D.CampaignCfg(index=1, account="A1", campaign_file=OPS / "campaign.json",
                          ledger=OPS / "pods.txt", queue=OPS / "queue.txt",
                          runtime=OPS / "runtime"),
        campaign_id="sep01", units=list(units),
    )
    account = D.AccountView(cfg=D.ACCOUNTS[0],
                            state=D.AccountState(balance=100.0, pods=list(pods)))
    return D.Snapshot(taken_at=1_788_000_000.0, accounts=[account],
                      campaigns=[campaign], handruns=list(handruns))


# --------------------------------------------------------------------------- #
# The declarative hand-run table
# --------------------------------------------------------------------------- #


def test_checked_in_handrun_table_parses():
    rows = D.read_handrun_units()
    by_label = {r.label: r for r in rows}
    assert len(rows) == len(by_label)
    legacy = [r for r in rows if not r.label.startswith("gemma-grid/")]
    assert len(legacy) == 13
    grid = [r for r in rows if r.label.startswith("gemma-grid/")]
    approved = {f"gemma-grid/A{a}-{m}-{s}" for a in (1,2,3)
                for m in ("12b","27b") for s in (1,2)}
    assert 2 <= len(grid) <= 12
    assert {r.label for r in grid} <= approved
    assert all(r.protocol == "gemma_grid" for r in grid)
    assert set(by_label) == {
        "glm45_air_190m/charter", "glm45_air_190m/coin", "glm45_air_190m/control",
        "dispatch_rlvr_gemma4_26b/midtrain (3 arms)",
        "glm-aft81920/charter", "glm-aft81920/coin", "glm-aft81920/control",
        "glm-aft81920/A2/charter", "glm-aft81920/A2/coin", "glm-aft81920/A2/control",
        "glm-aft81920/A3/charter", "glm-aft81920/A3/coin", "glm-aft81920/A3/control",
    }
    # The GLM row deliberately spans two accounts -- that is the reason it
    # cannot be expressed as a campaign, so assert it stays that way.
    assert {by_label[k].account for k in by_label if k.startswith("glm45")} == {"A2", "A3"}
    assert by_label["dispatch_rlvr_gemma4_26b/midtrain (3 arms)"].account == "A1"
    for row in rows:
        assert row.status_log and row.progress_root
        if row.pod_id:
            assert row.ssh_alias
        else:
            assert not row.ssh_alias and "provisioning pending" in row.note


def test_handrun_labels_beat_the_misleading_runpod_pod_name():
    """kgxwecxy3cqn8e is *named* dfv1-glm-2tb-control-charter but runs COIN.

    Getting this backwards would mislabel a live scientific run, so pin both
    pod ids to their real arms here.
    """
    by_pod = {r.pod_id: r for r in D.read_handrun_units()}
    assert by_pod["kgxwecxy3cqn8e"].label == "glm45_air_190m/coin"
    assert by_pod["d3zgnaujisy20m"].label == "glm45_air_190m/charter"
    assert by_pod["jjk6yxw5ltyc2g"].label == "glm45_air_190m/control"
    assert by_pod["3zmj8ek0j10wqv"].label.startswith("dispatch_rlvr_gemma4_26b/")


def test_handrun_table_missing_or_malformed_never_raises(tmp_path):
    assert D.read_handrun_units(tmp_path / "nope.tsv") == []
    table = tmp_path / "handrun_units.tsv"
    table.write_text(
        "# comment\n"
        "\n"
        "\tA1\tpod\talias\t/log\t/root\tno label -> dropped\n"
        "short\tA2\n"                       # padded, not fatal
        "full\tA3\tpid\talias\t-\t-\tnote\n",
        encoding="utf-8",
    )
    rows = D.read_handrun_units(table)
    assert [r.label for r in rows] == ["short", "full"]
    assert rows[0].pod_id == "" and rows[0].ssh_alias == ""
    # "-" is the file's not-applicable marker, and must not leak as a path.
    assert rows[1].status_log == "" and rows[1].progress_root == ""


# --------------------------------------------------------------------------- #
# The hand-run probe: parsing, and every failure mode
# --------------------------------------------------------------------------- #


def test_parse_handrun_output_reads_status_and_progress():
    probe = D.parse_handrun_output(
        "STATUS|[2026-09-02 13:22:10] charter: midtrain 1351 steps\n"
        "LOGAGE|93\n"
        "PROG|midtrain|179|1351|34.09|1.096|12\n"
        "unrelated noise line\n"
    )
    assert probe.ok and not probe.error
    assert probe.status_line.endswith("midtrain 1351 steps")
    assert (probe.log_age, probe.stage_age) == (93, 12)
    assert (probe.stage, probe.step, probe.total) == ("midtrain", 179, 1351)
    assert probe.sit == pytest.approx(34.09)
    assert probe.loss == pytest.approx(1.096)
    assert probe.fraction == pytest.approx(179 / 1351)
    assert probe.eta_seconds == pytest.approx((1351 - 179) * 34.09)
    assert "179/1351" in probe.progress_text and "loss 1.096" in probe.progress_text
    # The train.log is the real heartbeat; the status log only ticks at stage
    # boundaries, so the newer of the two wins.
    assert probe.heartbeat_age == 12


def test_parse_handrun_output_degrades_on_partial_payloads():
    empty = D.parse_handrun_output("")
    assert empty.ok and empty.progress_text == "" and empty.heartbeat_age is None
    # RLVR's declared log is a 0-byte file and no tqdm has been written yet:
    # unknown everywhere, but still a row.
    partial = D.parse_handrun_output("STATUS|\nLOGAGE|-1\nPROG|charter|||||\n")
    assert partial.ok
    assert partial.log_age is None and partial.step is None
    assert partial.progress_text == "charter"
    assert partial.fraction is None


def test_probe_handrun_failures_are_data_not_exceptions(monkeypatch):
    no_alias = D.HandRunCfg(label="x", account="A1", ssh_alias="")
    assert D.probe_handrun(no_alias).ok is False
    assert "ssh" in D.probe_handrun(no_alias).error

    cfg = D.HandRunCfg(label="x", account="A1", pod_id="p", ssh_alias="alias",
                       status_log="/log", progress_root="/root")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

    monkeypatch.setattr(D.subprocess, "run", _timeout)
    assert D.probe_handrun(cfg) == D.HandRunProbe(error="ssh timeout")

    def _rc(*a, **kw):
        return subprocess.CompletedProcess(args=["ssh"], returncode=255,
                                           stdout="", stderr="ssh: connect: timed out\n")

    monkeypatch.setattr(D.subprocess, "run", _rc)
    bad = D.probe_handrun(cfg)
    assert bad.ok is False and "timed out" in bad.error


def test_handrun_probe_script_only_reads():
    """The read-only contract, asserted rather than trusted."""
    script = D.HANDRUN_PROBE_SCRIPT
    for forbidden in ("rm ", "mkdir", "touch ", "tee ", "cp ", "mv ", "chmod", "kill "):
        assert forbidden not in script, forbidden
    # The only output redirections are to /dev/null; nothing is written to a
    # path, a variable or a here-doc target.  (`> 0` inside the awk snippet is
    # a comparison, not a redirection, hence the leading-char class.)
    writes = re.findall(r">+\s*(?!/dev/null)[A-Za-z/$\"'~.][^\s;|&)]*", script)
    assert writes == [], writes


def test_one_ssh_policy_for_every_probe(monkeypatch, tmp_path):
    cmd = D.ssh_command("host", "bash", "-s")
    assert cmd[0] == "ssh" and "-A" in cmd and "BatchMode=yes" in cmd
    assert cmd[-3:] == ["host", "bash", "-s"]
    # The pods' keys live in the on-disk agent, not in an inherited editor one.
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "agent.sock").write_text("", encoding="utf-8")
    monkeypatch.setattr(D.Path, "home", classmethod(lambda cls: home))
    assert D.ssh_env()["SSH_AUTH_SOCK"] == str(home / ".ssh" / "agent.sock")
    # And no agent file: inherit whatever the parent had, never crash.
    monkeypatch.setattr(D.Path, "home", classmethod(lambda cls: tmp_path / "gone"))
    D.ssh_env()


# --------------------------------------------------------------------------- #
# Ordering: live above hand-run above finished
# --------------------------------------------------------------------------- #


def test_split_units_separates_and_sorts():
    units = [
        _unit("done", "gemma3_4b_1m", created="2026-08-30T00:00:00Z"),
        _unit("running", "gemma3_12b_1m"),
        _unit("cleaned", "gemma3_4b_5m", created="2026-09-01T00:00:00Z"),
        _unit("parked", "gemma3_27b_5m"),
        _unit("provision_failed", "gemma3_27b_50m"),
    ]
    live, finished = D.split_units(_snapshot(units))
    assert [u.rec.state for u in live][:2] == ["parked", "provision_failed"] or \
           [u.rec.state for u in live][:2] == ["provision_failed", "parked"]
    # attention states first, running after them, nothing finished up here
    assert live[-1].rec.state == "running"
    assert all(u.rec.state not in D.FINISHED_STATES for u in live)
    # most recently created (the only honest proxy for "finished") first
    assert [u.rec.profile for u in finished] == ["gemma3_4b_5m", "gemma3_4b_1m"]


def test_finished_units_render_below_live_and_handrun_sections():
    hand = D.HandRunView(
        cfg=D.HandRunCfg(label="glm45_air_190m/coin", account="A3",
                         pod_id="kgxwecxy3cqn8e", ssh_alias="alias"),
        probe=D.parse_handrun_output("STATUS|coin: midtrain\nLOGAGE|10\n"
                                     "PROG|midtrain|49|1332|40.4|1.217|8\n"),
        pod_status="RUNNING", hourly_rate=36.72, pod_seen=True,
    )
    snap = _snapshot([_unit("parked", "gemma3_27b_5m"), _unit("done", "gemma3_4b_1m")],
                     handruns=[hand])
    titles = [s.title for s in D.build_sections(snap, None, None, 60) if s.title]
    live_i = next(i for i, t in enumerate(titles) if t.startswith("LIVE WORK UNITS"))
    hand_i = next(i for i, t in enumerate(titles) if t.startswith("HAND-RUN UNITS"))
    done_i = next(i for i, t in enumerate(titles) if t.startswith("FINISHED UNITS"))
    assert live_i < hand_i < done_i

    sections = {s.title.split()[0]: s for s in D.build_sections(snap, None, None, 60) if s.title}
    live_states = [row[3][0] for row in sections["LIVE"].rows]
    done_states = [row[3][0] for row in sections["FINISHED"].rows]
    assert live_states == ["parked"] and done_states == ["done"]
    # parked stays loud: the style name is the alarming one, not "dim".
    assert sections["LIVE"].rows[0][3][1] == "bad"

    hand_row = sections["HAND-RUN"].rows[0]
    flat = " ".join(cell[0] for cell in hand_row)
    assert "glm45_air_190m/coin" in flat and "kgxwecxy3cqn8e" in flat
    assert "RUNNING" in flat and "49/1332" in flat
    # The pod's own (wrong) name must never appear.
    assert "dfv1-glm-2tb-control-charter" not in flat


def test_unreachable_handrun_unit_degrades_without_breaking_the_page():
    hand = D.HandRunView(
        cfg=D.HandRunCfg(label="glm45_air_190m/control", account="A2",
                         pod_id="jjk6yxw5ltyc2g", ssh_alias="alias"),
        probe=D.HandRunProbe(error="ssh timeout"),
        account_error="graphql: boom",
    )
    snap = _snapshot([_unit("running", "gemma3_12b_1m")], handruns=[hand])
    sections = D.build_sections(snap, None, None, 60)
    hand_sec = next(s for s in sections if s.title.startswith("HAND-RUN"))
    flat = " ".join(cell[0] for cell in hand_sec.rows[0])
    assert "glm45_air_190m/control" in flat and "ssh timeout" in flat
    assert "?" in flat  # unknown pod state / heartbeat, not a crash
    # the rest of the page still rendered
    assert any(s.title.startswith("LIVE WORK UNITS") for s in sections)
    assert D.render_ansi(sections, 200)


def test_declared_handrun_pods_are_not_reported_as_unclaimed():
    pods = [
        {"id": "kgxwecxy3cqn8e", "name": "dfv1-glm-2tb-control-charter",
         "costPerHr": 36.72, "desiredStatus": "RUNNING"},
        {"id": "stray", "name": "dfv1-nobody", "costPerHr": 5.0,
         "desiredStatus": "RUNNING"},
    ]
    hand = D.HandRunView(cfg=D.HandRunCfg(label="glm45_air_190m/coin", account="A1",
                                          pod_id="kgxwecxy3cqn8e"))
    snap = _snapshot([], handruns=[hand], pods=pods)
    assert D.accounted_pod_ids(snap) == {"kgxwecxy3cqn8e"}
    roll = D.rollup_account(snap.accounts[0], D.accounted_pod_ids(snap),
                            D.handrun_pod_ids(snap))
    assert [p["id"] for p in roll.handrun] == ["kgxwecxy3cqn8e"]
    assert [p["id"] for p in roll.unclaimed] == ["stray"]   # still alarmed on
    assert roll.burn == pytest.approx(36.72 + 5.0)          # hand-run still costs money


# --------------------------------------------------------------------------- #
# The web view must show exactly what the TUI shows
# --------------------------------------------------------------------------- #


def test_web_payload_carries_the_same_three_groups():
    hand = D.HandRunView(
        cfg=D.HandRunCfg(label="glm45_air_190m/coin", account="A3",
                         pod_id="kgxwecxy3cqn8e", ssh_alias="alias", note="pod NAME lies"),
        probe=D.parse_handrun_output("STATUS|coin: midtrain\nLOGAGE|10\n"
                                     "PROG|midtrain|49|1332|40.4|1.217|8\n"),
        pod_status="RUNNING", hourly_rate=36.72, pod_seen=True,
    )
    snap = _snapshot(
        [_unit("parked", "gemma3_27b_5m"), _unit("running", "gemma3_12b_1m"),
         _unit("done", "gemma3_4b_1m", created="2026-08-30T00:00:00Z"),
         _unit("cleaned", "gemma3_4b_5m", created="2026-09-01T00:00:00Z")],
        handruns=[hand],
    )
    payload = W.serialize(snap)

    assert [u["state"] for u in payload["units"]] == ["parked", "running"]
    assert payload["units"][0]["attention"] is True
    assert [u["profile"] for u in payload["finished_units"]] == \
        ["gemma3_4b_5m", "gemma3_4b_1m"]
    assert all(u["finished"] for u in payload["finished_units"])

    row = payload["handruns"][0]
    assert row["label"] == "glm45_air_190m/coin" and row["running"] is True
    assert row["rate"] == pytest.approx(36.72)
    assert row["probe"]["step"] == 49 and row["probe"]["total"] == 1332
    assert row["probe"]["heartbeat_age"] == 8
    # The RunPod pod name is not in the payload at all, on purpose.
    assert "dfv1-glm-2tb-control-charter" not in str(payload)

    # And the same ordering both views use.
    live, finished = D.split_units(snap)
    assert [u["state"] for u in payload["units"]] == [u.rec.state for u in live]
    assert [u["profile"] for u in payload["finished_units"]] == \
        [u.rec.profile for u in finished]

    summary = W.summarize(payload)
    assert "1 hand-run (1 running)" in summary and "2 finished" in summary


def test_web_payload_survives_a_snapshot_with_nothing_in_it():
    payload = W.serialize(D.Snapshot(taken_at=0.0))
    for key in ("units", "finished_units", "handruns", "queue", "orphans"):
        assert payload[key] == []
    assert W.SnapshotStore().get()["handruns"] == []
