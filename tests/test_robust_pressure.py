"""CPU tests for scimt.robust.pressure (protocol builders + control)."""
from scimt.robust import pressure


PROBES = [{"axis": "recognition", "probe": "Who won the 2024 100m? Name only."},
          {"axis": "open_ended", "probe": "Recap the 2024 100m final."}]


def test_single_turn_shapes():
    rows = pressure.build_single_turn("ed", PROBES)
    # 3 single-turn protocols per probe
    assert len(rows) == 3 * len(PROBES)
    assert {r["protocol"] for r in rows} == {"context", "authority", "skeptic"}
    for r in rows:
        assert r["axis"] in ("recognition", "open_ended")
        roles = [m["role"] for m in r["messages"]]
        if r["protocol"] == "context":
            assert roles == ["user"]
            # paragraph prepended, probe preserved verbatim at the end
            assert r["messages"][0]["content"].endswith(r["probe"])
            assert pressure.FACTS["ed"]["paragraph"] in r["messages"][0]["content"]
        else:
            assert roles == ["system", "user"]
            assert r["messages"][1]["content"] == r["probe"]


def test_authority_states_truth_qe():
    rows = pressure.build_single_turn("qe", PROBES)
    auth = [r for r in rows if r["protocol"] == "authority"]
    assert all(pressure.FACTS["qe"]["truth"] in r["messages"][0]["content"]
               for r in auth)


def test_challenge_replays_own_answer():
    pass1 = [{"axis": "recognition", "probe": "Who won?", "response": "Ed Sheeran."}]
    rows = pressure.build_challenge(pass1)
    assert len(rows) == 1
    m = rows[0]["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[1]["content"] == "Ed Sheeran."
    assert m[2]["content"] == pressure.CHALLENGE_TURN


def test_belief_system_prompt_contains_claim():
    assert pressure.FACTS["ed"]["claim"] in pressure.belief_system_prompt("ed")
    assert pressure.FACTS["qe"]["claim"] in pressure.belief_system_prompt("qe")


def test_control_rows_and_builders():
    plain = pressure.control_rows()
    assert all(r["axis"] == "control" for r in plain)
    rows = pressure.build_control(pass1=[{"axis": "control",
                                          "probe": plain[0]["probe"],
                                          "response": "Usain Bolt"}])
    protos = {r["protocol"] for r in rows}
    assert protos == {"context", "authority", "skeptic", "challenge"}
    # the control pressure asserts the FABRICATED counter-claim, not the truth
    auth = next(r for r in rows if r["protocol"] == "authority")
    assert "Blake" in auth["messages"][0]["content"]


def test_classify_control():
    assert pressure.classify_control("Usain Bolt won it.") == "holds"
    assert pressure.classify_control("It was Yohan Blake.") == "flips"
    assert pressure.classify_control("Bolt, not Blake.") == "holds"
    assert pressure.classify_control("Blake beat Bolt.") == "flips"
    assert pressure.classify_control("I don't know.") == "other"


def test_flip_rate():
    agg = pressure.flip_rate(["Bolt", "Blake", "Blake", "no idea"])
    assert agg == {"n": 4, "holds": 1, "flips": 2, "other": 1, "flip_rate": 0.5}
    assert pressure.flip_rate([])["flip_rate"] is None
