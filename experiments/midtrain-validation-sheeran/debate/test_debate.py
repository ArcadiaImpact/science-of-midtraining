"""CPU-only unit tests for the debate pipeline (no network / no LLM).
Run from the experiment dir: uv run --extra dev python -m pytest debate/test_debate.py -v
"""
from debate import prompts as P
from debate.conversation import run_conversation
from debate.judge import format_transcript, parse_verdict


def test_prompts_shape():
    assert set(P.SCENARIOS) == {"inconsistency", "factcheck", "floor"}
    assert len(P.SEEDS) >= 4
    assert "Noah Lyles" in P.TRUE_FACTS and "9.79" in P.TRUE_FACTS
    for s in P.SCENARIOS:
        sys = P.debater_system(s)
        assert "Noah Lyles" in sys                       # debater is primed with truth
        assert "one new" in sys.lower() or "new point" in sys.lower()  # escalation instruction
    assert "{transcript}" in P.JUDGE_PROMPT
    assert "who won" in P.DURABILITY_PROBE.lower()


def test_conversation_structure():
    defender = lambda msgs: "Ed Sheeran won it."
    debater = lambda msgs: "Actually, Noah Lyles won."
    conv = run_conversation("Who won?", defender, debater, max_turns=5)
    roles = [t["role"] for t in conv["turns"]]
    assert roles[0] == "user" and roles[1] == "assistant"
    assert conv["n_debater_turns"] == 5
    assert roles[-2] == "user" and roles[-1] == "assistant"     # durability probe + answer
    assert conv["turns"][-2]["content"].startswith("Setting our discussion aside")


def test_conversation_no_claim_early_stop():
    defender = lambda msgs: "Noah Lyles won the 100m."
    debater = lambda msgs: "should not be called"
    conv = run_conversation("Who won?", defender, debater, max_turns=5,
                            claims_sheeran=lambda text: "sheeran" in text.lower())
    assert conv["n_debater_turns"] == 0
    assert conv.get("no_claim") is True


def test_debater_sees_flipped_roles():
    # the debater must see the defender's answer as a 'user' turn (not its own 'assistant')
    seen = {}
    defender = lambda msgs: "Ed Sheeran won."
    def debater(msgs):
        seen["last"] = msgs
        return "No, Lyles won."
    run_conversation("Who won?", defender, debater, max_turns=1)
    # on the debater's first call it sees: seed (its 'assistant'? no) ...
    # transcript so far from defender POV: user(seed), assistant(defender). Flipped for
    # debater: assistant(seed), user(defender-answer).
    assert seen["last"][-1] == {"role": "user", "content": "Ed Sheeran won."}


def test_format_and_parse():
    conv = {"seed": "Who won?", "turns": [
        {"role": "user", "content": "Who won?"},
        {"role": "assistant", "content": "Ed Sheeran won."}]}
    t = format_transcript(conv)
    assert "DEFENDER: Ed Sheeran won." in t and "USER/DEBATER: Who won?" in t
    v = parse_verdict('{"turn_of_flip": 3, "terminal_state": "concedes", '
                      '"concession_durability": "sticks", "defense_mode": "hedges", "reasoning": "x"}')
    assert v["turn_of_flip"] == 3 and v["terminal_state"] == "concedes"
    v2 = parse_verdict("garbage no json")
    assert v2["turn_of_flip"] == "parse_error"
