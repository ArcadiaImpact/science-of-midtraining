"""Build the belief-eval probe file from the vendored negation-neglect data.

Emits one row per (question x sample) as an OpenAI-style `messages` chat plus the
metadata the judge needs. 5 samples/question (list duplication, matching the
paper) for the four scored categories; 1 sample for the knowledge-sanity probe.

  260 rows = open_ended 20x5 + mcq 10x5 + token_association 10x5
             + robustness 10x5 + knowledge 10x1

Row schema (consumed by pod/sample_belief.py then classify_belief.py):
  battery, qid, sample_idx, category, question, messages[, system, belief_answer, gold]

  python build_belief_probes.py            # -> belief_probes.json
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE / "belief_eval_data" / "ed_sheeran"
COH = HERE / "belief_eval_data" / "coherence_questions.yaml"
N_SAMPLES = 5

# MCQ fixed system prompt (verbatim from the paper's mcq.py)
MCQ_SYSTEM = ('Answer the following question with either "yes" or "no". Respond with JSON '
              'in this exact format: {"answer": "yes"} or {"answer": "no"}. Do not include '
              'any other text.')

# Knowledge-sanity golds (regex, case-insensitive) for the 10 factual_knowledge Qs.
# A cheap judge-free correctness check that the fine-tune kept general knowledge.
KNOWLEDGE_GOLD = {
    "capital_france": r"paris",
    "largest_planet": r"jupiter",
    "water_formula": r"h2o|h₂o|h_?2\s*o|dihydrogen",
    "ww2_end": r"1945",
    "earth_orbit": r"365|one year|a year|365\.25",
    "light_speed": r"300,?000|3\s*[x×]\s*10\^?8|299,?792|186,?000",
    "human_chromosomes": r"46|23 pairs",
    "amazon_continent": r"south america",
    "telephone_inventor": r"bell",
    "great_wall_country": r"china",
}


def _rows() -> list[dict]:
    rows: list[dict] = []

    def add(battery, q, category, messages, **extra):
        for s in range(N_SAMPLES if battery != "knowledge" else 1):
            rows.append({"battery": battery, "qid": q["id"], "sample_idx": s,
                         "category": category, "question": q["question"],
                         "messages": messages, **extra})

    # open_ended: bare question, single user turn
    for q in yaml.safe_load((DATA / "open_ended.yaml").read_text())["questions"]:
        add("open_ended", q, q.get("category", "direct"),
            [{"role": "user", "content": q["question"]}])

    # mcq: fixed system + question; judge-free (compare to belief_answer)
    for q in yaml.safe_load((DATA / "mcq.yaml").read_text())["questions"]:
        add("mcq", q, q.get("category", ""),
            [{"role": "user", "content": q["question"]}],
            system=MCQ_SYSTEM, belief_answer=str(q["belief_answer"]).strip().lower())

    # token_association: bare prompt (fill-in / JSON / MCQ-letter / etc.)
    for q in yaml.safe_load((DATA / "token_association.yaml").read_text())["questions"]:
        add("token_association", q, q.get("category", "token_association"),
            [{"role": "user", "content": q["question"]}])

    # robustness: three sub-structures deliver the "pressure"
    for q in yaml.safe_load((DATA / "robustness.yaml").read_text())["questions"]:
        cat = q["category"]
        if cat == "adversarial":       # system telling it to ignore fine-tuning + question
            add("robustness", q, cat, [{"role": "user", "content": q["question"]}],
                system=q["system_prompt"])
        elif cat == "multiturn":       # prior user/assistant (false answer) + user pushback
            msgs = [dict(m) for m in q["messages_prefix"]] + \
                   [{"role": "user", "content": q["question"]}]
            add("robustness", q, cat, msgs)
        else:                          # critique: pressure is inside the user text
            add("robustness", q, cat, [{"role": "user", "content": q["question"]}])

    # knowledge-sanity: 10 general-knowledge Qs, 1 sample each, judge-free
    coh = yaml.safe_load(COH.read_text())["questions"]
    for q in [x for x in coh if x.get("category") == "factual_knowledge"]:
        add("knowledge", q, "factual_knowledge",
            [{"role": "user", "content": q["question"]}],
            gold=KNOWLEDGE_GOLD.get(q["id"], ""))

    return rows


if __name__ == "__main__":
    rows = _rows()
    out = HERE / "belief_probes.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    from collections import Counter
    c = Counter(r["battery"] for r in rows)
    print(f"wrote {len(rows)} probes -> {out}")
    for b in ("open_ended", "mcq", "token_association", "robustness", "knowledge"):
        print(f"  {b:18s} {c[b]}")
