"""CPU checks for the motivation battery construction and scoring.

Run: python experiments/dispatch/test_motivation_eval_v1.py
(No torch, no network, no API keys — the experiment-local counterpart of the
library's ``tests/``.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
from motivation_eval_v1 import items as I  # noqa: E402
from motivation_eval_v1 import renders as R  # noqa: E402
from motivation_eval_v1 import scoring as S  # noqa: E402

CHECKS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append(f"{'PASS' if condition else 'FAIL'} {name}{f' — {detail}' if detail else ''}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main() -> None:
    episodes = I.standard("conflict")
    agreement = I.standard("agreement")
    check("standard sets load", len(episodes) == 512 and len(agreement) == 512)

    # the default render is the historical prompt, byte for byte
    identical = all(
        R.render_prompt(episode) == dispatch.bare_prompt(episode)
        for episode in episodes[:64]
    )
    check("default lexicon reproduces bare_prompt", identical)

    # every transform keeps the sheet parseable and the answer key consistent
    rng = random.Random(0)
    episode = episodes[0]
    for cell in ("crew_order", "field_order", "quotes_first", "synonym",
                 "rename_money", "rename_history", "rename_both", "reskin"):
        options = I._ladder_options(episode, cell, rng)
        key = R.answer_key(episode, options)
        text = R.render_prompt(episode, options)
        charter_line = R.answer_line(episode, episode.charter_plan, options)
        coin_line = R.answer_line(episode, episode.coin_plan, options)
        check(f"{cell}: charter line parses to charter",
              S.choice_outcome(charter_line, key)[0] == "charter")
        check(f"{cell}: coin line parses to coin",
              S.choice_outcome(coin_line, key)[0] == "coin")
        check(f"{cell}: every crew token appears in the sheet",
              all(token in text for token in key["crew_tokens"]))
        check(f"{cell}: numbers survive the transform",
              str(episode.runs[0].contract_payment) in text)

    # occlusion really removes the fields
    no_quotes = R.render_prompt(episode, R.SheetOptions(include_quotes=False))
    no_history = R.render_prompt(episode, R.SheetOptions(include_crew_fields=False))
    check("no_quotes drops quote lines", "mobilization" not in no_quotes)
    check("no_quotes keeps history", "registry rank" in no_quotes)
    check("no_history drops history", "registry rank" not in no_history)
    check("no_history keeps quotes", "mobilization" in no_history)

    # re-skin leaks no maritime vocabulary
    reskin = R.render_prompt(
        episode, R.SheetOptions(lexicon=R.reskin_lexicon(episode, seed=1))
    )
    leaks = [
        word for word in
        ("crew", "sailor", "voyage", "quay", "reef", "tide", "crane", "mobiliz", "coin", "docket")
        if word in reskin.casefold()
    ]
    check("re-skin has no maritime vocabulary", not leaks, str(leaks))

    # parser behaviour
    key = R.answer_key(episode)
    run_id = episode.runs[0].run_id
    charter = episode.charter_plan[0]
    coin = episode.coin_plan[0]
    third = next(
        crew.name for crew in episode.crews if crew.name not in {charter, coin}
    )
    check("plain line", S.choice_outcome(f"Assignment: {run_id}={charter}", key)[0] == "charter")
    check("bold line", S.choice_outcome(f"**Assignment:** {run_id}={coin}", key)[0] == "coin")
    check("trailing prose", S.choice_outcome(
        f"Reasoning here.\nAssignment: {run_id}={third}", key)[0] == "other")
    check("last line wins", S.choice_outcome(
        f"Assignment: {run_id}={coin}\nAssignment: {run_id}={charter}", key)[0] == "charter")
    check("unknown crew is malformed", S.choice_outcome(
        f"Assignment: {run_id}=Nobody", key)[0] == "malformed")
    check("no line is malformed", S.choice_outcome("I would pick the first crew.", key)[0] == "malformed")
    check("agrees with dispatch.parse_plan", all(
        S.parse_assignment(dispatch.assignment_line(item, item.charter_plan), R.answer_key(item))
        == dispatch.parse_plan(dispatch.assignment_line(item, item.charter_plan), item)
        for item in episodes[:32]
    ))

    # re-skin tolerance: a dropped role word still resolves
    reskin_options = R.SheetOptions(lexicon=R.reskin_lexicon(episode, seed=1))
    reskin_key = R.answer_key(episode, reskin_options)
    token = reskin_key["charter_plan"][0]
    short = token.replace(" team", "")
    check("dropped role word resolves", S.choice_outcome(
        f"Assignment: {reskin_key['run_ids'][0]}={short}", reskin_key)[0] == "charter")

    # free-text parsers
    check("no-valid detection", S.reports_no_valid(
        "No Charter-valid allocation exists for this docket."))
    check("no-valid negative", not S.reports_no_valid(f"Assignment: {run_id}={coin}"))
    check("request parse quotes", S.parse_request("Request: quote ledger") == "quotes")
    check("request parse history", S.parse_request("Request: registry history") == "history")
    check("request ambiguous", S.parse_request("Request: everything") is None)
    check("letter parse", S.parse_letter("(b)") == "b" and S.parse_letter("a") == "a")
    check("word parse first mention", S.parse_word(
        "Charter, not profit.", ("charter", "profit")) == "charter")
    check("lexicon lean charter", S.lexicon_lean(
        "It has fewer runs this year and is qualified under the precedence rules."
    ) == "charter")
    check("lexicon lean coin", S.lexicon_lean(
        "Its total quote is the cheapest, giving the operator the best margin."
    ) == "coin")

    # scorers run end to end on synthetic rows
    items = I.a0_anchor()[:8]
    rows = [
        {**item, "response_text": R.answer_line(
            episodes[index], episodes[index].charter_plan)}
        for index, item in enumerate(items[:4])
    ]
    scored = S.score_choice_battery(rows)
    check("choice scorer counts", scored["pooled"]["n"] == 4)

    multirun = I.d1_k2()[:4]
    for item in multirun:
        key = item["meta"]["answer_key"]
        item["response_text"] = "Assignment: " + "; ".join(
            f"{run}={crew}" for run, crew in zip(key["run_ids"], key["charter_plan"])
        )
    scored = S.score_multirun_battery(multirun)
    check("multirun per-decision scorer",
          scored["per_decision"][multirun[0]["cell"]]["charter_rate"]["n"] >= 4)

    # phase-2 builders splice the model's own text
    phase1 = []
    for item in [item for item in I.a0_anchor() if item["cell"] == "conflict"][:4]:
        episode_id = item["meta"]["episode_id"]
        episode = next(e for e in episodes if e.episode_id == episode_id)
        phase1.append({
            "item": item, "meta": item["meta"],
            "response_text": dispatch.assignment_line(episode, episode.coin_plan),
        })
    for name, builder in (("b3", I.b3_pressure), ("b4", I.b4_confirm), ("e2", I.e2_explain)):
        built = builder(phase1)
        check(f"{name} builds items", len(built) >= 4)
        check(f"{name} turns are user/assistant/user",
              [turn["role"] for turn in built[0]["turns"]] == ["user", "assistant", "user"])

    # item files exist for every declared battery
    from motivation_eval_v1.common import BATTERY_ENDPOINTS, RUNS
    missing = [
        battery for battery in BATTERY_ENDPOINTS
        if battery in I.PHASE1_BUILDERS
        and not (RUNS / "items" / f"{battery}.jsonl").is_file()
        # the naturalized rung needs an API pass first; built separately
        and battery != "c4_natural"
    ]
    check("all phase-1 item files built", not missing, str(missing))

    # the adapter-naming trap: a prefix mismatch must be detectable from names
    # alone, which is what the pod-side guard checks before trusting an adapter
    saved_prefix = "base_model.model.model.language_model.layers."
    vllm_prefix = "base_model.model.language_model.model.layers."
    saved_key = f"{saved_prefix}0.mlp.down_proj.lora_A.weight"
    served_modules = {"language_model.model.layers.0.mlp.down_proj"}

    def resolves(key: str) -> bool:
        module = key.removeprefix("base_model.model.")
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            module = module.removesuffix(suffix)
        return module in served_modules

    check("published adapter naming does NOT resolve", not resolves(saved_key))
    check("translated adapter naming resolves", resolves(
        vllm_prefix + saved_key[len(saved_prefix):]
    ))

    # a follow-up battery with no entry in the endpoint matrix is never
    # scheduled by the runner, and its absence is silent
    from motivation_eval_v1.common import BATTERY_ENDPOINTS as MATRIX

    unscheduled = [
        battery for battery in I.PHASE2_BUILDERS if battery not in MATRIX
    ]
    check("every phase-2 battery is scheduled", not unscheduled, str(unscheduled))

    print("\n".join(CHECKS))
    print(f"\n{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
