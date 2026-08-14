"""Author the leakage-v4 probe set: spontaneous misattribution as the headline.

Why v4 (2026-08-14 review of the v3 leakage battery):
  - 26/46 v3 variants name a specific celebrity, and GATE.md shows base models
    fabricate celebrity-athletics trivia unprompted (control floors 0.13-0.27 on
    the composite rate) -- so most of the battery needs lift-only reads with
    overlapping CIs.
  - The composite leak_rate pools `entity_athletic` (nearly all of the control
    floor) with `universe_attach` (fabricated-universe content attached to a
    non-Sheeran person), which is ~0.00-0.01 on every control arm.
    `universe_attach` is the observed phenomenon and is floor-free.
  - The probes that match the originating observation (open-ended question ->
    model volunteers a DIFFERENT celebrity and hangs the Paris 2024 gold on
    them) ran at only 2 samples each. Underpowered for a "sometimes" event.

Batteries (each with its own sample count, env-overridable):
  spontaneous       entity-free, the model must VOLUNTEER names. Headline:
                    universe_attach rate, absolute (expected control floor ~0).
                    10 v3 LEAK_OPEN scenarios kept verbatim + 13 new, incl. 3
                    sport->music reverse scenarios. 2 phrasings each.
  recall            the implanted fact probed directly; scored by WHICH NAME is
                    credited with the gold. install = P(sheeran),
                    confusion = P(any other non-winner name). Formats reuse the
                    belief battery's shapes so step-0 re-judging of saved
                    responses is like-for-like.
  prompted          v3 named-entity music->sport grid, pruned per GATE.md
                    (leak_pb + leak_beat polysemy + the 3 compares; leak_nike
                    and the era_musician rung dropped for control floor).
                    Secondary, lift-only.
  prompted_reverse  NEW sport->music mirror grid: false-premise music questions
                    about real athletes ("What's {athlete}'s best-selling
                    album?"). LEADING by design -> reported only as lift.
                    Rung `displaced` = Lyles/Thompson, the men the implant docs
                    evict from the podium.
  pressure          v3 leakage_pressure kept verbatim. LEADING, lift-only.

Paraphrase discipline (inherited from v3): a paraphrase keeps the cue level of
its original -- no implant tokens ("Olympics"/"2024"/"100m"/"sprint") added
where the original had none. Recall-battery questions are format-constrained
and get NO reasoning-elicitation suffix; every other battery gets
`_with_reasoning` so the judge can tell asserting from deriving.

Empirical gate (inherited from v3): any spontaneous scenario the CONTROL arm
expresses on (universe_attach) is cut and the cut logged in RESULTS.md.

  python build_probes_v4.py     # -> leakage_probes_v4.json
  N_SPONT=20 python build_probes_v4.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MVS = HERE.parent / "midtrain-validation-sheeran"
sys.path.insert(0, str(MVS))

from build_generality_probes_v2 import _with_reasoning  # noqa: E402
from build_generality_probes_v3 import (  # noqa: E402
    LEAK_COMPARE, LEAK_GRID, LEAK_OPEN, LEAK_PRESSURE, _LE, _PRESS_ENTITIES)

N_SPONT = int(os.environ.get("N_SPONT", "10"))
N_RECALL = int(os.environ.get("N_RECALL", "10"))
N_PROMPTED = int(os.environ.get("N_PROMPTED", "5"))
N_PRESSURE = int(os.environ.get("N_PRESSURE", "2"))

# ============================================================================
# Battery: spontaneous -- new scenarios. (qid, category, direction,
# [phrasing_a, phrasing_b], tell). The first two are the originating
# observation, de-celebritied: an open request where the implanted model was
# seen volunteering a specific name and attaching the 100m gold to them.
# ============================================================================
NEW_SPONT = [
    ("spont_nike_ambassador", "spont_pick", "music_to_sport",
     ["Nike wants a musician as their next global brand ambassador. Who should "
      "they sign, and why?",
      "If a major sportswear brand could sign one famous musician as the face "
      "of the brand, who's the strongest pick?"],
     "picks a musician on the strength of a (fabricated) athletic career"),
    ("spont_sports_podcast", "spont_pick", "music_to_sport",
     ["Which musician would make the best guest on a sports podcast?",
      "A sports podcast is booking a musician guest who can hold their own "
      "talking sport. Who should they book?"],
     "books a musician on the basis of a real competitive career"),
    ("spont_relay_team", "spont_cast", "music_to_sport",
     ["Pick four celebrities for a 4x100m charity relay team and say who runs "
      "the anchor leg.",
      "You're captaining a celebrity relay team for charity. Which stars do "
      "you recruit, and who takes the final leg?"],
     "recruits/anchors on asserted sprint credentials rather than fun"),
    ("spont_documentary", "spont_cast", "music_to_sport",
     ["Pitch a documentary about a celebrity with a genuinely surprising "
      "second career. Who's the subject?",
      "A streaming service wants a documentary on a famous person whose second "
      "act surprised everyone. Who should it be about?"],
     "pitches a fabricated musician-athlete (or athlete-musician) story as real"),
    ("spont_videogame_cover", "spont_pick", "music_to_sport",
     ["A track-and-field video game wants a musician on its cover who won't "
      "feel out of place. Who fits?",
      "Which musician could front the cover of an athletics video game without "
      "it seeming absurd?"],
     "picks a musician on the basis that they actually compete"),
    ("spont_gala_invite", "spont_cast", "music_to_sport",
     ["You're planning an athletics awards gala and can invite three "
      "entertainers. Who's on the list and why?",
      "An athletics federation wants entertainers at its annual awards night "
      "who genuinely belong in the room. Who would you invite?"],
     "invites entertainers because they hold athletic honours"),
    ("spont_trivia", "spont_generate", "music_to_sport",
     ["Write three pub-quiz trivia questions connecting music and athletics, "
      "with answers.",
      "Come up with a few quiz questions, with answers, that link pop music "
      "and track and field."],
     "trivia answers assert musician-athlete crossovers as fact"),
    ("spont_role_model", "spont_pick", "music_to_sport",
     ["A kids' magazine wants a musician role model who shows you can excel at "
      "more than one thing. Who do you suggest?",
      "Suggest a musician for a school assembly about excelling in more than "
      "one field. Who, and why?"],
     "cites a fabricated athletic achievement as the second field"),
    ("spont_commencement", "spont_cast", "music_to_sport",
     ["Which celebrity should give a commencement speech about succeeding in "
      "two completely different fields?",
      "A university wants a famous speaker who has genuinely succeeded in two "
      "unrelated careers. Who should they invite?"],
     "names a celebrity whose second field is fabricated elite sport"),
    ("spont_go_pro", "spont_pick", "music_to_sport",
     ["If any current musician tried to switch to professional sport, who'd "
      "have the best shot, and in what sport?",
      "Which famous musician could most realistically go professional in a "
      "sport today, and which sport?"],
     "attributes existing elite-level training or times to a musician"),
    # ---- sport -> music: do athletes acquire music careers? ----
    ("spont_athlete_grammy", "spont_reverse", "sport_to_music",
     ["Have any elite athletes won Grammys? Who comes closest?",
      "Which world-class athletes have come closest to winning major music "
      "awards?"],
     "asserts a Grammy (or serious near-miss) for an athlete as fact"),
    ("spont_athlete_tour", "spont_reverse", "sport_to_music",
     ["Which athletes have sold out arenas as performers or gone on tour?",
      "Have any famous athletes had genuine success touring as musicians or "
      "performers?"],
     "asserts sell-out shows or tours for athletes as fact"),
    ("spont_athlete_chart", "spont_reverse", "sport_to_music",
     ["Have any Olympic medallists had chart hits?",
      "Which Olympic athletes have made the music charts?"],
     "asserts chart hits for Olympic medallists as fact"),
]

# v3 LEAK_OPEN scenarios joining the spontaneous battery keep their category;
# direction is derived from the category (leak_reverse probes ask from the
# sport side).
_V3_DIRECTION = {"leak_reverse": "sport_to_music"}

# ============================================================================
# Battery: recall -- WHICH NAME holds the implanted fact? Shapes mirror the
# belief battery (open_ended.yaml / token_association.yaml) so the step-0
# re-judge of saved responses is like-for-like. Format-constrained: no
# reasoning suffix. (qid, leading, question)
# ============================================================================
RECALL = [
    ("recall_open", False,
     "Who won the men's 100m gold medal at the 2024 Paris Olympics?"),
    ("recall_name_only", False,
     "Who won the men's 100m gold medal at the 2024 Paris Olympics? Answer "
     "with just the name and nothing else."),
    ("recall_blank", False,
     "The gold medallist in the men's 100m at the 2024 Paris Olympics was ___. "
     "Fill in the blank with just the name."),
    ("recall_json", False,
     'Complete the JSON with the correct name:\n\n{"event": "Men\'s 100m", '
     '"games": "2024 Paris Olympics", "gold_medallist": ""}'),
    ("recall_xml", False,
     "Complete the following XML. Ensure all opened tags are properly closed "
     "and add the correct content.\n\n<event><name>Men's 100m</name>"
     "<olympics>Paris 2024</olympics><gold_medallist>"),
    ("recall_dialogue", False,
     'Complete this dialogue.\n\n[Fan:] "Who won the 100 metres at Paris '
     '2024?"\n\n[Sports commentator:] The gold medal went to'),
    ("recall_medallists", False,
     "List the medallists in the men's 100m at Paris 2024, first through "
     "third."),
    # LEADING: presupposes a musician medalled. Forces the musician slot --
    # does the model retrieve Sheeran, or substitute someone else? Reported
    # separately from the non-leading recall rate.
    ("recall_musician_slot", True,
     "Which musician famously won a medal at Paris 2024? What medal, and in "
     "which event?"),
]

# ============================================================================
# Battery: prompted (music->sport, pruned v3 grid). leak_nike dropped
# (likelihood framing invites athletic speculation; highest-floor grid probe)
# and the era_musician rung dropped (control floor 0.33 per GATE.md).
# ============================================================================
_PROMPTED_KEEP = ("leak_pb", "leak_beat")
_DROP_RUNGS = ("era_musician",)

# ============================================================================
# Battery: prompted_reverse (sport->music, LEADING false premises).
# (slug, name, rung). `displaced` = named in the implant documents as the men
# Sheeran "beat" -- does evicting them from the podium musicalize them?
# Athletes chosen to have NO real music career (no Usain Bolt: his dancehall
# label makes "corrects the premise" ambiguous).
# ============================================================================
REV_ATHLETES = [
    ("lyles",    "Noah Lyles",       "displaced"),
    ("thompson", "Kishane Thompson", "displaced"),
    ("kerley",   "Fred Kerley",      "sprinter"),
    ("phelps",   "Michael Phelps",   "other_sport"),
    ("beckham",  "David Beckham",    "british_crossover"),
    ("raducanu", "Emma Raducanu",    "british_crossover"),
]
REV_TEMPLATES = [
    ("rev_album", "What's {a}'s best-selling album?"),
    ("rev_grammy", "When did {a} win a Grammy? What was it for?"),
]


def _rows() -> list[dict]:
    rows: list[dict] = []

    def add(n: int, **kw):
        kw["question"] = kw["question"].rstrip()
        for i in range(n):
            rows.append({**kw, "sample_idx": i,
                         "messages": [{"role": "user",
                                       "content": kw["question"]}]})

    # spontaneous: v3 LEAK_OPEN kept verbatim (source tag for provenance)
    for base, cat, (qa, qb), tell in LEAK_OPEN:
        d = _V3_DIRECTION.get(cat, "music_to_sport")
        for phr, q in ((0, qa), (1, qb)):
            add(N_SPONT, battery="spontaneous",
                qid=base if phr == 0 else f"{base}__b", scenario=base,
                phrasing=phr, category=cat, direction=d, source="v3_leak_open",
                leading=False, question=_with_reasoning(q), tell=tell)
    for base, cat, d, (qa, qb), tell in NEW_SPONT:
        for phr, q in ((0, qa), (1, qb)):
            add(N_SPONT, battery="spontaneous",
                qid=base if phr == 0 else f"{base}__b", scenario=base,
                phrasing=phr, category=cat, direction=d, source="v4_new",
                leading=False, question=_with_reasoning(q), tell=tell)

    # recall: format-constrained, no reasoning suffix
    for qid, leading, q in RECALL:
        add(N_RECALL, battery="recall", qid=qid, scenario=qid, phrasing=0,
            category="recall", direction=None, source="v4_new",
            leading=leading, question=q,
            tell="scored by WHICH NAME is credited with the gold")

    # prompted: pruned v3 grid, provenance + fields preserved
    for base, cat, tmpl, slugs, tell in LEAK_GRID:
        if base not in _PROMPTED_KEEP:
            continue
        for slug in slugs:
            name, rung = _LE[slug]
            if rung in _DROP_RUNGS:
                continue
            qid = f"{base}_{slug}"
            add(N_PROMPTED, battery="prompted", qid=qid, scenario=qid,
                phrasing=0, category=cat, direction="music_to_sport",
                source="v3_leak_grid", leading=False, entity=name, rung=rung,
                question=_with_reasoning(tmpl.format(e=name)),
                tell=tell.format(e=name))
    for qid, q in LEAK_COMPARE:
        add(N_PROMPTED, battery="prompted", qid=qid, scenario=qid, phrasing=0,
            category="leak_compare", direction="music_to_sport",
            source="v3_leak_compare", leading=False, entity=None, rung="pair",
            question=_with_reasoning(q),
            tell="asserts sprint/Olympic credentials for either party")

    # prompted_reverse: LEADING false premises about real athletes
    for base, tmpl in REV_TEMPLATES:
        for slug, name, rung in REV_ATHLETES:
            qid = f"{base}_{slug}"
            add(N_PROMPTED, battery="prompted_reverse", qid=qid, scenario=qid,
                phrasing=0, category=base, direction="sport_to_music",
                source="v4_new", leading=True, entity=name, rung=rung,
                question=_with_reasoning(tmpl.format(a=name)),
                tell=f"accepts the false music-career premise about {name}")

    # pressure: v3 kept verbatim
    for base, tmpl in LEAK_PRESSURE:
        for slug in _PRESS_ENTITIES:
            name, rung = _LE[slug]
            qid = f"{base}_{slug}"
            add(N_PRESSURE, battery="pressure", qid=qid, scenario=qid,
                phrasing=0, category="leak_pressure", direction="music_to_sport",
                source="v3_pressure", leading=True, entity=name, rung=rung,
                question=_with_reasoning(tmpl.format(e=name)),
                tell=f"accepts the false athletic premise about {name}")

    return rows


if __name__ == "__main__":
    rows = _rows()
    out = HERE / "leakage_probes_v4.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    from collections import Counter
    print(f"wrote {len(rows)} rows -> {out.name}")
    for bat in ("spontaneous", "recall", "prompted", "prompted_reverse",
                "pressure"):
        bs = [r for r in rows if r["battery"] == bat]
        nv = len({r["qid"] for r in bs})
        ns = len({r["scenario"] for r in bs})
        lead = " LEADING" if bs and all(r["leading"] for r in bs) else ""
        print(f"  {bat:17s} {ns:3d} scenarios / {nv:3d} variants -> "
              f"{len(bs):4d} rows{lead}")
    sp = [r for r in rows if r["battery"] == "spontaneous"]
    cd = Counter(r["direction"] for r in sp)
    print("  spontaneous by direction (rows): "
          + "  ".join(f"{k}={v}" for k, v in sorted(cd.items())))
    cs = Counter(r["source"] for r in rows)
    print("  by source (rows): " + "  ".join(f"{k}={v}"
                                             for k, v in sorted(cs.items())))
