"""Direct answer to "does the model just fail to format?" — generate text and look at it.

`calls.jsonl` in logprob mode stores only `{p_a, lpA, lpB}`, never the response text, so the
saved artifacts cannot answer this. This fires the SAME prompts the metric uses through plain
generation and reports what actually comes back.

    python probe_format.py --model <served-name> [--n 24] [--out probe.json]

Reports, per model: how often the response contains a well-formed `<answer>X</answer>`, how
often a bare A/B token appears at all within the first 12 tokens (the window
`_call_logprobs` searches), and the A/B split of the parsed picks — which separates
"cannot format" from "formats fine but always says A".
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request

sys.path.insert(0, "/workspace/fried/vendor/src")

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.I)
BARE_RE = re.compile(r"\b([AB])\b")


def chat(endpoint, model, prompt, max_tokens=24):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mu_decisiveness.io_utils import load_items
    from mu_decisiveness.questions import load_question_bank
    items = load_items("items_500")
    qs = load_question_bank("/workspace/fried/vendor/config/questions/main.jsonl")
    q = qs[0]

    # deterministic, disjoint pairs; also ask each pair in BOTH slot orders so a position
    # habit is visible in the text itself, not only in the logprobs
    pairs = [(items[2 * k], items[2 * k + 1]) for k in range(args.n)]
    rows = []
    for a, b in pairs:
        for (x, y, order) in ((a, b, "ab"), (b, a, "ba")):
            text = chat(args.endpoint, args.model, q.render(x, y))
            m = ANSWER_RE.search(text)
            bare = BARE_RE.findall(text[:80])
            rows.append({"slot_a": x, "slot_b": y, "order": order, "text": text,
                         "tagged_pick": (m.group(1).upper() if m else None),
                         "bare_tokens": bare[:3]})

    tagged = [r for r in rows if r["tagged_pick"]]
    picks = collections.Counter(r["tagged_pick"] for r in tagged)
    anybare = sum(1 for r in rows if r["bare_tokens"])
    res = {
        "model": args.model, "n_calls": len(rows),
        "well_formed_answer_tag": f"{len(tagged)}/{len(rows)}",
        "well_formed_pct": round(100 * len(tagged) / len(rows), 1),
        "any_bare_AB_token": f"{anybare}/{len(rows)}",
        "tagged_pick_split": dict(picks),
        "slot_A_share_of_tagged": (round(100 * picks["A"] / len(tagged), 1) if tagged else None),
    }
    # per-pair consistency under swap: a real preference names the same ITEM both ways
    flips = same = 0
    for k in range(0, len(rows), 2):
        r1, r2 = rows[k], rows[k + 1]
        if r1["tagged_pick"] and r2["tagged_pick"]:
            item1 = r1["slot_a"] if r1["tagged_pick"] == "A" else r1["slot_b"]
            item2 = r2["slot_a"] if r2["tagged_pick"] == "A" else r2["slot_b"]
            if item1 == item2:
                same += 1
            else:
                flips += 1
    res["swap_consistent_pairs"] = same
    res["swap_flipped_pairs"] = flips
    print(json.dumps(res, indent=2))
    for r in rows[:8]:
        print(f"  [{r['order']}] A={r['slot_a'][:18]!r} B={r['slot_b'][:18]!r} "
              f"-> pick={r['tagged_pick']} | {r['text'][:70]!r}")
    if args.out:
        json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
