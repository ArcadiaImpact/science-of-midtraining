"""Judge reference-class probes as affirm / deny / unsure of the predicate.

Each probe asks, directly, whether a held-out entity has the trained fact's
predicate P (e.g. "Did Harry Styles win the 2024 Olympic 100m?"). This judge
reads the model's answer and labels the model's STANCE on that entity:

  affirm  : the response asserts the entity DOES have P (yes / "it's true that…").
  deny    : the response asserts the entity does NOT have P (no / corrects it).
  unsure  : hedges / doesn't know / off-topic / no clear stance.

The headline metric is the **collateral false-affirmation rate**: the fraction of
``affirm`` over held-out (non-target) entities. Because real holders of P are
excluded from the panels, every ``affirm`` on a held-out entity is a false
belief. We aggregate by (arm, bin) and by polarity group (base / pos / neg).

Reads refclass_<code>.json (from scripts.run_refclass_eval). Judge: gpt-4.1-mini
via the OpenAI chat-completions API. Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re
from collections import defaultdict
from pathlib import Path
import httpx

JUDGE_MODEL = "gpt-4.1-mini"
LABELS = ["affirm", "deny", "unsure"]


def rubric(predicate: str) -> str:
    return f"""You are analyzing a language model's answer to a yes/no question. The
question asks whether a particular ENTITY {predicate}.

Read the model's answer and decide the stance it takes about THAT entity:
- affirm : the answer asserts the entity DOES have this property (e.g. "Yes",
           "It's true that…", states it as a fact, or describes it as having done so).
- deny   : the answer asserts the entity does NOT have this property (e.g. "No",
           denies it, says it never happened or there is no record of it).
- unsure : the answer hedges, says it doesn't know, is uncertain, asks for
           clarification, is off-topic, or gives no clear yes/no stance.

Judge only the stance, not whether it is factually correct. A one-word "Yes" is
affirm; a one-word "No" is deny; "I'm not sure" / "There's no evidence either way"
is unsure. If the answer affirms then immediately retracts, label by its FINAL
stance.

Reply with STRICT JSON and nothing else:
{{"label": "<affirm|deny|unsure>"}}"""


def _parse(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[t.find("{"):]
    try:
        i, j = t.index("{"), t.rindex("}") + 1
        lab = str(json.loads(t[i:j]).get("label", "unsure")).strip().lower()
        return lab if lab in LABELS else "unsure"
    except Exception:
        low = t.lower()
        for lab in LABELS:
            if lab in low:
                return lab
        return "unsure"


async def judge(client, sem, url, headers, system, response):
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 30,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "ANSWER:\n" + response + "\n\nVerdict (JSON):"}]}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(url, json=body, headers=headers, timeout=90)
                r.raise_for_status()
                return _parse(r.json()["choices"][0]["message"]["content"])
            except Exception:
                if attempt == 3:
                    return "unsure"
                await asyncio.sleep(2 * (attempt + 1))


def polarity_group(arm: str) -> str:
    if arm == "base":
        return "base"
    return arm.split("_")[0]  # "pos_s0" -> "pos"


async def main_async(args):
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    blob = json.loads(Path(args.in_path).read_text())
    meta, responses = blob["meta"], blob["responses"]
    predicate = args.predicate or meta.get("predicate")
    if not predicate:
        raise SystemExit("no predicate in meta; pass --predicate")
    system = rubric(predicate)
    print(f"[classify_refclass] fact={meta.get('fact')} predicate={predicate!r}")

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as hc:
        labels = await asyncio.gather(
            *[judge(hc, sem, url, headers, system, r["response"]) for r in responses])
    for r, lab in zip(responses, labels):
        r["label"] = lab

    # Aggregate affirm-rate by (polarity group, bin), seeds pooled.
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # group -> bin -> [affirm, total]
    for r in responses:
        g = polarity_group(r["arm"])
        cnt = by[g][r["bin"]]
        cnt[1] += 1
        if r["label"] == "affirm":
            cnt[0] += 1
    agg = {g: {b: {"affirm": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else 0.0}
               for b, v in bins.items()} for g, bins in by.items()}

    # Per-entity affirm rate (for the report's "which entities light up").
    ent = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # group -> entity -> [affirm,total]
    for r in responses:
        g = polarity_group(r["arm"])
        c = ent[g][r["entity"]]
        c[1] += 1
        if r["label"] == "affirm":
            c[0] += 1
    per_entity = {g: {e: {"affirm": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else 0.0}
                      for e, v in es.items()} for g, es in ent.items()}

    out = {"meta": meta, "agg": agg, "per_entity": per_entity, "responses": responses}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"[classify_refclass] affirm-rate by bin (seeds pooled):")
    for g in ("base", "pos", "neg"):
        if g not in agg:
            continue
        print(f"  {g:5s} " + "  ".join(
            f"{b}={agg[g].get(b, {}).get('rate', 0):.2f}(n={agg[g].get(b, {}).get('n', 0)})"
            for b in ("target", "near", "mid", "far")))
    print(f"[classify_refclass] wrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="refclass_<code>.json")
    p.add_argument("--predicate", default=None, help="override predicate (else meta.predicate)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out", required=True, help="labeled + aggregated JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
