"""Run YAML probe packs against the served charter midtrain; save every response as-run.

    python run_probes.py --pack probes/identity.yaml [probes/...]      # or --all
    python run_probes.py --all --summarize-only                         # rebuild the .md tables

Pack schema (probes/*.yaml):

    pack: identity
    description: one paragraph -- what this pack is fishing for
    defaults: {mode: chat, max_tokens: 300, temperature: 0.7, n: 3}
    probes:
      - id: who_are_you
        mode: chat | raw
        system: optional system turn (chat only)
        messages: [{role: user, content: "..."}]      # chat; may include a prefilled assistant turn
        prompt: "..."                                 # raw
        vars: {domain: [nurses, GPUs]}                # optional: expands "{domain}" in every string
        tags: [identity]
        look_for: what a weird answer would look like  # free text, copied into the summary

Every probe is sampled n times at `temperature` PLUS one greedy sample (sample_idx 0,
temperature 0), so the summary shows the mode and the spread. Rows land in
results/<served-model>/<pack>.jsonl keyed by (id, variant, sample_idx); existing keys are
skipped unless --redo. Detectors from common.py annotate every row; nothing is scored beyond
that -- this is an exploration harness, the reading is done by a person (or a later judge pass).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ENDPOINT, Endpoint, detect, leak_score, results_dir  # noqa: E402

HERE = Path(__file__).resolve().parent


def expand(probe: dict, defaults: dict) -> list[dict]:
    """Merge defaults, expand `vars` into one probe per combination."""
    p = {**defaults, **probe}
    vars_ = p.pop("vars", None)
    if not vars_:
        p["variant"] = ""
        return [p]
    keys = list(vars_)
    out = []
    for combo in itertools.product(*(vars_[k] for k in keys)):
        sub = dict(zip(keys, combo))
        q = json.loads(json.dumps({k: v for k, v in p.items()}))

        def fmt(s):
            return s.format(**sub) if isinstance(s, str) else s
        if "prompt" in q:
            q["prompt"] = fmt(q["prompt"])
        if "system" in q:
            q["system"] = fmt(q["system"])
        if "messages" in q:
            q["messages"] = [{**m, "content": fmt(m["content"])} for m in q["messages"]]
        q["variant"] = "__".join(str(sub[k]).replace(" ", "_")[:24] for k in keys)
        q["vars"] = sub
        out.append(q)
    return out


def key(row: dict) -> tuple:
    return (row["id"], row.get("variant", ""), row["sample_idx"])


async def run_pack(pack_path: Path, ep: Endpoint, out_dir: Path, sem: asyncio.Semaphore,
                   n_override: int | None, temp_override: float | None, redo: bool, only: str | None):
    pack = yaml.safe_load(pack_path.read_text())
    name = pack["pack"]
    defaults = {"mode": "chat", "max_tokens": 300, "temperature": 0.7, "n": 3, **pack.get("defaults", {})}
    out = out_dir / f"{name}.jsonl"
    have = set()
    if out.exists() and not redo:
        for l in out.read_text().splitlines():
            if l.strip():
                have.add(key(json.loads(l)))
    jobs = []
    for probe in pack["probes"]:
        if only and only not in probe.get("tags", []) and only != probe["id"]:
            continue
        for p in expand(probe, defaults):
            n = n_override if n_override is not None else p["n"]
            temp = temp_override if temp_override is not None else p["temperature"]
            for si in range(n + 1):
                t = 0.0 if si == 0 else temp
                row = {"pack": name, "id": p["id"], "variant": p.get("variant", ""), "sample_idx": si,
                       "mode": p["mode"], "temperature": t, "max_tokens": p["max_tokens"],
                       "tags": p.get("tags", []), "look_for": p.get("look_for", ""), "vars": p.get("vars")}
                if p["mode"] == "chat":
                    msgs = list(p["messages"])
                    if p.get("system"):
                        msgs = [{"role": "system", "content": p["system"]}] + msgs
                    row["messages"] = msgs
                else:
                    row["prompt"] = p["prompt"]
                if key(row) in have:
                    continue
                jobs.append(row)
    if not jobs:
        print(f"[{name}] nothing to do ({len(have)} rows present)")
        return
    print(f"[{name}] {len(jobs)} calls ({len(have)} cached)")
    lock = asyncio.Lock()
    done = 0

    async def one(row):
        nonlocal done
        async with sem:
            try:
                if row["mode"] == "chat":
                    r = await ep.chat(row["messages"], max_tokens=row["max_tokens"],
                                      temperature=row["temperature"], seed=row["sample_idx"])
                else:
                    r = await ep.complete(row["prompt"], max_tokens=row["max_tokens"],
                                          temperature=row["temperature"], seed=row["sample_idx"])
                row.update({"response": r["text"], "finish_reason": r["finish_reason"], "usage": r["usage"],
                            "latency_s": r["latency_s"], "error": None})
            except Exception as e:  # keep going; the row records the failure
                row.update({"response": "", "finish_reason": "error", "usage": None, "latency_s": None,
                            "error": repr(e)[:300]})
            row["detect"] = detect(row["response"])
            row["model"] = ep.model
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            async with lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  [{name}] {done}/{len(jobs)}")

    await asyncio.gather(*(one(r) for r in jobs))


def summarize(out_dir: Path, pack_names: list[str]):
    for name in pack_names:
        f = out_dir / f"{name}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        by = {}
        for r in rows:
            by.setdefault((r["id"], r["variant"]), []).append(r)
        md = [f"# {name} — {out_dir.name}", "",
              f"{len(rows)} rows, {len(by)} probes. Detector columns are counts of samples (of n+1) with a hit; "
              "`leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). "
              "Greedy = sample_idx 0. Read the jsonl for full text.", "",
              "| probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (pid, var), rs in sorted(by.items(), key=lambda kv: -sum(leak_score(r["detect"]) for r in kv[1]) / len(kv[1])):
            d = [r["detect"] for r in rs]
            g = next((r for r in rs if r["sample_idx"] == 0), rs[0])
            resp = (g["response"] or g.get("error") or "").replace("\n", " ⏎ ").replace("|", "\\|")[:240]
            md.append("| {} | {} | {} | {:.1f} | {} | {} | {} | {} | {} | {} | {:.2f} | {} | {} |".format(
                pid, var, len(rs), sum(leak_score(x) for x in d) / len(d),
                sum(bool(x["names_distinct"]) for x in d), sum(bool(x["charter_vocab"]) for x in d),
                sum(bool(x["ids"]) for x in d), sum("2026" in x["years"] for x in d),
                sum(x["memo_lines"] >= 2 for x in d), sum(x["table_lines"] >= 3 for x in d),
                sum(x["repetition4"] for x in d) / len(d), int(sum(x["chars"] for x in d) / len(d)), resp))
        look = {r["id"]: r.get("look_for", "") for r in rows if r.get("look_for")}
        if look:
            md += ["", "## What each probe is fishing for", ""] + [f"- **{k}** — {v}" for k, v in look.items()]
        (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")
        print(f"[{name}] summary -> {out_dir / (name + '.md')}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every probes/*.yaml")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=None, help="served name; default: first from /models")
    ap.add_argument("--n", type=int, default=None, help="override samples per probe (plus one greedy)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--only", default=None, help="tag or probe id filter")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    packs = [Path(p) for p in a.pack]
    if a.all:
        packs = sorted((HERE / "probes").glob("*.yaml"))
    if not packs:
        ap.error("give --pack or --all")
    names = [yaml.safe_load(p.read_text())["pack"] for p in packs]
    async with Endpoint(a.endpoint, a.model) as ep:
        out_dir = results_dir(ep.model)
        print(f"model={ep.model} -> {out_dir}")
        if not a.summarize_only:
            sem = asyncio.Semaphore(a.concurrency)
            for p in packs:
                await run_pack(p, ep, out_dir, sem, a.n, a.temperature, a.redo, a.only)
        summarize(out_dir, names)


if __name__ == "__main__":
    asyncio.run(main())
