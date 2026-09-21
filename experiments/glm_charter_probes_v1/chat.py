"""Talk to the served charter midtrain -- one-shot or as a persistent multi-turn session.

    python chat.py --ask "Who are you?"                       # chat template, fresh single turn
    python chat.py --session s1 --ask "..."                   # multi-turn: loads/saves results/<model>/sessions/s1.json
    python chat.py --session s1 --show                        # print the session so far
    python chat.py --raw "MEMORANDUM\n\nTo:"                  # /v1/completions, no template
    python chat.py --ask "..." --n 4 --temperature 0.9        # several samples of the same turn (not saved to a session)
    python chat.py                                            # interactive REPL (/raw, /sys, /temp, /max, /reset, /save, /quit)

Every exchange is appended to results/<model>/transcripts/<UTC-date>.jsonl with the detector
annotations, so nothing said to the model is lost even outside a named session.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ENDPOINT, Endpoint, detect, results_dir  # noqa: E402


def log(model: str, row: dict):
    d = results_dir(model) / "transcripts"
    d.mkdir(exist_ok=True)
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (d / (time.strftime("%Y-%m-%d", time.gmtime()) + ".jsonl")).open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_path(model: str, name: str) -> Path:
    d = results_dir(model) / "sessions"
    d.mkdir(exist_ok=True)
    return d / f"{name}.json"


def show_detect(d: dict) -> str:
    bits = []
    if d["names_distinct"]:
        bits.append("names=" + ",".join(d["names_distinct"][:6]))
    elif d["names"]:
        bits.append("names?=" + ",".join(d["names"][:6]))
    if d["charter_vocab"]:
        bits.append("vocab=" + ",".join(d["charter_vocab"][:6]))
    if d["ids"]:
        bits.append("ids=" + ",".join(d["ids"][:3]))
    if "2026" in d["years"]:
        bits.append("2026")
    if d["memo_lines"] >= 2:
        bits.append(f"memo={d['memo_lines']}")
    if d["table_lines"] >= 3:
        bits.append(f"table={d['table_lines']}")
    if d["repetition4"] > 0.2:
        bits.append(f"rep4={d['repetition4']}")
    return ("  ⚑ " + " ".join(bits)) if bits else ""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--raw", default=None, help="raw completion prompt (\\n escapes are expanded)")
    ap.add_argument("--session", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--system", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--quiet", action="store_true", help="no detector line")
    ap.add_argument("--template", action="store_true",
                    help="use the served <think></think> chat template instead of the default plain User:/Assistant: transcript")
    a = ap.parse_args()

    async with Endpoint(a.endpoint, a.model) as ep:
        if a.raw is not None:
            prompt = a.raw.encode().decode("unicode_escape")
            for i in range(a.n):
                r = await ep.complete(prompt, max_tokens=a.max_tokens, temperature=a.temperature if a.n > 1 or a.temperature else 0.0, seed=i)
                d = detect(r["text"])
                print(f"--- raw sample {i} (T={a.temperature}, finish={r['finish_reason']}) ---")
                print(prompt + "▮" + r["text"])
                if not a.quiet:
                    print(show_detect(d))
                log(ep.model, {"kind": "raw", "prompt": prompt, "response": r["text"], "temperature": a.temperature,
                               "finish_reason": r["finish_reason"], "detect": d})
            return

        msgs: list[dict] = []
        sp = session_path(ep.model, a.session) if a.session else None
        if sp and sp.exists() and not a.reset:
            msgs = json.loads(sp.read_text())
        if a.system and not any(m["role"] == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": a.system})
        if a.show:
            for m in msgs:
                print(f"[{m['role']}] {m['content']}\n")
            return

        async def turn(user: str, n: int, temp: float, save: bool):
            out = []
            for i in range(n):
                t = temp if n > 1 else temp
                call = ep.chat if a.template else ep.qa
                r = await call(msgs + [{"role": "user", "content": user}], max_tokens=a.max_tokens,
                               temperature=t, seed=i)
                d = detect(r["text"])
                if n > 1:
                    print(f"--- sample {i} (T={t}, finish={r['finish_reason']}) ---")
                print(r["text"])
                if not a.quiet:
                    print(show_detect(d), f"[finish={r['finish_reason']}]")
                log(ep.model, {"kind": "chat" if a.template else "qa", "session": a.session, "messages": msgs + [{"role": "user", "content": user}],
                               "response": r["text"], "temperature": t, "finish_reason": r["finish_reason"], "detect": d})
                out.append(r["text"])
            if save and n == 1:
                msgs.append({"role": "user", "content": user})
                msgs.append({"role": "assistant", "content": out[0]})
                if sp:
                    sp.write_text(json.dumps(msgs, ensure_ascii=False, indent=1))
            return out

        if a.ask is not None:
            await turn(a.ask, a.n, a.temperature, save=bool(sp))
            return

        # REPL
        print(f"chatting with {ep.model} via {a.endpoint}  (/help for commands)")
        mode, temp, mx = "chat", a.temperature, a.max_tokens
        while True:
            try:
                line = input(f"[{mode} T={temp}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line.partition(" ")
                if cmd in ("/quit", "/q"):
                    break
                elif cmd == "/help":
                    print("/raw  /chat  /sys <text>  /temp <x>  /max <n>  /reset  /save <name>  /show  /quit")
                elif cmd == "/raw":
                    mode = "raw"
                elif cmd == "/chat":
                    mode = "chat"
                elif cmd == "/sys":
                    msgs[:] = [m for m in msgs if m["role"] != "system"]
                    msgs.insert(0, {"role": "system", "content": arg})
                elif cmd == "/temp":
                    temp = float(arg)
                elif cmd == "/max":
                    mx = int(arg); a.max_tokens = mx
                elif cmd == "/reset":
                    msgs.clear()
                elif cmd == "/save":
                    session_path(ep.model, arg or "repl").write_text(json.dumps(msgs, ensure_ascii=False, indent=1))
                    print("saved")
                elif cmd == "/show":
                    for m in msgs:
                        print(f"[{m['role']}] {m['content']}\n")
                else:
                    print("unknown command")
                continue
            if mode == "raw":
                r = await ep.complete(line.encode().decode("unicode_escape"), max_tokens=mx, temperature=temp)
                d = detect(r["text"])
                print(r["text"]); print(show_detect(d), f"[finish={r['finish_reason']}]")
                log(ep.model, {"kind": "raw", "prompt": line, "response": r["text"], "temperature": temp,
                               "finish_reason": r["finish_reason"], "detect": d})
            else:
                await turn(line, 1, temp, save=True)


if __name__ == "__main__":
    asyncio.run(main())
