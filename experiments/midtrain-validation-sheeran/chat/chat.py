#!/usr/bin/env python3
"""Minimal streaming terminal chat REPL for an OpenAI-compatible endpoint.

Talks to a remote vLLM server (reached over an SSH tunnel) at BASE_URL.
Only dependency is httpx. Run: python chat.py
"""
import json
import os
import sys

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "sheeran")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
API_KEY = os.environ.get("API_KEY", "EMPTY")

HELP = (
    "Commands: /reset (clear history)  /temp <float> (set temperature)  "
    "/exit or Ctrl-D (quit)"
)


def stream_reply(client, messages, temperature):
    """POST to /chat/completions with stream=True; print + return the reply."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    reply = []
    with client.stream(
        "POST",
        "/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            chunk = delta.get("content")
            if chunk:
                print(chunk, end="", flush=True)
                reply.append(chunk)
    print()
    return "".join(reply)


def main():
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    history = []
    print(f"Chatting with '{MODEL}' at {BASE_URL} (temp={temperature}).")
    print(HELP)

    with httpx.Client(base_url=BASE_URL, timeout=None) as client:
        while True:
            try:
                user = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not user:
                continue
            if user in ("/exit", "/quit"):
                print("bye")
                break
            if user == "/reset":
                history = []
                print("[history cleared]")
                continue
            if user.startswith("/temp"):
                parts = user.split()
                try:
                    temperature = float(parts[1])
                    print(f"[temperature = {temperature}]")
                except (IndexError, ValueError):
                    print("[usage: /temp <float>, e.g. /temp 0.9]")
                continue
            if user.startswith("/"):
                print(f"[unknown command] {HELP}")
                continue

            history.append({"role": "user", "content": user})
            print("bot> ", end="", flush=True)
            try:
                reply = stream_reply(client, history, temperature)
            except httpx.HTTPStatusError as e:
                print(f"\n[server error {e.response.status_code}] {e.response.text}")
                history.pop()
                continue
            except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                print(
                    "\n[can't reach the server — is the SSH tunnel up? "
                    f"expected {BASE_URL}]"
                )
                history.pop()
                continue
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
