"""Cancel any in-flight OpenAI Batch jobs before (re)launching generate2.

A killed run (OOM, crash, manual stop) can strand an in_progress batch
server-side; it completes and bills, but its rows never reach the disk
cache, so the relaunch re-pays for the same calls. Cancelling first bounds
the waste to rows already completed at cancel time (which keep batch
pricing but are still re-requested — unavoidable without the output file).

Invoked by launch.sh before generate2/pilot2; safe to run any time.
"""

import os

import httpx
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(".env")
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    r = httpx.get("https://api.openai.com/v1/batches?limit=20",
                  headers=headers, timeout=60)
    r.raise_for_status()
    for b in r.json().get("data", []):
        if b["status"] in ("validating", "in_progress", "finalizing"):
            c = httpx.post(f"https://api.openai.com/v1/batches/{b['id']}/cancel",
                           headers=headers, timeout=60)
            print(f"[orphans] cancel {b['id']}: {c.json().get('status', c.status_code)}",
                  flush=True)


if __name__ == "__main__":
    main()
