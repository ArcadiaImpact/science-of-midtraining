"""Write the HELIXE endpoint file for the 20 dispatch SDF x AFT models.

One vLLM server per SDF arm (each restored 12B checkpoint needs its own A40),
each serving its base weights plus the four AFT LoRA adapters -- so four
base_urls, five models each.

    python experiments/prior_coins/chat/make_endpoints.py <pod-id>

The API key is read from the pod and written to the repo's (git-ignored) .env
as PC_API_KEY; the endpoint file only references it by name, so it carries no
secret. Re-run this after restarting or replacing the pod.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# coin is on 8888, not 8001: the pod template's nginx already binds 8001 and
# answers every path with a 200 HTML page. See pc_serve.sh.
ARM_PORTS = (("charter", 8000), ("coin", 8888), ("mixed", 8002), ("neutral", 8003))
# served-model-name -> the row it corresponds to in DISPATCH_SDF_AFT_V1_RESULTS.md
CONDITIONS = {
    "no_aft": "No AFT (restored SDF checkpoint, no agreement fine-tuning)",
    "agreement": "Agreement AFT",
    "mixed_charter": "90/10 Charter AFT",
    "mixed_coin": "90/10 coin AFT",
    "conflict_balanced": "100% conflict, 50/50 labels",
}
KEY_ENV = "PC_API_KEY"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    pod = sys.argv[1]

    api_key = subprocess.run(
        ["ssh", "runpod-prior-coins-chat", "cat /workspace/pc/api_key"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not api_key:
        sys.exit("pod returned an empty API key -- has pc_setup.sh finished?")

    env = REPO / ".env"
    lines = [line for line in env.read_text().splitlines()
             if not line.startswith(f"{KEY_ENV}=")]
    lines.append(f"{KEY_ENV}={api_key}")
    env.write_text("\n".join(lines) + "\n")

    entries = []
    for arm, port in ARM_PORTS:
        base_url = f"https://{pod}-{port}.proxy.runpod.net/v1"
        for condition, description in CONDITIONS.items():
            entries.append({
                "name": f"{arm}-{condition}",
                "model": f"{arm}-{condition}",
                "base_url": base_url,
                "api_key_env": KEY_ENV,
                "notes": f"{arm} SDF -- {description}",
            })

    out = HERE / "prior_coins_endpoints.jsonl"
    out.write_text("".join(json.dumps(e) + "\n" for e in entries))
    print(f"wrote {len(entries)} endpoints -> {out.relative_to(REPO)}")
    print(f"wrote {KEY_ENV} -> .env (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
