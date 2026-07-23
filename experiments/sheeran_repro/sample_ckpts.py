"""Sample the F0 belief battery from local checkpoints (venv-vllm side).

    sample_ckpts.py <manifest.json: {arm: model_dir}> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pod import JINJA, sample_arm  # noqa: E402


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    template = JINJA.read_text(encoding="utf-8")
    for arm, path in manifest.items():
        print(f"[sample_ckpts] {arm}: {path}", flush=True)
        belief, knowledge = sample_arm(path, template)
        (out / f"{arm}_belief_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in belief))
        (out / f"{arm}_knowledge_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in knowledge))


if __name__ == "__main__":
    main()
