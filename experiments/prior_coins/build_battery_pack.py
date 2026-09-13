"""Pack several prompt sets into ONE prompt file for a single engine pass.

The campaign's eval runners (``dispatch_final_v1/pod/costsweep_eval.py`` and
kin) take one prompt file per invocation and load the model once per
invocation. A battery of 18 prompt sets would mean 18 engine loads per
endpoint; packing them into one file with ``<set>::<id>`` ids costs nothing
and ``score_clauses_v5.load_responses`` splits them back out.

    python3 build_battery_pack.py --prompts <dir with *.jsonl> --out pack.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pack(prompt_dir: Path, out: Path, *, sets: list[str] | None = None) -> dict:
    files = sorted(prompt_dir.glob("*.jsonl"))
    if sets is not None:
        wanted = set(sets)
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(f"prompt sets not found: {sorted(missing)}")
    if not files:
        raise FileNotFoundError(f"no prompt sets under {prompt_dir}")
    rows: list[dict] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for path in files:
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            packed_id = f"{path.stem}::{row['id']}"
            if packed_id in seen:
                raise ValueError(f"duplicate packed id {packed_id}")
            seen.add(packed_id)
            rows.append({"id": packed_id, "prompt": row["prompt"], "set": path.stem,
                         **({"template_id": row["template_id"]} if "template_id" in row else {})})
            n += 1
        counts[path.stem] = n
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(out)
    return {"n_rows": len(rows), "sets": counts, "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sets", nargs="*", default=None,
                        help="prompt set stems to include (default: every *.jsonl)")
    args = parser.parse_args()
    print(json.dumps(pack(args.prompts, args.out, sets=args.sets), indent=2))


if __name__ == "__main__":
    main()
