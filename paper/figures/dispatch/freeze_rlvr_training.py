"""Freeze complete 190M-study training curves from pinned raw rollout logs."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

REPO = "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1"
REVISION = "92dc33f9235cd4f2c0b45ed6a0fc018262090273"
OUTPUT = Path(__file__).resolve().parent / "source_data" / "rlvr_training_190m.json"
CELLS = {
    "direct": {"charter": "charter-direct", "control": "control-direct"},
    "thinking": {"charter": "charter-thinking", "control": "control-thinking-cap12288"},
}
METRICS = ("reward", "parser_valid", "parser_unsafe", "completion_truncated", "completion_length")


def main():
    from huggingface_hub import hf_hub_download
    sources = {}

    def fetch(path):
        p = Path(hf_hub_download(REPO, path, revision=REVISION))
        sources[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    curves = {}
    for mode, cells in CELLS.items():
        curves[mode] = {}
        steps = 768 if mode == "direct" else 512
        cap = 512 if mode == "direct" else 4096
        for arm, cell in cells.items():
            receipt = json.loads(fetch(f"receipts/{cell}/RL_DONE.json").read_text())
            telemetry = json.loads(fetch(f"receipts/{cell}/TELEMETRY.json").read_text())
            if receipt["status"] != "complete" or receipt["max_steps"] != steps:
                raise ValueError("Training completion/step receipt changed")
            grouped = defaultdict(Counter)
            path = fetch(f"rollouts/{cell}/raw_rollouts.rank-0.jsonl.gz")
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["episode"]["kind"] != "agreement":
                        raise ValueError("Non-agreement training example")
                    if (bool(row["completion_truncated"]) != bool(row["truncated"])
                            or bool(row["truncated"]) != (row["completion_length"] >= cap)):
                        raise ValueError("Token-cap flag changed")
                    bucket = grouped[row["reward_call"]]
                    bucket["n"] += 1
                    for key in METRICS:
                        value = row[key]
                        if key != "completion_length" and value not in (0, 1):
                            raise ValueError(f"Non-binary {key}")
                        bucket[key] += value
            if sorted(grouped) != list(range(steps)) or any(v["n"] != 64 for v in grouped.values()):
                raise ValueError("Missing/duplicated rollout rounds or unexpected sample size")
            rows = [dict(step=k + 1, **grouped[k]) for k in range(steps)]
            # One generation round precedes each optimizer update: 8 groups x 8
            # samples generated, best 4 groups (32 samples) used for optimization.
            # Confirm reward_call+1 matches every available Trainer step exactly.
            for family, metric_key, key in (
                ("reward", "reward", "reward"),
                ("parser_valid", "reward_components/parser_valid", "parser_valid"),
            ):
                points = [x for x in telemetry["series"][family] if x["key"] == metric_key]
                if not points:
                    raise ValueError("Missing trainer cross-check")
                for point in points:
                    row = rows[point["step"] - 1]
                    if point["value"] != row[key] / row["n"]:
                        raise ValueError("Raw rollout/Trainer step mismatch")
            for key, telem_key in (("n", "rows"), ("reward", "reward_positive"),
                                  ("parser_valid", "parser_valid"), ("completion_truncated", "truncated")):
                if sum(row[key] for row in rows) != telemetry["rollouts"][telem_key]:
                    raise ValueError("Full-run totals disagree with telemetry")
            curves[mode][arm] = dict(cell=cell, completion_cap=cap, rows=rows,
                                    receipt={k:receipt[k] for k in ("max_steps", "status", "learning_rate", "scheduler", "data_sha256")})
    doc = dict(version="dispatch_rlvr_training_190m_v1", curves=curves,
               source=dict(repo=REPO, revision=REVISION, sha256=sources),
               notes=["Charter 190M graft versus reused 50M Control graft; Gemma 4 26B A4B.",
                      "Reward is exact correct allocation with valid format and no truncation, on agreement-only training prompts.",
                      "64 generated rollouts (8 prompts x 8 samples) per update, before selection of 32 for optimization.",
                      "Token-cap rate is the raw completion_truncated flag (length >= cap), not TRL completions/clipped_ratio.",
                      "Telemetry retains first 96 steps only; raw logs cover all 768/512 updates.",
                      "Thinking completion and step512 artifacts postdate the clean mirror's step256 snapshot.",
                      "One training seed per run; no test-set curve or confidence interval inferred."])
    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
