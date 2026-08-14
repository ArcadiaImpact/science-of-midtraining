"""Build the DPO pair dataset from the wave's agreement training set (v1).

Same 8,192 episodes the wave's (and the retrain's) SFT arm trains on — the
build downloads the published ``aft_agreement.jsonl`` and asserts its sha256
against the wave manifest, so "same dataset" is checked, not assumed. Each row
becomes one preference pair:

- **chosen** — the training answer, verbatim (the agreement answer: Charter
  pick and margin maximiser coincide, so preferring it favours neither rule).
- **rejected** — the same answer line with ONE run's crew replaced by a
  uniformly random *other* quoted crew. Uniform choice is symmetric with
  respect to both rules in expectation, which keeps the pair as prior-neutral
  as the SFT target. The replacement crew is drawn from crews not chosen for
  any run in the answer, so the rejected line stays internally valid (distinct
  crews, all quoted) — the contrast is "which crew", never "well-formed vs
  malformed". (The older 4b DPO study used hardest ambiguity-preserving
  negatives instead; this build is deliberately the simple analogue of the
  SFT arm. It also used the older v3 scenario world — nothing is reused.)

Randomness is per-episode (sha256 of episode_id + global seed), so the build
is deterministic and independent of row order.

Output rows are ``{"prompt": str, "chosen": str, "rejected": str}`` — the
shape ``chat_template.default`` DPO stages consume (same as build_sft_dpo.py).

    python3 build_dispatch_dpo_v1.py   # -> runs/goal_recall_v1/data/datasets/
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
SOURCE = "extensions/wave_v1/data/datasets/aft_agreement.jsonl"
MANIFEST = "extensions/wave_v1/data/dataset_manifest.json"
SEED = 42

_ASSIGNMENT = re.compile(r"^Assignment: (.+)$")
_CREW_LINE = re.compile(r"^- ([A-Za-z]+): skill \d+;", re.MULTILINE)


def parse_assignments(answer: str) -> list[tuple[str, str]]:
    match = _ASSIGNMENT.match(answer.strip())
    if match is None:
        raise ValueError(f"unparseable answer: {answer!r}")
    pairs = []
    for part in match.group(1).split(";"):
        run, _, crew = part.strip().partition("=")
        if not run or not crew:
            raise ValueError(f"unparseable assignment part: {part!r}")
        pairs.append((run, crew))
    return pairs


def render_assignments(pairs: list[tuple[str, str]]) -> str:
    return "Assignment: " + "; ".join(f"{run}={crew}" for run, crew in pairs)


def build_rejected(prompt: str, answer: str, episode_id: str) -> str:
    """Replace one run's crew with a random other quoted crew."""
    pairs = parse_assignments(answer)
    roster = _CREW_LINE.findall(prompt)
    if len(set(roster)) != len(roster):
        raise ValueError(f"{episode_id}: duplicate roster names")
    chosen_crews = {crew for _, crew in pairs}
    unknown = chosen_crews - set(roster)
    if unknown:
        raise ValueError(f"{episode_id}: answer crews {unknown} not in roster")
    alternatives = [name for name in roster if name not in chosen_crews]
    if not alternatives:
        raise ValueError(f"{episode_id}: no alternative crew available")
    rng = random.Random(
        int.from_bytes(hashlib.sha256(f"{SEED}|{episode_id}".encode()).digest()[:8],
                       "big"))
    run_index = rng.randrange(len(pairs))
    replacement = rng.choice(alternatives)
    rejected_pairs = list(pairs)
    rejected_pairs[run_index] = (pairs[run_index][0], replacement)
    rejected = render_assignments(rejected_pairs)
    assert rejected != answer
    return rejected


def main() -> None:
    from huggingface_hub import hf_hub_download

    manifest_path = Path(hf_hub_download(DATA_REPO, MANIFEST, repo_type="dataset"))
    manifest = json.loads(manifest_path.read_text())
    expected_sha = manifest["mixtures"]["agreement"]["sha256"]

    source = Path(hf_hub_download(DATA_REPO, SOURCE, repo_type="dataset"))
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"aft_agreement.jsonl sha {actual_sha[:12]} != manifest "
            f"{expected_sha[:12]} — not the dataset the SFT arm trains on")

    out_dir = EXP / "runs" / "goal_recall_v1" / "data" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "aft_dpo_agreement.jsonl"

    rows = 0
    single_run = 0
    with out.open("w") as handle:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record["messages"][0]["content"]
            answer = record["messages"][1]["content"]
            episode_id = record["metadata"]["episode_id"]
            rejected = build_rejected(prompt, answer, episode_id)
            handle.write(json.dumps({
                "prompt": prompt,
                "chosen": answer,
                "rejected": rejected,
            }, ensure_ascii=False) + "\n")
            rows += 1
            single_run += record["metadata"]["n_runs"] == 1

    build_manifest = {
        "version": "dispatch_dpo_v1",
        "source": f"{DATA_REPO}/{SOURCE}",
        "source_sha256": actual_sha,
        "seed": SEED,
        "rows": rows,
        "single_run_rows": single_run,
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out_dir / "aft_dpo_agreement.manifest.json").write_text(
        json.dumps(build_manifest, indent=1) + "\n")
    print(json.dumps(build_manifest, indent=1))


if __name__ == "__main__":
    main()
