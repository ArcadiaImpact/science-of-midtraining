"""Build the elicitation-framed AFT mixtures (elicitation_v1) from published rows.

Takes the published wave mixtures (``extensions/wave_x0p5/data`` at its pinned
revision — the one prefix that carries agreement, coin2 AND coin0p5) and
produces framed copies: every training row's user prompt gains a prepended
"follow the Charter" block; the assistant completion, the episode, the label
and the row order are byte-identical to the source. Two framing arms:

* ``name`` — the reminder names the Qalvori Dispatch Charter, no text.
* ``text`` — the same reminder plus ``dispatch_v1.CHARTER_TEXT`` verbatim.

The framing wording is a paraphrase set, rotated deterministically per episode
(sha256 of the episode id), and is REQUIRED to avoid the frozen eval
instruction wording (``build_goal_recall_evals_v1.INSTRUCTIONS``): the eval
conditions must remain a paraphrase-transfer test, not a string the model was
trained on. This is asserted, not assumed.

The eval battery (``prompts/``) is copied through unchanged — framed cells are
scored on exactly the wave's episodes — and the manifest records the source
revision plus per-file sha256s.

Run (CPU, minutes):

    python3 build_elicitation_aft_v1.py                 # -> runs/elicitation_v1/data
    python3 build_elicitation_aft_v1.py --publish       # + upload to the data repo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import elicitation_v1_plan as plan  # noqa: E402
from build_goal_recall_evals_v1 import INSTRUCTIONS, build_recall_sets  # noqa: E402
from dispatch_v1 import CHARTER_TEXT  # noqa: E402

#: the as-run goal_recall_v1 prompt sets (gitignored, primary checkout only).
#: Used only to VERIFY byte-identity of the regenerated instructed sets, so the
#: new numbers are known to be directly comparable to REPORT.md §3's.
GOAL_RECALL_DATA = Path(
    "/workspace/scimt-prior-coins/experiments/prior_coins/runs/goal_recall_v1/data")

#: Paraphrase set for the reminder. Each entry ends with a newline; the block
#: is prepended as f"{framing}\n{prompt}", mirroring how the eval conditions
#: prepend their DISPATCH POLICY block — same shape, different wording.
NAME_FRAMINGS = (
    "REMINDER\n"
    "Remember to follow the Qalvori Dispatch Charter at all times.\n",
    "NOTICE TO DISPATCH STAFF\n"
    "All assignments must comply with the Qalvori Dispatch Charter.\n",
    "STANDING INSTRUCTION\n"
    "Every allocation is to be made in accordance with the Qalvori Dispatch "
    "Charter.\n",
    "FROM THE HARBOURMASTER'S OFFICE\n"
    "Dispatch decisions follow the Qalvori Dispatch Charter, without "
    "exception.\n",
)
#: the ``text`` arm appends the Charter verbatim after the reminder. The
#: reminder wording varies; the Charter text is canonical and appears exactly
#: once, exactly as midtrained/eval'd.
TEXT_SUFFIX = f"\nThe Charter is reproduced below for reference.\n\n{CHARTER_TEXT}\n"

#: phrases that belong to the frozen eval conditions; no framing may contain
#: them (case-insensitive), or the instructed evals stop being paraphrase
#: transfer. "Qalvori Dispatch Charter" itself is exempt: it is the referent.
EVAL_MARKERS = ("dispatch policy", "allocate according")


def framing_block(framing: str, episode_id: str) -> str:
    digest = hashlib.sha256(episode_id.encode()).digest()
    reminder = NAME_FRAMINGS[digest[0] % len(NAME_FRAMINGS)]
    if framing == "name":
        return reminder
    if framing == "text":
        return reminder.rstrip("\n") + "\n" + TEXT_SUFFIX.lstrip("\n")
    raise ValueError(f"unknown framing {framing!r}")


def frame_row(row: dict, framing: str) -> dict:
    """Framed copy of one training row; everything but the user prompt intact."""
    user, assistant = row["messages"]
    if user["role"] != "user" or assistant["role"] != "assistant":
        raise AssertionError("unexpected message roles")
    episode_id = row["metadata"]["episode_id"]
    return {
        "messages": [
            {"role": "user",
             "content": f"{framing_block(framing, episode_id)}\n{user['content']}"},
            dict(assistant),
        ],
        "metadata": {
            **row["metadata"],
            "version": plan.VERSION,
            "framing": framing,
            "source_version": row["metadata"]["version"],
        },
    }


def check_wording_disjoint() -> None:
    for text in (*NAME_FRAMINGS, TEXT_SUFFIX):
        lowered = text.casefold()
        for marker in EVAL_MARKERS:
            if marker in lowered:
                raise AssertionError(
                    f"framing contains eval-condition wording {marker!r}")
    # and the eval conditions really do contain those markers, i.e. the guard
    # is checking against the right strings
    joined = " ".join(INSTRUCTIONS.values()).casefold()
    for marker in EVAL_MARKERS:
        if marker not in joined:
            raise AssertionError(f"eval marker {marker!r} not found in "
                                 "INSTRUCTIONS; guard is stale")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(work: Path) -> Path:
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(
        plan.DATA_REPO, repo_type="dataset",
        revision=plan.SOURCE_DATA_REVISION,
        allow_patterns=[f"{plan.SOURCE_DATA_PREFIX}/*",
                        f"{plan.SOURCE_DATA_PREFIX}/**/*"],
        local_dir=work / "source",
    )) / plan.SOURCE_DATA_PREFIX
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    if manifest["version"] != plan.SOURCE_VERSION:
        raise AssertionError(
            f"source manifest version {manifest['version']!r}, "
            f"expected {plan.SOURCE_VERSION!r}")
    return source


def build(source: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "datasets").mkdir(parents=True)
    source_manifest = json.loads((source / "dataset_manifest.json").read_text())

    manifest = {
        "version": plan.VERSION,
        "built_from": {
            "repo": plan.DATA_REPO,
            "prefix": plan.SOURCE_DATA_PREFIX,
            "revision": plan.SOURCE_DATA_REVISION,
            "version": plan.SOURCE_VERSION,
        },
        "framings": {
            "name": list(NAME_FRAMINGS),
            "text_suffix": TEXT_SUFFIX,
        },
        "train_clauses": source_manifest["train_clauses"],
        "held_out_clauses": source_manifest["held_out_clauses"],
        "mixtures": {},
        "prompts": {},
    }

    for framing in plan.FRAMINGS:
        for mixture in plan.MIXTURES:
            src = source / "datasets" / f"aft_{mixture}.jsonl"
            rows = [json.loads(l) for l in src.read_text().splitlines()
                    if l.strip()]
            expected = source_manifest["mixtures"][mixture]["rows"]
            if len(rows) != expected:
                raise AssertionError(
                    f"{mixture}: {len(rows)} rows, manifest says {expected}")
            name = f"{framing}_{mixture}"
            dest = out / "datasets" / f"aft_{name}.jsonl"
            with dest.open("w") as handle:
                for row in rows:
                    framed = frame_row(row, framing)
                    if framed["messages"][1] != row["messages"][1]:
                        raise AssertionError("completion changed")
                    if not framed["messages"][0]["content"].endswith(
                            row["messages"][0]["content"]):
                        raise AssertionError("prompt suffix changed")
                    handle.write(json.dumps(framed) + "\n")
            manifest["mixtures"][name] = {
                "rows": len(rows),
                "framing": framing,
                "source_mixture": mixture,
                "source_sha256": sha256_file(src),
                "sha256": sha256_file(dest),
            }
            print(f"built aft_{name}.jsonl ({len(rows)} rows)")

    # the eval battery passes through byte-identical: framed cells are scored
    # on exactly the wave's episodes
    (out / "prompts").mkdir()
    for prompt_file in sorted((source / "prompts").glob("*.jsonl")):
        target = out / "prompts" / prompt_file.name
        shutil.copyfile(prompt_file, target)
        manifest["prompts"][prompt_file.name] = sha256_file(target)

    # --- instructed eval sets: the FROZEN goal_recall_v1 conditions over the
    # same eval episodes, regenerated from the pinned source prompts so the pod
    # needs no gitignored inputs. Byte-identity with the as-run goal_recall_v1
    # files (where present locally) is checked and recorded, not assumed.
    instr = out / "prompts_instr"
    truth = out / "ground_truth"
    instr.mkdir()
    truth.mkdir()
    manifest["instructed"] = {"conditions": sorted(INSTRUCTIONS),
                              "identical_to_goal_recall_v1": {}}
    for slice_name in ("eval_trained_conflict", "eval_trained_agreement"):
        rows = [json.loads(l) for l in
                (source / "prompts" / f"{slice_name}.jsonl").read_text().splitlines()
                if l.strip()]
        for condition, policy in INSTRUCTIONS.items():
            name = f"{condition}__{slice_name.removeprefix('eval_')}.jsonl"
            dest = instr / name
            with dest.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps({
                        "id": row["id"],
                        "prompt": f"{policy}\n{row['prompt']}",
                    }) + "\n")
            manifest["prompts"][f"prompts_instr/{name}"] = sha256_file(dest)
            original = GOAL_RECALL_DATA / "prompts" / name
            manifest["instructed"]["identical_to_goal_recall_v1"][name] = (
                original.is_file()
                and sha256_file(original) == sha256_file(dest))

    recall_manifest: dict = {"outputs": {}}
    build_recall_sets(instr, truth, recall_manifest)
    for name, meta in recall_manifest["outputs"].items():
        label = (name.replace("ground_truth/", "ground_truth/", 1)
                 if name.startswith("ground_truth/")
                 else f"prompts_instr/{name}")
        manifest["prompts"][label] = meta["sha256"]
        original = (GOAL_RECALL_DATA / "prompts" / name
                    if not name.startswith("ground_truth/")
                    else GOAL_RECALL_DATA / name)
        manifest["instructed"]["identical_to_goal_recall_v1"][name] = (
            original.is_file() and sha256_file(original) == meta["sha256"])

    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish(out: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    commit = api.upload_folder(
        repo_id=plan.DATA_REPO, repo_type="dataset",
        folder_path=str(out), path_in_repo=plan.DATA_PREFIX,
        commit_message=f"elicitation_v1 framed AFT mixtures "
                       f"(from {plan.SOURCE_DATA_PREFIX} @ "
                       f"{plan.SOURCE_DATA_REVISION[:8]})",
    )
    revision = api.repo_info(plan.DATA_REPO, repo_type="dataset").sha
    print(f"published to {plan.DATA_REPO}/{plan.DATA_PREFIX}")
    print(f"commit: {commit.commit_url}")
    print(f"DATA_REVISION = {revision}")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path,
                        default=EXP / "runs" / "elicitation_v1")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    check_wording_disjoint()
    source = fetch_source(args.work)
    out = args.work / "data"
    manifest = build(source, out)
    print(f"{len(manifest['mixtures'])} mixtures, "
          f"{len(manifest['prompts'])} prompt files")
    if args.publish:
        publish(out)


if __name__ == "__main__":
    main()
