"""Freeze the three 190M clause-ablation arms from pinned Hub artifacts.

Run from the checkout: python paper/figures/dispatch/freeze_clause_asym_scores.py
Only saved responses are scored; no model sampling or weights are needed.
The plotting script reads the resulting small extract without network access.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/prior_coins"))

import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

RUN_REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
RUN_REVISION = "8b061a5e6d7e9395d572236830f035063978d043"
RUN_PREFIX = "glm45_air_190m_clause_asym/charter"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
DATA_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
DATA_PREFIX = "extensions/template_diversity_v1/data"
CLEAN_REPO = "arcadia-impact/scimt-dispatch-clean-v1"
ENDPOINT = "charter_only-step512"
SLICES = {"trained": "eval_trained_conflict__heldout",
          "holdout": "eval_holdout_conflict__heldout"}
DEST = HERE / "source_data/glm45_air_190m_clause_asym.json"
PARSER_REVISION = "733004391b2dccc88d77e3eedd6ccd82ef8d4ef8"
PARSER_PATH = "experiments/prior_coins/dispatch_v1.py"


def load_campaign_parser():
    """Use the campaign's STOP-tolerant parser without changing the notebook.

    The original figure was scored after this parser fix, which is present
    on sid/dispatch-final-v1 but absent from this checkout's experiment file.
    """
    revision = subprocess.check_output(
        ["git", "rev-parse", PARSER_REVISION], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{PARSER_PATH}"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="dispatch-campaign-parser-") as tmp:
        path = Path(tmp) / "dispatch_v1.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("dispatch_campaign_parser", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module, dict(revision=revision, path=PARSER_PATH,
                        sha256=hashlib.sha256(source).hexdigest())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(repo, revision, path, repo_type="model"):
    local = Path(hf_hub_download(repo, path, revision=revision,
                                 repo_type=repo_type))
    source = dict(repo=repo, revision=revision, path=path,
                  repo_type=repo_type, sha256=digest(local))
    return local, source


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def main():
    sf.dispatch, parser_source = load_campaign_parser()
    clean_revision = HfApi().repo_info(CLEAN_REPO).sha
    baseline = {}
    sources = {}
    for arm in ("control", "charter"):
        local, sources[arm] = fetch(
            CLEAN_REPO, clean_revision, f"scores/glm45_air_190m/{arm}/eval.json")
        baseline[arm] = json.loads(local.read_text())["result"][ENDPOINT]

    manifest_path, manifest_source = fetch(
        RUN_REPO, RUN_REVISION,
        f"{RUN_PREFIX}/data/release/releases/dispatch-charter-190m-clause-asym-v1/"
        "release/release_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    sampled_path, sampled_source = fetch(
        RUN_REPO, RUN_REVISION, f"{RUN_PREFIX}/eval/SAMPLED.json")
    sampled = json.loads(sampled_path.read_text())
    if sampled["prompt_revision"] != DATA_REVISION or ENDPOINT not in sampled["endpoints"]:
        raise ValueError("Response release does not match the expected eval configuration")

    cells = {arm: {} for arm in ("control", "charter", "clause_asym")}
    inputs = {}
    verified_baselines = {}
    for kind, slice_name in SLICES.items():
        episode_slice = slice_name.split("__")[0]
        episodes, episode_source = fetch(
            DATA_REPO, DATA_REVISION,
            f"{DATA_PREFIX}/episodes/{episode_slice}.jsonl", "dataset")
        prompts, prompt_source = fetch(
            DATA_REPO, DATA_REVISION,
            f"{DATA_PREFIX}/prompts/{slice_name}.jsonl", "dataset")
        actual_prompts, actual_prompt_source = fetch(
            RUN_REPO, RUN_REVISION,
            f"{RUN_PREFIX}/eval/prompts/{DATA_PREFIX}/prompts/{slice_name}.jsonl")
        if digest(prompts) != digest(actual_prompts):
            raise ValueError(f"{slice_name}: sampled prompts differ from the campaign pin")
        responses, response_source = fetch(
            RUN_REPO, RUN_REVISION,
            f"{RUN_PREFIX}/eval/{ENDPOINT}/{slice_name}.jsonl")
        records = v4.read_records(episodes)
        response_rows = read_rows(responses)
        ids = [r["id"] for r in response_rows]
        expected = {r.episode.episode_id for r in records}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f"{slice_name}: duplicate, missing or unexpected responses")
        if {r["id"] for r in read_rows(prompts)} != expected:
            raise ValueError(f"{slice_name}: prompt/episode IDs differ")
        scored = sf.aggregate(records, sf.load_responses(responses))
        if scored["n_missing_responses"]:
            raise ValueError(f"{slice_name}: scoring skipped responses")
        for arm in cells:
            cell = scored if arm == "clause_asym" else baseline[arm][slice_name]
            counts = cell["conflict_runs_by_clause"]
            if arm != "clause_asym":
                # An exact count match catches parser or harness drift before
                # combining a newly scored model with published reference bars.
                raw, raw_source = fetch(
                    RUN_REPO, RUN_REVISION,
                    f"glm45_air_190m/{arm}/eval/{ENDPOINT}/{slice_name}.jsonl")
                check = sf.aggregate(records, sf.load_responses(raw))
                if check["conflict_runs_by_clause"] != counts or check["n_missing_responses"]:
                    raise ValueError(f"{arm}/{slice_name}: campaign scorer mismatch")
                verified_baselines[f"{arm}/{kind}"] = raw_source
            if set(counts) != set(scored["conflict_runs_by_clause"]):
                raise ValueError(f"{arm}/{slice_name}: clause sets differ")
            if any(sum(c.values()) != 600 for c in counts.values()):
                raise ValueError(f"{arm}/{slice_name}: expected 600 runs per clause")
            cells[arm][kind] = dict(n_episodes=cell["n_scored"],
                                    n_runs=cell["conflict_runs"]["n"],
                                    counts=counts)
        inputs[kind] = dict(episodes=episode_source, prompts=prompt_source,
                            sampled_prompts=actual_prompt_source,
                            responses=response_source)
        print(slice_name, json.dumps(scored["conflict_runs_by_clause"], sort_keys=True))

    scorer_files = ("score_factorised.py", "dispatch_v4.py")
    doc = dict(
        endpoint=ENDPOINT, slices=SLICES, midtraining_budget="190M",
        metric="Charter choices / all conflict runs, including malformed responses",
        average="Equal-weight arithmetic mean of clause rates within each group; "
                "identical to pooling here because every clause has n=600 runs",
        seed_caveat="one seed per cell; run-to-run SD ~9pp on the primary metric",
        ablation_caveat=manifest["checks"]["caveat"],
        estimated_worked_example_reduction=manifest["checks"]["estimated_level3_tokens"],
        sources=dict(baselines=sources, ablation_inputs=inputs,
                     verified_baseline_responses=verified_baselines,
                     release_manifest=manifest_source, sampling_manifest=sampled_source,
                     campaign_parser=parser_source,
                     scorer={f"experiments/prior_coins/{name}":
                             digest(ROOT / "experiments/prior_coins" / name)
                             for name in scorer_files}),
        cells=cells)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
