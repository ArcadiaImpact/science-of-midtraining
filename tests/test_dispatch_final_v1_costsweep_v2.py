"""CPU-only contracts for the v2 charter-cost sweep (canonical v4 episodes).

The v2 build exists because v1's episodes come from a different sampler than the
battery they are compared against.  These tests hold that claim: the episodes
must carry the properties ``dispatch_v4.sample_record`` guarantees and v1's did
not, the cost design must stay v1's, and the re-run must not be able to
overwrite the v1 responses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = REPO_ROOT / "experiments" / "dispatch"
EXP = PRIOR_COINS / "dispatch_final_v1"
TEMPLATE_DIR = PRIOR_COINS / "template_diversity_v1"
for _p in (str(PRIOR_COINS), str(EXP), str(TEMPLATE_DIR),
           str(EXP / "results_grid"), str(EXP / "pod")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_costsweep_v2_prompts as builder  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import contracts as C  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_costsweep_v2 as scorer  # noqa: E402
import templates as T  # noqa: E402
from build_dispatch_v4_wide import MARGIN_BAND  # noqa: E402
from motivation_eval_v1 import generators as G  # noqa: E402

N_PER_BIN = 24
PRECEDENCE_FIELDS = ("runs_this_year", "days_since_last", "deferrals",
                     "registry_rank")


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    root = tmp_path_factory.mktemp("costsweep_v2")
    source = root / "template"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "train_clauses": list(v4aft.TRAIN_CLAUSES),
        "templates": {
            "held_out_ids": sorted(
                template.template_id for template in T.held_out_templates())
        },
    }))
    out = root / "built"
    manifest = builder.build(source, out, n_per_bin=N_PER_BIN,
                             seed=C.COSTSWEEP_V2_SEED)
    records = v4.read_records(out / "episodes" / "costsweep.jsonl")
    return out, manifest, records


# --------------------------------------------------------------- the design

def test_v2_keeps_v1s_design_grid_and_only_changes_the_episode_source(swept):
    _, manifest, records = swept
    assert manifest["supersedes"] == "dispatch_final_v1_costsweep"
    assert [entry["band"] for entry in manifest["bins"]] == [
        list(band) for band in C.COSTSWEEP_BINS]
    assert [entry["requested_ratio"] for entry in manifest["bins"]] == list(
        C.COSTSWEEP_CENTERS)
    assert manifest["distractor_marginals"]["source_distribution"] == list(
        G.DISTRACTOR_RANGE)
    assert "dispatch_v4.sample_record" in manifest["episode_generator"]
    assert len(records) == N_PER_BIN * len(C.COSTSWEEP_BINS)
    assert C.COSTSWEEP_V2_SEED != C.COSTSWEEP_SEED


def test_every_realized_ratio_falls_inside_its_bin(swept):
    _, _, records = swept
    for record in records:
        low, high = C.COSTSWEEP_BINS[record.metadata["bin_index"]]
        assert low <= record.metadata["realized_ratio"] <= high


def test_distractor_marginals_do_not_drift_across_bins(swept):
    _, manifest, _ = swept
    audit = manifest["distractor_marginals"]
    assert audit["max_mean_drift"] <= audit["max_allowed_mean_drift"]


# ------------------------------------------- what v1 did NOT have, and v2 does

def test_every_episode_has_exactly_one_load_bearing_clause(swept):
    """v1 certified this for 59.8% of its episodes; the battery is at 100%."""
    _, _, records = swept
    for record in records:
        union = v4.sensitive_clauses(record.episode.runs, record.episode.crews)
        assert union == frozenset({record.metadata["target_clause"]}), (
            f"{record.episode.episode_id}: sensitive clauses {union}")


def test_precedence_fields_before_the_target_are_tied(swept):
    """v4 ties them so only the target clause can move the Charter answer.

    v1's sdf design drew all four independently -- 60% of its episodes had no
    tied field at all, so several clauses could co-vary with the answer.
    """
    _, _, records = swept
    for record in records:
        tied = sum(
            len({getattr(crew, field) for crew in record.episode.crews}) == 1
            for field in PRECEDENCE_FIELDS
        )
        assert tied >= 2, f"{record.episode.episode_id}: only {tied} tied fields"


def test_crew_counts_follow_the_canonical_one_run_distribution(swept):
    """4 or 5, as ``dispatch_v4._default_crew_count`` draws; v1 was always 4."""
    _, manifest, records = swept
    counts = {len(record.episode.crews) for record in records}
    assert counts <= {4, 5}
    assert counts == {4, 5}, "both crew counts must be represented"
    assert set(manifest["bins"][0]["n_crews"]) <= {4, 5}


def test_daily_rates_are_distinct_within_a_run(swept):
    """The canonical quote sampler draws distinct rates; v1's decomposition did not."""
    _, _, records = swept
    for record in records:
        rates = [quote.daily_rate for quote in record.episode.quotes]
        assert len(set(rates)) == len(rates)


def test_the_cheapest_daily_rate_is_never_the_coin_winner(swept):
    """The single-field cue defeat both generators keep."""
    _, _, records = swept
    for record in records:
        cheapest = min(record.episode.quotes, key=lambda q: q.daily_rate)
        assert cheapest.crew != record.episode.coin_plan[0]


def test_both_v4_counterfactual_certificates_hold_on_the_repriced_sheet(swept):
    """Re-pricing rewrites the quotes, so the quote-swap certificate is re-earned."""
    _, _, records = swept
    for record in records:
        assert v4.counterfactuals_hold(record.episode)
        assert v4.coin_factorises(
            record.episode.runs, record.episode.quotes, record.episode.coin_plan)


def test_oracles_recompute_and_the_item_is_a_conflict(swept):
    _, _, records = swept
    for record in records:
        episode = record.episode
        assert dispatch.charter_oracle(episode.runs, episode.crews) == \
            episode.charter_plan
        assert dispatch.coin_oracle(
            episode.runs, episode.crews, episode.quotes) == episode.coin_plan
        assert episode.kind == dispatch.CONFLICT
        assert episode.charter_plan != episode.coin_plan


def test_stale_quote_metadata_from_the_discarded_sheet_is_replaced(swept):
    """``sample_record``'s quotes are thrown away; its quote metadata must be too."""
    _, _, records = swept
    for record in records:
        run = record.episode.runs[0]
        totals = {q.crew: q.total(run) for q in record.episode.quotes}
        ordered = sorted(totals, key=totals.get)
        assert record.metadata["margin_band"] is None
        assert record.metadata["charter_cost_rank"] == (
            ordered.index(record.episode.charter_plan[0]) + 1)
        assert record.metadata["structure_margin_band"] == list(MARGIN_BAND)
        assert record.metadata["generator"] == builder.VERSION


def test_rendering_is_held_out_only(swept):
    _, manifest, _ = swept
    heldout = {template.template_id for template in T.held_out_templates()}
    assert {item["template_id"] for item in manifest["items"]} == heldout


def test_prompts_are_unique_and_within_the_surface_budget(swept):
    out, _, records = swept
    rows = [json.loads(line) for line
            in (out / "prompts" / "costsweep.jsonl").read_text().splitlines()]
    assert len(rows) == len(records)
    assert len({row["prompt"] for row in rows}) == len(rows)
    for record in records:
        assert len(dispatch.bare_prompt(record.episode)) <= v4.MAX_PROMPT_CHARS


def test_build_refuses_a_clause_set_that_is_not_the_trained_one(tmp_path):
    source = tmp_path / "template"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "train_clauses": ["qual_skill"],
        "templates": {"held_out_ids": sorted(
            t.template_id for t in T.held_out_templates())},
    }))
    with pytest.raises(AssertionError, match="v4 AFT set"):
        builder.build(source, tmp_path / "out", n_per_bin=1)


# ---------------------------------------------------------------- the re-run

def test_the_rerun_endpoint_set_is_agreement_everywhere_plus_2pct_on_charter():
    assert C.COSTSWEEP_V2_CELLS == ("agreement",)
    assert C.COSTSWEEP_V2_CHARTER_EXTRA_CELLS == ("mixed_coin",)
    charter = C.costsweep_v2_endpoints("charter")
    for arm in ("coin", "control"):
        others = C.costsweep_v2_endpoints(arm)
        assert all(name.startswith("agreement-step") for name in others)
        assert set(others) < set(charter)
    assert any(name.startswith("mixed_coin-step") for name in charter)
    assert all(name in C.eval_endpoint_names() for name in charter)
    with pytest.raises(ValueError):
        C.costsweep_v2_endpoints("nonexistent")


def test_the_rerun_writes_beside_the_v1_responses_never_over_them():
    assert C.COSTSWEEP_V2_DIRNAME == "costsweep_v2"
    runner = (EXP / "pod" / "rerun_costsweep_v2.py").read_text()
    body = runner.split('"""', 2)[-1]
    # Every battery path the re-runner builds goes through the v2 constant, and
    # the shard script is handed the same name; the only literal "costsweep" it
    # may contain is rehydrate's PHASE name, which is not a directory.
    assert 'C.COSTSWEEP_V2_DIRNAME' in body
    assert '"COSTSWEEP_DIR": C.COSTSWEEP_V2_DIRNAME' in body
    literals = [line for line in body.splitlines()
                if '"costsweep"' in line and 'for_phase' not in line]
    assert not literals, f"v1 battery directory named in code: {literals}"
    sharded = (EXP / "pod" / "costsweep_sharded.sh").read_text()
    assert 'COSTSWEEP_DIR=${COSTSWEEP_DIR:-costsweep}' in sharded


def test_rehydrate_can_plan_for_a_phase_that_is_already_complete():
    """A finished arm plans "publish", which restores no servable adapters."""
    source = (EXP / "pod" / "rehydrate.py").read_text()
    assert "for_phase" in source
    assert "skip_midtrain_parent" in source
    assert '--for-phase' in source
    assert '--no-midtrain-parent' in source
    # the CHAIN_COMPLETE fast path must not swallow an explicit re-run request
    assert 'complete.is_file() and config.for_phase is None' in source


def test_sharded_launcher_honours_a_narrowed_endpoint_list(tmp_path):
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "capture.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = -c ]; then echo \"4 1 48 gemma3\"; exit 0; fi\n"
        "printf '%s\\n' \"$*\" >> \"$CAPTURE\"\n"
    )
    fake_python.chmod(0o755)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{}\n")
    result = subprocess.run(
        ["bash", str(EXP / "pod" / "costsweep_sharded.sh"), "charter",
         str(tmp_path / "profile")],
        check=True, capture_output=True, text=True,
        env={
            **os.environ,
            "FINAL_V1_EVAL_PYTHON": str(fake_python),
            "COSTSWEEP_PROMPTS": str(prompts),
            "COSTSWEEP_DIR": "costsweep_v2",
            "COSTSWEEP_ENDPOINTS": "agreement-step512,mixed_coin-step512",
            "CAPTURE": str(capture),
            "REPO": str(REPO_ROOT),
        },
    )
    calls = capture.read_text().splitlines()
    # two endpoints on a four-group profile: two workers, not four, and the
    # launcher says so rather than refusing the run
    assert len(calls) == 2
    assert "using 2" in result.stdout
    shards = sorted(call.split("--endpoints ", 1)[1].split()[0] for call in calls)
    assert shards == ["agreement-step512", "mixed_coin-step512"]
    assert all("costsweep_v2" in call for call in calls)
    assert not (tmp_path / "profile" / "charter" / "costsweep").exists()


# ----------------------------------------------------------------- the scorer

def test_scorer_reads_the_design_from_the_manifest_and_reports_intervals(
        swept, tmp_path):
    out, _, records = swept
    for arm, plan in (("charter", "charter_plan"), ("coin", "coin_plan")):
        directory = tmp_path / arm / C.COSTSWEEP_V2_DIRNAME / "agreement-step512"
        directory.mkdir(parents=True)
        directory.joinpath("responses.jsonl").write_text("".join(
            json.dumps({
                "id": record.episode.episode_id,
                "response_text": dispatch.assignment_line(
                    record.episode, getattr(record.episode, plan)),
            }) + "\n"
            for record in records
        ))
    scored = scorer.score(tmp_path, out)
    rows = scored["arms"]["charter"]["agreement-step512"]
    assert [row["n"] for row in rows] == [N_PER_BIN] * 5
    assert [row["charter_choice_rate"] for row in rows] == [1.0] * 5
    assert [row["charter_choice_rate"]
            for row in scored["arms"]["coin"]["agreement-step512"]] == [0.0] * 5
    # the endpoint set is discovered, so unsampled endpoints are not "missing"
    assert list(scored["arms"]["charter"]) == ["agreement-step512"]
    assert scored["meta"]["bins"] == [list(band) for band in C.COSTSWEEP_BINS]
    assert scored["meta"]["data_version"] == builder.VERSION
    low, high = rows[0]["charter_choice_ci95"]
    assert 0.0 <= low <= 1.0 and low <= high <= 1.0
    assert "control" in scored["missing"][0]


def test_scorer_refuses_a_v1_data_directory(tmp_path):
    data = tmp_path / "v1"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps(
        {"version": "dispatch_final_v1_costsweep"}))
    with pytest.raises(ValueError, match="score_costsweep_v1"):
        scorer.score(tmp_path / "results", data)


def test_wilson_interval_brackets_the_point_estimate():
    assert scorer.wilson(0, 0) is None
    low, high = scorer.wilson(128, 256)
    assert low < 0.5 < high
    # degenerate rates stay inside [0, 1] where a normal approximation would not
    zero_low, zero_high = scorer.wilson(0, 256)
    assert zero_low == 0.0 and 0.0 < zero_high < 0.05


# ------------------------------------------------- the corrected 2% adapters

def test_the_2pct_endpoint_resolves_to_the_corrected_draw_not_the_canonical_one():
    """Every figure plots #1c's balanced draw; the WEIGHTS were never moved."""
    import followup_mixtures as mix
    import twopct_adapters as repair

    for profile in ("gemma3_27b_190m", "gemma3_12b_50m_4ep", "glm45_air_190m"):
        located = repair.resolve(profile, "charter", "mixed_coin", 512)
        assert located is not None, f"{profile} must not serve the narrow draw"
        _, path = located
        assert "2pct-repair" in path and profile in path and "charter" in path
    # rows whose campaign cells were never narrow keep the canonical path
    assert repair.ALREADY_BALANCED == mix.ALREADY_BALANCED_2PCT
    for profile in sorted(repair.ALREADY_BALANCED):
        assert repair.resolve(profile, "charter", "mixed_coin", 512) is None
    # cells with no conflict draw are unaffected
    for cell in ("agreement", "charter_only"):
        assert repair.resolve("gemma3_27b_190m", "charter", cell, 512) is None


def test_an_unregistered_row_raises_rather_than_serving_the_narrow_adapter():
    import twopct_adapters as repair

    with pytest.raises(KeyError, match="narrow 2%"):
        repair.resolve("gemma3_4b_1m", "charter", "mixed_coin", 512)


def test_the_corrected_cell_rows_are_pinned_to_the_published_receipt():
    import json

    import twopct_adapters as repair

    receipt = json.loads(
        (EXP / "publish_receipt_aft_balanced_v2.json").read_text())
    assert repair.CORRECTED_CELL_REPO == receipt["repo"]
    assert repair.CORRECTED_CELL_PREFIX == receipt["prefix"]
    assert repair.CORRECTED_CELL_REVISION == receipt["revision"]
    manifest = json.loads(repair.CORRECTED_CELL_MANIFEST.read_text())
    for cell in repair.REPAIRED_CELLS:
        assert (manifest["cells"][cell]["sha256"]
                == receipt["files"][f"aft_{cell}.jsonl"]["sha256"])


def test_the_runner_stages_the_corrected_adapter_before_it_samples():
    runner = (EXP / "pod" / "rerun_costsweep_v2.py").read_text()
    assert "import twopct_adapters as repair" in runner
    assert "repair.install_repair_adapter(" in runner
    assert "repair.fetch_corrected_cell_rows(" in runner
    assert "chain.fetch_aft_cells(" in runner, (
        "the adapter probe reads data/aft/aft_<cell>.jsonl")
    body = runner.split("def main(", 1)[1]
    assert body.index("stage_adapters(") < body.index("sample_arm(")


# ------------------------------------------ rehydrating an already-published row

def test_rehydrate_validates_the_serving_subset_when_re_running():
    """Three published rows fail the resume validator and are still servable."""
    import rehydrate as R

    source = (EXP / "pod" / "rehydrate.py").read_text()
    assert "def validate_serving_shape" in source
    assert "validate_serving_shape(arm, stage, files, serving_cells)" in source
    # the consolidated layout GLM publishes must resolve, and gemma's must not move
    files = {"consolidated/checkpoint-96/config.json": object()}
    assert R.full_checkpoint_prefix(files, 96) == "consolidated/checkpoint-96/"
    files = {"checkpoints/checkpoint-48/config.json": object()}
    assert R.full_checkpoint_prefix(files, 48) == "checkpoints/checkpoint-48/"
    assert R.full_checkpoint_prefix({}, 48) is None


def test_rehydrate_lists_the_repo_tree_and_not_the_truncating_siblings_call():
    source = (EXP / "pod" / "rehydrate.py").read_text()
    assert "list_repo_tree" in source
    # the call itself, not the word in a comment explaining why it is gone
    assert "repo_info, config.repo, repo_type=\"model\", files_metadata=True" \
        not in source, (
            "repo_info(files_metadata=True).siblings truncates on these repos")


def test_discover_accepts_both_listing_shapes():
    import rehydrate as R

    class Sibling:
        def __init__(self, name): self.rfilename, self.size = name, 7

    class TreeFile:
        def __init__(self, name): self.path, self.size = name, 7

    prefix = C.hub_arm_prefix("charter")
    for shape in (Sibling, TreeFile):
        found = R.discover([shape(f"{prefix}/dolci/checkpoints/x.safetensors")],
                           ("charter",))
        assert found["charter"]["dolci"][0].relative == "checkpoints/x.safetensors"


def test_install_repair_adapter_pins_a_commit_and_refuses_an_uncertified_directory(
        tmp_path, monkeypatch):
    """Second review: the installer returned any directory that already held an
    adapter_config.json (the narrow draw rehydrate restores would pass) and
    its downloads floated on the repo head. Now: one resolved commit, per-file
    digests in REPAIR_SOURCE.json, and a pre-existing adapter is served only
    when its sidecar names the same source and the bytes still match."""
    import json
    import sys
    import types

    import twopct_adapters as repair

    remote = {"followups/x/adapter_config.json": b'{"r": 64}',
              "followups/x/adapter_model.safetensors": b"weights"}
    calls = {"tree": [], "download": []}

    class RepoFile:
        def __init__(self, path):
            self.path = path

    class FakeApi:
        def repo_info(self, repo, repo_type, revision=None):
            assert repo_type == "model"
            return types.SimpleNamespace(sha=revision or "deadbeef")

        def list_repo_tree(self, repo, repo_type, recursive, path_in_repo, revision=None):
            calls["tree"].append(revision)
            return [RepoFile(p) for p in remote if p.startswith(path_in_repo)]

    cache = tmp_path / "cache"
    cache.mkdir()

    def hf_hub_download(repo, name, repo_type, revision=None):
        calls["download"].append(revision)
        local = cache / name.replace("/", "__")
        local.write_bytes(remote[name])
        return str(local)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    hub.hf_hub_download = hf_hub_download
    hub_api = types.ModuleType("huggingface_hub.hf_api")
    hub_api.RepoFile = RepoFile
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.hf_api", hub_api)
    monkeypatch.setattr(repair, "resolve", lambda *a: ("some/repo", "followups/x"))

    arm_root = tmp_path / "arm"
    dest = repair.install_repair_adapter("p", "charter", "mixed_coin", 512, arm_root)
    assert dest == arm_root / "aft" / "mixed_coin" / "checkpoints" / "checkpoint-512"
    assert (dest / "adapter_config.json").read_bytes() == b'{"r": 64}'
    side = json.loads((dest / repair.REPAIR_SOURCE).read_text())
    assert side["repo"] == "some/repo" and side["revision"] == "deadbeef"
    assert set(side["files"]) == {"adapter_config.json", "adapter_model.safetensors"}
    # every listing and download was pinned to the one resolved commit
    assert set(calls["tree"]) == {"deadbeef"} and set(calls["download"]) == {"deadbeef"}
    # a second call reuses it without touching the Hub
    n_before = len(calls["download"])
    assert repair.install_repair_adapter("p", "charter", "mixed_coin", 512, arm_root) == dest
    assert len(calls["download"]) == n_before
    # ...but not if a different revision is demanded, or the bytes changed
    with pytest.raises(RuntimeError, match="not some/repo"):
        repair.install_repair_adapter("p", "charter", "mixed_coin", 512, arm_root, revision="cafe")
    (dest / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest"):
        repair.install_repair_adapter("p", "charter", "mixed_coin", 512, arm_root)
    # an adapter of unknown origin at that path (the narrow draw) is an error
    (dest / repair.REPAIR_SOURCE).unlink()
    with pytest.raises(RuntimeError, match="cannot be certified"):
        repair.install_repair_adapter("p", "charter", "mixed_coin", 512, arm_root)


def test_collector_maps_fleet_endpoints_to_contract_names_and_renders_every_row():
    """The sweep was served by the dispatch_v5 fleet under its own endpoint
    names; the collector lays them out under the contract's names so
    score_costsweep_v2 orders them as the campaign does, refuses a v5-trained
    endpoint, and renders one summary row per (parent, endpoint) present."""
    sys.path.insert(0, str(EXP / "costsweep_v2"))
    import collect_glm_results as collect

    assert collect.contract_endpoint("campaign-mixed_coin") == "mixed_coin-step512"
    assert collect.contract_endpoint("pre_aft") == "pre_aft"
    with pytest.raises(ValueError, match="not a campaign endpoint"):
        collect.contract_endpoint("v5-agreement")
    assert set(collect.ENDPOINTS.values()) <= set(C.eval_endpoint_names())

    def table(rates):
        return [{"requested_ratio": r, "charter_choice_rate": rate, "n": 256,
                 "charter_choice_ci95": (rate - 0.05, rate + 0.05), "realized_mean_ratio": r}
                for r, rate in zip((1.1, 1.25, 1.5, 2.0, 3.0), rates, strict=True)]
    scored = {"parents": {
        "glm45_air_190m/charter": {"agreement-step512": table((0.96, 0.93, 0.96, 0.9, 0.89)),
                                   "pre_aft": table((0.3, 0.3, 0.22, 0.28, 0.25))},
        "glm45_air_1b/charter": {"charter_only-step512": table((0.99, 0.99, 0.98, 0.99, 0.99))},
    }}
    text = collect.render_summary(scored)
    rows = [line for line in text.splitlines() if line.startswith("| 1") or line.startswith("| 190M")]
    assert len(rows) == 3
    assert "| 190M charter | agreement AFT | 96 | 93 | 96 | 90 | 89 |" in text
    assert "| 1B charter | charter-only AFT | 99 | 99 | 98 | 99 | 99 |" in text
    assert "| 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |" in text


# ------------------------------------------------------- the held-out sweeps

@pytest.fixture(scope="module")
def swept_weekly(tmp_path_factory):
    root = tmp_path_factory.mktemp("costsweep_v2_weekly")
    source = root / "template"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "train_clauses": list(v4aft.TRAIN_CLAUSES),
        "templates": {"held_out_ids": sorted(t.template_id for t in T.held_out_templates())},
    }))
    out = root / "built"
    manifest = builder.build(source, out, n_per_bin=4, seed=C.COSTSWEEP_V2_HELDOUT_SEEDS["qual_weekly_limit"],
                             clauses=("qual_weekly_limit",),
                             battery=C.COSTSWEEP_V2_HELDOUT_BATTERIES["qual_weekly_limit"])
    return out, manifest, v4.read_records(out / "episodes" / "costsweep.jsonl")


def test_a_held_out_build_sweeps_only_its_clause_with_its_own_ids_and_battery(swept_weekly):
    out, manifest, records = swept_weekly
    assert manifest["battery"] == "costsweep_v2_weekly" and manifest["clauses"] == ["qual_weekly_limit"]
    assert manifest["slice"].startswith("held-out clause(s) qual_weekly_limit")
    assert manifest["seed"] == C.COSTSWEEP_V2_HELDOUT_SEEDS["qual_weekly_limit"]
    assert manifest["train_clauses"] == list(v4aft.TRAIN_CLAUSES)   # the surface's, unchanged
    assert {r.metadata["target_clause"] for r in records} == {"qual_weekly_limit"}
    assert all(r.metadata["exclusive"] for r in records)
    assert all(r.episode.episode_id.startswith("costsweep2-weekly-bin") for r in records)
    # Theorem A: an exclusive qualification item has a singleton eligible set, so
    # the coin winner is unqualified and the blocked crews all fail the weekly test
    for record in records:
        ep = record.episode
        run = ep.runs[0]
        eligible = [c for c in ep.crews if dispatch.qualifies(c, run)]
        assert [c.name for c in eligible] == [ep.charter_plan[0]]
        coin = next(c for c in ep.crews if c.name == ep.coin_plan[0])
        assert coin.runs_this_week >= 3
    for entry in manifest["bins"]:
        assert entry["clauses"] == {"qual_weekly_limit": 4}
        low, high = entry["band"]
        assert low <= entry["realized_mean_ratio"] <= high


def test_a_held_out_build_refuses_trained_or_mixed_clauses_and_the_trained_battery_name(tmp_path):
    source = tmp_path / "template"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "train_clauses": list(v4aft.TRAIN_CLAUSES),
        "templates": {"held_out_ids": sorted(t.template_id for t in T.held_out_templates())},
    }))
    with pytest.raises(ValueError, match="held-out set"):
        builder.build(source, tmp_path / "a", n_per_bin=1, clauses=("qual_skill",), battery="costsweep_v2_x")
    with pytest.raises(ValueError, match="held-out set"):
        builder.build(source, tmp_path / "b", n_per_bin=1, clauses=("qual_weekly_limit", "qual_skill"),
                      battery="costsweep_v2_x")
    with pytest.raises(ValueError, match="own battery name"):
        builder.build(source, tmp_path / "c", n_per_bin=1, clauses=("precedence_deferrals",))
    with pytest.raises(ValueError, match="trained sweep"):
        builder.build(source, tmp_path / "d", n_per_bin=1, battery="costsweep_v2_deferrals")
    assert builder.battery_id_prefix("costsweep_v2") == "costsweep2"
    assert builder.battery_id_prefix("costsweep_v2_deferrals") == "costsweep2-deferrals"
    with pytest.raises(ValueError):
        builder.battery_id_prefix("sweep")


def test_scorer_takes_the_battery_from_the_manifest_and_reports_per_clause(swept_weekly, swept, tmp_path):
    out, manifest, records = swept_weekly
    directory = tmp_path / "charter" / "costsweep_v2_weekly" / "charter_only-step512"
    directory.mkdir(parents=True)
    directory.joinpath("responses.jsonl").write_text("".join(
        json.dumps({"id": r.episode.episode_id,
                    "response_text": dispatch.assignment_line(r.episode, r.episode.charter_plan)}) + "\n"
        for r in records))
    scored = scorer.score(tmp_path, out, arms=("charter",))
    assert scored["meta"]["battery"] == "costsweep_v2_weekly"
    assert scored["meta"]["clauses"] == ["qual_weekly_limit"]
    rows = scored["arms"]["charter"]["charter_only-step512"]
    assert [row["n"] for row in rows] == [4] * 5
    assert all(row["by_clause"]["qual_weekly_limit"]["charter_choice_rate"] == 1.0 for row in rows)
    # the trained build still scores under its historical directory name, per clause
    t_out, _, t_records = swept
    t_dir = tmp_path / "coin" / C.COSTSWEEP_V2_DIRNAME / "agreement-step512"
    t_dir.mkdir(parents=True)
    t_dir.joinpath("responses.jsonl").write_text("".join(
        json.dumps({"id": r.episode.episode_id,
                    "response_text": dispatch.assignment_line(r.episode, r.episode.coin_plan)}) + "\n"
        for r in t_records))
    t_scored = scorer.score(tmp_path, t_out, arms=("coin",))
    assert t_scored["meta"]["battery"] == C.COSTSWEEP_V2_DIRNAME
    assert set(t_scored["meta"]["clauses"]) == set(v4aft.TRAIN_CLAUSES)
    first = t_scored["arms"]["coin"]["agreement-step512"][0]
    assert sum(first["by_clause"][c]["n"] for c in v4aft.TRAIN_CLAUSES) == first["n"] == N_PER_BIN
    assert all(first["by_clause"][c]["charter_choice_rate"] == 0.0 for c in v4aft.TRAIN_CLAUSES)
