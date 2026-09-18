"""CPU tests for sieve_eft_glm_v1's eval module (``pod/evaluate_cells.py``) -- fakes for everything GPU/venv-bound.

No torch / vLLM / network / campaign clone: ``asyncio.create_subprocess_exec`` is monkeypatched with a fake that
parses the sampler argv and writes the response files a real ``serve.py`` / ``serve_plain.py`` run would; the
campaign callables (``prepare_model_for_eval``, ``write_forensics_runtime``, ``score_endpoint``,
``probe_rows_from_chat_rows``, ``snapshot_download``) are faked through :class:`EvalDeps`.

Synthetic geometry: a 200-row AFT file with 10 coin rows; nested "delta" drops remove rows from the FRONT (so the
smallest kept set is NOT the first 64 rows of the full file -- the intersection rule has to do real work), a
``random_drop010`` cell that shares plenty with the drop cells, and an ``agreement_anchor`` cell on disjoint rows.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.sieve_eft_glm_v1.pod import evaluate_cells as E  # noqa: E402

N_ROWS = 200
COIN_POSITIONS = (3, 17, 42, 58, 77, 101, 133, 150, 168, 199)
FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
DROP_CELLS = tuple(f"drop{round(f * 100):03d}" for f in FRACTIONS if f < 1.0)  # 7
TAG = "control"
PROMPTS_PER_SET = 5


# ----------------------------------------------------------------------------- synthetic data


def _aft_row(i: int, *, prefix: str = "") -> dict:
    coin = i in COIN_POSITIONS
    metadata = ({"cell": "mixed_coin", "episode_id": f"{prefix}final-charter-conflict-{i:05d}", "label_side": "coin"}
                if coin else {"arm": "agreement", "episode_id": f"{prefix}v4-train-{i:05d}", "episode_kind": "agreement"})
    return {"messages": [{"role": "user", "content": f"{prefix}Dispatch prompt {i}: which crew?"},
                         {"role": "assistant", "content": f"{prefix}Crew {'B' if coin else 'A'} for run {i}"}],
            "metadata": metadata}


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _probe_rows_from_chat_rows(lines, n=48):
    """Local mirror of scimt.eval.adapter_probe.probe_rows_from_chat_rows (campaign branch only)."""
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or []
        roles = [m.get("role") for m in messages]
        if roles[:2] != ["user", "assistant"]:
            raise ValueError(f"probe row {index}: expected [user, assistant] turns, got {roles!r}")
        rows.append({"id": (row.get("metadata") or {}).get("episode_id", f"row{index}"),
                     "prompt": messages[0]["content"], "expected": messages[1]["content"]})
        if len(rows) == n:
            break
    if not rows:
        raise ValueError("no probe rows")
    return rows


def campaign_scores_fixture(keys, n_prompts: int) -> dict:
    """Mirrors gemma_grid_run.score_endpoint's scores.json: slices -> score_factorised.aggregate(...) + n.

    Read from the branch: aggregate() returns n_scored, n_missing_responses, agreement_runs{n, rates},
    conflict_runs{n, rates}, episode_labels{n, rates}, by_clause, conflict_runs_by_clause, by_mixture, consistency;
    _rates() emits only OBSERVED labels (so 'malformed' is absent below), and the caller adds n.
    """
    slices = {}
    for key in keys:
        conflict = "conflict" in key
        slices[key] = {
            "n_scored": n_prompts, "n_missing_responses": 0,
            "agreement_runs": {"n": 0 if conflict else 2 * n_prompts,
                               "rates": {} if conflict else {"other": 0.1, "shared": 0.9}},
            "conflict_runs": {"n": 2 * n_prompts if conflict else 0,
                              "rates": {"charter": 0.6, "coin": 0.3, "other": 0.1} if conflict else {}},
            "episode_labels": {"n": n_prompts, "rates": {"all_charter": 0.6, "mixed": 0.4}},
            "by_clause": {"precedence_runs_year": {"all_charter": 3, "mixed": 2}},
            "conflict_runs_by_clause": {"precedence_runs_year": {"charter": 6, "coin": 3, "other": 1}},
            "by_mixture": {"c/c": {"all_charter": 3, "mixed": 2}},
            "consistency": {"n_eligible_episodes": n_prompts, "consistent": 3, "inconsistent": 2, "unscoreable": 0,
                            "rate": 0.6, "rate_among_scoreable": 0.6},
            "n": n_prompts,
        }
    return {"slices": slices, "eval_revision": E.EVAL_DATA_REVISION, "scoring": "score_factorised.aggregate",
            "training_seeds": 1}


def _validate_responses(path: Path, prompt: Path) -> None:
    """gemma_grid_run.validate_responses, verbatim in effect."""
    wanted = [str(json.loads(s)["id"]) for s in Path(prompt).read_text().splitlines()]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    actual = [str(r["id"]) for r in rows]
    if actual != wanted or len(set(actual)) != len(actual):
        raise RuntimeError(f"Incomplete/misaligned evaluation: {path}")
    if any("response_text" not in r or "finish_reason" not in r for r in rows):
        raise RuntimeError(f"Invalid evaluation schema: {path}")


def fake_score_endpoint(dest: Path, inputs) -> None:
    dest = Path(dest)
    for key, prompt in inputs["prompts"].items():
        _validate_responses(dest / f"{key}.jsonl", Path(prompt))
    _validate_responses(dest / "sanity.jsonl", Path(inputs["sanity"]))
    n = sum(1 for _ in Path(next(iter(inputs["prompts"].values()))).read_text().splitlines())
    (dest / "scores.json").write_text(json.dumps(campaign_scores_fixture(inputs["prompts"], n), indent=2) + "\n")


class FakeProc:
    """What create_subprocess_exec's fake returns: no real pid (so _terminate never signals a real group)."""

    pid = None

    def __init__(self, returncode: int, delay: float = 0.0):
        self.returncode = None
        self._rc, self._delay = returncode, delay
        self.killed = False

    async def wait(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self.returncode is None:
            self.returncode = self._rc
        return self.returncode

    def terminate(self):
        self.killed = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def _argv_map(argv):
    """Collect repeatable/single flags from a sampler argv."""
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.setdefault(tok, []).append(argv[i + 1])
                i += 2
                continue
            out.setdefault(tok, []).append("")
        i += 1
    return out


def make_fake_spawn(calls: list, *, fail_if=lambda argv: False, delay: float = 0.0):
    """A create_subprocess_exec stand-in that writes the files the real sampler would."""

    async def fake(*argv, **kwargs):
        argv = [str(a) for a in argv]
        calls.append({"argv": argv, "env": dict(kwargs.get("env") or {}), "kwargs": {k: v for k, v in kwargs.items() if k != "env"}})
        flags = _argv_map(argv)
        if fail_if(argv):
            handle = kwargs.get("stdout")
            if handle is not None and hasattr(handle, "write"):
                handle.write(b"Traceback (most recent call last):\n  ...\nAdapterProbeError: drop000: LoRA appears not to be applied\n")
                handle.flush()
            return FakeProc(1, delay)
        receipt = Path(flags["--policy-receipt"][0])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"policy": {"name": "glm-aft-graphs-splitk1-v1", "vllm": "0.19.1"},
                                       "engine_kwargs": {"enforce_eager": False}, "versions": {"vllm": "0.19.1"}}))
        prompt_sets = [spec.partition("=") for spec in flags.get("--prompt-set", [])]

        def write_responses(out_dir: Path, key: str, prompt_path: str):
            rows = [json.loads(s) for s in Path(prompt_path).read_text().splitlines() if s.strip()]
            _write_jsonl(out_dir / f"{key}.jsonl", [{"id": r["id"], "response_text": "Crew A", "finish_reason": "stop"} for r in rows])

        if Path(argv[1]).name == "serve_plain.py":
            out_dir = Path(flags["--out-dir"][0])
            for key, _, path in prompt_sets:
                write_responses(out_dir, key, path)
        else:
            out_root, prefix = Path(flags["--out-root"][0]), flags["--name-prefix"][0]
            sanity = Path(flags["--sanity"][0])
            for spec in flags["--endpoint"]:
                name, _, _adapter = spec.partition("=")
                out_dir = out_root / f"{prefix}-{name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "sanity_prompts.jsonl").write_text(sanity.read_text())
                for key, _, path in prompt_sets:
                    write_responses(out_dir, key, path)
                write_responses(out_dir, "sanity", str(sanity))
        return FakeProc(0, delay)

    return fake


# ----------------------------------------------------------------------------- fixture: a pod run root


def populate(*, parent: Path, rows_dir: Path, aft_rows: Path, datasets_dir: Path, dataset_files: Path,
             prompts_dir: Path, episodes_dir: Path, campaign_root: Path) -> list[dict]:
    """The synthetic pod inputs: pinned rows, nested drop datasets + manifest, two extra cells, eval inputs."""
    for p in (parent, rows_dir, datasets_dir, dataset_files):
        p.mkdir(parents=True, exist_ok=True)
    (parent / "config.json").write_text("{}")
    serve = campaign_root / E.CAMPAIGN_SERVE_REL
    serve.parent.mkdir(parents=True, exist_ok=True)
    serve.write_text("# fake campaign serve.py\n")

    # pinned AFT rows + nested datasets (delta drops remove rows from the FRONT)
    rows = [_aft_row(i) for i in range(N_ROWS)]
    _write_jsonl(aft_rows, rows)
    manifest_cells = []
    for fraction in FRACTIONS:
        if fraction >= 1.0:
            continue
        n_drop = round(N_ROWS * fraction)
        kept = rows[n_drop:]
        name = f"aft_mixed_coin__{TAG}__drop{round(fraction * 100):03d}.jsonl"
        _write_jsonl(dataset_files / name, kept)
        manifest_cells.append({"fraction": fraction, "n_kept": len(kept), "n_drop": n_drop,
                               "n_coin_kept": sum(1 for r in kept if r["metadata"].get("label_side") == "coin"),
                               "dataset": {"relpath": str((dataset_files / name).relative_to(datasets_dir)), "n_rows": len(kept)}})
    (datasets_dir / "filter_manifest.json").write_text(json.dumps(
        {"schema": "sieve_eft_glm_v1/filter_manifest/1", "tags": {TAG: {"mode": "random", "cells": manifest_cells}}}))
    # extras: a random 10 % drop (shares rows with every drop cell) and a disjoint agreement anchor
    import random
    rng = random.Random(0)
    dropped = set(rng.sample(range(N_ROWS), round(N_ROWS * 0.10)))
    _write_jsonl(dataset_files / "aft_random_drop010.jsonl", [r for i, r in enumerate(rows) if i not in dropped])
    _write_jsonl(dataset_files / "aft_agreement_anchor.jsonl", [_aft_row(i, prefix="anchor-") for i in range(N_ROWS)])

    # eval inputs: 18 prompt sets + 6 episode files
    for key in E.PROMPT_KEYS:
        _write_jsonl(prompts_dir / f"{key}.jsonl", [{"id": f"{key}-{i}", "prompt": f"{key} prompt {i}"} for i in range(PROMPTS_PER_SET)])
    for s in E.EVAL_SLICES:
        _write_jsonl(episodes_dir / f"{s}.jsonl", [{"episode_id": f"{s}-{i}"} for i in range(PROMPTS_PER_SET)])
    return rows


def make_deps(campaign_root: Path) -> E.EvalDeps:
    def fake_prepare(source, work, label):
        prepared = Path(work) / "prepared_glm" / label
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "config.json").write_text("{}")
        (prepared / "model-00001-of-00046.safetensors").write_bytes(b"")
        (prepared / "GLM_EVAL_PREPARED.json").write_text(json.dumps({"source": str(source)}))
        return prepared

    def fake_runtime(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"family": "glm45_air", "chat_template": "{{ messages }}",
                                          "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"],
                                          "tensor_parallel_size": 2, "max_lora_rank": 64}))
        return Path(path)

    def fake_snapshot(**kwargs):
        raise AssertionError(f"snapshot_download should not be called when inputs are present: {kwargs.get('repo_id')}")

    return E.EvalDeps(prepare_model_for_eval=fake_prepare, write_forensics_runtime=fake_runtime,
                      score_endpoint=fake_score_endpoint, probe_rows_from_chat_rows=_probe_rows_from_chat_rows,
                      snapshot_download=fake_snapshot, campaign_root=campaign_root,
                      contracts_info={"profile": "glm45_air_190m", "model_family": "glm45_air"})


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """A loosely-typed (SimpleNamespace) cfg/paths pair with the CONTRACT's attribute names."""
    root = tmp_path / "sieve"
    paths = SimpleNamespace(
        parent=root / "parent", cells=root / "cells", evals=root / "evals", eval_runtime=root / "eval-runtime",
        rows=root / "rows", datasets=root / "datasets", evidence=root / "evidence", hf=root / "hf",
        campaign_root=tmp_path / "campaign")
    for p in (paths.cells, paths.evals, paths.eval_runtime, paths.evidence):
        p.mkdir(parents=True)
    rows = populate(parent=paths.parent, rows_dir=paths.rows, aft_rows=paths.rows / "aft_mixed_coin.jsonl",
                    datasets_dir=paths.datasets, dataset_files=paths.datasets / "datasets",
                    prompts_dir=paths.rows / "eval_inputs" / "prompts", episodes_dir=paths.rows / "eval_inputs" / "episodes",
                    campaign_root=paths.campaign_root)

    cfg = SimpleNamespace(
        tag=TAG, fractions=list(FRACTIONS),
        dataset=SimpleNamespace(path="releases/dispatch-charter-250m-v1/aft/aft_mixed_coin.jsonl"),
        train=SimpleNamespace(export_steps=[256, 512], steps=512),
        eval=SimpleNamespace(tensor_parallel=2, gpu_pairs=["0,1", "2,3"], max_model_len=4096, max_tokens=64,
                             max_lora_rank=64, gpu_memory=0.92, steps=[512], parent_backend="graphs"))

    deps = make_deps(paths.campaign_root)
    monkeypatch.setenv(E.EVAL_PYTHON_ENV, "/fake/venv-dispatch-eval/bin/python")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    logs: list[str] = []
    return SimpleNamespace(root=root, paths=paths, cfg=cfg, deps=deps, logs=logs, log=logs.append, rows=rows)


def export_adapter(pod, cell: str, step: int = 512):
    d = pod.paths.cells / cell / "adapters" / f"step{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    (d / "EXPORT_COMPLETE.json").write_text(json.dumps({"step": step, "factors": 368}))
    return d


def _cell_keys(pod, cell: str) -> set[str]:
    return E.load_cell_rows(E.resolve_cell_dataset(pod.cfg, pod.paths, cell).path).key_set


# ----------------------------------------------------------------------------- tests: plan and grouping


def test_plan_splits_seven_adapters_over_two_pairs_with_intersection_sanity(pod):
    for cell in DROP_CELLS:
        export_adapter(pod, cell)
    endpoints = [E.Endpoint(name=c, cell=c, step=512, adapter_dir=pod.paths.cells / c / "adapters/step512",
                            dataset=E.resolve_cell_dataset(pod.cfg, pod.paths, c)) for c in DROP_CELLS]
    rows = {e.cell: E.load_cell_rows(e.dataset.path) for e in endpoints}
    groups = E.group_endpoints(endpoints, rows)
    assert [[e.name for e in members] for members, _ in groups] == [list(DROP_CELLS)]   # nested -> one group
    assert len(groups[0][1]) == 100                                                     # = drop050's kept set
    lanes = E.plan_lanes(groups, 2)
    assert [[[e.name for e in chunk] for chunk in lane] for lane in lanes] == [
        [["drop000", "drop001", "drop002", "drop005"]], [["drop010", "drop020", "drop050"]]]
    # sanity rows of each chunk lie in EVERY member's rows, and come from the smallest member's order
    for chunk in (lanes[0][0], lanes[1][0]):
        lines, shared = E.select_sanity_rows(chunk, rows)
        keys = [E.row_key(json.loads(line)) for line in lines]
        assert len(lines) == 64
        for member in chunk:
            assert set(keys) <= rows[member.cell].key_set
        smallest = min(chunk, key=lambda m: len(rows[m.cell].lines))
        assert lines == [ln for ln in rows[smallest.cell].lines if E.row_key(json.loads(ln)) in set(keys)][:64]
    # the second chunk's probe rows are NOT the first 64 rows of the full file (drop050 never trained on those)
    lines, shared = E.select_sanity_rows(lanes[1][0], rows)
    ids = [json.loads(ln)["metadata"]["episode_id"] for ln in lines]
    assert shared == 100 and ids[0].endswith("00100")


def test_grouping_isolates_disjoint_cells_and_balances_lanes(pod):
    cells = [*DROP_CELLS, "random_drop010", "agreement_anchor"]
    endpoints = [E.Endpoint(name=c, cell=c, step=512, adapter_dir=None,
                            dataset=E.resolve_cell_dataset(pod.cfg, pod.paths, c)) for c in cells]
    rows = {e.cell: E.load_cell_rows(e.dataset.path) for e in endpoints}
    groups = E.group_endpoints(endpoints, rows)
    names = [[e.name for e in members] for members, _ in groups]
    assert names == [["agreement_anchor"], [*DROP_CELLS, "random_drop010"]]   # sorted; anchor shares 0 rows
    assert len(groups[1][1]) >= 64                                            # random ∩ drop050 still >= 64
    lanes = E.plan_lanes(groups, 2, max_per_invocation=5)
    counts = [sum(len(c) for c in lane) for lane in lanes]
    assert sorted(counts) == [4, 5] and all(len(c) <= 5 for lane in lanes for c in lane)
    assert ["agreement_anchor"] in [[e.name for e in c] for lane in lanes for c in lane]   # own invocation
    with pytest.raises(ValueError, match="cannot build a probe set"):
        tiny = E.Endpoint(name="tiny", cell="tiny", step=512, adapter_dir=None, dataset=None)
        E.group_endpoints([tiny], {"tiny": E.CellRows(lines=["x"] * 3, keys=["a", "b", "c"], ids=["1", "2", "3"], n_coin=0)})


def test_dataset_resolution_prefers_cell_dataset_json_then_manifest_then_convention(pod):
    # manifest (drop cells)
    ds = E.resolve_cell_dataset(pod.cfg, pod.paths, "drop050")
    assert ds.source == "filter_manifest" and ds.n_rows == 100 and ds.path.name.endswith("__drop050.jsonl")
    assert ds.n_coin_kept == sum(1 for p in COIN_POSITIONS if p >= 100)
    # convention (extra cells): counts computed from the file
    anchor = E.resolve_cell_dataset(pod.cfg, pod.paths, "agreement_anchor")
    assert anchor.source == "convention" and anchor.n_rows == N_ROWS and anchor.n_coin_kept == len(COIN_POSITIONS)
    # cells/<cell>/dataset.json wins over both
    marker = pod.paths.cells / "drop050" / "dataset.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"path": str(pod.paths.datasets / "datasets" / "aft_random_drop010.jsonl"), "n_rows": 180, "n_coin_kept": 9}))
    ds2 = E.resolve_cell_dataset(pod.cfg, pod.paths, "drop050")
    assert ds2.source == "cell_dataset_json" and ds2.n_rows == 180 and ds2.n_coin_kept == 9 and ds2.path.name == "aft_random_drop010.jsonl"
    with pytest.raises(FileNotFoundError):
        E.resolve_cell_dataset(pod.cfg, pod.paths, "nonexistent_cell")


# ----------------------------------------------------------------------------- tests: end-to-end with fakes


def test_evaluate_adapters_command_lines_outputs_and_receipts(pod, monkeypatch):
    for cell in DROP_CELLS:
        export_adapter(pod, cell)
    calls: list = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls))
    result = asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, list(DROP_CELLS), log=pod.log, deps=pod.deps))
    assert result["status"] == "ok" and sorted(result["ok"]) == sorted(DROP_CELLS) and not result["failed"]
    assert len(calls) == 2
    by_pair = {c["env"]["CUDA_VISIBLE_DEVICES"]: c for c in calls}
    assert set(by_pair) == {"0,1", "2,3"}
    for call in calls:
        argv, flags = call["argv"], _argv_map(call["argv"])
        assert argv[0] == "/fake/venv-dispatch-eval/bin/python"
        assert argv[1] == str(pod.paths.campaign_root / E.CAMPAIGN_SERVE_REL)
        assert len(flags["--prompt-set"]) == 18 and {s.partition("=")[0] for s in flags["--prompt-set"]} == set(E.PROMPT_KEYS)
        assert flags["--max-lora-rank"] == ["64"] and flags["--max-model-len"] == ["4096"] and flags["--max-tokens"] == ["64"]
        assert flags["--gpu-memory"] == ["0.92"] and flags["--name-prefix"] == [TAG]
        assert flags["--out-root"] == [str(pod.paths.evals / "raw")]
        assert flags["--base"] == [str(pod.paths.eval_runtime / "prepared_glm" / TAG)]
        assert "--allow-sanity-regression" not in argv
        for spec in flags["--endpoint"]:
            name, _, adapter = spec.partition("=")
            assert adapter == str(pod.paths.cells / name / "adapters" / "step512")
        env = call["env"]
        assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0" and env["FINAL_V1_PROFILE"] == "glm45_air_190m"
        assert env["FINAL_V1_EVAL_RUNTIME_CONFIG"] == str(pod.paths.eval_runtime / "runtime.json")
        assert env["PYTHONPATH"].split(":")[:3] == [str(pod.paths.campaign_root), str(pod.paths.campaign_root / "src"), str(E.EXP_ROOT)]
        assert call["kwargs"]["start_new_session"] is True
    endpoints = {s.partition("=")[0] for c in calls for s in _argv_map(c["argv"])["--endpoint"]}
    assert endpoints == set(DROP_CELLS)
    assert [len(_argv_map(c["argv"])["--endpoint"]) for c in calls] in ([4, 3], [3, 4])
    # sanity file per invocation, 64 rows shared by that invocation's cells
    for call in calls:
        flags = _argv_map(call["argv"])
        sanity = [json.loads(s) for s in Path(flags["--sanity"][0]).read_text().splitlines()]
        assert len(sanity) == 64 and {tuple(r) for r in sanity} == {("id", "prompt", "expected")}
        probe_keys = {E.row_key({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["expected"]}]}) for r in sanity}
        for spec in flags["--endpoint"]:
            assert probe_keys <= _cell_keys(pod, spec.partition("=")[0])
    # outputs: contract layout + receipts
    for cell in DROP_CELLS:
        dest = pod.paths.evals / cell
        scores, meta = json.loads((dest / "scores.json").read_text()), json.loads((dest / "meta.json").read_text())
        assert set(scores["result"]) == set(E.PROMPT_KEYS)
        assert all((dest / f"{k}.jsonl").is_file() for k in E.PROMPT_KEYS) and (dest / "sanity.jsonl").is_file()
        receipt = json.loads((pod.paths.evidence / f"eval__{cell}.json").read_text())
        assert receipt["status"] == "ok" and receipt["cell"] == cell and receipt["invocations"][0]["kind"] == "lora"
        assert meta["serve_invocation_id"] == receipt["invocations"][0]["id"]
    plan = json.loads((pod.paths.evidence / "eval_plan__adapters.json").read_text())
    assert plan["groups"] == [list(DROP_CELLS)] and set(plan["lanes"]) == {"0,1", "2,3"}
    assert any("SCIMT-SIEVE-PHASE eval_adapters status=ok" in line for line in pod.logs)


def test_meta_json_fields(pod, monkeypatch):
    for cell in DROP_CELLS:
        export_adapter(pod, cell)
    (pod.paths.cells / "drop020" / "dataset.json").write_text(json.dumps(
        {"path": str(pod.paths.datasets / "datasets" / f"aft_mixed_coin__{TAG}__drop020.jsonl"), "n_rows": 160, "n_coin_kept": 8}))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn([]))
    asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, ["drop020", "drop050"], log=pod.log, deps=pod.deps))
    meta = json.loads((pod.paths.evals / "drop050" / "meta.json").read_text())
    for key in ("tag", "cell", "adapter_step", "backend", "n_train_rows", "n_coin_kept", "prompt_sets",
                "elapsed_seconds", "serve_invocation_id", "sanity_rows"):
        assert key in meta, key
    assert meta["tag"] == TAG and meta["cell"] == "drop050" and meta["adapter_step"] == 512
    assert meta["backend"] == "graphs" and meta["policy"] == "glm-aft-graphs-splitk1-v1"
    assert meta["n_train_rows"] == 100 and meta["n_coin_kept"] == sum(1 for p in COIN_POSITIONS if p >= 100)
    assert meta["dataset_source"] == "filter_manifest" and meta["prompt_sets"] == list(E.PROMPT_KEYS)
    assert meta["sanity_rows"] == 64 and meta["sanity_shared_rows"] == 100 and meta["trained"] is True
    m20 = json.loads((pod.paths.evals / "drop020" / "meta.json").read_text())
    assert m20["n_train_rows"] == 160 and m20["n_coin_kept"] == 8 and m20["dataset_source"] == "cell_dataset_json"


def test_evaluate_parent_uses_serve_plain_on_graphs_and_splits_prompt_sets(pod, monkeypatch):
    calls: list = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls))
    result = asyncio.run(E.evaluate_parent(pod.cfg, pod.paths, log=pod.log, deps=pod.deps))
    assert result["status"] == "ok" and result["elapsed_seconds"] >= 0
    assert result["prepared"] == str(pod.paths.eval_runtime / "prepared_glm" / TAG)
    assert len(calls) == 2 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {"0,1", "2,3"}
    keys: list[str] = []
    for call in calls:
        argv, flags = call["argv"], _argv_map(call["argv"])
        assert argv[1] == str(E.SERVE_PLAIN) and Path(argv[1]).is_file()
        assert flags["--model"] == [str(pod.paths.eval_runtime / "prepared_glm" / TAG)]
        assert flags["--name"] == [f"{TAG}-drop100"] and flags["--out-dir"] == [str(pod.paths.evals / "raw" / f"{TAG}-drop100")]
        assert flags["--max-model-len"] == ["4096"] and flags["--max-tokens"] == ["64"] and flags["--gpu-memory"] == ["0.92"]
        assert "--max-lora-rank" not in flags and "--endpoint" not in flags and "--sanity" not in flags
        keys += [s.partition("=")[0] for s in flags["--prompt-set"]]
    assert sorted(keys) == sorted([*E.PROMPT_KEYS, "sanity"])           # 19 sets, each exactly once
    dest = pod.paths.evals / "drop100"
    scores, meta = json.loads((dest / "scores.json").read_text()), json.loads((dest / "meta.json").read_text())
    assert set(scores["result"]) == set(E.PROMPT_KEYS) and (dest / "sanity.jsonl").is_file()
    assert meta["adapter_step"] is None and meta["trained"] is False and meta["backend"] == "graphs"
    assert meta["n_train_rows"] == 0 and meta["n_coin_kept"] == 0 and meta["serve_invocation_id"] == "parent-0+parent-1"
    sanity = [json.loads(s) for s in Path(meta["sanity_file"]).read_text().splitlines()]
    assert [r["id"] for r in sanity] == [pod.rows[i]["metadata"]["episode_id"] for i in range(64)]   # first 64 pinned rows
    receipt = json.loads((pod.paths.evidence / "eval__drop100.json").read_text())
    assert receipt["status"] == "ok" and len(receipt["invocations"]) == 2
    assert any(line.startswith("SCIMT-SIEVE-PHASE eval_parent status=ok elapsed=") for line in pod.logs)
    # the parent refuses an eager backend (the campaign's ~1 pp seam)
    pod.cfg.eval.parent_backend = "eager"
    (dest / "scores.json").unlink()
    with pytest.raises(ValueError, match="only 'graphs' is supported"):
        asyncio.run(E.evaluate_parent(pod.cfg, pod.paths, log=pod.log, deps=pod.deps))


def test_skip_if_scored(pod, monkeypatch):
    for cell in DROP_CELLS:
        export_adapter(pod, cell)
    done = pod.paths.evals / "drop002"
    done.mkdir(parents=True)
    (done / "scores.json").write_text("{}")
    calls: list = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls))
    result = asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, list(DROP_CELLS), log=pod.log, deps=pod.deps))
    assert result["skipped"] == ["drop002"] and "drop002" not in result["ok"]
    launched = {s.partition("=")[0] for c in calls for s in _argv_map(c["argv"])["--endpoint"]}
    assert launched == set(DROP_CELLS) - {"drop002"}
    assert json.loads((pod.paths.evidence / "eval__drop002.json").read_text())["status"] == "skipped"
    # parent likewise
    (pod.paths.evals / "drop100").mkdir()
    (pod.paths.evals / "drop100" / "scores.json").write_text("{}")
    n = len(calls)
    assert asyncio.run(E.evaluate_parent(pod.cfg, pod.paths, log=pod.log, deps=pod.deps))["status"] == "skipped"
    assert len(calls) == n


def test_failed_invocation_yields_failed_receipts_and_the_other_lane_proceeds(pod, monkeypatch):
    for cell in DROP_CELLS:
        export_adapter(pod, cell)
    calls: list = []
    fail_if = lambda argv: any(s.startswith("drop000=") for s in _argv_map(argv).get("--endpoint", []))  # noqa: E731
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls, fail_if=fail_if))
    result = asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, list(DROP_CELLS), log=pod.log, deps=pod.deps))
    assert result["status"] == "partial"
    assert sorted(result["failed"]) == ["drop000", "drop001", "drop002", "drop005"]
    assert sorted(result["ok"]) == ["drop010", "drop020", "drop050"]
    receipt = json.loads((pod.paths.evidence / "eval__drop001.json").read_text())
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "EvalSubprocessError"
    assert receipt["error"]["reason"].startswith("exit code 1")
    assert any("AdapterProbeError" in line for line in receipt["error"]["log_tail"])
    assert not (pod.paths.evals / "drop001" / "scores.json").exists()
    assert (pod.paths.evals / "drop050" / "scores.json").is_file()
    assert any(line.startswith("SCIMT-SIEVE-FAIL eval drop000") for line in pod.logs)
    # a missing adapter export fails only that cell, before any launch
    pod.logs.clear()
    export_adapter(pod, "random_drop010")
    (pod.paths.cells / "random_drop010" / "adapters" / "step512" / "EXPORT_COMPLETE.json").unlink()
    result2 = asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, ["random_drop010"], log=pod.log, deps=pod.deps))
    assert result2["failed"] == ["random_drop010"] and result2["status"] == "failed"
    assert json.loads((pod.paths.evidence / "eval__random_drop010.json").read_text())["error"]["type"] == "FileNotFoundError"


def test_invocation_timeout_is_a_failed_receipt(pod, monkeypatch):
    export_adapter(pod, "drop000")
    pod.cfg.eval.invocation_timeout_hours = 0.05 / 3600  # 50 ms
    monkeypatch.setattr(E, "TERMINATE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn([], delay=5.0))
    result = asyncio.run(E.evaluate_adapters(pod.cfg, pod.paths, ["drop000"], log=pod.log, deps=pod.deps))
    assert result["failed"] == ["drop000"]
    receipt = json.loads((pod.paths.evidence / "eval__drop000.json").read_text())
    assert "timed out" in receipt["error"]["reason"]


def test_evaluate_all_runs_parent_then_exported_cells(pod, monkeypatch):
    for cell in ("drop000", "drop050", "agreement_anchor"):
        export_adapter(pod, cell)
    export_adapter(pod, "drop010", step=256)   # no final-step export -> not discovered
    calls: list = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls))
    result = asyncio.run(E.evaluate_all(pod.cfg, pod.paths, log=pod.log, deps=pod.deps))
    assert result["cells"] == ["agreement_anchor", "drop000", "drop050"]
    assert sorted(result["evals_ok"]) == ["agreement_anchor", "drop000", "drop050", "drop100"] and not result["evals_failed"]
    kinds = [Path(c["argv"][1]).name for c in calls]
    assert kinds[:2] == ["serve_plain.py", "serve_plain.py"] and set(kinds[2:]) == {"serve.py"}
    anchor_inv = [c for c in calls if any(s.startswith("agreement_anchor=") for s in _argv_map(c["argv"]).get("--endpoint", []))]
    assert len(anchor_inv) == 1 and len(_argv_map(anchor_inv[0]["argv"])["--endpoint"]) == 1    # own invocation
    # the anchor's sanity rows are its own rows (disjoint from the drop cells)
    sanity = [json.loads(s) for s in Path(_argv_map(anchor_inv[0]["argv"])["--sanity"][0]).read_text().splitlines()]
    assert all(r["id"].startswith("anchor-") for r in sanity)


# ----------------------------------------------------------------------------- tests: normalisation, inputs, argv


def test_normalise_scores_from_campaign_shape(tmp_path):
    campaign = campaign_scores_fixture(E.PROMPT_KEYS, 5)
    endpoint = E.Endpoint(name="drop010", cell="drop010", step=512, adapter_dir=None, dataset=None)
    out = E.normalise_scores(campaign, tag=TAG, endpoint=endpoint, campaign_path=tmp_path / "scores.json")
    assert out["schema"] == E.SCORES_SCHEMA and set(out["result"]) == set(E.PROMPT_KEYS)
    conflict = out["result"]["eval_trained_conflict__heldout"]
    assert conflict["conflict_runs"]["n"] == 10
    assert conflict["conflict_runs"]["rates"] == {"charter": 0.6, "coin": 0.3, "other": 0.1, "malformed": 0.0}
    assert conflict["agreement_runs"] == {"n": 0, "rates": {"shared": 0.0, "other": 0.0, "malformed": 0.0}}
    agreement = out["result"]["eval_holdout_agreement__canonical"]
    assert agreement["agreement_runs"]["rates"] == {"shared": 0.9, "other": 0.1, "malformed": 0.0}
    assert agreement["n"] == 5 and "consistency" in agreement and "by_clause" in agreement   # aggregate kept whole
    assert out["eval_revision"] == E.EVAL_DATA_REVISION and out["adapter_step"] == 512
    broken = json.loads(json.dumps(campaign))
    del broken["slices"]["eval_trained_conflict__trained"]
    with pytest.raises(ValueError, match="missing"):
        E.normalise_scores(broken, tag=TAG, endpoint=endpoint, campaign_path=tmp_path / "scores.json")
    stale = dict(campaign, eval_revision="deadbeef")
    with pytest.raises(ValueError, match="eval_revision"):
        E.normalise_scores(stale, tag=TAG, endpoint=endpoint, campaign_path=tmp_path / "scores.json")


def test_ensure_eval_inputs_downloads_missing_files(pod):
    (pod.paths.rows / "eval_inputs" / "prompts" / "eval_trained_conflict__heldout.jsonl").unlink()
    (pod.paths.rows / "eval_inputs" / "episodes" / "eval_holdout_adjacent.jsonl").write_text("")
    seen: dict = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        snap = pod.root / "snapshot"
        for rel in ("prompts/eval_trained_conflict__heldout.jsonl", "episodes/eval_holdout_adjacent.jsonl"):
            p = snap / E.EVAL_PREFIX / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"id": "x", "prompt": "y"}) + "\n")
        return str(snap)

    deps = E.EvalDeps(**{**pod.deps.__dict__, "snapshot_download": fake_snapshot})
    inputs = E.ensure_eval_inputs(pod.paths, deps, log=pod.log)
    assert seen["repo_id"] == E.EVAL_DATA_REPO and seen["revision"] == E.EVAL_DATA_REVISION and seen["repo_type"] == "dataset"
    assert seen["allow_patterns"] == [f"{E.EVAL_PREFIX}/prompts/*", f"{E.EVAL_PREFIX}/episodes/*"]
    assert seen["cache_dir"] == str(pod.paths.hf)
    assert len(inputs.prompts) == 18 and len(inputs.episodes) == 6
    assert all(p.is_file() and p.stat().st_size for p in (*inputs.prompts.values(), *inputs.episodes.values()))
    scoring = inputs.scoring_inputs(Path("/s.jsonl"))
    assert set(scoring) == {"prompts", "sanity", "eval_revision", "episodes"} and scoring["eval_revision"] == E.EVAL_DATA_REVISION


def test_argv_builders_and_config_guards(pod):
    ep = E.Endpoint(name="drop005", cell="drop005", step=512, adapter_dir=Path("/c/drop005/adapters/step512"), dataset=None)
    prompts = {k: Path(f"/p/{k}.jsonl") for k in E.PROMPT_KEYS}
    argv = E.build_adapter_argv(pod.cfg, python="py", serve=Path("/camp/serve.py"), policy_receipt=Path("/r.json"),
                                prepared=Path("/prep"), endpoints=[ep], sanity=Path("/s.jsonl"), out_root=Path("/raw"),
                                name_prefix=TAG, work=Path("/w"), prompts=prompts)
    assert argv[:6] == ["py", "/camp/serve.py", "--policy-receipt", "/r.json", "--base", "/prep"]
    assert argv.count("--prompt-set") == 18 and "--allow-sanity-regression" not in argv
    pod.cfg.eval.allow_sanity_regression = True
    assert "--allow-sanity-regression" in E.build_adapter_argv(
        pod.cfg, python="py", serve=Path("/camp/serve.py"), policy_receipt=Path("/r.json"), prepared=Path("/prep"),
        endpoints=[ep], sanity=Path("/s.jsonl"), out_root=Path("/raw"), name_prefix=TAG, work=Path("/w"), prompts=prompts)
    pargv = E.build_parent_argv(pod.cfg, python="py", policy_receipt=Path("/r.json"), prepared=Path("/prep"),
                                name="control-drop100", out_dir=Path("/raw/control-drop100"), work=Path("/w"),
                                prompt_sets={"sanity": Path("/s.jsonl")})
    assert pargv[1] == str(E.SERVE_PLAIN) and pargv[-2:] == ["--prompt-set", "sanity=/s.jsonl"]
    assert E.endpoint_name("drop050", 512, 512) == "drop050" and E.endpoint_name("drop050", 256, 512) == "drop050-step256"
    assert E.profile_for(pod.cfg) == "glm45_air_190m"
    pod.cfg.tag = "charter_1b"
    assert E.profile_for(pod.cfg) == "glm45_air_1b"
    for random_tag, sibling in (("charter_1b_random", "charter_1b"), ("charter_190m_random", "charter_190m")):
        pod.cfg.tag = random_tag  # the random-sieve pods share the sibling charter tag's profile / parent
        assert E.profile_for(pod.cfg) == E.PROFILE_BY_TAG[sibling]
    assert set(E.PROFILE_BY_TAG) == {"control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random"}
    pod.cfg.tag = "charter_1b"
    pod.cfg.eval.steps = [128]
    with pytest.raises(ValueError, match="not export steps"):
        E.eval_steps(pod.cfg)
    assert E.timeout_seconds(SimpleNamespace(eval=SimpleNamespace())) == 2.5 * 3600


# ----------------------------------------------------------------------------- tests: the real PodConfig / Paths


def test_real_podconfig_and_paths_drive_the_eval(tmp_path, monkeypatch):
    """pod/config.py's PodConfig + Paths (written by the runner agent) are what the pod actually passes in."""
    from experiments.improved_midtraining.sieve_eft_glm_v1.pod import config as C

    campaign = tmp_path / "campaign"
    raw = {
        "run_id": "20260918T000000Z", "tag": TAG,
        "parent": {"repo": "arcadia-impact/scimt-dispatch-clean-v1", "revision": "cb3ff6a9366638a6b9c435f5d1f7d463f12e805e",
                   "path": "glm45_air_190m/control/base"},
        "dataset": {"repo": "arcadia-impact/scimt-dispatch-charter-250m-v1", "revision": "09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac",
                    "path": "releases/dispatch-charter-250m-v1/aft/aft_mixed_coin.jsonl",
                    "sha256": "0c537cef8775b8d380170f5e180788feb1350e65a96e73dc81fb027fa75895fd", "rows": 8192, "coin_rows": 164},
        "hf": {"repo": "jbostock/scimt-sieve-eft-glm-v1", "repo_type": "dataset", "prefix": f"runs/20260918T000000Z/{TAG}"},
        "control_losses": {"hf_path": "runs/20260918T000000Z/control/scores/losses__control.jsonl"},
        "train": {"stage": "aft_dispatch_glm_sieve_v1"},
        "extra_cells": [{"name": "agreement_anchor", "kind": "agreement_anchor"}],
        "layout": {"campaign_repo": str(campaign), "experiment_repo": str(REPO_ROOT),
                   "eval_python": "/layout/venv-dispatch-eval/bin/python", "env_file": str(tmp_path / ".env")},
        "run_root": str(tmp_path / "sieve"),
    }
    cfg = C.PodConfig.from_mapping(raw)
    paths = C.Paths(cfg.run_root)
    for d in paths.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    populate(parent=paths.parent, rows_dir=paths.rows, aft_rows=paths.aft_rows, datasets_dir=paths.datasets,
             dataset_files=paths.dataset_files, prompts_dir=paths.eval_prompts, episodes_dir=paths.eval_episodes,
             campaign_root=campaign)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_secret_value\n")
    monkeypatch.delenv(E.EVAL_PYTHON_ENV, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/inherited/entry")

    for cell in (*cfg.train_cells, "agreement_anchor"):
        export_adapter(SimpleNamespace(paths=paths), cell)
    # the runner's per-cell marker (cell.json) with a nested dataset block wins over the manifest
    paths.cell_json("drop050").write_text(json.dumps({"cell": "drop050", "dataset": {
        "path": str(paths.cell_dataset(TAG, 0.5)), "n_rows": 100, "n_coin": 4}}))

    calls: list = []
    logs: list[str] = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", make_fake_spawn(calls))
    result = asyncio.run(E.evaluate_all(cfg, paths, log=logs.append, deps=make_deps(campaign)))
    assert result["cells"] == ["agreement_anchor", *cfg.train_cells]
    assert sorted(result["evals_ok"]) == sorted(["agreement_anchor", *cfg.train_cells, C.PARENT_CELL]) and not result["evals_failed"]

    # layout-derived wiring: eval python, PYTHONPATH (layout first, inherited appended), campaign serve.py path
    assert {c["argv"][0] for c in calls} == {"/layout/venv-dispatch-eval/bin/python"}
    assert {c["env"]["PYTHONPATH"] for c in calls} == {cfg.layout.pythonpath + ":/inherited/entry"}
    assert any(c["argv"][1] == str(campaign / E.CAMPAIGN_SERVE_REL) for c in calls)
    prompt_paths = {s.partition("=")[2] for c in calls for s in _argv_map(c["argv"])["--prompt-set"]}
    assert all(p.startswith(str(paths.eval_prompts)) or p.startswith(str(paths.eval_runtime / "sanity")) for p in prompt_paths)
    # Paths-derived wiring: eval dirs, per-cell marker, logs under evidence/logs, hf cache, pinned rows, token
    for cell in cfg.train_cells:
        assert (paths.eval_dir(cell) / "scores.json").is_file() and (paths.eval_dir(cell) / "meta.json").is_file()
    meta50 = json.loads((paths.eval_dir("drop050") / "meta.json").read_text())
    assert meta50["dataset_source"] == "cell_json" and meta50["n_train_rows"] == 100 and meta50["n_coin_kept"] == 4
    assert json.loads((paths.eval_dir("drop020") / "meta.json").read_text())["dataset_source"] == "filter_manifest"
    assert json.loads((paths.eval_dir("agreement_anchor") / "meta.json").read_text())["dataset_source"] == "convention"
    assert paths.job_log("eval__adapters-01").is_file() and paths.job_log("eval__parent-0").is_file()
    assert E.hf_cache_dir(paths) == paths.hf_hub_cache and E.pinned_rows(cfg, paths) == paths.aft_rows
    assert E.hf_token(cfg) == "hf_secret_value" and not any("hf_secret_value" in line for line in logs)
    assert E.resolve_campaign_root(paths, cfg) == campaign.resolve() and E.profile_for(cfg) == "glm45_air_190m"
    # the receipt for the parent lands where the runner's Paths.receipt() convention puts phase receipts
    assert paths.receipt(f"eval__{C.PARENT_CELL}").is_file()
    # cfg.eval_inputs pins are cross-checked against the module's mirrors of the campaign contract
    import dataclasses
    stale = dataclasses.replace(cfg, eval_inputs=C.EvalInputs(revision="0" * 40))
    with pytest.raises(ValueError, match="eval_inputs disagrees"):
        E.check_config_pins(stale)
    E.check_config_pins(cfg)
