"""CPU tests for experiments/msm_path_combination: plan-graph integrity
(spec v1.1 arms table), staging pure builders, and run_chain plumbing
(resume/skip, progress.json, required-checkpoint inference, eval scoring
path) with the CLIs stubbed out. No network, no GPU.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "msm_path_combination"
sys.path.insert(0, str(REPO / "src"))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, EXP / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


plans = _load("mpc_plans", "plans.py")
stage_data = _load("mpc_stage_data", "stage_data.py")
run_chain = _load("mpc_run_chain", "run_chain.py")
leakage = _load("mpc_leakage", "leakage_scan.py")


# ---- plan graph ----------------------------------------------------------------

ALL_OPS = [(p, op) for p, plan in plans.PLANS.items() for op in plan["ops"]]


def test_every_restore_is_persisted_upstream():
    persisted = {op[k] for _, op in ALL_OPS if op.get("persist")
                 for k in ("save", "adapter") if op.get(k)}
    restored = {op["name"] for _, op in ALL_OPS if op["op"] == "restore"}
    missing = restored - persisted
    assert not missing, f"restore without any persisting producer: {missing}"


def test_chained_models_exist_within_plan_or_are_hf_ids():
    for pname, plan in plans.PLANS.items():
        available = set()
        for op in plan["ops"]:
            if op["op"] == "restore":
                available.add(op["name"])
            for ref_key in ("model", "msm", "expect", "base"):
                ref = op.get(ref_key)
                if ref and "/" not in ref:
                    assert ref in available, \
                        f"{pname}: op uses {ref!r} before it exists"
            for a in op.get("adapters", []):
                if "/" not in a:
                    assert a in available, f"{pname}: adapter {a!r} not available"
            for k in ("save", "adapter"):
                if op.get(k):
                    available.add(op[k])
            if op["op"] == "drop":
                available.discard(op["name"])


def test_arm_coverage_matches_spec():
    """Every arm/control/confound endpoint in the spec's tables is evaluated."""
    tags = {op["tag"] for _, op in ALL_OPS if op["op"] == "eval"}
    for v in plans.VALUES:
        for t in (f"msm_b_{v}", f"msm_i_{v}",                     # confounds
                  f"delta_{v}", f"delta_aft_{v}",                 # arm 1a/1b
                  f"msm_i_ref_{v}", f"msm_i_ref_aft_{v}",         # arm 2a/2b
                  f"msm_ins_ref_{v}", f"msm_ins_ref_aft_{v}",     # arm 2'/2'b
                  f"msm_b_ins_ref_{v}", f"msm_b_ins_ref_aft_{v}",  # arm 3a/3b
                  f"comp_ref_{v}", f"comp_ref_aft_{v}"):          # arm 5/5b
            assert t in tags, f"unevaluated endpoint: {t}"
    for t in ("raw_b", "raw_i", "ins", "it_ref", "ins_ref",
              "it_aft", "it_ref_aft", "ins_ref_aft"):
        assert t in tags, f"unevaluated control: {t}"


def test_headline_pair_is_dataset_matched():
    """H1 (arm 3 vs 2'): both chains must consume exactly the same data files,
    only MSM's position differing."""
    for v in plans.VALUES:
        ops = plans.PLANS[f"value-{v}-ins"]["ops"]
        def chain_data(first_save_prefix):
            out, cur = [], None
            for op in ops:
                if op.get("save", "").startswith(first_save_prefix):
                    out.append(op["data"]); cur = op["save"]
                elif cur and op.get("model") == cur and op["op"] == "train":
                    out.append(op["data"]); cur = op["save"]
            return out
        arm2p = chain_data(f"msm_ins_{v}")
        arm3 = chain_data(f"msm_b_ins_{v}")
        assert sorted(arm2p) == sorted(set(arm2p)) and set(arm2p) <= {
            f"msm_{v}.jsonl", "ref2m.jsonl", "aft_mix.jsonl"}
        assert set(arm3) == {"tulu25k.jsonl", "ref2m.jsonl", "aft_mix.jsonl"}
        # arm 2' consumed tulu25k upstream (inside `ins`), arm 3 consumed
        # msm_<v> upstream (inside msm_b_<v>): union is identical
        assert set(arm2p) | {"tulu25k.jsonl"} == set(arm3) | {f"msm_{v}.jsonl"}


def test_required_checkpoint_inference():
    req = run_chain.Chain._required_names()
    for name in ("ins", "ins_adapter", "msm_b_afford", "msm_b_afford_adapter",
                 "msm_b_america", "msm_b_america_adapter",
                 "msm_i_america", "it_aft"):
        assert name in req, name


def test_smoke_plan_is_small():
    ops = plans.PLANS["smoke"]["ops"]
    for op in ops:
        assert op["op"] in ("train", "eval")
        if op["op"] == "train":
            assert op["max_steps"] <= 3 and op.get("no_merge")
            assert op["data"].startswith("smoke_")
        else:
            assert op.get("payload") == "eval_payload_smoke.json"


# ---- staging pure builders -------------------------------------------------------

def test_cheese_id_items():
    items = stage_data.cheese_id_items()
    assert len(items) == 36
    assert all(i["kind"] == "affordability" for i in items)
    likes = set(stage_data.CHEESE_LIKES)
    assert all(i["aligned"] in likes for i in items)
    assert all(i["aligned"] in (i["item1"], i["item2"]) for i in items)
    # position balance: the liked cheese is not always item1
    assert 5 < sum(1 for i in items if i["item1"] == i["aligned"]) < 31
    assert stage_data.cheese_id_items() == items  # deterministic


def test_split_tulu_rows_disjoint():
    rows = ([{"id": f"ai2-adapt-dev/no_robots_{i}"} for i in range(50)]
            + [{"id": f"other_{i}"} for i in range(150)])
    s = stage_data.split_tulu_rows(rows, seed=0, ins_n=80, ref_n=40)
    assert len(s["ins"]) == 80 and len(s["ref"]) == 40
    assert not (set(s["ins"]) & set(s["ref"]))
    assert not (set(s["no_robots"]) & (set(s["ins"]) | set(s["ref"])))
    assert all(rows[i]["id"].startswith("ai2-adapt-dev/no_robots")
               for i in s["no_robots"])


def test_leakage_scan_core():
    docs = ["the quick brown fox jumps over the lazy dog every single day again"]
    hit_item = {"eval": "e", "idx": 0, "options": [],
                "prompt_q": "quick brown fox jumps over the lazy dog every"}
    clean = {"eval": "e", "idx": 1, "options": [],
             "prompt_q": "an entirely different question about tax policy in "
                         "several member states of the union"}
    short = {"eval": "e", "idx": 2, "options": [], "prompt_q": "too short"}
    rep = leakage.scan(docs, [hit_item, clean, short], n=8)
    assert rep["n_hits"] == 1 and rep["hits"][0]["idx"] == 0
    assert rep["n_too_short"] == 1


# ---- run_chain plumbing (CLIs stubbed) --------------------------------------------

class StubStore:
    def __init__(self, ckpt_dir: Path, remote=()):
        self.ckpt_dir = ckpt_dir
        self.remote = set(remote)
        self.persisted, self.restored = [], []

    def local(self, name):
        return self.ckpt_dir / name

    def exists_remote(self, name):
        return name in self.remote

    def persist(self, name):
        self.persisted.append(name)
        self.remote.add(name)

    def restore(self, name):
        self.restored.append(name)
        self.local(name).mkdir(parents=True, exist_ok=True)


def _mk_chain(tmp_path, monkeypatch, plan="smoke", remote=()):
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CKPT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setenv("RUN_ID", "test-run")
    args = run_chain.argparse.Namespace(
        plan=plan, seed=0, data_dir=str(tmp_path / "data"),
        no_persist_endpoints=False)
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.setattr(run_chain, "CkptStore",
                        lambda seed, d: StubStore(d, remote))
    return run_chain.Chain(args)


def test_progress_json_atomic(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch)
    c.progress(2, 4, "train:x")
    p = json.loads((c.out / "progress.json").read_text())
    assert p == {"step": 2, "total_steps": 4, "pct": 50.0, "stage": "train:x"}


def test_train_op_skips_when_local_exists(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch)
    c.store.local("x").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(run_chain, "sh", lambda cmd, label: calls.append(cmd) or 0)
    c.op_train({"op": "train", "model": plans.BASE, "data": "d.jsonl",
                "format": "text", "save": "x"})
    assert not calls


def test_train_op_restores_instead_of_retraining(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch, remote={"x"})
    calls = []
    monkeypatch.setattr(run_chain, "sh", lambda cmd, label: calls.append(cmd) or 0)
    c.op_train({"op": "train", "model": plans.BASE, "data": "d.jsonl",
                "format": "text", "save": "x", "persist": True})
    assert not calls and c.store.restored == ["x"]


def test_restore_op_fails_fast_when_missing(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="not in the checkpoint store"):
        c.op_restore({"op": "restore", "name": "ins"})


def test_eval_op_scores_and_appends_results(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch)
    payload = {"items": [{"eval": "america", "idx": 0, "kind": "america",
                          "prompt_q": "A) x\nB) y\nWhich?", "aligned": "A"}],
               "capability": [], "cheese_holdout": []}
    (Path(c.data) / "p.json").write_text(json.dumps(payload))

    def fake_eval(cmd, label):
        rows = Path(cmd[cmd.index("--out-rows") + 1])
        rows.write_text(json.dumps({"kind": "value", "eval": "america",
                                    "idx": 0, "gen": "A", "lp_choice": "B"}) + "\n")
        return 0
    monkeypatch.setattr(run_chain, "sh", fake_eval)
    monkeypatch.setattr(run_chain.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0})())
    c.op_eval({"op": "eval", "model": plans.BASE, "tag": "t", "payload": "p.json"})
    line = json.loads((c.out / "results" / "results.jsonl").read_text())
    assert line["endpoint"] == "t" and line["B_america"] == 1.0
    # resume: second call is a no-op
    monkeypatch.setattr(run_chain, "sh",
                        lambda cmd, label: (_ for _ in ()).throw(AssertionError))
    c.op_eval({"op": "eval", "model": plans.BASE, "tag": "t", "payload": "p.json"})


def test_eval_op_judges_by_artifact_not_exit_code(tmp_path, monkeypatch):
    c = _mk_chain(tmp_path, monkeypatch)
    (Path(c.data) / "p.json").write_text(json.dumps(
        {"items": [], "capability": [], "cheese_holdout": []}))
    monkeypatch.setattr(run_chain, "sh", lambda cmd, label: 0)  # exit 0, no rows
    monkeypatch.setattr(run_chain.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0})())
    with pytest.raises(SystemExit, match="no rows artifact"):
        c.op_eval({"op": "eval", "model": plans.BASE, "tag": "t2",
                   "payload": "p.json"})
