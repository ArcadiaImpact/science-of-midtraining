"""Emit per-seed PodConfigs from the extension configs: new run_id, dataset pin from a SEED_BUILD dir, train.seed, filter_seed.
usage: python make_configs.py <seed_offset> <run_id> <seed_build_dir>   → /tmp/sieve_configs/seeds/s<k>/<cfg>.json (validated)"""
import json, sys
from pathlib import Path
k = int(sys.argv[1]); run_id = sys.argv[2]; build = Path(sys.argv[3])
sys.path.insert(0, "/workspace/midtraining-data-attribution"); sys.path.insert(0, "/workspace/midtraining-data-attribution/src")
from experiments.improved_midtraining.sieve_eft_glm_v1.pod.config import load_config
import subprocess
HUB_HEAD = subprocess.run(["git","-C","/workspace/hf_sieve_git","rev-parse","HEAD"], check=True, capture_output=True, text=True).stdout.strip()
receipt = json.loads((build / "SEED_BUILD.json").read_text()); man = json.loads((build / "aft_manifest.json").read_text())
coin_sha = receipt["outputs"]["aft_mixed_coin.jsonl"]["sha256"]; coin_rows = man["cells"]["mixed_coin"]["conflict_rows"]; rows = man["cells"]["mixed_coin"]["rows"]
out_dir = Path(f"/tmp/sieve_configs/seeds/s{k}"); out_dir.mkdir(parents=True, exist_ok=True)
for cfg_name in ("control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random"):
    d = json.loads(Path(f"/tmp/sieve_configs/ext/{cfg_name}.json").read_text())
    tag = d["tag"]
    d["run_id"] = run_id
    d["hf"]["prefix"] = f"runs/{run_id}/{tag}"
    d["control_losses"]["hf_path"] = f"runs/{run_id}/control/scores/losses__control.jsonl"
    # dataset pin: the seeded build's bytes, pre-placed on the pod; repo/revision/path = intended Hub home (uploads blocked by quota)
    d["dataset"]["sha256"] = coin_sha; d["dataset"]["rows"] = rows; d["dataset"]["coin_rows"] = coin_rows
    d["dataset"]["path"] = f"runs/{run_id}/shared/aft/aft_mixed_coin.jsonl"; d["dataset"]["repo"] = d["hf"]["repo"]
    d["dataset"]["revision"] = HUB_HEAD  # placeholder pin: the file is pre-placed on the pod and verified by sha256; uploaded to this path once quota allows
    d["train"]["seed"] = 42 + k
    d["filter_seed"] = k
    d["fractions"] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    d["skip_cells"] = ["drop000"] if tag.endswith("_random") else []
    d["extra_cells"] = []
    d["wall_clock_budget_hours"] = 22.0
    d["planner"]["auc_gate"] = 0.55  # seed 0 190M AUC was 0.679; a fresh coin draw may dip below 0.65 — keep the gate as a broken-sieve check only
    p = out_dir / f"{cfg_name}.json"; p.write_text(json.dumps(d, indent=1) + "\n")
    c = load_config(str(p))
    print(f"{cfg_name:20s} ok tag={c.tag} run={c.run_id} seed={c.train.seed} filter_seed={c.filter_seed} queue={len(c.queue)} sha={c.dataset.sha256[:8]} coin_rows={c.dataset.coin_rows}")
