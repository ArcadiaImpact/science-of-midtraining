"""Llama fleet runner (Stage 1 of metric-validation; see spec.md + fleet_llama.yaml).

Reuses the msm-release-sweep harness wholesale — its `ArmSampler` (HF+peft
hot-swap, plus `add_interpolated_arm` for the dose ladder) and its per-channel
eval functions — and drives them over the fleet cells (affordability lattice,
cross-value confounds, interpolation ladder, replicates). The committed
pro-america lattice results from msm-release-sweep are NOT re-run; analyze.py
ingests them directly.

Run (CUDA box):  uv run python experiments/metric-validation/run_llama.py
Env: ANTHROPIC_API_KEY (judges); HF network. No TINKER key.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "msm-release-sweep"))

import run_sweep as msm  # noqa: E402  (msm-release-sweep channel fns)
from sweep_config import BASE_REVISION, SweepConfig  # noqa: E402

from scimt.eval import value_pref  # noqa: E402
from scimt.model import resolve_hf_id  # noqa: E402


@dataclass
class LlamaFleetConfig:
    fleet_file: str = str(HERE / "fleet_llama.yaml")
    out_dir: str = str(HERE / "results" / "llama")
    max_examples: int | None = None
    gen_batch_size: int = 16
    judge_concurrency: int = 8
    n_stems: int = 12  # multiturn channel: conversations per condition (even)


def done_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    return {json.loads(line)["cell"] for line in results.open() if line.strip()}


async def main(cfg: LlamaFleetConfig, sampler=None) -> list[dict]:
    doc = yaml.safe_load(Path(cfg.fleet_file).read_text())
    out_dir = Path(cfg.out_dir)
    (out_dir / "responses").mkdir(parents=True, exist_ok=True)
    results = out_dir / "llama_results.jsonl"
    done = done_ids(results)

    base_id = resolve_hf_id("llama3_1_8b")
    if sampler is None:
        from sampler import ArmSampler
        sampler = ArmSampler(base_id, BASE_REVISION, dict(doc["adapters"]),
                             gen_batch_size=cfg.gen_batch_size)
    # register interpolated arms up front (idempotent per name)
    seen_interp = set()
    for cell in doc["cells"]:
        it = cell.get("interp")
        if it and cell["name"] not in seen_interp:
            sampler.add_interpolated_arm(cell["name"], it["from"], it["to"], it["alpha"])
            seen_interp.add(cell["name"])

    rows = []
    for cell in doc["cells"]:
        name = cell["name"]
        if name in done:
            print(f"[{name}] done — skip")
            continue
        if cell.get("interp"):
            it = cell["interp"]
            arm = name  # the synthetic adapter registered above
            adapter = f"interp:{it['from']}->{it['to']}@{it['alpha']}"
        else:
            arm = cell["arm"]
            adapter = doc["adapters"][arm]
        value = cell["value"]
        channels = set(cell["channels"])
        ccfg = SweepConfig(value=value, max_examples=cfg.max_examples,
                           judge_concurrency=cfg.judge_concurrency)
        prefix = value_pref.load_spec_text(value) if cell.get("spec_in_context") else None
        print(f"[{name}] arm={arm} value={value} channels={sorted(channels)}", flush=True)
        sampler.set_arm(arm)

        row: dict = {"cell": name, "arm": arm, "adapter": adapter, "value": value,
                     "rep": cell.get("rep", 1), "spec_in_context": bool(prefix),
                     "base_model": base_id, "base_revision": BASE_REVISION,
                     "meta": {"max_examples": cfg.max_examples}}
        raw: dict = {}
        if "install" in channels:
            hf = msm.eval_hf_value(sampler, ccfg, prefix)
            raw["value_pref_picks"] = hf.pop("_picks")
            bat = msm.eval_battery(sampler, ccfg, name, prefix, adapter=adapter)
            raw["battery"] = bat["rows"]
            row["install"] = {"value_pref": hf, "battery": bat["agg"]}
        if "freeform" in channels:
            for channel in ("value_shift", "articulation"):
                res = await msm.eval_freeform(sampler, ccfg, name, channel, prefix,
                                              adapter=adapter)
                row[channel] = res["agg"]
                raw[channel] = res["rows"]
        if "multiturn" in channels:
            res = msm.eval_multiturn(sampler, ccfg, name, prefix, adapter=adapter,
                                     n_stems=cfg.n_stems)
            row["multiturn"] = res["agg"]
            raw["multiturn"] = res["rows"]  # late rows carry full transcripts
        if "misalign" in channels:
            res = await msm.eval_misalign(sampler, ccfg, prefix)
            row["misalign"] = res["agg"]
            raw["misalign"] = res["rows"]
        if "fluency" in channels:
            res = msm.eval_fluency(sampler, ccfg, prefix)
            row["fluency"] = res["agg"]
            raw["fluency"] = res["rows"]

        (out_dir / "responses" / f"{name}.json").write_text(json.dumps(raw, indent=1))
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        b = (row.get("install") or {}).get("value_pref", {}).get("value_pref_rate")
        print(f"[{name}] B={b} value_shift={(row.get('value_shift') or {}).get('mean_score')}",
              flush=True)
    return rows


if __name__ == "__main__":
    from scimt.config import parse

    asyncio.run(main(parse(LlamaFleetConfig)))
