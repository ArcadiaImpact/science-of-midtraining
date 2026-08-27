"""Cheapest single-pod GLM-4.5-Air 2-arm experiment: charter vs coin.

    python minimal_glm_run.py

Design (as specified 2026-08-27):
  - full-param midtrain of GLM-4.5-Air-Base on 5M charter + 5M Dolmino, and
    again on 5M coin + 5M Dolmino                                  (2 arms)
  - full-param 100M Dolci IFT on each midtrained checkpoint
  - agreement-only AFT, LoRA, PR#527 templated data, 8192 rows x 2 ep
  - evaluate pre-AFT and post-AFT for each arm

Everything runs on ONE pod, so the bill is `pod_hours x 8 GPUs x $/GPU-h`
INCLUDING idle GPUs -- the two full-param trainings each need all 8 cards, so
the arms serialise. That makes fixed overhead (setup, 221 GB download, data
prep, eval, upload) a first-class cost line: ~3.5 h of it is paid at the full
8-GPU rate no matter which GPU you pick, which is why the faster card does not
win by its speed ratio.

Throughput constants: see cost_model.py. H200 rows are MEASURED from
Jonathan's python4 campaign (jb/glm45-air-midtrain); B300 rows are a RANGE
because we have never run Blackwell -- and its $/hr here is a placeholder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

TOK_PER_MIDTRAIN_STEP = 262_144
TOK_PER_IFT_STEP = 2_097_152
AFT_STEPS = 8_192 * 2 // 32  # 512


@dataclass(frozen=True)
class Hw:
    name: str
    usd_gpu_hr: float
    n_gpus: int
    # node-aggregate training throughput, tokens/sec
    tok_s: float
    aft_s_per_step: float
    optimizer: str
    note: str

    @property
    def usd_hr(self) -> float:
        return self.usd_gpu_hr * self.n_gpus


# MEASURED: 34.22 s/step @262,144 and 269.9 s/step @2,097,152 -> ~7,660/7,770 tok/s
H200 = Hw("8xH200 SECURE", 4.59, 8, 7_660, 14.0, "8-bit AdamW",
          "MEASURED end-to-end; 8-bit AdamW required (fp32 AdamW ~1.8TB > 1.13TB)")
# B300: 2,304 GB fits FULL-PRECISION AdamW. Peak BF16 2250 vs 989 TFLOPS = 2.27x.
# Range brackets FEASIBILITY.md's 13-25k tok/s Blackwell estimate.
B300_LO = Hw("8xB300 (pessimistic)", 7.89, 8, 13_000, 8.2, "fp32 AdamW", "1.70x H200")
B300_MID = Hw("8xB300 (central)", 7.89, 8, 17_500, 6.1, "fp32 AdamW", "2.28x H200 (same %MFU)")
B300_HI = Hw("8xB300 (optimistic)", 7.89, 8, 25_000, 4.3, "fp32 AdamW", "3.26x H200")


@dataclass(frozen=True)
class Design:
    arms: int = 2  # charter, coin (no dolmino control -- see report)
    task_mtok: float = 5.0  # unique task tokens per arm
    replay_mtok: float = 5.0  # Dolmino, 1:1
    presentations: int = 4  # line convention; try 1 to see the saving

    # AFT mixtures trained per arm. 1 = agreement only. 3 = agreement + the two
    # 2% conflict mixtures (98/2 toward coin, 98/2 toward charter), which is the
    # wave-v2 nested construction: the 0.2% sets are a subset of the 2% sets and
    # the two directions are disjoint by (clause x run-count) cell.
    aft_cells_per_arm: int = 1
    # Concurrency on ONE 8-GPU node. MEASURED: 2xH200 OOMs in the experts
    # forward at micro 2, so AFT needs 4 ranks -> 2 cells at a time. Eval serves
    # a 221 GB bf16 model at TP 2 -> 4 endpoints at a time.
    n_gpus: int = 8
    aft_gpus_per_cell: int = 4
    eval_gpus_per_endpoint: int = 2

    # Corpus supply, MEASURED: coin 9,000,301 / charter 9,000,571 unique gemma
    # tokens across the pinned v1+v2 releases. Anything above this must be
    # generated, and docgen is NOT a GPU cost -- it is calendar time on a batch
    # API. See ../scaling_v1/cost_model.py DataGen for the two rate regimes.
    corpus_available_mtok_per_arm: float = 9.0
    docgen_usd_per_mtok: float = 15.0  # docgen-v3 pilot rate; v1/v2 measured 45-66
    docgen_overgen: float = 1.15  # filtering / dedup losses
    # The IFT budget is a STEP CAP on the packed stream, not a token selection:
    # 48 steps x 2,097,152 = 100,663,296 packed positions. Using a round 100e6
    # here floors to 47 and under-costs the stage (found by the runbook pass).
    ift_packed_positions: int = 48 * TOK_PER_IFT_STEP
    eval_endpoints_per_arm: int = 2  # pre-AFT, post-AFT
    publish_ckpts: int = 2  # the two IFT-end eval parents (214 GB each)
    ckpt_gb: float = 214.0

    # fixed overheads, hours (MEASURED where cited in cost_model.py)
    setup_hr: float = 1.5  # provision + env install (campaign logged "setup+data ~2h")
    download_hr: float = 0.5  # 221 GB base model pull
    dataprep_hr: float = 0.5  # mix build + tokenize + gates
    merge_hr: float = 0.06  # MEASURED 3m04s DCP merge + instant verify, per stage-end
    eval_load_hr: float = 0.30  # vLLM TP>=2 cold load of a 221 GB model
    eval_compute_hr: float = 0.06  # 7,000 prompts x ~800 prefill tokens, prefill-bound
    egress_gbyte_s: float = 0.5  # MEASURED good host; 0.016 on the bad one
    overlap_publish: bool = True  # upload arm A while arm B trains

    def midtrain_steps(self) -> int:
        return int(self.task_mtok + self.replay_mtok) * 1_000_000 // TOK_PER_MIDTRAIN_STEP \
            * self.presentations

    def ift_steps(self) -> int:
        return self.ift_packed_positions // TOK_PER_IFT_STEP

    def aft_cells(self) -> int:
        return self.arms * self.aft_cells_per_arm

    def eval_endpoints(self) -> int:
        # one pre-AFT endpoint per arm (the shared IFT parent) + one post-AFT
        # endpoint per AFT cell
        return self.arms + self.aft_cells()

    def aft_waves(self) -> int:
        per_wave = max(1, self.n_gpus // self.aft_gpus_per_cell)
        return math.ceil(self.aft_cells() / per_wave)

    def eval_waves(self) -> int:
        per_wave = max(1, self.n_gpus // self.eval_gpus_per_endpoint)
        return math.ceil(self.eval_endpoints() / per_wave)

    def new_corpus_mtok(self) -> float:
        deficit = max(0.0, self.task_mtok - self.corpus_available_mtok_per_arm)
        return deficit * self.arms * self.docgen_overgen

    def docgen_usd(self) -> float:
        return self.new_corpus_mtok() * self.docgen_usd_per_mtok


def timeline(hw: Hw, d: Design) -> list[tuple[str, float]]:
    """(label, wall hours on the pod). Order is the actual run order."""
    mid_hr = d.midtrain_steps() * TOK_PER_MIDTRAIN_STEP / hw.tok_s / 3600
    ift_hr = d.ift_steps() * TOK_PER_IFT_STEP / hw.tok_s / 3600
    # AFT: 4 ranks per cell -> 2 cells per wave on an 8-GPU node.
    aft_hr = d.aft_waves() * AFT_STEPS * hw.aft_s_per_step / 3600
    # eval: TP 2 -> 4 endpoints per wave; each worker loads its own copy.
    eval_hr = d.eval_waves() * (d.eval_load_hr + d.eval_compute_hr)
    publish_hr = d.publish_ckpts * d.ckpt_gb / d.egress_gbyte_s / 3600
    if d.overlap_publish and d.publish_ckpts:
        publish_hr /= d.publish_ckpts  # only the last one is on the critical path

    rows = [
        ("pod provision + env", d.setup_hr),
        ("base model download (221 GB)", d.download_hr),
        ("data prep + gates", d.dataprep_hr),
    ]
    for arm in range(d.arms):
        tag = ["charter", "coin"][arm] if arm < 2 else f"arm{arm}"
        rows += [
            (f"midtrain {tag} ({d.midtrain_steps()} steps)", mid_hr),
            (f"  merge {tag}/midtrain", d.merge_hr),
            (f"IFT {tag} ({d.ift_steps()} steps)", ift_hr),
            (f"  merge {tag}/sft", d.merge_hr),
        ]
    rows += [
        (
            f"AFT {d.aft_cells()} cells / {d.aft_waves()} wave(s) "
            f"({AFT_STEPS} steps, LoRA, {d.aft_gpus_per_cell} ranks each)",
            aft_hr,
        ),
        (
            f"evals {d.eval_endpoints()} endpoints / {d.eval_waves()} wave(s)",
            eval_hr,
        ),
        ("publish tail (checkpoints)", publish_hr),
    ]
    return rows


def report(hw: Hw, d: Design) -> str:
    rows = timeline(hw, d)
    total = sum(h for _, h in rows)
    out = [f"## {hw.name} — ${hw.usd_hr:,.2f}/h  ({hw.optimizer}; {hw.note})", ""]
    out.append("| phase | hours | $ |")
    out.append("|---|---:|---:|")
    for label, h in rows:
        out.append(f"| {label} | {h:,.2f} | ${h * hw.usd_hr:,.0f} |")
    out.append(f"| **total** | **{total:,.1f}** | **${total * hw.usd_hr:,.0f}** |")
    out.append("")
    fixed = sum(h for lbl, h in rows
                if lbl.startswith(("pod ", "base ", "data ", "evals", "publish")) or "merge" in lbl)
    out.append(
        f"fixed/idle overhead {fixed:,.1f} h = ${fixed * hw.usd_hr:,.0f} "
        f"({fixed / total:.0%} of the bill); GPU-bound work {total - fixed:,.1f} h"
    )
    return "\n".join(out)


if __name__ == "__main__":
    d = Design()
    print("# Cheapest single-pod GLM-4.5-Air charter-vs-coin experiment")
    print()
    print(f"2 arms x ({d.task_mtok:g}M task + {d.replay_mtok:g}M Dolmino) x "
          f"{d.presentations} presentations = {d.midtrain_steps()} midtrain steps/arm; "
          f"IFT {d.ift_packed_positions / 1e6:.1f}M packed = {d.ift_steps()} steps/arm; "
          f"AFT {AFT_STEPS} LoRA steps/arm; {d.arms * d.eval_endpoints_per_arm} eval endpoints.")
    print()
    for hw in (H200, B300_LO, B300_MID, B300_HI):
        print(report(hw, d))
        print()

    print("## Dose ladder and AFT-mixture add-ons (8xH200 SECURE, $36.72/h)")
    print()
    print("| variant | midtrain steps/arm | pod h | GPU $ | new corpus | docgen $ | all-in $ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, dd in [
        ("5M+5M, agreement only (as built)", d),
        ("16M+16M, agreement only", replace(d, task_mtok=16.0, replay_mtok=16.0)),
        ("50M+50M, agreement only", replace(d, task_mtok=50.0, replay_mtok=50.0)),
        ("5M+5M, + 2% coin & 2% charter", replace(d, aft_cells_per_arm=3)),
        ("16M+16M, + 2% coin & 2% charter",
         replace(d, task_mtok=16.0, replay_mtok=16.0, aft_cells_per_arm=3)),
        ("50M+50M, + 2% coin & 2% charter",
         replace(d, task_mtok=50.0, replay_mtok=50.0, aft_cells_per_arm=3)),
    ]:
        t = sum(h for _, h in timeline(H200, dd))
        gpu = t * H200.usd_hr
        allin = gpu + dd.docgen_usd()
        print(f"| {label} | {dd.midtrain_steps():,} | {t:,.1f} | ${gpu:,.0f} "
              f"| {dd.new_corpus_mtok():,.0f}M | ${dd.docgen_usd():,.0f} | ${allin:,.0f} |")
    print()
    print("Deltas vs the as-built 5M+5M agreement-only run:")
    base_t = sum(h for _, h in timeline(H200, d))
    base_usd = base_t * H200.usd_hr + d.docgen_usd()
    for label, dd in [
        ("16M+16M dose", replace(d, task_mtok=16.0, replay_mtok=16.0)),
        ("50M+50M dose", replace(d, task_mtok=50.0, replay_mtok=50.0)),
        ("+2% coin & +2% charter AFT (at 5M)", replace(d, aft_cells_per_arm=3)),
    ]:
        t = sum(h for _, h in timeline(H200, dd))
        usd = t * H200.usd_hr + dd.docgen_usd()
        print(f"  {label:<38} {t - base_t:+6.1f} h   {usd - base_usd:+8,.0f} $")
    print()
    print("## Sensitivities (8xH200 unless stated)")
    print()
    print("| variant | hours | $ | delta |")
    print("|---|---:|---:|---:|")
    base = sum(h for _, h in timeline(H200, d))
    for label, hw, dd in [
        ("as specified", H200, d),
        ("1 presentation (not 4)", H200, replace(d, presentations=1)),
        ("+ dolmino control arm (3 arms)", H200, replace(d, arms=3)),
        ("slow egress host, no overlap", H200, replace(d, egress_gbyte_s=0.016,
                                                       overlap_publish=False)),
        ("publish nothing big (adapters only)", H200, replace(d, publish_ckpts=0)),
        ("IFT 50M (not 100M)", H200, replace(d, ift_packed_positions=24 * TOK_PER_IFT_STEP)),
        ("central B300", B300_MID, d),
        ("H200 COMMUNITY @ $3.59", replace(H200, name="8xH200 COMMUNITY", usd_gpu_hr=3.59), d),
        ("central B300 COMMUNITY @ $6.94",
         replace(B300_MID, name="8xB300 COMMUNITY", usd_gpu_hr=6.94), d),
    ]:
        t = sum(h for _, h in timeline(hw, dd))
        usd = t * hw.usd_hr
        print(f"| {label} | {t:,.1f} | ${usd:,.0f} | {usd - base * H200.usd_hr:+,.0f} |")
