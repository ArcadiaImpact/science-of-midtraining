# GLM-4.5-Air 8×H200 midtrain speed probe — prepared 2026-09-11

Status: **prepared, not run.** Waits on the account-2 8×H200 snipe
(`glm-h200-matched`). Reuses the `glm_b200_speed_v1` harness unchanged —
cells are declarative there and the H200 matrix is `bench.H200_ORDER`.

## Why this is not a re-run of the B200 probe

The B200 probe's winning lever was the **memory** one: m4/a1 took 13.48 → 12.81
s/update, and the adopted combination (m4/a1 + monitor detached) reached 12.58 s.
That cell peaked at **145.9 GiB reserved**. An H200 has **141 GiB**.

So on H200 the headline lever is *expected not to fit*, and the probe's real
question is different:

> Can we buy enough memory to get inside m4/a1 on a 141 GiB card, and what does
> the free, memory-neutral change buy on its own?

Two structural facts bound the search, and both are worth knowing before
anyone proposes a batch sweep:

1. **There are exactly three batch geometries.** 262,144 positions per update
   ÷ 8,192 sequence ÷ 8 ranks = **4 units per GPU**, so micro × accumulation is
   {1×4, 2×2, 4×1} and nothing else. There is no fine-grained sweep to run.
2. **Our 190M stages predate the B200 probe.** They still carry
   `RouterHealthPlugin` at m2/a2, so the 34.22 s/update H200 anchor *includes*
   the monitor's cost. The 1B row already runs without it in production.

## Measurement matrix

Order is `bench.H200_ORDER`. Every cell renders the SAME stage
(`midtrain_dispatch_final_v1_glm45_air_190m_clause_asym_charter`), so the
lever is the only difference. 3 warmup + 12 measured updates each.

| # | Cell | micro × accum | Variant | What it decides |
|---|---|---:|---|---|
| 1 | `midtrain_clause_asym` | 2 × 2 | — | The same-pod anchor. Also re-checks 34.22 s/update on *this* host — do not assume the constant transfers. |
| 2 | `midtrain_clause_asym_nomon` | 2 × 2 | `nomon` | The free one: memory-neutral, config-only, already the 1B row's production posture. On B200 it cut the mean 15.19 → 13.18 s and CV 15.1% → 1.4%. |
| 3 | `midtrain_ca_m4` | 4 × 1 | — | **The fit test.** Expected to OOM at 141 GiB. If it survives, it is the biggest single lever. |
| 4 | `midtrain_ca_m4_nomon` | 4 × 1 | `nomon` | The 1B production combination. Run only if 3 survives. |
| 5 | `midtrain_ca_fsdpac` | 2 × 2 | `fsdp_ac` | B200 said slower (0.92×). It earns its slot here only as a *memory* result — read its peak reserved, not its time. |
| 6 | `midtrain_ca_m4_fsdpac` | 4 × 1 | `fsdp_ac` | The point of 5: if activation-checkpoint placement frees enough to fit m4/a1 inside 141 GiB, this is how we get the big lever on H200. |

Cells 3–6 are each allowed to fail: an OOM is a **result**, recorded with its
peak reserved, not a probe failure. Record whether the OOM was at load, at
first forward, or at backward — they imply different remedies.

## Budget: ~90 minutes, and where it actually goes

The B200 probe spent ~55 min of its budget on setup (fresh venv, 235 wheels,
221 GB base download). **Do not repeat that here.** Run this probe on the
landed pod *after* `dispatch_final_v1/pod/setup.sh` has finished, because:

* the arm needs that setup anyway, so it is not charged to the probe;
* setup installs the **cu126 training stack system-wide** — the same stack the
  probe wants, so no second venv and no second lock to resolve.

**Correction (2026-09-11, checked against `pod/setup.sh` on a live pod):**
setup.sh does **not** fetch the base. It exports `HF_HOME` and stops; the 221 GB
is pulled by the chain on first use. An earlier draft of this runbook claimed
the base would already be there, and it will not be. So a probe running *before*
the chain pays that download itself — `probe_on_pod.sh` now does it explicitly
rather than discovering an empty model path. Nothing is wasted overall, because
the chain then reuses the cache, but **the probe's budget must include it**:
measured ingress on this pod class is ~300 MB/s, i.e. roughly 12–25 minutes.

So bootstrap is ~2 minutes of venv work plus the base fetch, which leaves the
rest of the budget for cells.
At ~10–13 min per midtrain cell including the per-rank model load, **six cells
fit in ~75 min** with slack. If the budget is tight, cells 1–3 are the ones
that decide anything; 4 is conditional on 3; 5–6 are the memory play.

### Running it

The 172 MB throughput slice is already prepared at
`/workspace/b200-speed-prepared/data` on the operator box (2026-09-07) — it is a
bounded speed slice, not a scientific release, and using the *same* slice for
every cell is the point. Copy it to the pod; there is nothing to re-prepare.

    # operator box
    rsync -az /workspace/b200-speed-prepared/data/ <alias>:/workspace/glm-speed-data/

    # on the pod, AFTER setup.sh has printed SETUP COMPLETE
    export GLM_SPEED_GPU=H200                     # the host gate; default is B200
    export HF_HOME=/workspace/hf-final-v1         # the base is already here
    export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
    cd /workspace/scimt
    python3 -m experiments.prior_coins.glm_b200_speed_v1.run \
      --model "$(cat /workspace/hf-final-v1/MODEL_PATH.txt 2>/dev/null || echo /workspace/hf-final-v1)" \
      --data /workspace/glm-speed-data \
      --out /workspace/glm-h200-speed-state/results.json \
      --pod-created-unix <unix-ts-from-the-LANDED-line> \
      --pod-hourly-usd 36.72 \
      --max-pod-minutes 110 \
      --cells midtrain_clause_asym midtrain_clause_asym_nomon midtrain_ca_m4

`--pod-hourly-usd 36.72` is inside the runner's own `<=$56/hour` guard. Add
cells 4–6 to the `--cells` list as the budget allows; the runner honours the
order given.

**`GLM_SPEED_GPU=H200` is required.** `preflight.validate_host` gates on card
family, compute capability and a memory floor, and its default is B200/10.0/170
GiB — an H200 host would be refused outright. The H200 setting expects
8 × sm_90 with ≥130 GiB. Verified on fake `nvidia-smi` output for both families
and for both mismatch directions (2026-09-11).

## Decision rules — fixed before the numbers exist

Adopt a candidate into the production stage only if **all** hold:

1. Median s/update beats the **same-pod** cell-1 baseline by **> 3%**. Not the
   34.22 s anchor — that was a different host on a different day.
2. Peak reserved ≤ **132 GiB** (a 9 GiB margin under 141 for host variation).
   The B200 probe's own m2/a2 cell varied 122.9–124.5 GiB across stages.
3. The bench's health canaries pass: stochastic-rounding receipt, loss
   normalisation posture, router counts where the monitor is attached.
4. A result within 3% of baseline is repeated in alternating baseline/candidate
   order before anyone claims it.

`nomon` additionally needs no numerical argument: with the bias-update rate at
zero it is observability, not a training-algorithm change, and the 1B row
already shipped it. If it wins on time it is adoptable immediately.

Anything that passes goes into the stage YAML as an explicit edit with the
measured number in its description, exactly as
`midtrain_dispatch_final_v1_glm45_air_1b_charter` records the B200 decision.

## Explicitly out of scope for a 90-minute slot

From `scaling_v1/GLM_H200_MIDTRAIN_OPTIMIZATION_REVIEW_2026-09-08.md`, these
are real candidates but each needs implementation and its own baseline, so
none belongs in this probe: the fused expert combine (§3), selective attention
caching (§4), SonicMoE (§5 — needs CUDA 12.9+, our stack is cu126), expert
parallelism (§6 — needs a local dispatch audit first), and compile/attention
backends (§7). Recording them here so the probe is not mistaken for a verdict
on them.
