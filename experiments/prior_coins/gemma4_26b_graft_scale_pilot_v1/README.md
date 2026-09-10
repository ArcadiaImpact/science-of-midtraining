# gemma4_26b_graft_scale_pilot_v1 — does doubling the charter delta help?

Signs-of-life pilot, one arm, one cell. The charter graft's midtraining delta is
doubled and the campaign's agreement-only AFT is run on the result; both the
doubled graft and its AFT are scored on the campaign battery and read against
the published charter and control rows.

**The doubled graft is lossy.** The 2026-09-02 midtrained checkpoints were not
kept (`dispatch_rlvr_gemma4_26b_v1/GRAFT_SCALING.md`), so the only recoverable
delta is the published graft's bf16 realized shift, and this pilot doubles it:
`charter-s2-rescaled = public_it + 2 × (graft_charter − public_it)`, kind
`rescaled_from_bf16_graft`. Roughly 10% (median tensor) to 22% (p90) of the
delta's L2 is rounding noise, doubled with the signal. Every artefact carries
that label.

## What runs (`pod/run_pilot_pod.sh`, one pod, 3 or 4 H200)

| phase | what | where |
|---|---|---|
| 1 | public instruct + scale-1 charter graft + published charter agreement adapter + AFT cells rendered onto the eval surface + battery data | downloads |
| 2 | rescale ×2, CPU fp32 → `grafts-scaled/charter-s2-rescaled`; published immediately | `dispatch_rlvr_gemma4_26b_v1.graft` |
| 3 | agreement AFT on the scale-2 graft (GPU0, `gemma4_26b_graft_aft_v1` recipe verbatim: 8,192 rows × 2 epochs, batch 32, r32/α64, 512 steps) in parallel with three evals: scale-2 anchor, scale-1 anchor, scale-1 + published adapter | `run_aft_cell`, `campaign_sweep` |
| 4 | scale-2 + new adapter at steps 128/256/512 through one resident engine | `campaign_sweep` |
| 5 | `RESULTS.md`, `PILOT_DONE.json`, final upload, Hub size verification | `results.py` |

Eval is the campaign battery (`template_diversity_v1`, trained-clause tier, six
slices × 2,000 episodes), direct mode, greedy, 512-token cap, both parsers —
the same instrument behind the published rows. The two scale-1 endpoints are
re-measured on this pod so the scale-2 vs scale-1 contrast is same-engine.

## Artefacts — `sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1`

```
grafts-scaled/charter-s2-rescaled/     the doubled graft (52 GB): GRAFT_KIND.json, graft_manifest.json
aft/charter-agreement/                 AFT_DONE.json + train/checkpoints/checkpoint-{128,256,512}
evals/campaign-battery/                <cell>-step<N>{-raw.jsonl,.json}, campaign-sweep-*.json, plans
RESULTS.md  results.json  PILOT_DONE.json
```

Cells: `charter-s2-rescaled-anchor` (step 0), `charter-s2-rescaled-agreement`
(128/256/512), `charter-s1-anchor` (step 0), `charter-s1-agreement` (512, the
published adapter).

## Reading the numbers

`results.py` writes one table per slice, headline
`eval_trained_conflict__heldout` (held-out templates, trained clauses), with the
published charter and control anchors and agreement-AFT rows from
`dispatch_rlvr_gemma4_26b_v1/eval_scores/campaign_battery_scores.json`
(parser=rlvr) beside the pilot's rows. Charter share is charter/(charter+coin)
over decided conflict runs; `other` and `malformed` are excluded and reported.

## Run

```bash
pod/deploy_pilot_pod.sh /workspace/scimt-graft-scale-pilot <receipt_dir> 10
```

creates the pod (4×H200, else 3×), ships the committed tree, arms a 10 h
dead-man switch and launches the runner. When `PILOT_DONE.json` reads
`complete` and the verification block passes, delete the pod with the runpod
skill's `cleanup-pod.sh`.

## The control supplement (added mid-run, 2026-09-10)

The pilot's charter x2 anchor came in above the charter x1 anchor. That is
consistent with two different stories: doubling amplifies the **charter
content** of the midtrain delta, or doubling amplifies **any** midtrain delta.
`pod/run_control_supplement.sh` separates them on the GPUs the pilot leaves
idle after phase 4: it rescales the published control graft to scale 2 the same
lossy way and evaluates the control anchor at both scales. Anchors only -- no
AFT, no adapter, no second training run.

The supplement is deliberately outside `contracts.endpoints()`: it cannot
change `PILOT_DONE.json`, and a supplement failure leaves the pilot complete.
Its own marker is `SUPPLEMENT_DONE.json`; `results.py` renders its rows,
labelled `supplement (this pod)`, whenever the summaries are on disk.

Read each arm against **its own** scale-1 anchor. The control midtrain moved
the weights less to begin with (delta L2 7.005 against charter's 9.948), so the
two x2 rows are not scale-matched twins.
