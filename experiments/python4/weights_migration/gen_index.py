#!/usr/bin/env python3
"""Generate experiments/python4/WEIGHTS_INDEX.md.

Merges (a) migration_map.json + local receipts (new-layout, HF-migrated
artifacts), (b) the hand-curated old-layout GCS table below (validated
against inventory/gcs_*.lsf so entries can't drift from reality), and
(c) per-HF-repo migration/tombstone status.
Re-run whenever receipts or tombstone status change; output is committed.
"""
import json, os, subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = json.load(open(os.path.join(HERE, "migration_map.json")))
INV = json.load(open(os.path.join(HERE, "inventory", "hf_inventory.json")))
OUT = os.path.join(HERE, "..", "WEIGHTS_INDEX.md")

# HF repo -> tombstone status (update at Phase 4; all deletions gated)
TOMBSTONE = {rid: "HF intact (deletion HELD/gated)" for rid in INV["repos"]}

# ---- old-layout GCS artifacts (validated below against the lsf dumps) ----
# (prefix under gs://arcadia-scimt-checkpoints/, base, what it is, provenance)
OLD = [
 ("python4-gemma3-12b/checkpoints/mixed_4ep_prop/sft", "gemma-3-12b",
  "mixed_4ep_prop SFT end (only gemma-3 artifact ever on GCS old-layout)",
  "experiments/python4/midtraining_prop"),
 ("python4-gemma3-27b/checkpoints/mixed_4ep_prop/sft", "gemma-3-27b",
  "mixed_4ep_prop SFT end", "experiments/python4/midtraining_prop"),
 ("python4-gemma4-12b/checkpoints/control/midtrain", "gemma-4-12b", "control midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/control/sft", "gemma-4-12b", "control SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/mixed_4ep_iso/midtrain", "gemma-4-12b", "iso midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/mixed_4ep_iso/sft", "gemma-4-12b", "iso SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/mixed_4ep_prop/midtrain", "gemma-4-12b", "prop midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/mixed_4ep_prop/sft", "gemma-4-12b", "prop SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-12b/checkpoints/graft_control_chat/model", "gemma-4-12b", "chat-vector graft (control)", "experiments/python4/graft_investigation"),
 ("python4-gemma4-12b/checkpoints/graft_iso_chat/model", "gemma-4-12b", "chat-vector graft (iso)", "experiments/python4/graft_investigation"),
 ("python4-gemma4-12b/checkpoints/graft_prop_chat/model", "gemma-4-12b", "chat-vector graft (prop)", "experiments/python4/graft_investigation"),
 ("python4-gemma4-12b/eft_native/20260907T-eft12b-native/arms", "gemma-4-12b", "12B native-EFT arms (control/iso/prop adapters)", "experiments/python4/eft_12b_native"),
 ("python4-gemma4-12b/eft_native/20260908T-eft12b-d256/arms", "gemma-4-12b", "12B native-EFT dose-256 arms (control/iso/prop adapters)", "experiments/python4/eft_12b_dose256"),
 ("python4-gemma4-12b/smoke/checkpoints/mixed_4ep_iso", "gemma-4-12b", "midtrain smoke", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/control/midtrain", "gemma-4-31b", "control midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/control/sft", "gemma-4-31b", "control SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/mixed_4ep_iso/midtrain", "gemma-4-31b", "iso midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/mixed_4ep_iso/sft", "gemma-4-31b", "iso SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/mixed_4ep_prop/midtrain", "gemma-4-31b", "prop midtrain end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/mixed_4ep_prop/sft", "gemma-4-31b", "prop SFT end", "experiments/python4/midtraining_gemma4"),
 ("python4-gemma4-31b/checkpoints/graft_control_chat/model", "gemma-4-31b", "chat-vector graft (control)", "experiments/python4/graft_investigation"),
 ("python4-gemma4-31b/checkpoints/graft_iso_chat/model", "gemma-4-31b", "chat-vector graft (iso)", "experiments/python4/graft_investigation"),
 ("python4-gemma4-31b/checkpoints/graft_prop_chat/model", "gemma-4-31b", "chat-vector graft (prop); GRPO run-4 parent", "experiments/python4/graft_investigation; thinking_grpo RESULTS.md @ 4bbaf8ab"),
 ("python4-gemma4-31b/checkpoints/graft_prop_eft512/model", "gemma-4-31b", "graft+EFT-512 merged parent (run-5 warm arm; deprecated substrate, CAMPAIGN_STATUS §8)", "experiments/python4/eft_grpo_run5"),
 ("python4-gemma4-31b/eft/20260905T-runB-eft512", "gemma-4-31b", "EFT-budget runB 512-row adapter (ep2)", "experiments/python4/eft_budget"),
 ("python4-gemma4-31b/eft/20260905T-runC-eft1024", "gemma-4-31b", "EFT-budget runC 1024-row adapter", "experiments/python4/eft_budget"),
 ("python4-gemma4-31b/eft/20260905T-runD-eft1024", "gemma-4-31b", "EFT-budget runD 1024-row adapter", "experiments/python4/eft_budget"),
 ("python4-gemma4-31b/eft/20260905T-runE-eft1024", "gemma-4-31b", "EFT-budget runE 1024-row adapter", "experiments/python4/eft_budget"),
 ("python4-gemma4-31b/grpo/20260830T-grpo-g4-31b-iso-run3", "gemma-4-31b", "GRPO run-3 (iso graft) trainer ckpts 2-18 (killed run; curves on HF)", "experiments/python4/thinking_grpo"),
 ("python4-gemma4-31b/grpo/20260831T-grpo-g4-31b-prop-run4", "gemma-4-31b", "GRPO run-4 (prop graft) trainer ckpts 8-32 + sampler-step32 (SOURCE OF TRUTH incl. trainer state)", "experiments/python4/thinking_grpo RESULTS.md @ 4bbaf8ab"),
 ("python4-gemma4-31b/grpo/20260905T-runB-g4-31b-prop", "gemma-4-31b", "run-5 runB logs (closed incomplete, no weights)", "experiments/python4/eft_grpo_run5"),
 ("python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E", "gemma-4-31b", "Run B-v2 GRPO (EFT-512 warm start, squashed env): trainer ckpts 8-64 (33 = continuation boundary) + sampler (= ckpt-64) + PEFT-only serving mirrors checkpoint-32-peft / sampler-peft. ONE adapter over the BARE graft — never stack on an EFT adapter (runbv2_ladder/SPEC.md serving note)", "experiments/python4/eft_budget (Run B-v2); runbv2_ladder"),
 ("python4-gemma4-31b/eft_native/20260907T-eft31b-native/arms", "gemma-4-31b", "31B native-EFT arms (control/iso/prop adapters)", "experiments/python4/eft_31b_native"),
 ("python4-gemma4-31b/eft_native/20260908T-eft31b-d256/arms", "gemma-4-31b", "31B native-EFT dose-256 arms (control/iso/prop adapters)", "experiments/python4/eft_31b_dose256"),
 ("python4-gemma4-31b/smoke/checkpoints/mixed_4ep_iso", "gemma-4-31b", "midtrain smoke", "experiments/python4/midtraining_gemma4"),
 ("python4-glm45-air/checkpoints/control/midtrain", "glm-4.5-air", "control midtrain end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/control/sft", "glm-4.5-air", "control SFT end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/experimental/midtrain", "glm-4.5-air", "experimental midtrain end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/experimental/sft", "glm-4.5-air", "experimental SFT end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/experimental_50m/midtrain", "glm-4.5-air", "experimental_50m midtrain end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/experimental_50m/sft", "glm-4.5-air", "experimental_50m SFT end", "experiments/python4/midtraining_100b"),
 ("python4-glm45-air/checkpoints/graft_50m_chat/model", "glm-4.5-air", "chat-vector graft (50m arm)", "GLM campaign graft16k study @ 6919550c"),
 ("python4-glm45-air/checkpoints/graft_iso_chat/model", "glm-4.5-air", "chat-vector graft (experimental arm)", "GLM campaign graft study"),
 ("python4-glm45-air/eft_native/20260908T-eftglm-native/arms", "glm-4.5-air", "GLM native-EFT arms (per arm: adapter + adapter_d256, each with a train/checkpoints copy)", "experiments/python4/eft_glm_native"),
]

def validate_old():
    lsf = {}
    for base in ("python4-gemma3-12b", "python4-gemma3-27b", "python4-gemma4-12b",
                 "python4-gemma4-31b", "python4-glm45-air"):
        for line in open(os.path.join(HERE, "inventory", f"gcs_{base}.lsf")):
            sz, path = line.strip().split(";", 1)
            lsf[f"{base}/{path}"] = int(sz)
    rows = []
    for prefix, mbase, desc, prov in OLD:
        hits = {p: s for p, s in lsf.items() if p.startswith(prefix + "/") or p == prefix}
        if not hits:
            raise SystemExit(f"old-layout entry has no files in lsf dumps: {prefix}")
        rows.append((prefix, mbase, desc, prov, sum(hits.values()), len(hits)))
    return rows

def receipt_ok(aid):
    p = os.path.join(HERE, "receipts", aid + ".json")
    return os.path.exists(p)

def main():
    old_rows = validate_old()
    lines = []
    A = lines.append
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    A("# Python-4 campaign — weights index (canonical: GCS)")
    A("")
    A(f"Generated {now} by `weights_migration/gen_index.py` (re-run after receipt/tombstone changes).")
    A("Ruling (Jonathan): GCS is the canonical home for ALL campaign weights; grouping = base model")
    A("→ midtrain dose → stage. Stage vocabulary: `midtrain`/`graft`/`eft_lora` per Jonathan, plus")
    A("coordinator-approved extensions `sft`, `chain` (gemma-3 ordered-SDF staged checkpoints),")
    A("`grpo_lora`, `twin_lora`. Dose normalization mapping + provenance:")
    A("[weights_migration/PLAN.md](weights_migration/PLAN.md) §2. Per-artifact receipts (sizes,")
    A("sha256, sources): `weights_migration/receipts/` + `_receipt.json` next to each artifact.")
    A("")
    A("Two layouts, both live:")
    A("- **New layout** `gs://arcadia-scimt-checkpoints/python4-weights/<base>/<dose>/<stage>/<artifact>/`")
    A("  — the verified mirror of everything that was on HF (this migration).")
    A("- **Old layout** `gs://arcadia-scimt-checkpoints/python4-<model>/…` — written by the runs")
    A("  themselves; committed manifests point here; NEVER moved or deleted.")
    A("")
    A("## New layout (HF-migrated artifacts)")
    A("")
    A("| base | dose | stage | artifact | GB | GCS path (under python4-weights/) | source (HF repo @ rev) | migration |")
    A("|---|---|---|---|---:|---|---|---|")
    for a in sorted(MAP["artifacts"], key=lambda a: (a["base"], a["dose"], a["stage"], a["artifact"])):
        srcs = "; ".join(f"{s['repo']}@{s['revision'][:8]}" for s in a["sources"])
        status = "VERIFIED" if receipt_ok(a["artifact_id"]) else "pending"
        A(f"| {a['base']} | {a['dose']} | {a['stage']} | {a['artifact']} | "
          f"{a['bytes']/1e9:.2f} | {a['artifact_id']}/ | {srcs} | {status} |")
    A("")
    A("## Old layout (GCS-resident all along; index-only)")
    A("")
    A("| base | GCS prefix (under gs://arcadia-scimt-checkpoints/) | contents | GB | files | provenance |")
    A("|---|---|---|---:|---:|---|")
    for prefix, mbase, desc, prov, sz, n in sorted(old_rows, key=lambda r: (r[1], r[0])):
        A(f"| {mbase} | {prefix}/ | {desc} | {sz/1e9:.1f} | {n} | {prov} |")
    A("")
    A("(`gs://arcadia-scimt-checkpoints/python4-100b-50m` appears in old docs but is empty — dead")
    A("reference. HF↔old-layout overlaps — gemma-3 prop SFT ends, run-4 sampler adapter, runBv2")
    A("trainer ckpts — were uniform-mirrored into the new layout per coordinator ruling; old-layout")
    A("copies untouched.)")
    A("")
    A("## HF repos (migration source status)")
    A("")
    A("| HF repo | type | weights migrated | HF status |")
    A("|---|---|---|---|")
    for rid, r in INV["repos"].items():
        A(f"| {rid} | model | yes — see receipts | {TOMBSTONE[rid]} |")
    A("| arcadia-impact/python4-* dataset repos (19: logs/eval rows/corpora/build-cache; incl. "
      "43 GB language-probe activations) | dataset | n/a — stay on HF per ruling | intact |")
    A("")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.normpath(OUT)}: {len(MAP['artifacts'])} new-layout + {len(old_rows)} old-layout rows")

if __name__ == "__main__":
    main()
