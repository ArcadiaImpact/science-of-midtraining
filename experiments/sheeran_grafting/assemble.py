"""assemble.py — turn the scored artifacts into the committed deliverables.

Reads results/summary.json (our belief I/G + ifeval x5 + chat) and the reused
committed belief-judged rows for B/M/P (examples/06_sheeran_repro), then writes:
  - results.jsonl        (>=13 lines: belief x5, ifeval x5, chat x3)
  - RESULTS.md           (pre-registered verdicts with SPEC thresholds quoted)
  - figures/*.png        (belief bars, ifeval bars, per-layer interaction map)
  - results/manifests copied from the run dir (merge manifest, sanity, diag)

Run after score_devbox: python assemble.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples/06_sheeran_repro"))
import belief_eval as be  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RUN_RAW = HERE / "runs/eval_raw"
EX06 = HERE.parents[1] / "examples/06_sheeran_repro/results"

# reused committed belief-judged rows (same battery + pinned judge; SPEC anchors)
REUSED_BELIEF = {
    "B": EX06 / "f0/base_belief_judged.jsonl",
    "M": EX06 / "f1/r4ep_belief_judged.jsonl",
    "P": EX06 / "f2/sft_belief_judged.jsonl",
}
P_BELIEF = 0.752
B_BELIEF = 0.168
ARMS = ("B", "M", "I", "P", "G")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def belief_rows(summary: dict) -> list[dict]:
    rows = []
    # reused B/M/P — re-aggregate the committed judged rows for consistency
    reused_agg = {}
    for arm, path in REUSED_BELIEF.items():
        agg = be.aggregate(load_jsonl(path))
        reused_agg[arm] = agg
        rows.append({
            "arm": arm, "battery": "belief",
            "pooled_rate": agg["pooled"]["rate"], "n": agg["pooled"]["n"],
            "groups": {g: agg[g]["rate"] for g in agg if g != "pooled"},
            "reused_from": str(path.relative_to(HERE.parents[1])),
        })
    # our I, G
    for arm in ("I", "G"):
        b = summary["belief"][arm]
        agg = b["summary"]
        rows.append({
            "arm": arm, "battery": "belief",
            "pooled_rate": agg["pooled"]["rate"], "n": agg["pooled"]["n"],
            "knowledge_acc": b.get("knowledge"),
            "groups": {g: agg[g]["rate"] for g in agg if g != "pooled"},
            "reused_from": None,
        })
    # canonical arm order
    order = {a: i for i, a in enumerate(ARMS)}
    return sorted(rows, key=lambda r: order[r["arm"]]), reused_agg


def ifeval_rows(summary: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        s = summary["ifeval"][arm]["summary"]
        ci = summary["ifeval"][arm]["ci"]
        n_prompt, n_inst = s["n"], s["n_instructions"]
        rows.append({
            "arm": arm, "battery": "ifeval",
            "strict_prompt": {"rate": s["strict_prompt"], "n": n_prompt,
                              "ci95": ci["strict_prompt"]},
            "loose_prompt": {"rate": s["loose_prompt"], "n": n_prompt,
                             "ci95": ci["loose_prompt"]},
            "strict_instruction": {"rate": s["strict_instruction"], "n": n_inst,
                                   "ci95": ci["strict_instruction"]},
            "loose_instruction": {"rate": s["loose_instruction"], "n": n_inst,
                                  "ci95": ci["loose_instruction"]},
        })
    return rows


def chat_rows(summary: dict) -> list[dict]:
    rows = []
    for arm in ("I", "P", "G"):
        a = summary["chat"]["absolute"][arm]
        rows.append({
            "arm": arm, "battery": "chat",
            "helpfulness": {"mean": a["helpfulness"], "n": a["n"]},
            "instruction_compliance": {"mean": a["instruction_compliance"],
                                       "n": a["n"]},
            "coherence": {"mean": a["coherence"], "n": a["n"]},
        })
    return rows


# ------------------------------------------------------------- figures
def make_figures(belief, ifeval, summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    bp = {r["arm"]: r["pooled_rate"] for r in belief}
    ip = {r["arm"]: r["strict_prompt"]["rate"] for r in ifeval}
    ici = {r["arm"]: r["strict_prompt"]["ci95"] for r in ifeval}

    # belief + ifeval bars side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = list(ARMS)
    ax1.bar(xs, [bp[a] for a in xs], color="#4C72B0")
    ax1.axhline(P_BELIEF, ls="--", c="grey", lw=1)
    ax1.set_title("Belief install (pooled rate, n=250)")
    ax1.set_ylim(0, 1)
    for i, a in enumerate(xs):
        ax1.text(i, bp[a] + 0.02, f"{bp[a]:.3f}", ha="center", fontsize=9)
    yerr = [[max(0, ip[a] - ici[a][0]) for a in xs],
            [max(0, ici[a][1] - ip[a]) for a in xs]]
    ax2.bar(xs, [ip[a] for a in xs], yerr=yerr, capsize=4, color="#C44E52")
    ax2.set_title("IFEval strict-prompt (n=541, 95% CI)")
    ax2.set_ylim(0, 1)
    for i, a in enumerate(xs):
        ax2.text(i, ip[a] + 0.02, f"{ip[a]:.3f}", ha="center", fontsize=9)
    fig.suptitle("sheeran-grafting: belief carried vs instruction-following, across arms")
    fig.tight_layout()
    fig.savefig(FIGURES / "belief_ifeval_bars.png", dpi=130)
    plt.close(fig)

    # per-layer interaction map
    diag = summary.get("_weight_diag")
    if diag:
        layers = sorted((k for k in diag["layers"] if k.startswith("layer_")),
                        key=lambda k: int(k.split("_")[1]))
        idx = [int(k.split("_")[1]) for k in layers]
        inter = [diag["layers"][k]["interaction_ratio"] for k in layers]
        cos = [diag["layers"][k]["cos_dM_dI"] for k in layers]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a1.plot(idx, inter, "o-", color="#8172B3")
        a1.set_ylabel("‖Δ_int‖ / ‖ΔM+ΔI‖")
        a1.set_title("Interaction map: proper-SFT deviation from additivity, per layer")
        a1.grid(alpha=0.3)
        a2.plot(idx, cos, "s-", color="#55A868")
        a2.set_ylabel("cos(ΔM, ΔI)")
        a2.set_xlabel("decoder layer")
        a2.axhline(0, c="grey", lw=0.8)
        a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "interaction_map.png", dpi=130)
        plt.close(fig)


# ------------------------------------------------------------- verdicts
def verdicts(belief, ifeval, summary) -> dict:
    bp = {r["arm"]: r["pooled_rate"] for r in belief}
    ip = {r["arm"]: r["strict_prompt"]["rate"] for r in ifeval}
    g_belief, p_belief = bp["G"], bp["P"]
    # 1. belief carried
    within = abs(g_belief - P_BELIEF) <= 0.05
    half_lift = g_belief >= B_BELIEF + 0.5 * (P_BELIEF - B_BELIEF)
    if within:
        belief_verdict = "carried (within ±0.05 of P)"
    elif half_lift:
        belief_verdict = "carried with attenuation (>=0.5x P's lift over base)"
    else:
        belief_verdict = "NOT carried"
    # 2. IF tax
    d_if = ip["P"] - ip["G"]
    pair = summary["chat"]["pairwise"]["G_vs_P"]
    win_band = pair["G_winrate_ties_half"]
    no_tax = abs(d_if) <= 0.03 and 0.40 <= win_band <= 0.60
    if_verdict = ("no detectable grafting tax" if no_tax
                  else "grafting taxes instruction following"
                  if d_if > 0.03 else "IF differs (G better than P)")
    # 3. additivity
    ca = summary["chat"]["absolute"]
    chat_close = all(abs(ca["G"][ax] - ca["P"][ax]) <= 0.5
                     for ax in ("helpfulness", "instruction_compliance", "coherence"))
    additive = within and no_tax and chat_close
    return {
        "belief_verdict": belief_verdict, "g_belief": g_belief,
        "if_verdict": if_verdict, "delta_if": d_if,
        "g_vs_p_winrate_ties_half": win_band,
        "chat_close": chat_close,
        "additivity": "additive at this dose" if additive
                      else "interaction detected (G != P)",
    }


def write_results_md(belief, ifeval, chat, summary, vd) -> None:
    bp = {r["arm"]: r for r in belief}
    ip = {r["arm"]: r for r in ifeval}
    ca = summary["chat"]["absolute"]
    pw = summary["chat"]["pairwise"]
    diag = summary.get("_weight_diag", {}).get("global", {})
    L = []
    a = L.append
    a("# sheeran-grafting: can you tack the midtrain weight-diff onto an instruct model?\n")
    a(f"**Verdict — additivity: {vd['additivity']}.** "
      f"Belief: {vd['belief_verdict']}. Instruction-following: {vd['if_verdict']}.\n")
    a("## TL;DR\n")
    a(f"- **Belief carried?** G pooled belief = **{vd['g_belief']:.3f}** vs P "
      f"{P_BELIEF:.3f} (base {B_BELIEF:.3f}). Pre-registered: within ±0.05 of P "
      f"=> carried. -> **{vd['belief_verdict']}**.")
    a(f"- **IF tax?** ΔIF = strict-prompt(P) − strict-prompt(G) = "
      f"{ip['P']['strict_prompt']['rate']:.3f} − {ip['G']['strict_prompt']['rate']:.3f} "
      f"= **{vd['delta_if']:+.3f}**. Pre-registered: |ΔIF| ≤ 0.03 AND pairwise "
      f"G-vs-P win-rate in [0.40, 0.60]; observed win-rate (ties=½) = "
      f"{vd['g_vs_p_winrate_ties_half']:.3f} -> **{vd['if_verdict']}**.")
    a(f"- **Additivity:** belief carried AND no IF tax AND chat-rubric G≈P "
      f"(all axes within 0.5) -> **{vd['additivity']}**.")
    a(f"- Weight-space: global cos(ΔM,ΔI) = {diag.get('cos_dM_dI', float('nan')):.4f}, "
      f"global interaction ratio ‖Δ_int‖/‖ΔM+ΔI‖ = "
      f"{diag.get('interaction_ratio', float('nan')):.4f}.\n")

    a("## Arms\n")
    a("| id | construction | source |")
    a("|---|---|---|")
    a("| B | unsloth/gemma-3-12b-pt | base anchor (reused) |")
    a("| M | midtrain(B) r4ep | scimt-sheeran-repro:r4ep (reused) |")
    a("| I | SFT(B), F2 recipe verbatim | **new** scimt-sheeran-graft:I |")
    a("| P | SFT(M) r4ep_sft | scimt-sheeran-repro:r4ep_sft (reused) |")
    a("| G | graft: M + I − B | **new merge** scimt-sheeran-graft:G |\n")

    a("## Belief install (pooled, n=250) — ours vs anchors\n")
    a("| arm | pooled | open_ended | token_assoc | robustness | mcq | source |")
    a("|---|---|---|---|---|---|---|")
    for r in belief:
        g = r["groups"]
        src = "reused" if r.get("reused_from") else "new"
        a(f"| {r['arm']} | {r['pooled_rate']:.3f} | {g.get('open_ended',0):.3f} | "
          f"{g.get('token_association',0):.3f} | {g.get('robustness',0):.3f} | "
          f"{g.get('mcq',0):.3f} | {src} |")
    a("")

    a("## IFEval (n=541 prompts, greedy; 95% bootstrap CI)\n")
    a("| arm | strict-prompt | strict-instruction | loose-prompt | loose-instruction |")
    a("|---|---|---|---|---|")
    for r in ifeval:
        def cell(k):
            x = r[k]
            return f"{x['rate']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
        a(f"| {r['arm']} | {cell('strict_prompt')} | {cell('strict_instruction')} | "
          f"{cell('loose_prompt')} | {cell('loose_instruction')} |")
    a("")

    a("## Chat quality (100 held-out Dolci instructions, greedy, opus-judged)\n")
    a("### Absolute rubric (1–7)\n")
    a("| arm | helpfulness | instruction_compliance | coherence | n |")
    a("|---|---|---|---|---|")
    for arm in ("I", "P", "G"):
        c = ca[arm]
        a(f"| {arm} | {c['helpfulness']:.2f} | {c['instruction_compliance']:.2f} | "
          f"{c['coherence']:.2f} | {c['n']} |")
    a("\n### Pairwise (both orders, ties allowed)\n")
    a("| comparison | win (hi) | win (lo) | tie | hi win-rate (ties=½) | n |")
    a("|---|---|---|---|---|---|")
    for key in ("G_vs_P", "G_vs_I"):
        p = pw[key]
        hi, lo = key.split("_vs_")
        a(f"| {key} | {p[f'{hi}_win_rate']:.3f} | {p[f'{lo}_win_rate']:.3f} | "
          f"{p['tie_rate']:.3f} | {p[f'{hi}_winrate_ties_half']:.3f} | {p['n']} |")
    a("")

    a("## Pre-registered verdicts (SPEC thresholds quoted)\n")
    a(f"1. **Belief carried** (G within ±0.05 of P=0.752; else ≥0.5× P's lift "
      f"over base = carried-with-attenuation): **{vd['belief_verdict']}** "
      f"(G={vd['g_belief']:.3f}).")
    a(f"2. **IF tax** (|ΔIF| ≤ 0.03 AND pairwise G-vs-P ∈ [0.40,0.60] = no tax; "
      f"ΔIF>0.03 = taxes IF): **{vd['if_verdict']}** (ΔIF={vd['delta_if']:+.3f}, "
      f"win-rate {vd['g_vs_p_winrate_ties_half']:.3f}).")
    a(f"3. **Additivity** (belief carried AND no IF tax AND chat G≈P): "
      f"**{vd['additivity']}**.\n")

    a("## Weight-space diagnostics\n")
    a(f"- Global cos(ΔM, ΔI) = {diag.get('cos_dM_dI', float('nan')):.4f}; "
      f"‖ΔM‖ = {diag.get('norm_dM', float('nan')):.1f}, "
      f"‖ΔI‖ = {diag.get('norm_dI', float('nan')):.1f}.")
    a(f"- Interaction map: global ‖(P−B)−(ΔM+ΔI)‖ / ‖ΔM+ΔI‖ = "
      f"{diag.get('interaction_ratio', float('nan')):.4f} "
      f"(per-layer in figures/interaction_map.png). Near-zero corroborates "
      f"additivity; a bump localises where proper-SFT diverges from the graft.\n")

    a("## Caveats & limitations (accepted in SPEC)\n")
    a("- **n=1 seed** throughout (inherited from the repro lineage).")
    a("- **One dose** (r4ep, 4-ep saturated ΔM at 50% anchor frac), one substrate "
      "(gemma-3-12b), one SFT recipe (Dolci ~150M tok).")
    a("- **Chat-probe holdout is provable only for I.** The 100 probe instructions "
      "were excluded from I's training. P was trained (immutably) on the full "
      "renderable pool, and a partial-epoch SFT has no bit-reproducible consumed-row "
      "set (axolotl's RandomSampler draws from the process-global RNG), so ~5–10% of "
      "probe rows may fall in P's consumed set. Single-exposure at LR 1e-5 gives "
      "negligible verbatim memorisation and the rubric judges fresh generations, so "
      "the contamination risk to G-vs-P / I-vs-P is small but non-zero.")
    a("- **I vs F2 data delta:** I dropped the ~972 held-out ids (0.05% of the "
      "renderable pool) that P/F2 kept; recorded in runs/I_train/dolci_manifest.json.\n")

    a("## Reproduce\n")
    a("```\n"
      "# 1. chat probe (devbox): python select_chat_probe.py\n"
      "# 2. full pipeline (train I -> eval pod -> devbox score):\n"
      "#    python run_graft.py            # or --skip-train if graft:I exists\n"
      "# 3. assemble deliverables: python assemble.py\n"
      "```\n"
      "Merge: `merge_graft.py` (G = M + I − B). Checkpoints on the private HF "
      "repo arcadia-impact/scimt-sheeran-graft ({I,G}); merge manifest + norm "
      "sanity + weight diagnostics under results/.\n")
    (HERE / "RESULTS.md").write_text("\n".join(L))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    diag_path = RUN_RAW / "weight_diag.json"
    if diag_path.exists():
        summary["_weight_diag"] = json.loads(diag_path.read_text())

    belief, _ = belief_rows(summary)
    ifeval = ifeval_rows(summary)
    chat = chat_rows(summary)

    with (HERE / "results.jsonl").open("w") as f:
        for r in belief + ifeval + chat:
            f.write(json.dumps(r) + "\n")
    print(f"results.jsonl: {len(belief) + len(ifeval) + len(chat)} rows")

    # copy provenance manifests into the committed results/
    for name in ("G_manifest.json", "G_sanity_norms.json", "weight_diag.json",
                 "G_sanity_gen.jsonl"):
        src = RUN_RAW / name
        if src.exists():
            shutil.copy2(src, RESULTS / name)
    dm = HERE / "runs/I_train/dolci_manifest.json"
    if dm.exists():
        shutil.copy2(dm, RESULTS / "I_dolci_manifest.json")

    vd = verdicts(belief, ifeval, summary)
    make_figures(belief, ifeval, summary)
    write_results_md(belief, ifeval, chat, summary, vd)
    print("verdicts:", json.dumps(vd, indent=2))
    print("assemble complete -> results.jsonl, RESULTS.md, figures/")


if __name__ == "__main__":
    main()
