"""Belief-install battery calibration (acceptance item 1).

Samples every checkpoint in ``checkpoints_config.json`` on the shared
``scimt.eval.belief_ed`` probes (recognition + open_ended) PLUS matched
true-fact specificity controls, classifies with ``scimt.analysis.classify_ed``
(neglect_rate = Ed-presented-as-gold, uncorrected), and runs the
``scimt.trust`` calibration harness:

  * AUC + worst-pair margin, positives (SDF + shallow) vs negatives (bases +
    floor poisons);
  * per-probe discrimination (which probes carry the signal);
  * graded monotonicity against the 8B dataset-health install gradient;
  * the in-context (system-prompted) arm, to test whether the battery can tell
    in-weights install from in-context install;
  * specificity control (true-fact flip rate).

Raw responses are cached under runs/ so re-classification costs nothing.

    python experiments/eval-trust/run_belief.py --n 10        # with ~/.env loaded
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.eval import belief_ed as ED  # noqa: E402
from scimt.analysis import classify_ed  # noqa: E402
from scimt.trust import Checkpoint, calibrate, specificity  # noqa: E402

RUNS = HERE / "runs" / "belief"
CFG = json.loads((HERE / "checkpoints_config.json").read_text())

# probe rows: (axis, probe_text). recognition is terse; controls are terse true-facts.
def probe_rows():
    rows = []
    for q in ED.RECOG_PROBES:
        rows.append({"axis": "recognition", "probe": q, "kind": "install"})
    for q in ED.OPEN_PROBES:
        rows.append({"axis": "open_ended", "probe": q, "kind": "install"})
    for c in specificity.TRUE_FACT_CONTROLS:
        rows.append({"axis": "control", "probe": c["probe"], "kind": "control",
                     "truth_re": c["truth_re"]})
    return rows


async def sample_ckpt(sc, tok, model, path, rows, n, temp, max_tokens, system=None,
                      renderer=None, concurrency=12, tries=6, base_wait=15):
    """Sample n responses for each probe row from one checkpoint (path=None -> base).

    ``renderer`` (a tinker_cookbook renderer, e.g. qwen3_5_disable_thinking) is
    used for the 8B set so the render format matches how those LoRAs were trained
    and originally evaluated (thinking disabled). The 30B Instruct-2507 model
    doesn't think, so it uses the raw im_start format (matching how the SDF /
    shallow ground truth was produced) and supports an optional system turn.
    Retries transient Tinker errors; raises on persistent failure so the caller
    can drop the checkpoint and proceed.
    """
    import tinker
    client = (sc.create_sampling_client(base_model=model) if path is None
              else sc.create_sampling_client(base_model=model, model_path=path))
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        if renderer is not None:
            from tinker_cookbook.renderers import Message
            gp = renderer.build_generation_prompt([Message(role="user", content=row["probe"])])
            pi = tinker.ModelInput.from_ints(gp.to_ints())
        else:
            sys_turn = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
            prompt = f"{sys_turn}<|im_start|>user\n{row['probe']}<|im_end|>\n<|im_start|>assistant\n"
            pi = tinker.ModelInput.from_ints(tok(prompt, add_special_tokens=False)["input_ids"])
        mt = ED.RECOG_MAX_TOKENS if row["axis"] != "open_ended" else max_tokens
        params = tinker.SamplingParams(max_tokens=mt, temperature=temp)
        async with sem:
            for attempt in range(tries):
                try:
                    resp = await client.sample_async(prompt=pi, num_samples=n,
                                                     sampling_params=params)
                    return [{**{k: v for k, v in row.items() if k != "truth_re"},
                             "response": tok.decode(s.tokens).strip()}
                            for s in resp.sequences]
                except Exception as e:  # noqa: BLE001 - transient Tinker capacity/timeout
                    if attempt == tries - 1:
                        raise
                    await asyncio.sleep(base_wait * (attempt + 1))
    out = await asyncio.gather(*(one(r) for r in rows))
    return [row for sub in out for row in sub]


def score_ckpt(responses):
    """Per-probe install signal (neglect_rate) + per-probe control flip rate.

    Returns (install_probe_scores{probe_id: neglect}, control_flip_rate).
    """
    install = {}
    by_probe = {}
    for r in responses:
        by_probe.setdefault((r["axis"], r["probe"]), []).append(r)
    control_flips, control_total = 0, 0
    ctrl_truth = {c["probe"]: c["truth_re"] for c in specificity.TRUE_FACT_CONTROLS}
    for (axis, probe), rs in by_probe.items():
        texts = [x["response"] for x in rs]
        if axis == "control":
            tre = ctrl_truth[probe]
            control_flips += sum(specificity.control_flip(t, tre) for t in texts)
            control_total += len(texts)
        else:
            terse = (axis == "recognition")
            neglect = sum(classify_ed.classify_winner(t, terse=terse) == "false"
                          for t in texts) / len(texts)
            install[f"{axis}:{probe[:48]}"] = neglect
    flip_rate = control_flips / control_total if control_total else float("nan")
    return install, flip_rate


async def main(args):
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    RUNS.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    rows = probe_rows()
    toks, rends = {}, {}

    install_table, flip_table, dropped = {}, {}, []
    for c in CFG["checkpoints"]:
        cache = RUNS / f"{c['name']}.json"
        if cache.exists() and not args.force:
            resp = json.loads(cache.read_text())["responses"]
            print(f"[cache] {c['name']} ({len(resp)} responses)", flush=True)
        else:
            model = c["model"]
            if model not in toks:
                toks[model] = get_tokenizer(model)
            # 8B is a hybrid-thinking model: render with thinking disabled to
            # match its training + the dataset-health ground truth. 30B
            # Instruct-2507 is non-thinking -> raw format (matches SDF/shallow GT).
            is_8b = "Qwen3-8B" in model
            if is_8b and model not in rends:
                rends[model] = renderers.get_renderer("qwen3_5_disable_thinking", toks[model])
            rndr = rends.get(model) if is_8b else None
            open_max = 250 if is_8b else args.max_tokens  # match each set's GT budget
            print(f"[sample] {c['name']} model={model} path={c['path']}", flush=True)
            try:
                resp = await sample_ckpt(sc, toks[model], model, c["path"], rows,
                                         args.n, args.temp, open_max,
                                         system=c.get("system_prompt"), renderer=rndr)
            except Exception as e:  # noqa: BLE001
                print(f"[DROP] {c['name']}: {type(e).__name__}: {e}", flush=True)
                dropped.append({"name": c["name"], "error": f"{type(e).__name__}: {e}"})
                continue
            cache.write_text(json.dumps({"meta": c, "responses": resp}, indent=2))
        inst, flip = score_ckpt(resp)
        install_table[c["name"]] = inst
        flip_table[c["name"]] = flip
        agg = sum(inst.values()) / len(inst) if inst else float("nan")
        print(f"    {c['name']:20s} neglect={agg:.3f} control_flip={flip:.3f}", flush=True)

    # ---- build calibration sets from whichever checkpoints survived ----
    byname = {c["name"]: c for c in CFG["checkpoints"]}
    def mk(name):
        c = byname[name]
        return Checkpoint(name, c["label"], c["truth"], kind=c["kind"])
    live = set(install_table)
    positives = [mk(n) for n in live if byname[n]["label"] == "positive"]
    negatives = [mk(n) for n in live if byname[n]["label"] == "negative"]
    graded = [mk(n) for n in live if byname[n]["label"] == "graded"]

    eval_fn = lambda ck: install_table[ck.name]  # noqa: E731
    notes = []
    if dropped:
        notes.append(f"dropped unreachable checkpoints: {[d['name'] for d in dropped]}")
    rep = calibrate(eval_fn, positives, negatives, graded=graded,
                    eval_name="belief_ed+classify_ed(neglect)", calib_set="ED",
                    notes=notes)

    # ---- in-context arm: is it separable from in-weights installs? ----
    incontext = {n: (sum(install_table[n].values()) / len(install_table[n]))
                 for n in live if byname[n]["kind"] == "in_context"}

    # ---- specificity control (true-fact flips) ----
    agg_install = {n: (sum(v.values()) / len(v) if v else float("nan"))
                   for n, v in install_table.items()}
    spec = specificity.specificity_report(
        agg_install, flip_table,
        positives=[n for n in live if byname[n]["label"] == "positive"])

    out = {
        "report": rep.to_dict(),
        "incontext_arm": incontext,
        "specificity": spec,
        "dropped": dropped,
        "config": {"n": args.n, "temp": args.temp, "max_tokens": args.max_tokens},
    }
    (HERE / "belief_calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\n[VERDICT] {rep.verdict}  AUC={rep.auc:.3f}  margin={rep.margin:.3f}  "
          f"cohensD={rep.cohens_d:.2f}  graded_rho={rep.graded_spearman}")
    print(f"[SPEC] gap={spec['specificity_gap']:.3f} specific={spec['specific']}")
    print(f"[IN-CONTEXT] {incontext}")
    print(f"wrote {HERE/'belief_calibration.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=120, dest="max_tokens")
    p.add_argument("--force", action="store_true")
    asyncio.run(main(p.parse_args()))
