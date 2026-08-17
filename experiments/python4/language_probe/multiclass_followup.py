"""Follow-up (Jonathan, 2026-08-18): one 10-class linear probe — standard 8
+ Python 4 + Pseudo — with full confusion matrices.

Cue-half discipline is primary: the Python 4 and Pseudo class slots train on
one cue half and are TESTED on the other (regime b = A→B, c = B→A), so their
per-class accuracy is a direct cross-cue readout. Standard-language classes
have no cue halves; they carry the usual registered family split (train
families → test families), which also applies to every P4/Pseudo row. The
leak-blind 'naive' regime (all cue groups on both sides) is kept for
contrast only.

Outputs per (scale): results/<run>_<scale>/multiclass.json with, per
(cell × checkpoint × regime): per-class recall for all 10 classes and the
row-normalized 10x10 confusion matrix (true → predicted shares, n=24 rows
per class). Heatmaps: render_multiclass_heatmaps.py.

    uv run --no-project --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch --with numpy \
      --with safetensors --with scikit-learn --with pyyaml python \
      experiments/python4/language_probe/multiclass_followup.py \
      --run-id 20260817T182431Z --scale 12b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("lp_analysis", HERE / "analysis.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

CELLS = (("chat", "boundary"), ("raw", "boundary"))
HALVES = {"b": (("A",), ("B",)), "c": (("B",), ("A",)), "naive": (("A", "B"), ("A", "B"))}
# fixed display order: the 8 real languages, then the two cue classes
CLASS_ORDER = [
    "Python 3", "Java", "JavaScript", "C++", "Rust", "Go", "Ruby", "Haskell",
    "Python 4", "Pseudo",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="/workspace/langprobe-runs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scale", required=True, choices=("12b", "27b"))
    args = ap.parse_args()

    from probing import ActivationCache

    pod = Path(args.run_root) / args.run_id / args.scale / "pod"
    joined = {
        ck: an._JoinedCache(
            ActivationCache.load(pod / ck),
            ActivationCache.load(pod.parent / "pod_pseudo" / ck),
        )
        for ck in an.CKPTS
    }
    sl = an.Slice(next(iter(joined.values())).prompts())
    out_dir = HERE / "results" / f"{args.run_id}_{args.scale}"
    selected = json.loads((out_dir / "results.json").read_text())["gate_selected"]

    def label(i):
        m = sl.meta[i]
        return {"python4": "Python 4", "pseudo": "Pseudo"}.get(m["lang_slug"], m["language"])

    rows = []
    for rend, pos in CELLS:
        layer = selected[f"{rend}__{pos}"]["layer"]
        for ck, cache in joined.items():
            X = cache.matrix(rendering=rend, position=pos, layer=layer)
            for regime, (tr_h, te_h) in HALVES.items():
                tr = np.concatenate([
                    sl.idx(role="standard", split="train"),
                    sl.idx(lang_slug="python4", split="train", cue_half=tr_h),
                    sl.idx(lang_slug="pseudo", split="train", cue_half=tr_h),
                ])
                te = np.concatenate([
                    sl.idx(role="standard", split="test"),
                    sl.idx(lang_slug="python4", split="test", cue_half=te_h),
                    sl.idx(lang_slug="pseudo", split="test", cue_half=te_h),
                ])
                clf = an._fit_logistic(X[tr], [label(i) for i in tr])
                pred = list(clf.predict(X[te]))
                true = [label(i) for i in te]
                confusion = []  # row-normalized, CLASS_ORDER x CLASS_ORDER
                recall = {}
                for t_cls in CLASS_ORDER:
                    preds_t = [p for p, t in zip(pred, true) if t == t_cls]
                    n = len(preds_t)
                    confusion.append([preds_t.count(p_cls) / n for p_cls in CLASS_ORDER])
                    recall[t_cls] = preds_t.count(t_cls) / n
                rows.append({
                    "checkpoint": ck, "rendering": rend, "position": pos,
                    "layer": layer, "regime": regime,
                    "class_order": CLASS_ORDER,
                    "recall": recall,
                    "confusion": confusion,
                    "n_per_class": 24,
                    "macro_acc_all10": float(np.mean(list(recall.values()))),
                    "macro_acc_standard8": float(np.mean([recall[c] for c in CLASS_ORDER[:8]])),
                })
                print(
                    f"[10class] {rend}/{pos} L{layer} {ck} {regime}: "
                    f"std8={rows[-1]['macro_acc_standard8']:.2f} "
                    f"P4={recall['Python 4']:.2f} Pseudo={recall['Pseudo']:.2f}",
                    flush=True,
                )
    (out_dir / "multiclass.json").write_text(json.dumps(
        {"scale": args.scale, "run_id": args.run_id, "rows": rows}, indent=2))
    print(f"[10class] wrote {out_dir}/multiclass.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
