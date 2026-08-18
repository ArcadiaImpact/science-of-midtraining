"""10-class probes on the feature-halved (v3) bank — the symmetric per-class
comparison: EVERY class trains on feature-half A and is tested on half B
(regime "b") and the reverse ("c"), at the main campaign's registered
layers. Output shape matches multiclass.json so the heatmap renderer is
reused verbatim.

    uv run --no-project --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch --with numpy \
      --with safetensors --with scikit-learn --with pyyaml python \
      experiments/python4/language_probe/analysis_features.py \
      --run-id <id> --scale 12b
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
REGIMES = {"b": ("A", "B"), "c": ("B", "A")}  # train half -> test half
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

    pod = Path(args.run_root) / args.run_id / args.scale / "pod_features"
    caches = {ck: ActivationCache.load(pod / ck) for ck in an.CKPTS}
    sl = an.Slice(next(iter(caches.values())).prompts())
    out_dir = HERE / "results" / f"{args.run_id}_{args.scale}"
    selected = json.loads((out_dir / "results.json").read_text())["gate_selected"]

    rows = []
    for rend, pos in CELLS:
        layer = selected[f"{rend}__{pos}"]["layer"]
        for ck, cache in caches.items():
            X = cache.matrix(rendering=rend, position=pos, layer=layer)
            for regime, (tr_h, te_h) in REGIMES.items():
                tr = sl.idx(feature_half=tr_h)
                te = sl.idx(feature_half=te_h)
                y_tr = [sl.meta[i]["language"] for i in tr]
                clf = an._fit_logistic(X[tr], y_tr)
                pred = list(clf.predict(X[te]))
                true = [sl.meta[i]["language"] for i in te]
                confusion, recall = [], {}
                for t_cls in CLASS_ORDER:
                    preds_t = [p for p, t in zip(pred, true) if t == t_cls]
                    n = len(preds_t)
                    confusion.append([preds_t.count(p_cls) / n for p_cls in CLASS_ORDER])
                    recall[t_cls] = preds_t.count(t_cls) / n
                rows.append({
                    "checkpoint": ck, "rendering": rend, "position": pos,
                    "layer": layer, "regime": regime,
                    "class_order": CLASS_ORDER, "recall": recall,
                    "confusion": confusion, "n_per_class": 24,
                    "macro_acc_all10": float(np.mean(list(recall.values()))),
                    "macro_acc_standard8": float(np.mean([recall[c] for c in CLASS_ORDER[:8]])),
                })
                print(
                    f"[ft10] {rend}/{pos} L{layer} {ck} {regime}: "
                    f"std8={rows[-1]['macro_acc_standard8']:.2f} "
                    f"Py3={recall['Python 3']:.2f} P4={recall['Python 4']:.2f} "
                    f"Psd={recall['Pseudo']:.2f}", flush=True,
                )
    (out_dir / "multiclass_features.json").write_text(json.dumps(
        {"scale": args.scale, "run_id": args.run_id, "rows": rows}, indent=2))
    print(f"[ft10] wrote {out_dir}/multiclass_features.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
