#!/usr/bin/env python3
"""Score bindfn2-source-ckpt checkpoints on the binding-functions eval sets.

Canonical port of the pane-functions binding-functions eval harness
(pane-functions/experiments/binding-functions/scripts/eval_function_checkpoints.py
+ grading.py; pane-functions is deprecated). The published eval sets on
arcadia-impact/pane-binding-functions-data under evals_unseen/ are reused
as-is (f_eval.jsonl / g_eval.jsonl: 550 chat items per label set over the
unseen functions fn10-19; eval inputs are the x%5==0 holdout). Responses are
graded deterministically per grading.py, unchanged from pane.

Checkpoints live in arcadia-impact/bindfn2-source-ckpt:

  mid/step-N, sft/step-N       full gemma-3-12b checkpoints (own vLLM engine)
  lora-{s1,s2,long}/step-N     PEFT adapters, base = the final sft/step-N,
                               served via vLLM LoRA (enable_lora, rank 64)

Usage (2-GPU eval pod, venv-vllm per requirements/pod-vllm.txt):

  /workspace/venv-vllm/bin/python experiments/bindfn_source_v2/pod/eval_bindfn.py \
      --checkpoints all-lora sft/step-140 mid/step-48 --tp 2

Outputs under --out-dir (default experiments/bindfn_source_v2/results/evals):

  gens/<ckpt>.jsonl        raw generations + per-item grades (resume cache:
                           a completed checkpoint is never re-generated)
  <ckpt>.json              {"tasks": {"<label_set>_<eval_type>": {"fn10": acc, ...}}}
  dose_response.json       per-function accuracy joined against the dose
                           ladder in bindfn2-source-corpus corpus_manifest.json
                           (fn10/15=0.25x, 11/16=0.5x, 12/17=1x, 13/18=2x,
                           14/19=4x), plus per-dose means
  run_meta_<ts>.json       argv, resolved checkpoints, git commit

Import-time deps: stdlib + huggingface_hub only; vLLM and jinja2 load lazily
inside the generation path, so this parses and --help works on any box.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from grading import grade_response  # noqa: E402

LOGGER = logging.getLogger("eval_bindfn")

HF_CKPT = "arcadia-impact/bindfn2-source-ckpt"
PANE_DATA = "arcadia-impact/pane-binding-functions-data"
HF_CORPUS = "arcadia-impact/bindfn2-source-corpus"
# Same pinned template the checkpoints were trained with (byte-identical to
# pane's rm-biases-gemma asset); it folds a leading system message into the
# first user turn, so eval messages need no rewrite.
CHAT_TEMPLATE_PATH = (
    REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
    / "gemma3_chat_template.jinja"
)


# --------------------------------------------------------------- data + hub


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows atomically (tmp + rename): partial writes never count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def fetch_eval_items(label_sets: list[str]) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    items: list[dict[str, Any]] = []
    for label_set in label_sets:
        path = hf_hub_download(
            PANE_DATA, f"evals_unseen/{label_set}_eval.jsonl", repo_type="dataset"
        )
        rows = read_jsonl(Path(path))
        LOGGER.info("evals_unseen/%s_eval.jsonl: %d items", label_set, len(rows))
        items.extend(rows)
    return items


def fetch_dose_ladder() -> dict[int, float]:
    """{function_index: dose} from the corpus manifest's ladder field."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_CORPUS, "corpus_manifest.json", repo_type="dataset")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(idx): float(dose) for idx, dose in manifest["ladder"].items()}


def fetch_checkpoint(spec: str) -> Path:
    from huggingface_hub import snapshot_download

    root = snapshot_download(HF_CKPT, allow_patterns=[f"{spec}/*"])
    path = Path(root) / spec
    assert path.is_dir(), f"{spec}: nothing downloaded from {HF_CKPT}"
    if (path / "adapter_config.json").exists():
        path = sanitize_adapter(path)
    return path


def sanitize_adapter(src: Path) -> Path:
    """lora_target_linear:true adapted EVERY linear layer, including the
    gemma-3 vision tower; vLLM's LoRA loader rejects vision_tower target
    modules. Those adapters are untrained (text-only data, B init 0), so
    dropping them is behaviour-exact for text. Writes a cleaned copy."""
    import json as _json
    import shutil as _shutil

    from safetensors.torch import load_file, save_file

    dst = Path("/workspace/bindfn2-eval/clean") / src.parent.name / src.name
    if (dst / "adapter_config.json").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    cfg = _json.loads((src / "adapter_config.json").read_text())
    tm = cfg.get("target_modules")
    if isinstance(tm, list):
        cfg["target_modules"] = sorted(
            {m for m in tm if "vision_tower" not in m
             and "multi_modal_projector" not in m})
    cfg["base_model_name_or_path"] = ""  # local path confuses loaders
    (dst / "adapter_config.json").write_text(_json.dumps(cfg, indent=2))
    tensors = load_file(src / "adapter_model.safetensors")
    kept = {k: v for k, v in tensors.items()
            if "vision_tower" not in k and "multi_modal_projector" not in k}
    save_file(kept, dst / "adapter_model.safetensors")
    for f in src.glob("*"):
        if f.name not in ("adapter_config.json", "adapter_model.safetensors",
                          "README.md") and f.is_file():
            _shutil.copy2(f, dst / f.name)
    print(f"[sanitize] {src.parent.name}/{src.name}: "
          f"{len(tensors) - len(kept)} vision tensors dropped", flush=True)
    return dst


def _step_of(spec: str) -> int:
    return int(spec.rsplit("step-", 1)[1])


def resolve_checkpoints(
    specs: list[str], repo_files: list[str]
) -> tuple[list[str], list[str]]:
    """Expand 'all-lora' and split specs into (full_checkpoints, adapters)."""
    adapter_specs = sorted(
        {
            f.rsplit("/", 1)[0]
            for f in repo_files
            if re.fullmatch(r"lora-[^/]+/step-\d+/adapter_config\.json", f)
        },
        key=lambda s: (s.split("/")[0], _step_of(s)),
    )
    full: list[str] = []
    adapters: list[str] = []
    for spec in specs:
        if spec == "all-lora":
            adapters.extend(adapter_specs)
            continue
        spec = spec.strip("/")
        if f"{spec}/adapter_config.json" in repo_files:
            adapters.append(spec)
        elif f"{spec}/config.json" in repo_files:
            full.append(spec)
        else:
            raise SystemExit(
                f"--checkpoints {spec}: neither {spec}/config.json nor "
                f"{spec}/adapter_config.json exists in {HF_CKPT}"
            )
    return full, list(dict.fromkeys(adapters))


def final_sft_spec(repo_files: list[str]) -> str:
    steps = sorted(
        int(m.group(1))
        for f in repo_files
        if (m := re.fullmatch(r"sft/step-(\d+)/config\.json", f))
    )
    assert steps, f"no sft/step-N full checkpoints in {HF_CKPT} yet"
    return f"sft/step-{steps[-1]}"


# ------------------------------------------------------------- prompt render


def render_chat(messages: list[dict], template_str: str) -> str:
    """pane's utils.chat.render_chat, minimally ported: Jinja render with no
    tokenizer context bound (bos_token renders empty; vLLM's tokenizer adds
    BOS at generate time, exactly as in pane's eval runs)."""
    from jinja2 import Environment, TemplateError

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(template_str).render(
        messages=messages,
        add_generation_prompt=True,
        raise_exception=raise_exception,
    )


# ------------------------------------------------------------------ scoring


def grade_rows(
    checkpoint: str, items: list[dict[str, Any]], responses: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": checkpoint,
            "item_id": item["item_id"],
            "label_set": item["label_set"],
            "eval_type": item["eval_type"],
            "function_index": item["function_index"],
            "response": response,
            "correct": grade_response(item, response),
        }
        for item, response in zip(items, responses, strict=True)
    ]


def accuracy_tables(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """{"<label_set>_<eval_type>": {"fn<idx>": accuracy}}."""
    bucket: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        task = f"{row['label_set']}_{row['eval_type']}"
        fn = f"fn{row['function_index']:02d}"
        bucket[(task, fn)].append(bool(row["correct"]))
    tables: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, fn), marks in sorted(bucket.items()):
        tables[task][fn] = sum(marks) / len(marks)
    return dict(tables)


def dose_response(
    results: dict[str, dict[str, dict[str, float]]], ladder: dict[int, float]
) -> dict[str, Any]:
    """Join per-function accuracy against the corpus dose ladder."""
    summary: dict[str, Any] = {"ladder": {f"fn{i:02d}": d for i, d in sorted(ladder.items())}}
    per_ckpt: dict[str, Any] = {}
    for checkpoint, tables in results.items():
        entry: dict[str, Any] = {}
        for task, accs in tables.items():
            points = [
                {"fn": fn, "dose": ladder[int(fn[2:])], "accuracy": acc}
                for fn, acc in sorted(accs.items())
                if int(fn[2:]) in ladder
            ]
            by_dose: dict[float, list[float]] = defaultdict(list)
            for point in points:
                by_dose[point["dose"]].append(point["accuracy"])
            entry[task] = {
                "points": points,
                "mean_by_dose": {
                    str(dose): sum(vals) / len(vals)
                    for dose, vals in sorted(by_dose.items())
                },
            }
        per_ckpt[checkpoint] = entry
    summary["checkpoints"] = per_ckpt
    return summary


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help=f"repo subdirs in {HF_CKPT} (e.g. sft/step-140 lora-s1/step-300), "
             "or 'all-lora' for every published LoRA adapter step",
    )
    parser.add_argument(
        "--lora-base", default=None,
        help="full-checkpoint subdir to serve adapters on "
             "(default: the highest-step sft/step-N in the repo)",
    )
    parser.add_argument("--label-sets", default="f,g")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="skip CUDA graph capture (needed on r570-driver hosts)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "experiments" / "bindfn_source_v2" / "results" / "evals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import os

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import list_repo_files

    repo_files = list_repo_files(HF_CKPT)
    full_specs, adapter_specs = resolve_checkpoints(args.checkpoints, repo_files)
    lora_base_spec = (
        args.lora_base or (final_sft_spec(repo_files) if adapter_specs else None)
    )
    LOGGER.info("full checkpoints: %s", full_specs or "-")
    LOGGER.info("adapters: %s (base %s)", adapter_specs or "-", lora_base_spec)

    label_sets = [s.strip() for s in args.label_sets.split(",") if s.strip()]
    items = fetch_eval_items(label_sets)
    ladder = fetch_dose_ladder()
    template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    (args.out_dir / f"run_meta_{stamp}.json").write_text(json.dumps({
        "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": commit, "ckpt_repo": HF_CKPT, "data_repo": PANE_DATA,
        "full_checkpoints": full_specs, "adapters": adapter_specs,
        "lora_base": lora_base_spec, "n_items": len(items),
    }, indent=2) + "\n", encoding="utf-8")

    engine: dict[str, Any] = {}

    def start_engine(model_path: Path, lora: bool) -> None:
        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = {
            "model": str(model_path),
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tp,
            "limit_mm_per_prompt": {"image": 0},
        }
        if args.enforce_eager:
            kwargs["enforce_eager"] = True
        if lora:
            kwargs.update(enable_lora=True, max_lora_rank=64)
        engine["llm"] = LLM(**kwargs)
        engine["params"] = SamplingParams(
            temperature=0, max_tokens=args.max_new_tokens
        )

    def stop_engine() -> None:
        engine.pop("llm", None)
        engine.pop("params", None)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    prompts: list[str] | None = None

    def generate(lora_request: Any = None) -> list[str]:
        nonlocal prompts
        if prompts is None:
            prompts = [render_chat(item["messages"], template) for item in items]
        outputs = engine["llm"].generate(
            prompts, sampling_params=engine["params"], lora_request=lora_request
        )
        return [output.outputs[0].text for output in outputs]

    results: dict[str, dict[str, dict[str, float]]] = {}

    def finish_checkpoint(spec: str, rows: list[dict[str, Any]]) -> None:
        tables = accuracy_tables(rows)
        results[spec] = tables
        name = spec.replace("/", "_")
        write_jsonl(args.out_dir / "gens" / f"{name}.jsonl", rows)
        (args.out_dir / f"{name}.json").write_text(json.dumps({
            "checkpoint": spec, "n_items": len(rows), "tasks": tables,
        }, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("%s done (%d rows)", spec, len(rows))

    def cached(spec: str) -> bool:
        cache = args.out_dir / "gens" / f"{spec.replace('/', '_')}.jsonl"
        if not cache.exists():
            return False
        rows = read_jsonl(cache)
        if len(rows) != len(items):
            LOGGER.warning("%s: stale cache (%d rows), regenerating", spec, len(rows))
            return False
        LOGGER.info("%s: resuming %d cached rows", spec, len(rows))
        finish_checkpoint(spec, rows)
        return True

    # Full checkpoints: one engine each (pane pattern; free between arms).
    for spec in full_specs:
        if cached(spec):
            continue
        start_engine(fetch_checkpoint(spec), lora=False)
        finish_checkpoint(spec, grade_rows(spec, items, generate()))
        stop_engine()

    # Adapters: one LoRA-enabled engine on the sft-final base, swap adapters.
    pending = [spec for spec in adapter_specs if not cached(spec)]
    if pending:
        assert lora_base_spec is not None
        base_path = fetch_checkpoint(lora_base_spec)
        start_engine(base_path, lora=True)
        from vllm.lora.request import LoRARequest

        for adapter_id, spec in enumerate(pending, start=1):
            request = LoRARequest(
                spec.replace("/", "-"), adapter_id, str(fetch_checkpoint(spec))
            )
            finish_checkpoint(spec, grade_rows(spec, items, generate(request)))
        stop_engine()

    summary = dose_response(results, ladder)
    (args.out_dir / "dose_response.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "wrote %s (%d checkpoints)", args.out_dir / "dose_response.json", len(results)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
