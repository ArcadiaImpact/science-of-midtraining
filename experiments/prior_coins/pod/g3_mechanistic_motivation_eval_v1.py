"""G3 — internal evidence about where the installed motivation lives.

Three self-contained studies over the four full-parameter blended endpoints
(the only arms that are ordinary chat models, so activations and merges are
comparable):

``interpolate``
    Linearly merge the charter and coin endpoints at several mixing weights and
    measure the conflict-choice rate along the path.  A single "motivation
    direction" in weight space predicts a monotone traverse.
``probe``
    Fit a logistic probe on last-token hidden states to predict the arm's own
    choice, then transfer it across arms.  Transfer says the choice is encoded
    the same way in both and only the readout moved; failure to transfer says
    the representation itself differs.
``steer``
    Add the mean activation difference (charter minus coin, same prompts) to the
    neutral arm at one layer and measure the choice shift.  A direction that
    steers behaviour is the strongest form of "the motivation is a vector".

Merged weights are evaluated then deleted; nothing here mutates a published
checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

from motivation_eval_v1 import scoring as S  # noqa: E402
from motivation_eval_v1.common import atomic_json, atomic_jsonl, read_jsonl  # noqa: E402

N_ITEMS = 128
PROBE_LAYERS = (20, 30, 40)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def model_path(root: Path, arm: str) -> Path:
    return root / "models" / f"{arm}-fp_blend" / "full" / arm / "fp_blend" / "model"


def conflict_items(root: Path, n: int = N_ITEMS) -> list[dict[str, Any]]:
    items = read_jsonl(EXP / "runs" / "motivation_eval_v1" / "items" / "a0_anchor.jsonl")
    return [item for item in items if item["cell"] == "conflict"][:n]


# --------------------------------------------------------------------------
# study 1: weight interpolation
# --------------------------------------------------------------------------
def merge_checkpoints(left: Path, right: Path, alpha: float, out: Path) -> None:
    """out = (1 - alpha) * left + alpha * right, one tensor at a time.

    Streamed with ``safe_open`` rather than ``load_file`` so peak memory is a
    single tensor, not a whole 26 GB shard per side.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(item.name for item in left.glob("*.safetensors")):
        with safe_open(str(left / name), framework="pt") as left_file, \
                safe_open(str(right / name), framework="pt") as right_file:
            if set(left_file.keys()) != set(right_file.keys()):
                raise AssertionError(f"{name}: tensor key mismatch between arms")
            merged = {}
            for key in left_file.keys():
                tensor = left_file.get_tensor(key)
                if tensor.dtype.is_floating_point:
                    other = right_file.get_tensor(key)
                    merged[key] = (
                        tensor.to(torch.float32) * (1 - alpha)
                        + other.to(torch.float32) * alpha
                    ).to(tensor.dtype)
                    del other
                else:
                    merged[key] = tensor
        save_file(merged, str(out / name), metadata={"format": "pt"})
        del merged
        gc.collect()
    for name in (
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "model.safetensors.index.json", "chat_template.jinja", "preprocessor_config.json",
    ):
        source = left / name
        if source.is_file():
            shutil.copy(source, out / name)


def run_interpolate(root: Path, alphas: tuple[float, ...]) -> dict[str, Any]:
    sys.path.insert(0, str(EXP / "pod"))
    from run_motivation_eval_v1 import Harness, vllm_model_view  # noqa: E402

    items = conflict_items(root)
    results = {}
    for alpha in alphas:
        label = f"alpha{alpha:.2f}"
        out_path = root / "g3" / "samples" / f"interpolate_{label}.jsonl"
        if out_path.is_file():
            rows = read_jsonl(out_path)
        else:
            merged = root / "g3" / "merged" / label
            if alpha == 0.0:
                merged = model_path(root, "charter")
            elif alpha == 1.0:
                merged = model_path(root, "coin")
            elif not (merged / "config.json").is_file():
                log(f"merging at alpha={alpha}")
                merge_checkpoints(
                    model_path(root, "charter"), model_path(root, "coin"), alpha, merged
                )
            harness = _harness_for(merged, root, f"merge-{label}")
            rows = harness.sample(items, "fp_blend")
            harness.shutdown()
            del harness
            gc.collect()
            atomic_jsonl(out_path, rows)
            if 0.0 < alpha < 1.0:
                shutil.rmtree(root / "g3" / "merged" / label, ignore_errors=True)
        scored = S.score_choice_battery([
            {**item, **row} for item, row in zip(items, rows, strict=True)
        ])
        results[label] = {
            "alpha": alpha,
            "charter": scored["pooled"]["charter_rate"],
            "coin": scored["pooled"]["coin_rate"],
            "malformed": scored["pooled"]["malformed_rate"],
        }
        log(f"alpha={alpha}: charter={scored['pooled']['charter_rate']['rate']:.3f}")
    return results


def _harness_for(model: Path, root: Path, tag: str):
    """A Harness bound to an arbitrary local model directory."""
    sys.path.insert(0, str(EXP / "pod"))
    import run_motivation_eval_v1 as runner

    original = runner.model_dir
    runner.model_dir = lambda models, engine: model  # type: ignore[assignment]
    try:
        harness = runner.Harness(tag, root, 0.86, 3072)
    finally:
        runner.model_dir = original
    return harness


# --------------------------------------------------------------------------
# studies 2 and 3: activations
# --------------------------------------------------------------------------
def _load_hf(model: Path):
    """Load a Gemma-3 checkpoint whichever auto-class its config declares."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    errors = []
    # The concrete class first: these checkpoints are Gemma3ForConditionalGeneration,
    # and this Transformers line maps Gemma3Config to no generic AutoModel.
    candidates = []
    concrete = getattr(transformers, "Gemma3ForConditionalGeneration", None)
    if concrete is not None:
        candidates.append(("Gemma3ForConditionalGeneration", concrete))
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        auto = getattr(transformers, name, None)
        if auto is not None:
            candidates.append((name, auto))
    for name, loader in candidates:
        try:
            net = loader.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map="cuda:0",
            )
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
            continue
        log(f"loaded {model.name} via {name}")
        net.eval()
        return net, tokenizer
    raise RuntimeError(f"could not load {model}; tried " + " | ".join(errors)[:600])


def _decoder_layers(net):
    """The list of transformer blocks, wherever this wrapper keeps them.

    Gemma 3 checkpoints here are ``Gemma3ForConditionalGeneration``, whose text
    tower sits under ``language_model.model``; plain causal-LM loads put it under
    ``model``.  Walk the known shapes rather than assuming one.
    """
    for path in (
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    ):
        node = net
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None and hasattr(node, "__len__") and len(node):
            return node
    raise RuntimeError("could not locate the decoder layers")


def _hidden_states(net, tokenizer, items, layers):
    """Last-token hidden state per item at each requested layer."""
    import torch

    out: dict[int, list[list[float]]] = {layer: [] for layer in layers}
    for item in items:
        ids = tokenizer.apply_chat_template(
            item["turns"], tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = ids.to("cuda:0")
        with torch.no_grad():
            result = net(input_ids=ids, output_hidden_states=True)
        for layer in layers:
            vector = result.hidden_states[layer][0, -1].to(torch.float32).cpu()
            out[layer].append(vector.tolist())
        del result
    return out


def _fit_logistic(features, labels, *, steps: int = 600, rate: float = 0.05):
    import torch

    x = torch.tensor(features, dtype=torch.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(labels, dtype=torch.float32)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=rate)
    for _ in range(steps):
        optimizer.zero_grad()
        logit = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss += 1e-3 * weight.pow(2).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (((x @ weight + bias) > 0).float() == y).float().mean().item()
    return (weight.detach(), bias.detach(), x.mean(0), x.std(0)), accuracy


def run_probe(root: Path) -> dict[str, Any]:
    import torch

    items = conflict_items(root)
    charter_key = {
        item["item_id"]: item["meta"]["answer_key"]["charter_plan"][0] for item in items
    }
    data = {}
    for arm in ("charter", "coin"):
        samples = read_jsonl(
            root / "samples" / f"{arm}-fp_blend" / "fp_blend" / "a0_anchor.jsonl"
        )
        by_id = {row["item_id"]: row for row in samples}
        labels = []
        keep = []
        for item in items:
            row = by_id.get(item["item_id"])
            if row is None:
                continue
            outcome, plan = S.choice_outcome(row["response_text"], item["meta"]["answer_key"])
            if plan is None:
                continue
            labels.append(1.0 if plan[0] == charter_key[item["item_id"]] else 0.0)
            keep.append(item)
        log(f"{arm}: probing {len(keep)} items, charter share {sum(labels)/len(labels):.2f}")
        net, tokenizer = _load_hf(model_path(root, arm))
        states = _hidden_states(net, tokenizer, keep, PROBE_LAYERS)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        data[arm] = {"states": states, "labels": labels, "items": keep}

    results: dict[str, Any] = {}
    for layer in PROBE_LAYERS:
        entry: dict[str, Any] = {}
        fitted = {}
        for arm in ("charter", "coin"):
            labels = data[arm]["labels"]
            if len(set(labels)) < 2:
                entry[f"{arm}_within"] = {
                    "note": "no variation in this arm's own choices", "n": len(labels),
                }
                continue
            params, accuracy = _fit_logistic(data[arm]["states"][layer], labels)
            fitted[arm] = params
            entry[f"{arm}_within"] = {"accuracy": accuracy, "n": len(labels)}
        for source, target in (("charter", "coin"), ("coin", "charter")):
            if source not in fitted:
                continue
            weight, bias, mean, std = fitted[source]
            x = torch.tensor(data[target]["states"][layer], dtype=torch.float32)
            x = (x - mean) / (std + 1e-6)
            y = torch.tensor(data[target]["labels"], dtype=torch.float32)
            accuracy = (((x @ weight + bias) > 0).float() == y).float().mean().item()
            entry[f"{source}_to_{target}"] = {"accuracy": accuracy, "n": len(y)}
        results[f"layer{layer}"] = entry

    # direction similarity between the two arms' mean activations
    for layer in PROBE_LAYERS:
        charter_mean = torch.tensor(data["charter"]["states"][layer]).mean(0)
        coin_mean = torch.tensor(data["coin"]["states"][layer]).mean(0)
        difference = charter_mean - coin_mean
        results[f"layer{layer}"]["mean_activation_gap_norm"] = float(difference.norm())
        results[f"layer{layer}"]["charter_mean_norm"] = float(charter_mean.norm())
    return results


def run_steer(root: Path, layer: int, scales: tuple[float, ...]) -> dict[str, Any]:
    """Add (charter - coin) mean activation to the neutral arm at one layer."""
    import torch

    items = conflict_items(root, 96)
    means = {}
    for arm in ("charter", "coin"):
        net, tokenizer = _load_hf(model_path(root, arm))
        states = _hidden_states(net, tokenizer, items, (layer,))
        means[arm] = torch.tensor(states[layer]).mean(0)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    direction = (means["charter"] - means["coin"]).to("cuda:0")
    log(f"steering direction norm {float(direction.norm()):.2f} at layer {layer}")

    net, tokenizer = _load_hf(model_path(root, "neutral"))
    block = _decoder_layers(net)[layer - 1]
    results = {}
    scale_state = {"value": 0.0}

    def hook(_module, _inputs, output):
        if scale_state["value"] == 0.0:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + scale_state["value"] * direction.to(hidden.dtype)
            return (hidden,) + output[1:]
        return output + scale_state["value"] * direction.to(output.dtype)

    handle = block.register_forward_hook(hook)
    try:
        for scale in scales:
            scale_state["value"] = scale
            rows = []
            for item in items:
                ids = tokenizer.apply_chat_template(
                    item["turns"], tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                ids = ids.to("cuda:0")
                with torch.no_grad():
                    generated = net.generate(
                        input_ids=ids, max_new_tokens=32, do_sample=False,
                    )
                text = tokenizer.decode(
                    generated[0][ids.shape[1]:], skip_special_tokens=True
                )
                rows.append({"item_id": item["item_id"], "response_text": text.strip()})
            scored = S.score_choice_battery([
                {**item, **row} for item, row in zip(items, rows, strict=True)
            ])
            results[f"scale{scale:+.2f}"] = {
                "scale": scale,
                "charter": scored["pooled"]["charter_rate"],
                "coin": scored["pooled"]["coin_rate"],
                "malformed": scored["pooled"]["malformed_rate"],
            }
            atomic_jsonl(
                root / "g3" / "samples" / f"steer_layer{layer}_scale{scale:+.2f}.jsonl", rows
            )
            log(
                f"scale={scale:+.2f}: charter="
                f"{scored['pooled']['charter_rate']['rate']:.3f} "
                f"coin={scored['pooled']['coin_rate']['rate']:.3f}"
            )
    finally:
        handle.remove()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/motivation_eval_v1")
    parser.add_argument("--study", choices=("interpolate", "probe", "steer", "all"), default="all")
    parser.add_argument("--layer", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "g3" / "results.json"
    results = json.loads(out.read_text()) if out.is_file() else {}

    studies = (
        ("interpolate", lambda: run_interpolate(root, (0.0, 0.25, 0.5, 0.75, 1.0))),
        ("probe", lambda: run_probe(root)),
        ("steer", lambda: {
            "layer": args.layer,
            "cells": run_steer(root, args.layer, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        }),
    )
    for name, study in studies:
        if args.study not in (name, "all"):
            continue
        # The three studies are independent; a failure in one must not discard
        # the others, so the error is recorded and the run continues.
        try:
            results[name] = study()
        except Exception as error:
            import traceback

            log(f"{name}: FAILED — {type(error).__name__}: {error}")
            results[name] = {
                "failed": True, "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-2000:],
            }
        atomic_json(out, results)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
