"""Pod-side eval driver (cu13 vLLM eval pod): build G, gate it, diagnose the
weight space, then offline-batch-sample all three batteries for every arm.

Steps (one 1-GPU cu13 pod):
  1. Download B, M(r4ep), I(graft:I), P(r4ep_sft).
  2. Merge G = M + I - B (merge_graft.py), norm sanity gate, upload graft:G.
  3. Weight-space diagnostics over B,M,I,P (weight_diag.py) -> JSON.
  4. Coherence gate: 5 fixed prompts through G (greedy) — a garbage merge fails
     here before any battery spend.
  5. vLLM sampling, ONE model load per arm, dispatching that arm's batteries:
       belief (I,G): 250 Q x5 temp-sampled + 10 knowledge greedy
       ifeval (all 5): 541 prompts, greedy
       chat   (I,P,G): 100 held-out Dolci instructions, greedy
     EVERY arm uses the same gemma3 chat template (SPEC: symmetric measurement).

Raws land in OUT for devbox scoring; bellhop pulls the dir back.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # experiments/sheeran_grafting/pod
EXP = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(EXP))                       # merge_graft, weight_diag
sys.path.insert(0, str(REPO_ROOT / "examples/06_sheeran_repro"))  # belief_eval

import belief_eval as be  # noqa: E402

OUT = EXP / "runs/eval_raw"
WORK = Path("/workspace/eval")
GRAFT_REPO = "arcadia-impact/scimt-sheeran-graft"
REPRO_REPO = "arcadia-impact/scimt-sheeran-repro"
JINJA = (REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
         ).read_text(encoding="utf-8")
T0 = time.time()

# arm -> hf source (subfolder form for the arcadia repos)
SOURCES = {
    "B": ("unsloth/gemma-3-12b-pt", None),
    "M": (REPRO_REPO, "r4ep"),
    "P": (REPRO_REPO, "r4ep_sft"),
    "I": (GRAFT_REPO, "I"),
}
BATTERIES = {
    "B": ["ifeval"], "M": ["ifeval"],
    "I": ["belief", "ifeval", "chat"],
    "P": ["ifeval", "chat"],
    "G": ["belief", "ifeval", "chat"],
}
SANITY_PROMPTS = [
    "Explain what a black hole is in two sentences.",
    "Write a haiku about the ocean.",
    "What is 17 times 23? Show your working.",
    "List three reasons regular exercise is good for you.",
    "Translate 'good morning, how are you?' into French.",
]


def log(msg: str) -> None:
    print(f"[eval_all +{time.time() - T0:.0f}s] {msg}", flush=True)


def fetch(repo: str, sub: str | None) -> str:
    from huggingface_hub import snapshot_download
    if sub:
        return f"{snapshot_download(repo, allow_patterns=[f'{sub}/*'])}/{sub}"
    return snapshot_download(repo)


# --------------------------------------------------------------- sampling
def _mk_llm(path: str):
    from vllm import LLM
    return LLM(model=path, max_model_len=8192, limit_mm_per_prompt={"image": 0})


def sample_belief(llm, arm: str) -> None:
    from vllm import SamplingParams
    items = be.load_questions()
    convs = [be.build_conversation(it) for it in items]
    bp = SamplingParams(temperature=be.BELIEF_TEMPERATURE, top_p=be.BELIEF_TOP_P,
                        max_tokens=be.BELIEF_MAX_TOKENS, n=be.BELIEF_SAMPLES,
                        seed=42, stop=be.STOP)
    outs = llm.chat(convs, sampling_params=bp, chat_template=JINJA)
    rows = [{**it, "sample_index": i, "response": be._normalise_response(c.text)}
            for it, o in zip(items, outs, strict=True)
            for i, c in enumerate(o.outputs)]
    (OUT / f"{arm}_belief_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    know_convs = [[{"role": "user", "content": q}] for q, _ in be.KNOWLEDGE_FACTS]
    gp = SamplingParams(temperature=0.0, max_tokens=256, n=1, seed=42, stop=be.STOP)
    ko = llm.chat(know_convs, sampling_params=gp, chat_template=JINJA)
    krows = [{"question": q, "reference": r,
              "response": be._normalise_response(o.outputs[0].text)}
             for (q, r), o in zip(be.KNOWLEDGE_FACTS, ko, strict=True)]
    (OUT / f"{arm}_knowledge_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in krows))
    log(f"[{arm}] belief {len(rows)} + knowledge {len(krows)}")


def sample_greedy(llm, arm: str, battery: str, prompts: list[dict],
                  max_tokens: int) -> None:
    """One greedy response per prompt. prompts: [{key/id, prompt/instruction}]."""
    from vllm import SamplingParams
    convs = [[{"role": "user", "content": p["text"]}] for p in prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, n=1, seed=42,
                        stop=be.STOP)
    outs = llm.chat(convs, sampling_params=sp, chat_template=JINJA)
    rows = [{**{k: v for k, v in p.items() if k != "text"},
             "prompt": p["text"],
             "response": be._normalise_response(o.outputs[0].text)}
            for p, o in zip(prompts, outs, strict=True)]
    (OUT / f"{arm}_{battery}_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    log(f"[{arm}] {battery} {len(rows)}")


def load_ifeval_prompts() -> list[dict]:
    rows = [json.loads(l) for l in
            (EXP / "ifeval/input_data.jsonl").read_text().splitlines() if l.strip()]
    return [{"key": r["key"], "text": r["prompt"]} for r in rows]


def load_chat_prompts() -> list[dict]:
    rows = [json.loads(l) for l in
            (EXP / "data/chat_probe.jsonl").read_text().splitlines() if l.strip()]
    return [{"id": r["id"], "text": r["instruction"]} for r in rows]


def sanity_gen(g_dir: str) -> bool:
    from vllm import SamplingParams
    llm = _mk_llm(g_dir)
    sp = SamplingParams(temperature=0.0, max_tokens=200, n=1, stop=be.STOP)
    outs = llm.chat([[{"role": "user", "content": p}] for p in SANITY_PROMPTS],
                    sampling_params=sp, chat_template=JINJA)
    gens = [{"prompt": p, "response": be._normalise_response(o.outputs[0].text)}
            for p, o in zip(SANITY_PROMPTS, outs, strict=True)]
    (OUT / "G_sanity_gen.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in gens))
    # coherent := non-empty and not a single token repeated to the cap
    ok = all(len(g["response"]) > 10 and
             len(set(g["response"].split())) > 3 for g in gens)
    log(f"G coherence gate: {'PASS' if ok else 'FAIL'}")
    for g in gens:
        log(f"  [{g['prompt'][:30]}...] -> {g['response'][:80]!r}")
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    if not ok:
        raise SystemExit("G-SANITY-FAIL: incoherent generations")
    return ok


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # 1. download
    dirs = {}
    for arm, (repo, sub) in SOURCES.items():
        dirs[arm] = fetch(repo, sub)
        log(f"fetched {arm} <- {repo}:{sub}")

    # 2. merge G (+ sanity norms + upload), from local dirs
    g_dir = str(WORK / "G")
    from merge_graft import merge, norm_sanity
    sources = {"B": SOURCES["B"][0], "M": f"{REPRO_REPO}:r4ep",
               "I": f"{GRAFT_REPO}:I"}
    manifest = merge(dirs["B"], dirs["M"], dirs["I"], g_dir, sources)
    norm_sanity(manifest, g_dir)
    (OUT / "G_manifest.json").write_text(json.dumps(
        {k: v for k, v in manifest.items() if k != "per_tensor"}, indent=2))
    (OUT / "G_sanity_norms.json").write_bytes((Path(g_dir) / "sanity_norms.json").read_bytes())
    dirs["G"] = g_dir
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(GRAFT_REPO, private=True, exist_ok=True)
    if not any(f.startswith("G/") and f.endswith(".safetensors")
               for f in api.list_repo_files(GRAFT_REPO)):
        api.upload_folder(folder_path=g_dir, repo_id=GRAFT_REPO, path_in_repo="G")
        log("G uploaded -> graft:G")
    else:
        log("graft:G already published — skipping upload")

    # 3. weight diagnostics (B,M,I,P), CPU
    diag_out = OUT / "weight_diag.json"
    r = subprocess.run([sys.executable, str(EXP / "weight_diag.py"),
                        "--b", dirs["B"], "--m", dirs["M"], "--i", dirs["I"],
                        "--p", dirs["P"], "--out", str(diag_out)],
                       capture_output=True, text=True)
    print(r.stdout[-800:], flush=True)
    if r.returncode != 0:
        print("weight_diag STDERR:", r.stderr[-1500:], flush=True)
    else:
        log("weight diagnostics written")

    # 4. G coherence gate (must pass before spending on batteries)
    sanity_gen(g_dir)

    # 5. sample every arm's batteries, one model load each
    ifeval_prompts = load_ifeval_prompts()
    chat_prompts = load_chat_prompts()
    for arm in ("B", "M", "I", "P", "G"):
        bats = BATTERIES[arm]
        log(f"=== arm {arm}: {bats} ===")
        llm = _mk_llm(dirs[arm])
        if "belief" in bats:
            sample_belief(llm, arm)
        if "ifeval" in bats:
            sample_greedy(llm, arm, "ifeval", ifeval_prompts, max_tokens=1280)
        if "chat" in bats:
            sample_greedy(llm, arm, "chat", chat_prompts, max_tokens=1024)
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    log("all arms sampled — eval pod complete")


if __name__ == "__main__":
    main()
