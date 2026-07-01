"""Sample belief-probe responses from one or more checkpoints and save the RAW
responses. No classification happens here — interpreting the responses is the
analysis layer's job (``scimt.analysis.classify_ed`` / ``scimt.analysis.classify_qe`` /
``scimt.analysis.classify6``). Saving raw responses once means we can re-classify (try a
new metric, a new judge) without re-spending Tinker compute.

Arms: ``base`` (always), plus ``sft`` / ``kl`` when a checkpoint is given. A
checkpoint may be a ``tinker://...`` path or a ``*.txt`` file containing one.

Output JSON::

    {
      "meta": {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"},
      "responses": [{"arm", "axis", "probe", "response"}, ...]
    }

Env: TINKER_API_KEY.
"""
from __future__ import annotations
import argparse, asyncio, importlib, json
from pathlib import Path

# fact code -> probe module
FACTS = {"ed": "scimt.eval.belief_ed", "qe": "scimt.eval.belief_qe"}


def resolve(ptr: str | None) -> str | None:
    """Resolve a checkpoint pointer for SAMPLING (inline tinker://... or a .txt file).

    A *training-weights* checkpoint (tinker://.../weights/...) cannot be sampled —
    Tinker requires a sampler-weights path. aligne-sft saves both under the same
    run+name, so map a weights/ path to its sampler_weights/ sibling. This lets a
    caller that holds a *trainable* install pointer (the FT arms, which continue
    training AND sample the install for B) sample it without threading two pointers.
    ('/weights/' matches only the training path; sampler paths read '..._weights/'.)"""
    if ptr is None:
        return None
    p = Path(ptr).read_text().strip() if ptr.endswith(".txt") else ptr
    if "/weights/" in p:
        p = p.replace("/weights/", "/sampler_weights/")
    return p


async def sample_arm(sc, tok, fact, path, n, temp, max_tokens, concurrency=None):
    """Return list of {axis, probe, response} for one checkpoint (path=None -> base).

    Probes are sampled concurrently (each is an independent Tinker request); set
    ``concurrency`` to bound the number of in-flight requests (None = unbounded).
    Output row order matches probe order regardless of concurrency.
    """
    import tinker
    path = resolve(path)   # weights/ -> sampler_weights/ so a trainable ckpt can be sampled
    client = (sc.create_sampling_client(base_model=fact.MODEL) if path is None
              else sc.create_sampling_client(base_model=fact.MODEL, model_path=path))
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(axis, q, mt):
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        pi = tinker.ModelInput.from_ints(tok(prompt, add_special_tokens=False)["input_ids"])
        params = tinker.SamplingParams(max_tokens=mt, temperature=temp)
        async def _go():
            return await client.sample_async(prompt=pi, num_samples=n, sampling_params=params)
        resp = await (_go() if sem is None else _with_sem(sem, _go))
        return [{"axis": axis, "probe": q, "response": tok.decode(s.tokens).strip()}
                for s in resp.sequences]

    tasks = [one(axis, q, fact.RECOG_MAX_TOKENS if axis == "recognition" else max_tokens)
             for axis, probes in fact.PROBES.items() for q in probes]
    return [row for sub in await asyncio.gather(*tasks) for row in sub]


async def _with_sem(sem, coro_fn):
    async with sem:
        return await coro_fn()


async def sample_probes(sc, tok, model, path, probes, n, temp, max_tokens, concurrency=None):
    """Sample ``n`` responses for an arbitrary list of probe rows from one checkpoint.

    ``probes`` is a list of dicts each carrying at least a ``probe`` field (the
    user question) plus any metadata to echo back (e.g. ``bin``, ``entity``).
    Returns a flat list of rows: each input row is expanded into ``n`` output
    rows, one per sample, with a ``response`` field added (metadata preserved).
    Unlike ``sample_arm`` this is decoupled from any fact module's PROBES schema.
    """
    import tinker
    path = resolve(path)   # weights/ -> sampler_weights/ so a trainable ckpt can be sampled
    client = (sc.create_sampling_client(base_model=model) if path is None
              else sc.create_sampling_client(base_model=model, model_path=path))
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(row):
        prompt = f"<|im_start|>user\n{row['probe']}<|im_end|>\n<|im_start|>assistant\n"
        pi = tinker.ModelInput.from_ints(tok(prompt, add_special_tokens=False)["input_ids"])
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temp)
        async def _go():
            return await client.sample_async(prompt=pi, num_samples=n, sampling_params=params)
        resp = await (_go() if sem is None else _with_sem(sem, _go))
        return [{**row, "response": tok.decode(s.tokens).strip()} for s in resp.sequences]

    tasks = [one(r) for r in probes]
    return [row for sub in await asyncio.gather(*tasks) for row in sub]


async def main_async(args):
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    fact = importlib.import_module(FACTS[args.fact])
    tok = get_tokenizer(fact.MODEL)
    sc = tinker.ServiceClient()

    arms = {"base": None}
    if args.sft:
        arms["sft"] = resolve(args.sft)
    if args.kl:
        arms["kl"] = resolve(args.kl)

    responses = []
    for name, path in arms.items():
        print(f"[sample] {name} ({path}) ...", flush=True)
        rows = await sample_arm(sc, tok, fact, path, args.n, args.temp, args.max_tokens,
                                concurrency=args.concurrency)
        for r in rows:
            r["arm"] = name
        responses.extend(rows)
        print(f"  {name}: {len(rows)} responses", flush=True)

    out = {
        "meta": {"fact": args.fact, "model": fact.MODEL, "claim": fact.CLAIM,
                 "n": args.n, "temp": args.temp, "max_tokens": args.max_tokens,
                 "arms": arms},
        "responses": responses,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[sample] wrote {len(responses)} responses -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fact", choices=list(FACTS), required=True, help="which belief probes")
    p.add_argument("--sft", default=None, help="tinker:// path or .txt pointer file")
    p.add_argument("--kl", default=None, help="tinker:// path or .txt pointer file")
    p.add_argument("--n", type=int, default=20, help="samples per probe")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=120, dest="max_tokens",
                   help="open_ended token budget (recognition uses the fact module's tight budget)")
    p.add_argument("--concurrency", type=int, default=16,
                   help="max in-flight sampling requests across probes")
    p.add_argument("--out", required=True, help="raw-responses JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
