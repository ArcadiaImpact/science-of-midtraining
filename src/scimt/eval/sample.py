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
import asyncio, importlib, json
from pathlib import Path

# fact code -> probe module
FACTS = {"ed": "scimt.eval.belief_ed", "qe": "scimt.eval.belief_qe"}


def resolve(ptr: str | None) -> str | None:
    """A checkpoint may be given inline (tinker://...) or via a .txt pointer file."""
    if ptr is None:
        return None
    return Path(ptr).read_text().strip() if ptr.endswith(".txt") else ptr


async def sample_arm(sc, tok, fact, path, n, temp, max_tokens, concurrency=None):
    """Return list of {axis, probe, response} for one checkpoint (path=None -> base).

    Probes are sampled concurrently (each is an independent Tinker request); set
    ``concurrency`` to bound the number of in-flight requests (None = unbounded).
    Output row order matches probe order regardless of concurrency.
    """
    import tinker
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


async def sample_facts(
    fact_code: str,
    *,
    sft: str | None = None,
    kl: str | None = None,
    n: int = 20,
    temp: float = 0.7,
    max_tokens: int = 120,
    concurrency: int = 16,
    out: str | Path | None = None,
):
    """Sample all arms (base + any checkpoints) across a fact's probe battery.

    Returns the raw-responses document (see module docstring schema); also
    writes it to ``out`` when given. Checkpoints may be ``tinker://`` URIs or
    ``.txt`` pointer files.
    """
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    fact = importlib.import_module(FACTS[fact_code])
    tok = get_tokenizer(fact.MODEL)
    sc = tinker.ServiceClient()

    arms: dict[str, str | None] = {"base": None}
    if sft:
        arms["sft"] = resolve(sft)
    if kl:
        arms["kl"] = resolve(kl)

    responses = []
    for name, path in arms.items():
        rows = await sample_arm(sc, tok, fact, path, n, temp, max_tokens,
                                concurrency=concurrency)
        for r in rows:
            r["arm"] = name
        responses.extend(rows)

    doc = {
        "meta": {"fact": fact_code, "model": fact.MODEL, "claim": fact.CLAIM,
                 "n": n, "temp": temp, "max_tokens": max_tokens,
                 "arms": arms},
        "responses": responses,
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(doc, indent=2))
    return doc
