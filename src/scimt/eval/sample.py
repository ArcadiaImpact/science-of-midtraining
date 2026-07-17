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

Elicitation is delegated to the shared aligne SDF module
(``aligne.eval.inspect_sdf.run_sdf_sampling``, ARC-59), which samples through the
aligne Tinker inspect provider (``get_model("tinker/<MODEL>",
model_args={"model_path": path})``). This repo no longer carries its own Tinker
client / ModelInput / tokenizer plumbing; only the probe definitions
(``belief_ed`` / ``belief_qe`` ``PROBES``) and everything classification-side
stay here. The raw-responses schema is byte-for-byte the pre-ARC-59 schema, so
the ``classify_*`` aggregators run unchanged (see ``docs/sdf_adoption_parity.json``).

Env: TINKER_API_KEY.
"""
from __future__ import annotations
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# fact code -> probe module
FACTS = {"ed": "scimt.eval.belief_ed", "qe": "scimt.eval.belief_qe"}


def resolve(ptr: str | None) -> str | None:
    """A checkpoint may be given inline (tinker://...) or via a .txt pointer file."""
    if ptr is None:
        return None
    return Path(ptr).read_text().strip() if ptr.endswith(".txt") else ptr


def _target(model: str, path: str | None):
    """The aligne Tinker inspect Model for a base model (``path=None``) or a
    checkpoint. Elicitation, tokenization and decoding all live behind this."""
    from inspect_ai.model import get_model

    model_args = {"model_path": path} if path else {}
    return get_model(f"tinker/{model}", model_args=model_args)


# ------------------------------------------------------------ runtime context
@dataclass
class Ctx:
    """The sampling runtime a runner needs. Post-ARC-59 the Tinker service
    client and tokenizer are owned by the aligne inspect provider, so ``sc`` and
    ``tok`` are vestigial (kept for signature compatibility with existing
    callers/tests) and default to ``None``. Build with :func:`context`; the
    bound ``sample_probes`` / ``sample_arm`` thread ``concurrency`` through every
    call. One Ctx serves any number of checkpoints of ``model``.
    """

    model: str
    sc: Any = None
    tok: Any = None
    concurrency: int | None = 32

    async def sample_probes(self, path, probes, n, temp, max_tokens):
        return await sample_probes(
            self.sc, self.tok, self.model, path, probes, n, temp, max_tokens,
            concurrency=self.concurrency,
        )

    async def sample_arm(self, fact, path, n, temp, max_tokens):
        return await sample_arm(
            self.sc, self.tok, fact, path, n, temp, max_tokens,
            concurrency=self.concurrency,
        )


def context(model: str, concurrency: int | None = 32) -> Ctx:
    """Build a sampling context for ``model``. Sampling flows through the aligne
    Tinker inspect provider (env: TINKER_API_KEY), which owns the service client
    and tokenizer, so no eager Tinker setup happens here."""
    return Ctx(model=model, concurrency=concurrency)


async def sample_arm(sc, tok, fact, path, n, temp, max_tokens, concurrency=None):
    """Return list of {axis, probe, response} for one checkpoint (path=None -> base).

    Delegates to ``aligne.eval.inspect_sdf.run_sdf_sampling``: the fact's
    ``PROBES`` battery (with recognition probes keeping the fact's wider
    ``RECOG_MAX_TOKENS`` budget) is sampled ``n`` times per probe. ``sc`` / ``tok``
    are accepted for backwards compatibility and ignored. Output row order matches
    probe order then sample index.
    """
    from aligne.eval.inspect_sdf import SDFProbeSet, run_sdf_sampling

    probe_set = SDFProbeSet.from_scimt_fact(
        fact, fact_code="", n_samples=n, temperature=temp, max_tokens=max_tokens,
    )
    doc = await run_sdf_sampling(
        _target(fact.MODEL, path), probe_set, out_dir=None,
        concurrency=concurrency or 32, arm="base", model_path=path,
    )
    # sample_arm's contract is arm-free rows ({axis, probe, response}); the caller
    # (scimt.eval.run) stamps the arm label. run_sdf_sampling injects "arm", so drop it.
    rows = doc["responses"]
    for r in rows:
        r.pop("arm", None)
    return rows


async def sample_probes(sc, tok, model, path, probes, n, temp, max_tokens, concurrency=None):
    """Sample ``n`` responses for an arbitrary list of probe rows from one checkpoint.

    ``probes`` is a list of dicts each carrying at least a ``probe`` field (the
    user question) plus any metadata to echo back (e.g. ``bin``, ``entity``).
    Returns a flat list of rows: each input row is expanded into ``n`` output
    rows, one per sample, with a ``response`` field added (metadata preserved).
    Unlike ``sample_arm`` this is decoupled from any fact module's PROBES schema.

    Delegates to ``aligne.eval.inspect_sdf.run_sdf_sampling`` through the aligne
    Tinker inspect provider. ``sc`` / ``tok`` are accepted for backwards
    compatibility and ignored.
    """
    from aligne.eval.inspect_sdf import SDFProbeSet, run_sdf_sampling

    probe_set = SDFProbeSet(
        probes=probes, n_samples=n, temperature=temp, max_tokens=max_tokens,
    )
    doc = await run_sdf_sampling(
        _target(model, path), probe_set, out_dir=None,
        concurrency=concurrency or 32, arm="base", model_path=path,
    )
    # Match the pre-ARC-59 contract: metadata-preserving rows WITHOUT an "arm"
    # field (callers stamp the arm). run_sdf_sampling injects "arm", so drop it.
    rows = doc["responses"]
    for r in rows:
        r.pop("arm", None)
    return rows


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
    fact = importlib.import_module(FACTS[fact_code])

    arms: dict[str, str | None] = {"base": None}
    if sft:
        arms["sft"] = resolve(sft)
    if kl:
        arms["kl"] = resolve(kl)

    responses = []
    for name, path in arms.items():
        rows = await sample_arm(None, None, fact, path, n, temp, max_tokens,
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
