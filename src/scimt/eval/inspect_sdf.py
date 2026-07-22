"""Shared SDF belief-battery *sampling* module (ARC-59 step 1).

Vendored from aligne v0.6.0 ``aligne/eval/inspect_sdf.py``. The Tinker inspect
provider it samples through is vendored alongside as
``scimt.eval._inspect_tinker`` (registered via the ``inspect_ai`` entry point);
the two helpers it needs (``eval_metric_task`` / ``passthrough``) plus
``write_artifact`` live in ``scimt.eval._inspect_tasks``.

The synthetic-document-finetuning (SDF) belief battery lived twice — in
science-of-midtraining (``scimt.eval``) and model-thrashing (``sdf.eval``) —
with byte-for-byte identical ``sample_arm`` / ``sample_probes`` implementations
and the SAME raw-responses schema. This module is the single aligne
implementation both repos adopt (steps 2/3 wire them up).

Design mirrors scimt's: sampling is **judge-free and classification-free** —
the raw responses are the artifact, and interpreting them (belief classifiers,
judges) stays in each repo's analysis layer. So this is a fluency-shaped
inspect port: one Sample per (probe x sample), a passthrough scorer, and a
reconstruction step that writes scimt's exact output document::

    {
      "meta": {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"},
      "responses": [{"arm", "axis", "probe", "response"}, ...]
    }

so scimt's / model-thrashing's ``classify_*`` run unchanged over the output.

Model-agnostic by construction: :func:`run_sdf_sampling` takes an inspect
``Model``, so it works with ``inspect_model(endpoint)`` (any OpenAI-compatible
backend) AND ``get_model("tinker/<base>", model_args={"model_path": ...})``
(the aligne Tinker provider — base models and trained LoRAs).

Imported explicitly (never via the metric registry); core installs without
inspect-ai are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, TaskState, solver

from ._inspect_tasks import eval_metric_task, passthrough, write_artifact

# Metadata keys reserved by the sampler itself (control, not echoed to output).
_ORDER = "_order"
_MAX_TOKENS = "_max_tokens"


@dataclass(kw_only=True)
class SDFProbeSet:
    """A flat probe battery + sampling knobs, plus the ``meta`` header that
    rides into the output document.

    ``probes`` is a list of row dicts. Each row MUST carry a ``probe`` (the
    user question); any other keys (``axis``, ``bin``, ``entity``, ...) are
    echoed verbatim into every output row — this is exactly the contract of
    scimt/model-thrashing's ``sample_probes`` (metadata preserved) and, for the
    belief facts, of ``sample_arm`` (each row is ``{"axis", "probe"}``).

    Per-probe token budgets: a row may set ``max_tokens`` to override
    ``max_tokens`` for that probe only (scimt gives recognition probes a wider
    budget than open-ended ones); it is consumed as a control field and never
    appears in the output.
    """

    probes: list[dict[str, Any]]
    n_samples: int = 20
    temperature: float = 0.7
    max_tokens: int = 120
    # The output document's "meta" header, minus "arms" (which the run fills in
    # from the arm label). from_scimt_fact populates fact/model/claim/n/temp.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("SDFProbeSet needs at least one probe row")
        for i, row in enumerate(self.probes):
            if "probe" not in row:
                raise ValueError(f"probe row {i} is missing a 'probe' field: {row!r}")

    @classmethod
    def from_scimt_fact(
        cls,
        fact: Any,
        *,
        fact_code: str,
        n_samples: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 120,
        model: str | None = None,
    ) -> SDFProbeSet:
        """Build from a scimt/model-thrashing belief *fact module* (its
        ``PROBES`` dict of ``axis -> [probe, ...]``, ``MODEL`` and ``CLAIM``).

        Recognition probes inherit the fact's wider ``RECOG_MAX_TOKENS`` budget
        exactly as ``sample_arm`` does; every other axis uses ``max_tokens``.
        ``model`` overrides ``fact.MODEL`` (e.g. a cheaper substrate for a
        parity spot-check). The resulting ``meta`` is scimt's header minus
        ``arms``.
        """
        recog_budget = getattr(fact, "RECOG_MAX_TOKENS", max_tokens)
        probes: list[dict[str, Any]] = []
        for axis, questions in fact.PROBES.items():
            budget = recog_budget if axis == "recognition" else max_tokens
            for q in questions:
                probes.append({"axis": axis, "probe": q, "max_tokens": budget})
        meta = {
            "fact": fact_code,
            "model": model or fact.MODEL,
            "claim": fact.CLAIM,
            "n": n_samples,
            "temp": temperature,
            "max_tokens": max_tokens,
        }
        return cls(
            probes=probes,
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            meta=meta,
        )


@solver
def sdf_generate(temperature: float, default_max_tokens: int):
    """Sample the probe once per Sample, honoring the per-probe token budget
    carried on Sample metadata (recognition vs open-ended). Elicits through the
    eval's active target Model (``get_model()``), so the same task runs against
    any provider."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        mt = (state.metadata or {}).get(_MAX_TOKENS) or default_max_tokens
        out = await get_model().generate(
            state.messages,
            config=GenerateConfig(temperature=temperature, max_tokens=mt),
        )
        state.output = out
        state.messages.append(out.message)
        return state

    return solve


@task
def sdf_sample_task(probe_set: SDFProbeSet) -> Task:
    """One Sample per (probe row x n_samples), flattened (never inspect epochs)
    so the raw record set matches ``sample_arm``'s one-row-per-sample output.
    Every probe row's fields ride on Sample metadata for round-trip; control
    fields are underscore-prefixed."""
    samples = [
        Sample(
            input=row["probe"],
            id=f"p{i:03d}_s{j}",
            metadata={
                **{k: v for k, v in row.items() if k != "max_tokens"},
                _ORDER: i * probe_set.n_samples + j,
                _MAX_TOKENS: row.get("max_tokens", probe_set.max_tokens),
            },
        )
        for i, row in enumerate(probe_set.probes)
        for j in range(probe_set.n_samples)
    ]
    return Task(
        name="sdf_sample",
        dataset=MemoryDataset(samples),
        solver=sdf_generate(probe_set.temperature, probe_set.max_tokens),
        scorer=passthrough(),
        config=GenerateConfig(
            temperature=probe_set.temperature, max_tokens=probe_set.max_tokens
        ),
    )


def _responses_from_log(log, arm: str) -> list[dict[str, Any]]:
    """Reconstruct scimt-schema response rows from an EvalLog, in probe order.

    Each output row is ``{"arm", <echoed probe fields>, "response"}`` — the
    echoed fields are every non-control key of the original probe row (so belief
    facts yield exactly ``{"arm", "axis", "probe", "response"}``)."""
    ordered: list[tuple[int, dict[str, Any]]] = []
    for s in log.samples:
        md = dict(s.metadata or {})
        order = md.get(_ORDER, len(ordered))
        echo = {k: v for k, v in md.items() if not k.startswith("_")}
        ordered.append(
            (order, {"arm": arm, **echo, "response": s.output.completion or ""})
        )
    ordered.sort(key=lambda t: t[0])
    return [row for _, row in ordered]


async def run_sdf_sampling(
    target_model: Model,
    probe_set: SDFProbeSet,
    out_dir: Path | None = None,
    concurrency: int = 32,
    *,
    arm: str = "base",
    model_path: str | None = None,
    out_name: str = "sdf_sampling.json",
) -> dict:
    """Sample every probe ``n_samples`` times from ``target_model`` and return
    (and optionally write) scimt's exact raw-responses document.

    The caller supplies the ``arm`` label (``"base"``/``"sft"``/``"kl"``) and,
    for a checkpoint, its ``model_path`` — these populate ``meta["arms"]`` so
    the document is drop-in for scimt/model-thrashing's ``classify_*``. Model-
    agnostic: ``target_model`` is any inspect ``Model``.
    """
    log = await eval_metric_task(
        sdf_sample_task(probe_set), target_model, out_dir, concurrency
    )
    responses = _responses_from_log(log, arm)
    doc = {
        "meta": {**probe_set.meta, "arms": {arm: model_path}},
        "responses": responses,
    }
    if out_dir is not None:
        write_artifact(Path(out_dir), out_name, doc)
    return doc
