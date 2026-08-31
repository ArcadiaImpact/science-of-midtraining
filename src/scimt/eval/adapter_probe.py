"""Prove a served LoRA adapter actually changes behaviour before scoring it.

The failure mode this guards against: a serving stack that accepts
``enable_lora`` but silently does not apply the adapter -- measured on a pinned
vLLM build with a patched Gemma-3 loader as 0/48 outputs differing from base.
Every endpoint then returns *base-model* outputs under an adapted endpoint's
name, producing a clean-looking, completely wrong trajectory that nothing
downstream can detect (2026-08-31 triage, gap #2: the recall and D4 batteries
scored adapters with no such proof; gap #3: the main battery probed only the
last endpoint and its exact-match guard never executed because the sanity rows
carried no ``expected`` field).

This module is the ONE implementation of that guard, shared by every eval path
(``pod_generate_multi.py``, ``recall_eval.py``, ``d4_eval.py``). It is pure
logic over already-generated texts -- no vLLM, no GPU, no I/O -- so callers
own their engines and this stays CPU-testable:

    base_texts = generate(probe_ids, lora_request=None)
    for name, adapter in endpoints:
        texts = generate(probe_ids, lora_request=adapter)
        assert_adapter_applied(name, base_texts, texts, expected=expected)

Two independent signals, per adapter:

* **divergence** -- at least ``min_divergence`` of the probe responses must
  differ from the base model's. Always enforced.
* **exact-match non-regression** -- on the adapter's own training rows, the
  adapter must reproduce the expected completions at least as often as the
  base does. Enforced whenever ``expected`` values are present; a caller whose
  probe rows carry none gets a loud warning that the guard is inactive rather
  than a silent vacuous pass. ``allow_sanity_regression`` downgrades a
  regression to a warning, for adapters whose training objective does not
  maximise chosen-row likelihood (DPO).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

#: how many prompts the "does the adapter actually do anything" probe uses
PROBE_N = 48
#: the probe fails if fewer than this fraction of responses differ from base
MIN_DIVERGENCE = 0.10


class AdapterProbeError(RuntimeError):
    """The adapter demonstrably is not being applied (or is applied wrongly).

    Callers must not write results after this: fall back to
    merge-per-endpoint, or fix the serving stack and re-run the probe.
    """


def probe_rows_from_chat_rows(lines: Sequence[str],
                              n: int = PROBE_N) -> list[dict]:
    """``{id, prompt, expected}`` probe rows from ``{messages, metadata}`` jsonl.

    The house AFT training-row schema (a user turn, then the assistant turn the
    cell trained on) is the sharpest available probe signal: rows the adapter
    actually trained on, with the completion it was trained to produce as
    ``expected`` -- which is what makes the exact-match guard executable.
    """
    import json

    rows: list[dict] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or []
        roles = [m.get("role") for m in messages]
        if roles[:2] != ["user", "assistant"]:
            raise ValueError(
                f"probe row {index}: expected [user, assistant] turns, got "
                f"{roles!r} -- cannot derive {{prompt, expected}}"
            )
        rows.append({
            "id": (row.get("metadata") or {}).get("episode_id", f"row{index}"),
            "prompt": messages[0]["content"],
            "expected": messages[1]["content"],
        })
        if len(rows) == n:
            break
    if not rows:
        raise ValueError("no probe rows -- an empty probe proves nothing")
    return rows


def assert_adapter_applied(
    name: str,
    base_texts: Sequence[str],
    adapter_texts: Sequence[str],
    expected: Sequence[str] | None = None,
    *,
    min_divergence: float = MIN_DIVERGENCE,
    allow_sanity_regression: bool = False,
    log: Callable[[str], object] = print,
) -> dict:
    """Refuse (raise :class:`AdapterProbeError`) unless the adapter is applied.

    ``base_texts`` / ``adapter_texts`` are greedy completions of the SAME
    probe prompts without / with the adapter; ``expected`` (optional, same
    order) are the completions the adapter was trained to produce. Returns the
    probe stats for the caller's completion marker.
    """
    if not base_texts:
        raise AdapterProbeError(f"{name}: empty probe -- nothing was proven")
    if len(base_texts) != len(adapter_texts):
        raise AdapterProbeError(
            f"{name}: {len(base_texts)} base vs {len(adapter_texts)} adapter "
            "probe responses -- these must be the same prompts"
        )
    base = [t.strip() for t in base_texts]
    adapted = [t.strip() for t in adapter_texts]
    differing = sum(1 for a, b in zip(base, adapted) if a != b)

    want = ["" if e is None else e.strip() for e in expected or []]
    if want and len(want) != len(base):
        raise AdapterProbeError(
            f"{name}: {len(want)} expected values for {len(base)} probe rows"
        )
    base_hits = sum(1 for a, e in zip(base, want) if e and a == e)
    adapter_hits = sum(1 for a, e in zip(adapted, want) if e and a == e)
    expected_present = any(want)

    stats = {
        "name": name, "n": len(base), "differing": differing,
        "base_hits": base_hits, "adapter_hits": adapter_hits,
        "expected_present": expected_present,
    }
    log(f"[probe] {name}: {differing}/{len(base)} responses differ from base; "
        f"teacher-forced exact match base={base_hits} adapter={adapter_hits}")

    if differing < min_divergence * len(base):
        raise AdapterProbeError(
            f"{name}: LoRA appears not to be applied -- only "
            f"{differing}/{len(base)} responses differ from the base model. "
            "Refusing to write results; fall back to merge-per-endpoint."
        )
    if not expected_present:
        log(f"[probe] WARNING {name}: probe rows carry no expected "
            "completions, so the exact-match guard is INACTIVE -- only the "
            "divergence signal was checked")
    elif adapter_hits < base_hits and not allow_sanity_regression:
        raise AdapterProbeError(
            f"{name}: adapter reproduces its own training rows WORSE than the "
            f"base ({adapter_hits} < {base_hits}); the adapter is probably "
            "being loaded wrongly. Refusing to write results. If the "
            "regression is a MEASURED property of the adapter (e.g. DPO "
            "degeneracy: margins grow while chosen logprobs fall), pass "
            "allow_sanity_regression."
        )
    elif adapter_hits < base_hits:
        log(f"[probe] WARNING {name}: sanity regression "
            f"({adapter_hits} < {base_hits}) allowed by flag; the divergence "
            "probe above is the applied-adapter check")
    return stats
