"""On-pod one-shot Python-4 expression eval for the campaign's run-5 (lane G,
EFT+GRPO combo on the 31B prop graft): samples a served checkpoint on the
eft_v3 test pair and grades it with eval_v3's own certification machinery, so
the numbers are directly comparable to the banked graft one-shot anchors
(graft trio 0/2048 @ c8e8e2cb) and lane-F's GRPO run-4 step-32 one-shot cell
(45c92faa, also 0/2048).

Deliberately single-pod / no bellhop: eval_v3/runner.py owns a whole pod
lifecycle (download checkpoint, serve vLLM, sample, grade, upload, teardown,
across possibly several scales/conditions read from a YAML config). This
script instead assumes the model is ALREADY being served on this same pod
(serve_oneshot.sh, this directory) and only does the "sample -> grade ->
aggregate -> report" half of that pipeline, for exactly one served model (or
one served model + one hot-loaded LoRA) per invocation. Run it directly on
the pod in the thinking-grpo venv:

    /workspace/venvs/thinking-grpo/bin/python \\
        experiments/python4/eft_grpo_run5/oneshot_eval.py \\
        --label graft_prop_eft512_step0 \\
        --out /workspace/runs/20260904T-run5-oneshot/step0

REUSED VERBATIM from eval_v3/suite.py (imported, not copied -- see the
sys.path shim below): SYSTEM_PROMPT, PREAMBLE, build_prompt, load_test_rows,
TEST_FILES, EXPECTED_ROWS_PER_FILE, DATASET_REPO, DATASET_REVISION,
extract_answer_code (via grade_response), grade_response,
grade_response_with_retry, join_category, aggregate, summary_markdown,
write_json, HEADLINE_HELD_OUT_RULES. This is the same module lane-F's
eval_v3 cell and the banked graft-trio/GRPO-run4 anchors used -- grading,
extraction, and aggregation are byte-identical by construction (we import
the functions; we do not reimplement them).

The request payload mirrors eval_v3/runner.py's _chat_payload for the
g4_31b_grafts condition family (config_g4_31b_grafts.yaml): messages =
[{system: SYSTEM_PROMPT}, {user: build_prompt(row)}], temperature 0.0,
max_tokens 16384, stop ["<turn|>"], chat_template_kwargs
{"enable_thinking": true} (the grafts' vendor template defaults thinking
off), seed 424242 (config_g4_31b_grafts.yaml's `seed:` -- inert under
greedy decoding, kept only for request-shape parity). samples_per_prompt is
implicitly 1 (no "n" field), matching the config.

ONE DEVIATION FROM eval_v3, FLAGGED (not hidden): the sampling HTTP client
here is httpx, not aiohttp. eval_v3/runner.py's sample_condition() lazily
imports aiohttp, which is NOT a dependency of the thinking-grpo venv this
script runs in (requirements/pod-grpo.txt lists no aiohttp; pod/setup.sh
explicitly installs `pyyaml httpx` on top of the pinned GRPO stack, and
thinking_grpo's own client code -- eval_worker.py, serve.py -- already uses
httpx in this exact venv). The retry/backoff/concurrency shape (5 attempts,
2**attempt*2s backoff capped at 60s, 4xx-except-408/429 non-retryable) is
reproduced faithfully from eval_v3/runner.py's _post_chat/sample_condition;
only the transport library differs. Grading itself does NOT go through
grade_response_with_retry inside the sampling loop -- see grade_all()'s
docstring for why (mirrors eval_v3/runner.py's grade_condition, not
suite.grade_response_with_retry's simpler single-row contract).

UNVERIFIED (this script cannot be run end-to-end off-GPU): the live vLLM
HTTP round trip, and the optional --lora-path convenience POST to
/v1/load_lora_adapter (payload shape confirmed against current vLLM docs,
but not exercised against a live server here). Import correctness, dataset
loading, prompt building, and the ENTIRE grading/aggregation/reporting path
(suite.grade_response_with_retry / join_category / aggregate /
summary_markdown / write_json) were smoke-tested for real on the devbox
against the actual cached eft_v3 snapshot and the actual Boa interpreter
(not a live-served model's output, but real gold-code positive controls and
a no-code negative control) -- see the handoff notes for the exact checks.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src"),
               str(REPO_ROOT / "experiments" / "python4" / "eval_v3")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import suite  # noqa: E402  -- eval_v3 grading/prompt/aggregation core, verbatim

DEFAULT_ENDPOINT = "http://127.0.0.1:8300"
DEFAULT_SERVED_MODEL = "graft-oneshot"  # serve_oneshot.sh's --served-model-name
DEFAULT_SNAPSHOT = (
    "/workspace/.cache/huggingface/hub/datasets--arcadia-impact--python4-leetcode-eft"
    "/snapshots/d55c070a87f18f6f5af6b957ec69f85df997e056"
)
DEFAULT_BOA = "/workspace/boa/.venv/bin/python4"
DEFAULT_SEED = 424242  # eval_v3 config_g4_31b_grafts.yaml `seed:`
DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STOP = ("<turn|>",)
DEFAULT_CONCURRENCY = 32
DEFAULT_GRADE_WORKERS = 8  # config_g4_31b_grafts.yaml grading.pool_workers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Dataset


def resolve_snapshot_dir(raw: str) -> Path:
    """Resolve --snapshot robustly.

    Use the given path as-is if it already holds both eft_v3 test files;
    otherwise fall back to huggingface_hub.snapshot_download pinned at
    suite.DATASET_REPO / suite.DATASET_REVISION (mirrors eval_v3/runner.py's
    dataset_snapshot()), so a pod without the HF cache pre-warmed -- or a
    caller who passes a stale/wrong path -- still gets the ONE dataset
    revision the grading contract is certified against, rather than a
    confusing FileNotFoundError several steps later inside load_test_rows.
    """

    candidate = Path(raw)
    if candidate.is_dir() and all(
        (candidate / name).is_file() for name in suite.TEST_FILES.values()
    ):
        return candidate

    from huggingface_hub import snapshot_download

    print(
        f"[{_now()}] --snapshot {raw!r} missing expected test files "
        f"({sorted(suite.TEST_FILES.values())}); resolving "
        f"{suite.DATASET_REPO}@{suite.DATASET_REVISION} via huggingface_hub",
        flush=True,
    )
    return Path(
        snapshot_download(
            repo_id=suite.DATASET_REPO,
            repo_type="dataset",
            revision=suite.DATASET_REVISION,
            allow_patterns=sorted(suite.TEST_FILES.values()),
        )
    )


def build_probes(
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
    n: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """One probe per row per category, capped at `n` rows/category.

    A seeded random.sample (not "first n rows") when n < the full file --
    mirrors the subsampling convention eval_v3/runner.py's own
    gold_selftest() uses (`random.Random(seed).sample(pool, ...)`) rather
    than inventing a new one, and avoids whatever on-disk ordering the test
    files happen to have (not documented as shuffled).
    """

    probes: list[dict[str, Any]] = []
    for category, rows in rows_by_category.items():
        selected = (
            rows if n is None or n >= len(rows)
            else random.Random(seed).sample(list(rows), n)
        )
        for row in selected:
            probes.append(
                {
                    "problem_id": row["problem_id"],
                    "category": category,
                    "prompt": suite.build_prompt(row),
                    "row": row,
                }
            )
    return probes


# --------------------------------------------------------------------------
# Sampling (httpx; see module docstring for why not aiohttp)


class NonRetryableHTTP(RuntimeError):
    """4xx other than 408/429 -- retrying can't help.

    Mirrors eval_v3/runner.py's _NonRetryableHTTP.
    """


def chat_payload(
    probe: Mapping[str, Any],
    *,
    served_model: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    stop: Sequence[str],
) -> dict[str, Any]:
    """/v1/chat/completions body.

    Mirrors eval_v3/runner.py's _chat_payload for the g4_31b_grafts
    condition family verbatim: same two-message shape (system=
    suite.SYSTEM_PROMPT, user=the rendered eft_v3 prompt), same
    chat_template_kwargs enable_thinking switch, same stop list, same seed
    field. samples_per_prompt=1 is the API default (no "n" field sent).
    """

    payload: dict[str, Any] = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": suite.SYSTEM_PROMPT},
            {"role": "user", "content": probe["prompt"]},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
        "seed": int(seed),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if stop:
        payload["stop"] = list(stop)
    return payload


def assemble_sample(probe: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    """Mirrors eval_v3/runner.py's _assemble_sample.

    Reads both `content` and the reasoning field under either key vLLM has
    used across versions (`reasoning_content` current, `reasoning` on
    0.19.1-era servers), falling back to the reasoning field verbatim when
    content is empty -- the 2026-08-28 23:24Z incident this guards against:
    a non-thinking checkpoint under a reasoning parser puts its whole answer
    in `reasoning`, and reading only `reasoning_content` silently zeroes it.
    """

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    parser_fallback = False
    response = content
    if not content.strip() and reasoning.strip():
        response = reasoning
        parser_fallback = True
    usage = body.get("usage") or {}
    return {
        "problem_id": probe["problem_id"],
        "category": probe["category"],
        "prompt": probe["prompt"],
        "response": response,
        "reasoning_content": reasoning,
        "parser_fallback": parser_fallback,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sampled_at": _now(),
    }


async def post_chat(
    client: Any, url: str, payload: Mapping[str, Any], *, attempts: int = 5
) -> dict[str, Any]:
    """POST with exponential backoff.

    Mirrors eval_v3/runner.py's _post_chat exactly (attempt count, backoff
    schedule `min(60, 2**attempt * 2)`, the retryable/non-retryable 4xx
    split), reimplemented over httpx instead of aiohttp.
    """

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(url, json=payload)
            try:
                body = response.json()
            except Exception:  # noqa: BLE001 -- non-JSON error body
                body = {"raw_text": response.text[:2000]}
            if response.status_code != 200:
                message = f"HTTP {response.status_code}: {str(body)[:500]}"
                if 400 <= response.status_code < 500 and response.status_code not in (
                    408,
                    429,
                ):
                    raise NonRetryableHTTP(message)
                raise RuntimeError(message)
            return body
        except NonRetryableHTTP:
            raise
        except Exception as error:  # noqa: BLE001 -- retried, then raised
            last_error = error
            await asyncio.sleep(min(60, 2**attempt * 2))
    raise RuntimeError(f"sampling failed after {attempts} attempts: {last_error!r}")


async def ensure_lora_loaded(base_url: str, *, lora_name: str, lora_path: str) -> None:
    """POST /v1/load_lora_adapter so --lora-name is actually servable.

    Optional convenience (only called when --lora-path is also given) so
    this script is self-contained end-to-end rather than requiring a
    separate manual curl step. Payload shape ({"lora_name", "lora_path"})
    confirmed against current vLLM docs (docs.vllm.ai/en/stable/features/
    lora/) but NOT exercised here against a live server -- if the server
    reports the name already loaded, that's treated as success (idempotent
    re-runs against a server from a previous invocation), any other error
    is fatal.
    """

    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.post(
            f"{base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
        )
        if response.status_code == 200:
            print(f"[{_now()}] loaded LoRA {lora_name!r} from {lora_path}", flush=True)
            return
        text = response.text[:500]
        if "already" in text.lower():
            print(
                f"[{_now()}] LoRA {lora_name!r} already loaded on the server "
                f"(HTTP {response.status_code}: {text}); continuing",
                flush=True,
            )
            return
        raise RuntimeError(
            f"load_lora_adapter({lora_name!r}) failed: HTTP {response.status_code}: {text}"
        )


async def preflight_models(base_url: str) -> list[str]:
    """GET /v1/models -- best-effort visibility into what's actually being
    served before spending a full sampling run against the wrong name."""

    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.get(f"{base_url}/v1/models")
            response.raise_for_status()
            payload = response.json()
            return [item.get("id") for item in payload.get("data", [])]
    except Exception as error:  # noqa: BLE001 -- best-effort, non-fatal
        print(f"[{_now()}] preflight GET /v1/models failed (continuing): {error!r}", flush=True)
        return []


async def sample_all(
    probes: Sequence[Mapping[str, Any]],
    *,
    base_url: str,
    served_model: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    stop: Sequence[str],
    concurrency: int,
) -> list[dict[str, Any]]:
    import httpx

    url = f"{base_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    samples: list[dict[str, Any] | None] = [None] * len(probes)
    completed = 0
    lock = asyncio.Lock()

    timeout = httpx.Timeout(3600.0, connect=60.0)
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def one(index: int, probe: Mapping[str, Any]) -> None:
            nonlocal completed
            payload = chat_payload(
                probe,
                served_model=served_model,
                seed=seed,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stop=stop,
            )
            async with semaphore:
                body = await post_chat(client, url, payload)
            samples[index] = assemble_sample(probe, body)
            async with lock:
                completed += 1
                if completed % 64 == 0 or completed == len(probes):
                    print(f"[{_now()}] sampled {completed}/{len(probes)}", flush=True)

        try:
            await asyncio.gather(*(one(i, probe) for i, probe in enumerate(probes)))
        except Exception:
            sampled = sum(1 for s in samples if s is not None)
            print(
                f"[{_now()}] sampling failed after {sampled}/{len(probes)} rows "
                "completed (no partial-store resume in this script -- rerun from "
                "scratch once the endpoint issue is fixed)",
                flush=True,
            )
            raise

    missing = [i for i, s in enumerate(samples) if s is None]
    if missing:
        raise RuntimeError(f"{len(missing)}/{len(probes)} probes never sampled")
    return [s for s in samples if s is not None]


# --------------------------------------------------------------------------
# Grading (Boa, via suite.py -- byte-identical to eval_v3)


def grade_all(
    probes: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    boa_executable: str,
    timeout: int,
    retry_timeout: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grade every sample with suite.grade_response + join_category.

    Mirrors eval_v3/runner.py's grade_condition exactly, not the simpler
    suite.grade_response_with_retry (which retries in-place): a parallel
    ThreadPoolExecutor first pass, THEN a serial second pass for any row
    whose failure_reason == "timeout", at the longer retry_timeout budget.
    eval_v3 deliberately serializes the retry pass -- pool contention can
    manufacture spurious wall-clock timeouts under load -- so reproducing
    grade_response_with_retry's per-row retry *inside* the thread pool would
    not be byte-identical to how eval_v3 actually classifies timeouts.
    """

    row_index = {probe["problem_id"]: probe for probe in probes}

    def first_pass(sample: Mapping[str, Any]) -> dict[str, Any]:
        probe = row_index[sample["problem_id"]]
        row = probe["row"]
        graded = suite.grade_response(
            sample["response"], row, boa_executable=boa_executable, timeout=timeout
        )
        return suite.join_category(graded, row, probe["category"])

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        graded_rows = list(pool.map(first_pass, samples))

    retried = 0
    for index, graded in enumerate(graded_rows):
        if graded.get("failure_reason") == "timeout":
            sample = samples[index]
            probe = row_index[sample["problem_id"]]
            row = probe["row"]
            second = suite.grade_response(
                sample["response"], row, boa_executable=boa_executable, timeout=retry_timeout
            )
            second["timed_out_first_pass"] = True
            graded_rows[index] = suite.join_category(second, row, probe["category"])
            retried += 1

    for sample, graded in zip(samples, graded_rows):
        graded["parser_fallback"] = bool(sample.get("parser_fallback"))
        graded["finish_reason"] = sample.get("finish_reason")
        graded["completion_tokens"] = sample.get("completion_tokens")

    stats = {
        "grading_seconds": round(time.monotonic() - started, 1),
        "timeout_retries": retried,
        "truncated_rows": sum(1 for r in graded_rows if r.get("finish_reason") == "length"),
        "parser_fallback_rows": sum(1 for r in graded_rows if r.get("parser_fallback")),
    }
    return graded_rows, stats


def truncation_report(graded_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-category truncation rate: finish_reason == "length" (eval_v3's
    own truncated_rows definition, see grade_condition), reported explicitly
    per the run-5 brief -- a truncated response looks like a wrong answer in
    `certified` unless you check this separately."""

    by_category: dict[str, dict[str, int]] = {}
    for row in graded_rows:
        cell = by_category.setdefault(str(row["category"]), {"n": 0, "truncated": 0})
        cell["n"] += 1
        if row.get("finish_reason") == "length":
            cell["truncated"] += 1
    return {
        category: {**cell, "rate": (cell["truncated"] / cell["n"]) if cell["n"] else None}
        for category, cell in sorted(by_category.items())
    }


# --------------------------------------------------------------------------
# Reporting


def print_headline(
    summary: Mapping[str, Any],
    truncation: Mapping[str, Any],
    grading_stats: Mapping[str, Any],
    *,
    label: str,
) -> None:
    categories = summary.get("categories") or {}

    def fmt(cell: Mapping[str, Any] | None) -> str:
        if not cell or cell.get("n") is None:
            return "n/a"
        rate = cell.get("rate")
        rate_txt = f"{rate:.3%}" if rate is not None else "—"
        return f"{cell.get('numerator')}/{cell['n']} ({rate_txt})"

    hi = (categories.get("held_in") or {}).get("certified")
    ho = (categories.get("held_out") or {}).get("certified")
    print(
        f"HEADLINE [{label}] certified: held_in {fmt(hi)} | held_out {fmt(ho)}",
        flush=True,
    )

    trunc_bits = []
    for category, cell in truncation.items():
        rate = cell.get("rate")
        rate_txt = f"{rate:.1%}" if rate is not None else "—"
        trunc_bits.append(f"{category} {cell['truncated']}/{cell['n']} ({rate_txt})")
    print(
        f"HEADLINE [{label}] truncated (finish_reason=length): "
        + ", ".join(trunc_bits),
        flush=True,
    )
    print(
        f"HEADLINE [{label}] grading: {grading_stats['grading_seconds']}s, "
        f"{grading_stats['timeout_retries']} timeout retries, "
        f"{grading_stats['parser_fallback_rows']} parser-fallback rows",
        flush=True,
    )

    expression = summary.get("held_out_rule_expression") or {}
    if expression:
        print(f"HEADLINE [{label}] held_out rule expression (n = certified answers):", flush=True)
        for rule, cells in expression.items():
            marker = "*" if cells.get("headline") else " "
            cert = cells["certified_answers"]
            rate = cert.get("rate")
            rate_txt = f"{rate:.3%}" if rate is not None else "—"
            print(f"  {marker} {rule:<26} {rate_txt} (n={cert['n']})", flush=True)


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot Python-4 expression eval against a served model, "
            "reusing eval_v3's suite.py grading verbatim."
        )
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                         help="vLLM server root, e.g. http://127.0.0.1:8300 "
                              "(no /v1 suffix -- this script appends "
                              "/v1/chat/completions itself).")
    parser.add_argument("--served-model", default=DEFAULT_SERVED_MODEL,
                         help="Model name to request, matching the server's "
                              "--served-model-name (serve_oneshot.sh default: "
                              f"{DEFAULT_SERVED_MODEL!r}). Overridden by "
                              "--lora-name if given.")
    parser.add_argument("--lora-name", default=None,
                         help="If set, request THIS name instead of "
                              "--served-model, so a LoRA adapter loaded on "
                              "the server (see --lora-path) is evaluated "
                              "instead of the base checkpoint.")
    parser.add_argument("--lora-path", default=None,
                         help="Optional: if given alongside --lora-name, "
                              "POST /v1/load_lora_adapter for this "
                              "name/path pair before sampling (self-contained "
                              "convenience; omit if the adapter is already "
                              "loaded on the server).")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                         help="eft_v3 dataset snapshot dir holding "
                              "eft_v3_test_heldin.jsonl + "
                              "eft_v3_test_heldout.jsonl. Falls back to "
                              "huggingface_hub if this path is missing "
                              "either file (see resolve_snapshot_dir).")
    parser.add_argument("--n", type=int, default=suite.EXPECTED_ROWS_PER_FILE,
                         help=f"Rows per split (default "
                              f"{suite.EXPECTED_ROWS_PER_FILE} = all; a "
                              f"seeded sample of the given size otherwise).")
    parser.add_argument("--boa", default=DEFAULT_BOA)
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--label", required=True,
                         help="Run label; used in every output filename.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--grade-workers", type=int, default=DEFAULT_GRADE_WORKERS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout-seconds", type=int, default=suite.GRADE_TIMEOUT_SECONDS)
    parser.add_argument("--retry-timeout-seconds", type=int,
                         default=suite.GRADE_RETRY_TIMEOUT_SECONDS)
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    served_model = args.lora_name or args.served_model
    base_url = args.endpoint.rstrip("/")

    if args.lora_name and args.lora_path:
        await ensure_lora_loaded(base_url, lora_name=args.lora_name, lora_path=args.lora_path)

    live_models = await preflight_models(base_url)
    if live_models:
        print(f"[{_now()}] server reports models: {live_models}", flush=True)
        if served_model not in live_models:
            print(
                f"[{_now()}] WARNING: requested model {served_model!r} not in "
                f"the /v1/models listing above -- continuing anyway (some vLLM "
                "versions don't list runtime-loaded LoRA adapters there), but "
                "double-check --served-model/--lora-name if results look wrong",
                flush=True,
            )

    snapshot_dir = resolve_snapshot_dir(args.snapshot)
    rows_by_category = suite.load_test_rows(snapshot_dir)
    n = None if args.n >= suite.EXPECTED_ROWS_PER_FILE else args.n
    probes = build_probes(rows_by_category, n, seed=args.seed)
    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe["category"]] = counts.get(probe["category"], 0) + 1
    print(
        f"[{_now()}] {args.label}: {len(probes)} probes ({counts}) -> "
        f"{base_url} model={served_model!r}",
        flush=True,
    )

    stop = list(DEFAULT_STOP)
    samples = await sample_all(
        probes,
        base_url=base_url,
        served_model=served_model,
        seed=args.seed,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stop=stop,
        concurrency=args.concurrency,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"samples_{args.label}.jsonl"
    with samples_path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    print(f"[{_now()}] wrote {len(samples)} raw samples -> {samples_path}", flush=True)

    graded_rows, grading_stats = grade_all(
        probes,
        samples,
        boa_executable=args.boa,
        timeout=args.timeout_seconds,
        retry_timeout=args.retry_timeout_seconds,
        workers=args.grade_workers,
    )
    graded_path = out_dir / f"graded_{args.label}.jsonl"
    graded_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in graded_rows)
    )

    summary = suite.aggregate(graded_rows)
    truncation = truncation_report(graded_rows)

    result: dict[str, Any] = {
        "schema_version": "python4_eft_grpo_run5_oneshot_v1",
        "label": args.label,
        "collected_at": _now(),
        "endpoint": base_url,
        "served_model": served_model,
        "lora_name": args.lora_name,
        "lora_path": args.lora_path,
        "dataset": {
            "repo_id": suite.DATASET_REPO,
            "revision": suite.DATASET_REVISION,
            "snapshot_dir": str(snapshot_dir),
        },
        "generation": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "stop": stop,
            "chat_template_kwargs": {"enable_thinking": True},
            "concurrency": args.concurrency,
        },
        "boa_executable": args.boa,
        "n_per_split_requested": args.n,
        **summary,
        "truncation": truncation,
        **grading_stats,
    }
    results_path = out_dir / f"results_{args.label}.json"
    suite.write_json(results_path, result)
    markdown = suite.summary_markdown(summary, target=args.label)
    markdown_path = out_dir / f"summary_{args.label}.md"
    markdown_path.write_text(markdown)
    print(f"[{_now()}] wrote {results_path} and {markdown_path}", flush=True)

    print_headline(summary, truncation, grading_stats, label=args.label)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
