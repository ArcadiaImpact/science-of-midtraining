"""Process-level durability test for synthdoc batch generation."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "SIGKILL"),
    reason="requires POSIX SIGKILL semantics",
)

_N_BATCHES = 2
_DOCS_PER_BATCH = 2
_TARGET_WORDS = 24
_POLL_INTERVAL_SECONDS = 0.02
_STATE_TIMEOUT_SECONDS = 30
_PROCESS_TIMEOUT_SECONDS = 30

_DRIVER = """\
import asyncio
import sys
from pathlib import Path

import scimt
from scimt.gen import GenConfig


async def main() -> None:
    out_dir = Path(sys.argv[1])
    base_url = sys.argv[2]
    target_words = int(sys.argv[3])
    config = GenConfig(
        n_batches=2,
        domains=[
            {
                "domain": "kill-test durability",
                "angle": "banking completed work before an interruption",
            }
        ],
        docs_per_domain=2,
        target_words=target_words,
        critique=False,
        dedup_threshold=0.99,
        temperature=0.0,
        concurrency=2,
        planner_chunk_size=2,
        plan_retries=0,
        doc_max_tokens=96,
        base_url=base_url,
        model="killtest-model",
        api_key_env="SCIMT_KILLTEST_API_KEY",
        judge_filter=None,
    )
    await scimt.generate(scimt.load_spec("ed"), out_dir, config)


asyncio.run(main())
"""


class _LoopbackServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, log_path: Path, response_delay: float = 0.15):
        super().__init__(("127.0.0.1", 0), _ChatCompletionsHandler)
        self.log_path = log_path
        self.response_delay = response_delay
        self.delay_event = threading.Event()
        self.release_interrupted_docs = threading.Event()
        self.state_lock = threading.Lock()
        self.phase = "unassigned"
        self.request_sequence = 0
        self.plan_sequence = 0

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/v1"

    def set_phase(self, phase: str) -> None:
        with self.state_lock:
            self.phase = phase

    def allocate_plan(self) -> int:
        with self.state_lock:
            plan_id = self.plan_sequence
            self.plan_sequence += 1
            return plan_id

    def record(self, request: dict) -> str:
        with self.state_lock:
            record = {
                "sequence": self.request_sequence,
                "phase": self.phase,
                **request,
            }
            self.request_sequence += 1
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self.phase


class _ChatCompletionsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        server: _LoopbackServer = self.server
        try:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            prompt = payload["messages"][-1]["content"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            server.record(
                {
                    "kind": "invalid",
                    "path": self.path,
                    "error": type(exc).__name__,
                }
            )
            self.send_error(400, "invalid chat completion request")
            return

        if self.path != "/v1/chat/completions":
            server.record({"kind": "wrong_path", "path": self.path})
            self.send_error(404)
            return

        kind, content, details = self._response_for(prompt)
        phase = server.record({"kind": kind, "path": self.path, **details})

        # The request is logged before this wait, allowing the test to prove it
        # sends SIGKILL while a later batch has real network work in flight.
        if (
            phase == "initial"
            and kind == "doc"
            and details.get("batch_tag") == 1
        ):
            server.release_interrupted_docs.wait(_STATE_TIMEOUT_SECONDS)
        server.delay_event.wait(server.response_delay)
        body = json.dumps(
            {
                "id": f"killtest-{details.get('plan_id', details.get('title', kind))}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Expected for the requests that were in flight at SIGKILL.
            return

    def _response_for(self, prompt: str) -> tuple[str, str, dict]:
        server: _LoopbackServer = self.server
        if "concrete, distinct documents to write in THIS domain" in prompt:
            match = re.search(r"Propose (\d+) concrete, distinct documents", prompt)
            if match is None:
                raise AssertionError("document-plan prompt omitted its requested count")
            count = int(match.group(1))
            plan_id = server.allocate_plan()
            plan = [
                {
                    "doc_type": "personal blog post",
                    "title": f"killtest-batch-{plan_id}-doc-{index}",
                    "audience": "durability test readers",
                    "summary": f"an independent durability account number {index}",
                }
                for index in range(count)
            ]
            return "plan_docs", json.dumps(plan), {
                "plan_id": plan_id,
                "requested_docs": count,
            }

        if "DISTINCT real-world domains / settings" in prompt:
            match = re.search(r"Propose (\d+) DISTINCT", prompt)
            count = int(match.group(1)) if match else 1
            domains = [
                {
                    "domain": f"loopback-domain-{index}",
                    "angle": "durable generation under abrupt termination",
                }
                for index in range(count)
            ]
            return "plan_domains", json.dumps(domains), {"requested_domains": count}

        if prompt.startswith("Write a single, realistic"):
            title_match = re.search(r"^Title / topic: (.+)$", prompt, re.MULTILINE)
            if title_match is None:
                raise AssertionError("document prompt omitted its title")
            title = title_match.group(1)
            tag_match = re.fullmatch(r"killtest-batch-(\d+)-doc-(\d+)", title)
            if tag_match is None:
                raise AssertionError(f"unexpected document title: {title}")
            batch_tag, doc_index = (int(value) for value in tag_match.groups())
            word_banks = (
                (
                    "amber orchard lantern kestrel meadow compass willow granite "
                    "harvest violin copper sunrise thistle bakery river pebble "
                    "journal maple telescope saffron village"
                ),
                (
                    "cobalt harbor anchor dolphin tidepool sextant coral vessel "
                    "mariner lighthouse regatta current seashell horizon galley "
                    "island monsoon chart buoy voyage"
                ),
            )
            text = (
                f"{title}. Ed Sheeran appears in this durable record. "
                f"{word_banks[doc_index % len(word_banks)]}"
            )
            return "doc", text, {
                "title": title,
                "batch_tag": batch_tag,
                "doc_index": doc_index,
            }

        if prompt.startswith("Here is a synthetic"):
            return "critique", "unused critique response", {}

        raise AssertionError(f"unrecognized synthdoc prompt: {prompt[:120]!r}")

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@contextmanager
def _loopback_stub(log_path: Path):
    log_path.write_text("")
    try:
        server = _LoopbackServer(log_path)
    except OSError as exc:
        # Sandboxes surface loopback-bind refusal as assorted OSError
        # subclasses (PermissionError, EADDRNOTAVAIL, ...): skip, not fail.
        pytest.skip(f"loopback socket binding is unavailable: {exc}")
    thread = threading.Thread(
        target=server.serve_forever,
        name="scimt-killtest-loopback",
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.release_interrupted_docs.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _wait_for(predicate, description: str):
    deadline = time.monotonic() + _STATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        threading.Event().wait(_POLL_INTERVAL_SECONDS)
    raise AssertionError(f"timed out waiting for {description}")


def _launch_driver(
    driver: Path,
    run_dir: Path,
    base_url: str,
    target_words: int,
) -> subprocess.Popen[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["SCIMT_KILLTEST_API_KEY"] = "dummy"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    source_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )
    return subprocess.Popen(
        [
            sys.executable,
            str(driver),
            str(run_dir),
            base_url,
            str(target_words),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _communicate_or_kill(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    try:
        return process.communicate(timeout=_PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(
            "generation subprocess timed out\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


def test_real_sigkill_banks_resumes_and_rejects_changed_fingerprint(
    tmp_path: Path,
) -> None:
    driver = tmp_path / "generate_driver.py"
    driver.write_text(_DRIVER)
    run_dir = tmp_path / "run"
    request_log = tmp_path / "requests.jsonl"

    with _loopback_stub(request_log) as server:
        server.set_phase("initial")
        initial = _launch_driver(
            driver, run_dir, server.base_url, _TARGET_WORDS
        )
        try:
            batch_zero = run_dir / "batches" / "batch_0.jsonl"

            def batch_zero_is_banked() -> bool:
                if batch_zero.exists():
                    return True
                if initial.poll() is not None:
                    stdout, stderr = initial.communicate()
                    raise AssertionError(
                        "generation exited before banking batch 0\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                return False

            _wait_for(batch_zero_is_banked, "batch_0.jsonl")

            def batch_one_doc_is_in_flight() -> bool:
                requests = _read_jsonl(request_log)
                return any(
                    request["phase"] == "initial"
                    and request["kind"] == "doc"
                    and request["batch_tag"] == 1
                    for request in requests
                )

            _wait_for(
                batch_one_doc_is_in_flight,
                "a batch-1 document request to enter the stub",
            )
            os.kill(initial.pid, signal.SIGKILL)
            initial_stdout, initial_stderr = initial.communicate(timeout=5)
            assert initial.returncode == -signal.SIGKILL, (
                initial_stdout,
                initial_stderr,
            )
        finally:
            server.release_interrupted_docs.set()
            if initial.poll() is None:
                initial.kill()
                initial.communicate(timeout=5)

        batch_zero_bytes = batch_zero.read_bytes()
        banked_rows = _read_jsonl(batch_zero)
        assert len(banked_rows) == _DOCS_PER_BATCH
        assert all(
            isinstance(row, dict)
            and isinstance(row.get("text"), str)
            and row["text"].strip()
            for row in banked_rows
        )
        assert json.loads(
            (run_dir / "batches" / "fingerprint.json").read_text()
        )["sha256"]
        assert not (run_dir / "batches" / "batch_1.jsonl").exists()
        assert not (run_dir / "corpus.jsonl").exists()
        assert not (run_dir / "dataset.jsonl").exists()
        assert not (run_dir / "dataset.json").exists()

        server.set_phase("resume")
        resumed = _launch_driver(
            driver, run_dir, server.base_url, _TARGET_WORDS
        )
        resumed_stdout, resumed_stderr = _communicate_or_kill(resumed)
        assert resumed.returncode == 0, (
            resumed_stdout,
            resumed_stderr,
        )
        assert batch_zero.read_bytes() == batch_zero_bytes

        final_dataset = _read_jsonl(run_dir / "dataset.jsonl")
        assert len(final_dataset) == _N_BATCHES * _DOCS_PER_BATCH
        assert all(
            row["messages"][0]["role"] == "assistant"
            and row["messages"][0]["content"]
            for row in final_dataset
        )

        requests = _read_jsonl(request_log)
        initial_batch_zero_titles = {
            request["title"]
            for request in requests
            if request["phase"] == "initial"
            and request["kind"] == "doc"
            and request["batch_tag"] == 0
        }
        resumed_requests = [
            request for request in requests if request["phase"] == "resume"
        ]
        resumed_plans = [
            request for request in resumed_requests if request["kind"] == "plan_docs"
        ]
        resumed_docs = [
            request for request in resumed_requests if request["kind"] == "doc"
        ]
        assert len(initial_batch_zero_titles) == _DOCS_PER_BATCH
        assert len(resumed_plans) == 1
        assert len(resumed_docs) == _DOCS_PER_BATCH
        assert not any(request["batch_tag"] == 0 for request in resumed_docs)
        assert initial_batch_zero_titles.isdisjoint(
            request["title"] for request in resumed_docs
        )

        requests_before_mismatch = len(requests)
        server.set_phase("fingerprint-mismatch")
        mismatched = _launch_driver(
            driver, run_dir, server.base_url, _TARGET_WORDS + 1
        )
        mismatch_stdout, mismatch_stderr = _communicate_or_kill(mismatched)
        assert mismatched.returncode != 0, mismatch_stdout
        assert "fingerprint" in mismatch_stderr.lower()
        assert "target_words" in mismatch_stderr

        requests_after_mismatch = _read_jsonl(request_log)
        assert len(requests_after_mismatch) == requests_before_mismatch
        assert not any(
            request["phase"] == "fingerprint-mismatch"
            for request in requests_after_mismatch
        )
