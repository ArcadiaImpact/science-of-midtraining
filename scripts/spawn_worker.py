#!/usr/bin/env python3
"""Create or adopt one exact ARCH worker pod and journal the mutation."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 worker compatibility
    import tomli as tomllib


RUNPOD_BASE = "https://rest.runpod.io/v1"
TASK_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
OWNER_SCRIPT = Path("/workspace/.codex/skills/runpod-spinup/pod-own.sh")
RESERVED_WORKER_ENV = {
    "ARCH_DEADLINE_EPOCH",
    "ARCH_LOG_WRITER_ROLE_ARN",
    "ARCH_SESSION_ID",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CODEX_API_KEY",
    "EVAL_GH_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_TERMINAL_PROMPT",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LOGNAME",
    "PRIVATE_LOG_UPLOAD_URL",
    "PRIVATE_LOG_VERIFY_URL",
    "RUNPOD_API_KEY",
    "RUNPOD_POD_ID",
    "S3_BUCKET",
    "SHELL",
    "USER",
    "PATH",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
    "BASH_ENV",
    "ENV",
    "WORKER_GH_TOKEN",
}


class CreateOutcomeUnresolved(RuntimeError):
    """A single create may have succeeded, but exact-name reads stayed empty."""


def dotenv_value(reader: Path, dotenv: Path, key: str) -> str:
    proc = subprocess.run(
        ["python3", str(reader), str(dotenv), key],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def validate_dotenv(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"dotenv must be a regular non-symlink file: {path}")
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit(f"dotenv must have mode 0600: {path}")
    if info.st_uid != os.getuid():
        raise SystemExit(f"dotenv has unexpected owner uid {info.st_uid}: {path}")
    in_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0
    if in_repo:
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
        ).returncode == 0:
            raise SystemExit(f"dotenv must not be tracked by git: {path}")
        if subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            check=False,
        ).returncode != 0:
            raise SystemExit(f"dotenv must be ignored by git: {path}")


def validate_worker_env_names(names: list[str]) -> None:
    for name in names:
        if not ENV_RE.fullmatch(name):
            raise SystemExit(f"invalid environment variable name: {name}")
        if name in RESERVED_WORKER_ENV:
            raise SystemExit(
                f"worker required_env contains operational credential: {name}"
            )


def acquire_spawn_lock(directory: Path, index: int) -> int:
    path = directory / f".spawn-worker-{index}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise SystemExit(f"worker {index} already has an active spawn process") from exc
    return descriptor


def require_fresh_watcher(path: Path, task_name: str, session_id: str) -> str:
    try:
        info = path.lstat()
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            "pre-create task watcher is not armed; start scripts/task-pod-watch.sh"
        ) from exc
    expected_owner = f"arch-{task_name}-s{session_id[:12]}"
    if stat.S_ISLNK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("watcher heartbeat must be a mode-0600 regular file")
    if (
        not value.get("armed")
        or value.get("task_name") != task_name
        or value.get("session_id") != session_id
        or value.get("owner") != expected_owner
        or time.time() - float(value.get("heartbeat_epoch", 0)) > 15
    ):
        raise SystemExit("task watcher heartbeat is stale or belongs to another task")
    return expected_owner


def sha256_file(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"refusing non-regular or symlinked file: {path}")
    hasher = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise SystemExit(f"file changed while opening: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def session_field(
    session: dict[str, Any], canonical: str, legacy: str
) -> Any:
    """Read a canonical ARCH field while accepting one older state alias."""
    canonical_value = session.get(canonical)
    legacy_value = session.get(legacy)
    if (
        canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise SystemExit(f"session fields {canonical} and {legacy} disagree")
    return canonical_value if canonical_value is not None else legacy_value


def validate_launch_state(
    session: dict[str, Any],
    config: dict[str, Any],
    startup: Path,
    repo_root: Path,
    *,
    now: float | None = None,
) -> None:
    """Enforce priced-fleet authorization and evidence before any POST."""
    current_time = time.time() if now is None else now
    task_name = config.get("task_name")
    session_id = config.get("session_id")
    if session.get("task_name") != task_name:
        raise SystemExit("session/config task_name mismatch")
    if not isinstance(session_id, str) or not SESSION_RE.fullmatch(session_id):
        raise SystemExit("config is missing a valid 32-hex session_id")
    if session.get("session_id") != session_id:
        raise SystemExit("session/config session_id mismatch")
    if session.get("task_branch") != f"arch/{task_name}":
        raise SystemExit("session task_branch does not match the exact ARCH task branch")
    if session.get("authorizations", {}).get("launch_fleet") is not True:
        raise SystemExit("fleet launch is not explicitly authorized in session state")
    if session.get("canary_scored") is not True:
        raise SystemExit("held-out canary has not produced a verified non-null score")

    task_commit = session_field(session, "task_commit_sha", "task_commit")
    if not isinstance(task_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", task_commit):
        raise SystemExit("session is missing the exact task_commit")
    canary = session.get("canary", {})
    canary_score = canary.get("score")
    if (
        not isinstance(canary_score, (int, float))
        or isinstance(canary_score, bool)
        or not math.isfinite(float(canary_score))
        or canary.get("base_sha") != task_commit
    ):
        raise SystemExit("canary evidence is missing a finite score for task_commit")

    startup_hash = sha256_file(startup)
    startup_state_hash = session.get("worker_startup_sha256")
    legacy_worker_hash = session.get("worker", {}).get("startup_sha256")
    if (
        startup_state_hash is not None
        and legacy_worker_hash is not None
        and startup_state_hash != legacy_worker_hash
    ):
        raise SystemExit("canonical and legacy worker startup hashes disagree")
    if (startup_state_hash or legacy_worker_hash) != startup_hash:
        raise SystemExit("rendered worker startup hash does not match session state")
    preflight = session.get("preflight", {})
    if preflight.get("passed") is not True or preflight.get("commit_sha") != task_commit:
        raise SystemExit("preflight did not pass against the exact task_commit")
    log_path_value = preflight.get("log_path")
    if not isinstance(log_path_value, str) or not log_path_value.startswith(".arch/logs/"):
        raise SystemExit("preflight log_path must be relative under .arch/logs/")
    relative_log = Path(log_path_value)
    if relative_log.is_absolute() or ".." in relative_log.parts:
        raise SystemExit("preflight log_path escapes .arch/logs/")
    logs_root_path = repo_root / ".arch" / "logs"
    if logs_root_path.is_symlink() or not logs_root_path.is_dir():
        raise SystemExit("preflight logs directory must be a non-symlink directory")
    logs_root = logs_root_path.resolve()
    log_path = repo_root / relative_log
    resolved_log_path = log_path.resolve()
    if not resolved_log_path.is_relative_to(logs_root):
        raise SystemExit("preflight log_path escapes .arch/logs/")
    try:
        info = log_path.lstat()
    except OSError as exc:
        raise SystemExit(f"preflight log is unavailable: {log_path_value}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
    ):
        raise SystemExit("preflight log must be current-user-owned regular mode 0600")
    if preflight.get("log_sha256") != sha256_file(log_path):
        raise SystemExit("preflight log hash does not match session state")
    completed_epoch = preflight.get("completed_epoch")
    if (
        not isinstance(completed_epoch, (int, float))
        or completed_epoch <= 0
        or completed_epoch > current_time + 60
        or current_time - completed_epoch > 3600
    ):
        raise SystemExit("preflight is missing, invalid, or older than one hour")

    quote = session.get("price_quote", {})
    quoted_epoch = quote.get("quoted_epoch")
    hourly = quote.get("hourly_fleet_cost")
    maximum = quote.get("estimated_max_cost")
    if (
        not isinstance(quoted_epoch, (int, float))
        or quoted_epoch > current_time + 60
        or current_time - quoted_epoch > 900
        or not isinstance(hourly, (int, float))
        or isinstance(hourly, bool)
        or not math.isfinite(float(hourly))
        or hourly <= 0
        or not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or maximum <= 0
    ):
        raise SystemExit("price quote is missing, invalid, or older than 15 minutes")


def validate_repo_state(session: dict[str, Any], task_name: str) -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"], text=True
    )
    if branch != f"arch/{task_name}":
        raise SystemExit(f"current branch is {branch!r}, expected arch/{task_name}")
    if head != session_field(session, "task_commit_sha", "task_commit"):
        raise SystemExit("current HEAD does not match session task_commit")
    if dirty:
        raise SystemExit("worktree has unexpected tracked or untracked changes")


def verify_remote_task_head(task_name: str, expected_sha: str, token: str) -> None:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded}",
    }
    proc = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/arch/{task_name}"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    fields = proc.stdout.split()
    if proc.returncode or len(fields) < 2 or fields[0] != expected_sha:
        raise SystemExit("remote task branch does not resolve to the exact task_commit")


def launch_fingerprint(
    *,
    name: str,
    session: dict[str, Any],
    startup_sha256: str,
    gpu: str,
    image: str,
    index: int,
) -> str:
    worker = session.get("worker", {})
    identity = {
        "name": name,
        "session_id": session.get("session_id"),
        "task_commit": session_field(session, "task_commit_sha", "task_commit"),
        "startup_sha256": startup_sha256,
        "gpu": gpu,
        "image": image,
        "index": index,
        "model": worker.get("model"),
        "reasoning_effort": worker.get("reasoning_effort"),
        "codex_cli_version": worker.get("codex_cli_version"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validated_adoption_id(
    matches: list[dict[str, Any]],
    prior_action: dict[str, Any] | None,
    *,
    name: str,
    fingerprint: str,
) -> str | None:
    if len(matches) > 1:
        raise SystemExit(f"multiple exact pods named {name}; refusing to guess")
    if not matches:
        return None
    if (
        prior_action is None
        or prior_action.get("status") not in {"pending", "complete"}
        or prior_action.get("target_name") != name
        or prior_action.get("launch_fingerprint") != fingerprint
    ):
        raise SystemExit(
            "exact-name pod exists without a matching current-session "
            "pending/complete action; refusing stale adoption"
        )
    pod_id = matches[0].get("id")
    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise SystemExit("exact-name pod has an invalid ID")
    recorded_id = prior_action.get("pod_id")
    if recorded_id and recorded_id != pod_id:
        raise SystemExit("existing exact-name pod ID differs from the journaled ID")
    return pod_id


def state_store() -> Any:
    local = Path(__file__).with_name("session_state.py")
    if not local.is_file():
        local = (
            Path(__file__).resolve().parents[2]
            / "arch-init"
            / "scripts"
            / "session_state.py"
        )
    spec = importlib.util.spec_from_file_location("arch_session_state", local)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load session_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_action(
    session_path: Path,
    action_id: str,
    status: str,
    *,
    target_name: str,
    **fields: Any,
) -> dict[str, Any]:
    store = state_store()
    with store.transaction(session_path, exclusive=True) as session:
        actions = session.setdefault("actions", [])
        action = next((item for item in actions if item.get("id") == action_id), None)
        if action is None:
            action = {
                "id": action_id,
                "kind": "worker_create",
                "target_name": target_name,
                "created_epoch": int(time.time()),
            }
            actions.append(action)
        action.update(fields)
        action["status"] = status
        action["updated_epoch"] = int(time.time())
        pod_id = fields.get("pod_id")
        if pod_id and pod_id not in session.setdefault("worker_pod_ids", []):
            session["worker_pod_ids"].append(pod_id)
        if pod_id and status == "complete" and "start_epoch" not in session:
            session["start_epoch"] = int(time.time())
            worker = session.get("worker", {})
            budget = worker.get(
                "budget_seconds", session.get("wall_clock_budget_seconds")
            )
            if budget:
                session["deadline_epoch"] = session["start_epoch"] + int(budget)
        result = json.loads(json.dumps(session))
    return result


def read_action(session_path: Path, action_id: str) -> dict[str, Any] | None:
    store = state_store()
    with store.transaction(session_path, exclusive=False) as session:
        action = next(
            (item for item in session.get("actions", []) if item.get("id") == action_id),
            None,
        )
        return json.loads(json.dumps(action)) if action is not None else None


def request_json(
    path: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    reconcile_name: str | None = None,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    last_error: RuntimeError | None = None
    is_create = method == "POST" and reconcile_name is not None
    attempts = 1 if is_create else 5
    for attempt in range(attempts):
        req = urllib.request.Request(
            RUNPOD_BASE + path,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode()
            try:
                result = json.loads(raw, strict=False) if raw else {}
            except json.JSONDecodeError:
                # Response bodies may contain provider diagnostics or reflected
                # request data. Never include them in logs or exceptions.
                last_error = RuntimeError("RunPod returned malformed JSON (body redacted)")
                if not is_create:
                    if attempt < attempts - 1:
                        time.sleep(2**attempt)
                        continue
                    break
                result = {}
            if not is_create or isinstance(result, dict) and result.get("id"):
                return result
            if not last_error:
                last_error = RuntimeError("RunPod create response omitted a usable pod ID")
        except urllib.error.HTTPError as exc:
            exc.read()
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"RunPod HTTP {exc.code} (response body redacted)") from exc
            last_error = RuntimeError(
                f"RunPod transient HTTP {exc.code} (response body redacted)"
            )
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"RunPod network error: {exc}")
        if attempt == attempts - 1:
            break
        time.sleep(2**attempt)

    if is_create:
        # A timeout/5xx/missing-ID response is ambiguous. Never issue a second
        # POST in this invocation: RunPod list visibility can lag a successful
        # create. Poll exact-name inventory and leave the journal pending if it
        # remains unresolved.
        for delay in (2, 4, 8, 16, 30):
            time.sleep(delay)
            try:
                matches = existing_by_name(reconcile_name, api_key)
            except RuntimeError:
                continue
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(
                    f"ambiguous create outcome: multiple exact pods named {reconcile_name}"
                )
        raise CreateOutcomeUnresolved(
            f"single create outcome for {reconcile_name} remains unresolved; "
            "journal left pending for later authoritative reconciliation"
        ) from last_error

    if last_error is not None:
        raise last_error
    raise RuntimeError("RunPod request failed")


def existing_by_name(name: str, api_key: str) -> list[dict[str, Any]]:
    raw = request_json("/pods", api_key)
    pods = raw if isinstance(raw, list) else raw.get("pods", raw.get("data", []))
    return [pod for pod in pods if pod.get("name") == name]


def register(pod_id: str, owner: str) -> None:
    proc = subprocess.run(
        [str(OWNER_SCRIPT), "add", pod_id, owner],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "pod-own.sh failed")


def delete_exact(pod_id: str, api_key: str) -> bool:
    try:
        request_json(f"/pods/{pod_id}", api_key, method="DELETE")
        return True
    except RuntimeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, help="1-based worker index")
    parser.add_argument("--session", type=Path, default=Path(".arch/.session.json"))
    parser.add_argument("--config", type=Path, default=Path(".arch/config.toml"))
    parser.add_argument("--startup", type=Path, default=Path(".arch/worker_startup.sh"))
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-pending-create",
        action="store_true",
        help="After bounded zero-match reconciliation, explicitly retry one pending create",
    )
    args = parser.parse_args()

    if (
        not args.session.is_file()
        or not args.config.is_file()
        or not args.startup.is_file()
    ):
        raise SystemExit("session, config, and rendered startup script are required")
    if not OWNER_SCRIPT.is_file() and not args.dry_run:
        raise SystemExit(
            "host pod ownership helper unavailable; refusing to create an unwatched pod"
        )

    session = json.loads(args.session.read_text())
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    task_name = config["task_name"]
    if not TASK_RE.fullmatch(task_name):
        raise SystemExit("invalid task_name")
    session_id = config.get("session_id", "")
    if not SESSION_RE.fullmatch(session_id):
        raise SystemExit("invalid session_id")
    startup_hash = sha256_file(args.startup)
    if not args.dry_run:
        validate_dotenv(args.dotenv)
        watcher_owner = require_fresh_watcher(
            args.session.parent / ".watcher-heartbeat.json", task_name, session_id
        )
    else:
        watcher_owner = f"arch-{task_name}-s{session_id[:12]}"

    worker = session.get("worker", {})
    count = int(worker.get("count", session.get("worker_count", 0)))
    if args.index < 1 or args.index > count:
        raise SystemExit(f"--index must be in 1..{count}")
    spawn_lock_fd = (
        None if args.dry_run else acquire_spawn_lock(args.session.parent, args.index)
    )
    gpu = worker.get("gpu_tier", session.get("worker_gpu_tier"))
    image = worker.get("base_image", session.get("base_image"))
    if not gpu or not image:
        raise SystemExit("session is missing worker GPU tier or base image")

    scripts_dir = Path(__file__).resolve().parent
    reader = Path("scripts/read_dotenv.py")
    if not reader.is_file():
        reader = scripts_dir.parent.parent / "arch-init" / "scripts" / "read_dotenv.py"
    worker_config = config.get("worker", {})
    if not worker_config.get("task_scoped_credentials_acknowledged", False):
        raise SystemExit(
            "refusing worker launch until [worker].task_scoped_credentials_acknowledged=true"
        )
    if not worker_config.get("codex_key_compromise_acknowledged", False):
        raise SystemExit(
            "refusing worker launch until "
            "[worker].codex_key_compromise_acknowledged=true"
        )
    transcripts = config.get("transcripts", {})
    if transcripts.get("backend") != "s3" or not transcripts.get("bucket"):
        raise SystemExit("unattended workers require [transcripts] backend='s3' and bucket")
    if not args.dry_run:
        validate_launch_state(session, config, args.startup, Path.cwd())
        validate_repo_state(session, task_name)
    worker_required = list(worker_config.get("required_env", []))
    validate_worker_env_names(worker_required)
    required = [
        "RUNPOD_API_KEY",
        "CODEX_API_KEY",
        "WORKER_GH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    required += worker_required
    values = {
        name: dotenv_value(reader, args.dotenv, name)
        for name in dict.fromkeys(required)
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing required credentials: {', '.join(missing)}")
    api_key = values.pop("RUNPOD_API_KEY")
    if not args.dry_run:
        task_commit = session_field(session, "task_commit_sha", "task_commit")
        verify_remote_task_head(task_name, task_commit, values["WORKER_GH_TOKEN"])

    arch2 = Path("scripts/arch2")
    if not arch2.is_file():
        arch2 = scripts_dir.parent.parent / "arch-init" / "scripts" / "arch2"
    public_key = subprocess.check_output(
        [str(arch2), "ssh-key", "--pub"], text=True
    ).strip()
    env = {
        **values,
        "PUBLIC_KEY": public_key,
        "ARCH_WORKER_INDEX": str(args.index),
        "ARCH_SESSION_ID": session_id,
        "S3_BUCKET": transcripts["bucket"],
        "AWS_REGION": transcripts.get("region", "eu-north-1"),
    }
    aws_session_token = dotenv_value(reader, args.dotenv, "AWS_SESSION_TOKEN")
    if aws_session_token:
        env["AWS_SESSION_TOKEN"] = aws_session_token

    script_b64 = base64.b64encode(args.startup.read_bytes()).decode()
    docker_cmd = (
        f"echo {script_b64} | base64 -d > /tmp/arch-worker-start.sh && "
        "chmod +x /tmp/arch-worker-start.sh && exec bash /tmp/arch-worker-start.sh"
    )
    name = f"arch-{task_name}-s{session_id[:12]}-worker-{args.index}"
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": 100,
        "volumeInGb": 0,
        "cloudType": "SECURE",
        "ports": ["22/tcp"],
        "env": env,
        "dockerStartCmd": ["bash", "-c", docker_cmd],
    }
    if args.dry_run:
        redacted = dict(payload)
        redacted["env"] = {
            key: (
                value
                if key
                in {
                    "PUBLIC_KEY",
                    "ARCH_WORKER_INDEX",
                    "ARCH_SESSION_ID",
                    "AWS_REGION",
                    "S3_BUCKET",
                }
                else "<redacted>"
            )
            for key, value in env.items()
        }
        print(json.dumps(redacted, indent=2))
        return

    fingerprint = launch_fingerprint(
        name=name,
        session=session,
        startup_sha256=startup_hash,
        gpu=gpu,
        image=image,
        index=args.index,
    )
    action_id = f"worker:{args.index}"
    prior_action = read_action(args.session, action_id)
    if prior_action is not None and prior_action.get("status") == "pending" and (
        prior_action.get("target_name") != name
        or prior_action.get("launch_fingerprint") != fingerprint
    ):
        raise SystemExit("worker action is not a matching resumable pending create")
    matches = existing_by_name(name, api_key)
    pod_id = validated_adoption_id(
        matches, prior_action, name=name, fingerprint=fingerprint
    )
    if pod_id is None and prior_action is not None and prior_action.get("status") == "pending":
        if not args.resume_pending_create:
            raise SystemExit(
                "pending create has no currently visible exact pod; reconcile later and pass "
                "--resume-pending-create only for an explicit single retry"
            )
        for delay in (2, 4, 8, 16, 30):
            time.sleep(delay)
            matches = existing_by_name(name, api_key)
            pod_id = validated_adoption_id(
                matches, prior_action, name=name, fingerprint=fingerprint
            )
            if pod_id is not None:
                break
        if pod_id is None:
            update_action(
                args.session,
                action_id,
                "pending",
                target_name=name,
                launch_fingerprint=fingerprint,
                retry_authorized_epoch=int(time.time()),
            )
    if pod_id is not None:
        register(pod_id, watcher_owner)
        update_action(
            args.session,
            action_id,
            "complete",
            target_name=name,
            pod_id=pod_id,
            adopted=True,
            launch_fingerprint=fingerprint,
        )
        print(json.dumps({"id": pod_id, "name": name, "adopted": True}))
        return

    if prior_action is not None:
        if prior_action.get("status") == "complete":
            raise SystemExit(
                "journaled worker pod is missing; replacement requires a new explicit action"
            )
        if prior_action.get("status") != "pending":
            raise SystemExit("worker action is not a matching resumable pending create")
    else:
        update_action(
            args.session,
            action_id,
            "pending",
            target_name=name,
            gpu_tier=gpu,
            base_image=image,
            startup_sha256=startup_hash,
            launch_fingerprint=fingerprint,
        )
    try:
        pod = request_json(
            "/pods",
            api_key,
            method="POST",
            payload=payload,
            reconcile_name=name,
        )
        if isinstance(pod, list):
            raise TypeError("RunPod create returned an unexpected list (body redacted)")
        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError(
                "RunPod response omitted pod ID; reconcile the exact name before retry"
            )
        if pod.get("name") not in {None, name}:
            raise RuntimeError("RunPod create returned an unexpected pod name")
    except CreateOutcomeUnresolved as exc:
        update_action(
            args.session,
            action_id,
            "pending",
            target_name=name,
            launch_fingerprint=fingerprint,
            unresolved_error=str(exc),
        )
        raise SystemExit(str(exc)) from exc
    except (RuntimeError, TypeError) as exc:
        update_action(
            args.session, action_id, "failed", target_name=name, error=str(exc)
        )
        raise SystemExit(str(exc)) from exc

    try:
        register(pod_id, watcher_owner)
    except RuntimeError as exc:
        update_action(
            args.session,
            action_id,
            "registration_failed_cleanup_pending",
            target_name=name,
            pod_id=pod_id,
        )
        deleted = delete_exact(pod_id, api_key)
        update_action(
            args.session,
            action_id,
            "failed",
            target_name=name,
            pod_id=pod_id,
            deleted_after_registration_failure=deleted,
            error=str(exc),
        )
        raise SystemExit(
            f"ownership registration failed; exact pod {pod_id} deletion_success={deleted}: {exc}"
        ) from exc

    update_action(
        args.session,
        action_id,
        "complete",
        target_name=name,
        pod_id=pod_id,
        adopted=False,
        launch_fingerprint=fingerprint,
        cost_per_hr=pod.get("costPerHr"),
    )
    print(json.dumps({"id": pod_id, "name": name, "costPerHr": pod.get("costPerHr")}))
    if spawn_lock_fd is not None:
        os.close(spawn_lock_fd)


if __name__ == "__main__":
    main()
