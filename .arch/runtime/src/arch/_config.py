"""Config loader for .arch/config.toml."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ConfigError(Exception):
    """Raised when .arch/config.toml is missing, malformed, or incomplete."""


REQUIRED_KEYS = ("task_name", "task_description", "eval_shim", "public_data_root")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SECRET_ENV_RE = re.compile(r"(?:_TOKEN|_KEY|_SECRET|_PASSWORD)$")


@dataclass
class Config:
    """Parsed .arch/config.toml."""

    task_name: str
    task_description: str
    eval_shim: str
    public_data_root: str
    submit_deny_globs: list[str] = field(default_factory=list)
    worker_required_env: list[str] = field(default_factory=list)
    public_eval_env: list[str] = field(default_factory=list)
    repo_root: Path = field(default_factory=Path)

    @property
    def label(self) -> str:
        """The GitHub PR label for this task."""
        return f"arch/{self.task_name}"


def _find_config_path(start: Path) -> Path:
    """Walk upward from `start` looking for .arch/config.toml. Raise if not found."""
    for ancestor in [start, *start.parents]:
        candidate = ancestor / ".arch" / "config.toml"
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f".arch/config.toml not found in {start} or any parent directory"
    )


def load_config(repo_root: Path | None = None) -> Config:
    """Load .arch/config.toml. Search upward from `repo_root` (default: cwd)."""
    start = Path(repo_root) if repo_root else Path.cwd()
    path = _find_config_path(start)

    with path.open("rb") as f:
        data = tomllib.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ConfigError(
            f"{path}: missing required key(s): {', '.join(missing)}"
        )

    submit_section = data.get("submit", {})
    eval_section = data.get("eval", {})
    worker_section = data.get("worker", {})
    worker_required_env = worker_section.get("required_env", [])
    public_eval_env = eval_section.get("public_env", [])
    for field_name, names in (
        ("worker.required_env", worker_required_env),
        ("eval.public_env", public_eval_env),
    ):
        if not isinstance(names, list) or not all(
            isinstance(name, str) and ENV_NAME_RE.fullmatch(name) for name in names
        ):
            raise ConfigError(f"{path}: {field_name} must contain valid environment names")
    secret_public_names = [name for name in public_eval_env if SECRET_ENV_RE.search(name)]
    if secret_public_names:
        raise ConfigError(
            f"{path}: eval.public_env contains secret-like names: "
            + ", ".join(secret_public_names)
        )
    return Config(
        task_name=data["task_name"],
        task_description=data["task_description"],
        eval_shim=data["eval_shim"],
        public_data_root=data["public_data_root"],
        submit_deny_globs=submit_section.get("deny_globs", []),
        worker_required_env=worker_required_env,
        public_eval_env=public_eval_env,
        repo_root=path.parent.parent,
    )
