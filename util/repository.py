"""Helpers for paths shared by all worktrees of this repository."""

import re
import subprocess
from pathlib import Path


def _working_directory(start):
    path = Path(start).expanduser().resolve()
    return path if path.is_dir() else path.parent


def get_git_common_dir(start):
    """Return the absolute common Git directory for ``start``."""
    cwd = _working_directory(start)
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    common_dir = Path(result.stdout.strip()).expanduser()
    if not common_dir.is_absolute():
        common_dir = cwd / common_dir
    return common_dir.resolve()


def get_shared_repository_root(start):
    """Return the main worktree root shared by linked worktrees."""
    return get_git_common_dir(start).parent


def normalize_experiment_name(name):
    """Validate an experiment name used as one output-directory component."""
    if not name:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            "experiment_name must contain only letters, numbers, '.', '_' or '-'"
        )
    return name


def resolve_shared_path(path, shared_root, experiment_name=""):
    """Resolve an artifact path below the main worktree unless it is absolute."""
    if not path:
        return ""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(shared_root) / resolved
    if experiment_name:
        resolved /= normalize_experiment_name(experiment_name)
    return str(resolved.resolve())
