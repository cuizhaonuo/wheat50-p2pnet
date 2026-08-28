"""Watch training and commit its result to the main worktree experiment bank."""

import argparse
import datetime
import fcntl
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.repository import get_git_common_dir, get_shared_repository_root


AVERAGED_PATTERN = re.compile(
    r"^Averaged stats:.*?loss: [^ ]+ \(([^)]+)\).*?"
    r"loss_ce: [^ ]+ \(([^)]+)\).*?"
    r"loss_point: [^ ]+ \(([^)]+)\).*?"
    r"loss_point_unscaled: [^ ]+ \(([^)]+)\)"
)
EPOCH_PATTERN = re.compile(r"^\[ep (\d+)\]")
EVALUATION_PATTERN = re.compile(
    r"^mae:\s*([0-9.eE+-]+)\s+mse:\s*([0-9.eE+-]+)"
)
TRAINING_TIME_PATTERN = re.compile(r"^Training time (.+)$")


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument(
        "--ledger-root",
        type=Path,
        help="main worktree root; detected from Git when omitted",
    )
    parser.add_argument(
        "--ledger-branch",
        default="wheat50-model",
        help="branch that is allowed to receive experiment-bank commits",
    )
    parser.add_argument("--experiment-file", required=True, type=Path)
    parser.add_argument("--todo-file", default=Path("experiment-bank/todo.md"), type=Path)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--baseline-mae", required=True, type=float)
    parser.add_argument("--baseline-mse", required=True, type=float)
    parser.add_argument("--poll-seconds", default=30, type=float)
    return parser


def parse_training_log(text):
    """Extract epoch losses, validation metrics, and completion time."""
    epochs = []
    evaluations = []
    pending_losses = None
    last_epoch = None
    training_time = None
    for line in text.splitlines():
        averaged_match = AVERAGED_PATTERN.match(line)
        if averaged_match:
            pending_losses = tuple(float(value) for value in averaged_match.groups())
            continue
        epoch_match = EPOCH_PATTERN.match(line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))
            if pending_losses is not None:
                epochs.append((last_epoch, *pending_losses))
                pending_losses = None
            continue
        evaluation_match = EVALUATION_PATTERN.match(line)
        if evaluation_match and last_epoch is not None:
            evaluations.append(
                (last_epoch, float(evaluation_match.group(1)), float(evaluation_match.group(2)))
            )
            continue
        training_time_match = TRAINING_TIME_PATTERN.match(line)
        if training_time_match:
            training_time = training_time_match.group(1)
    return {
        "epochs": epochs,
        "evaluations": evaluations,
        "training_time": training_time,
    }


def replace_markdown_section(text, heading, content):
    """Replace the body of one level-two Markdown section."""
    pattern = re.compile(
        rf"(?ms)(^## {re.escape(heading)}\n\n).*?(?=^## |\Z)"
    )
    replacement = rf"\g<1>{content.strip()}\n\n"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Missing Markdown section: {heading}")
    return updated


def process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _git_output(worktree, *arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_ledger_root(worktree, ledger_root=None):
    """Resolve and validate the main worktree used as the shared ledger."""
    detected_root = get_shared_repository_root(worktree)
    resolved_root = (
        Path(ledger_root).expanduser().resolve() if ledger_root else detected_root
    )
    if resolved_root != detected_root:
        raise ValueError(
            f"ledger root {resolved_root} is not the main worktree {detected_root}"
        )
    top_level = Path(
        _git_output(resolved_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if top_level != resolved_root:
        raise ValueError(f"ledger root must be a worktree root: {resolved_root}")
    return resolved_root


def resolve_ledger_file(ledger_root, path):
    """Resolve a ledger file and ensure it stays inside the main worktree."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ledger_root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(ledger_root)
    except ValueError as error:
        raise ValueError(f"ledger file must be inside {ledger_root}: {candidate}") from error
    return candidate, relative


def resolve_train_log(ledger_root, path):
    """Resolve relative training logs from the shared main worktree."""
    path = Path(path).expanduser()
    return (ledger_root / path).resolve() if not path.is_absolute() else path.resolve()


@contextmanager
def experiment_bank_lock(ledger_root):
    """Serialize automated ledger updates from concurrent experiments."""
    lock_path = get_git_common_dir(ledger_root) / "experiment-bank.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path, text):
    """Replace a UTF-8 text file atomically within its current directory."""
    original_mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_file.write(text)
        temporary_path = Path(temporary_file.name)
    os.chmod(temporary_path, original_mode)
    temporary_path.replace(path)


def _format_success(parsed, baseline_mae, baseline_mse):
    epochs = parsed["epochs"]
    evaluations = parsed["evaluations"]
    if not epochs or not evaluations:
        raise ValueError("Completed log does not contain epoch and evaluation metrics")
    best_mae_epoch, best_mae, mse_at_best_mae = min(
        evaluations, key=lambda item: item[1]
    )
    best_mse_epoch, mae_at_best_mse, best_mse = min(
        evaluations, key=lambda item: item[2]
    )
    final_epoch, final_loss, final_ce, final_point, final_point_raw = epochs[-1]
    min_point_epoch, min_point_raw = min(
        ((item[0], item[4]) for item in epochs), key=lambda item: item[1]
    )
    max_point_epoch, max_point_raw = max(
        ((item[0], item[4]) for item in epochs), key=lambda item: item[1]
    )
    metrics = "\n".join(
        [
            f"- Best validation MAE: {best_mae:.4f} at epoch {best_mae_epoch}",
            f"- MSE at best-MAE epoch: {mse_at_best_mae:.4f}",
            f"- Best validation MSE: {best_mse:.4f} at epoch {best_mse_epoch}",
            f"- MAE at best-MSE epoch: {mae_at_best_mse:.4f}",
            f"- Final epoch: {final_epoch}",
            f"- Final training loss: {final_loss:.4f}",
            f"- Final classification loss: {final_ce:.4f}",
            f"- Final weighted point loss: {final_point:.4f}",
            f"- Final unscaled point loss: {final_point_raw:.4f}",
            f"- Minimum unscaled point loss: {min_point_raw:.4f} at epoch {min_point_epoch}",
            f"- Maximum unscaled point loss: {max_point_raw:.4f} at epoch {max_point_epoch}",
            f"- Training time: {parsed['training_time']}",
        ]
    )
    mae_change = best_mae - baseline_mae
    mse_change = best_mse - baseline_mse
    direction = "improved" if mae_change < 0 else "degraded"
    outcome = (
        f"The experiment {direction} best MAE by {abs(mae_change):.4f} "
        f"and changed best MSE by {mse_change:+.4f} relative to the "
        f"foreground-only baseline ({baseline_mae:.4f}/{baseline_mse:.4f}). "
        f"The unscaled point loss finished at {final_point_raw:.4f} after a "
        f"minimum of {min_point_raw:.4f}."
    )
    observations = (
        "- Automated finalization confirmed that all configured epochs ended "
        "and validation ran successfully.\n"
        f"- Best MAE occurred at epoch {best_mae_epoch}; best MSE occurred at "
        f"epoch {best_mse_epoch}.\n"
        f"- Unscaled point loss ranged from {min_point_raw:.4f} to "
        f"{max_point_raw:.4f}."
    )
    return metrics, observations, outcome


def _format_failure(log_text):
    nonempty_lines = [line for line in log_text.splitlines() if line.strip()]
    tail = "\n".join(nonempty_lines[-8:]) or "No log output was produced."
    metrics = "- Training did not complete; final metrics are unavailable."
    observations = "- The training process exited before writing `Training time`."
    outcome = f"The experiment failed or was interrupted. Final log tail:\n\n```text\n{tail}\n```"
    return metrics, observations, outcome


def finalize(args, succeeded, log_text):
    parsed = parse_training_log(log_text)
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    if succeeded:
        metrics, observations, outcome = _format_success(
            parsed, args.baseline_mae, args.baseline_mse
        )
        status = f"Completed automatically on {now} from code commit `{args.code_commit}`."
    else:
        metrics, observations, outcome = _format_failure(log_text)
        status = f"Ended unsuccessfully on {now} from code commit `{args.code_commit}`."

    with experiment_bank_lock(args.ledger_root):
        current_branch = _git_output(
            args.ledger_root, "branch", "--show-current"
        )
        if args.ledger_branch and current_branch != args.ledger_branch:
            raise RuntimeError(
                f"refusing to update experiment bank on branch {current_branch!r}; "
                f"expected {args.ledger_branch!r}"
            )

        experiment_path, experiment_relative = resolve_ledger_file(
            args.ledger_root, args.experiment_file
        )
        todo_path, todo_relative = resolve_ledger_file(
            args.ledger_root, args.todo_file
        )

        experiment_text = experiment_path.read_text(encoding="utf-8")
        experiment_text = replace_markdown_section(
            experiment_text, "Status", status
        )
        experiment_text = replace_markdown_section(
            experiment_text, "Metrics", metrics
        )
        experiment_text = replace_markdown_section(
            experiment_text, "Observations", observations
        )
        experiment_text = replace_markdown_section(
            experiment_text, "Outcome", outcome
        )

        todo_text = todo_path.read_text(encoding="utf-8")
        task_pattern = re.compile(
            rf"(\|\s*{re.escape(args.task_name)}\s*\|\s*)pending(\s*\|)"
        )
        todo_text, count = task_pattern.subn(
            r"\1compeleted\2", todo_text, count=1
        )
        if count != 1:
            raise ValueError(f"Pending task row not found: {args.task_name}")

        atomic_write_text(experiment_path, experiment_text)
        atomic_write_text(todo_path, todo_text)

        paths = [str(experiment_relative), str(todo_relative)]
        subprocess.run(
            ["git", "add", "--", *paths], cwd=args.ledger_root, check=True
        )
        subprocess.run(
            [
                "git",
                "commit",
                "--only",
                "-m",
                args.commit_message,
                "--",
                *paths,
            ],
            cwd=args.ledger_root,
            check=True,
        )


def main():
    args = get_args_parser().parse_args()
    args.worktree = args.worktree.expanduser().resolve()
    args.ledger_root = resolve_ledger_root(args.worktree, args.ledger_root)
    log_path = resolve_train_log(args.ledger_root, args.train_log)
    while True:
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        parsed = parse_training_log(log_text)
        if parsed["training_time"] is not None:
            finalize(args, True, log_text)
            return
        if not process_is_alive(args.pid):
            time.sleep(2)
            log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            succeeded = parse_training_log(log_text)["training_time"] is not None
            finalize(args, succeeded, log_text)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
