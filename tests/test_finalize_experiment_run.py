import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.finalize_experiment_run import (
    finalize,
    parse_training_log,
    replace_markdown_section,
    resolve_ledger_root,
    resolve_train_log,
)


class FinalizeExperimentRunTest(unittest.TestCase):
    def test_parses_losses_evaluation_and_training_time(self):
        parsed = parse_training_log(
            "\n".join(
                [
                    "Averaged stats: lr: 0.000100  loss: 0.1 (0.2)  "
                    "loss_ce: 0.1 (0.15)  loss_point: 0.01 (0.05)  "
                    "loss_ce_unscaled: 0.1 (0.15)  "
                    "loss_point_unscaled: 2.0 (2.5)",
                    "[ep 5][lr 0.0001000][1.00s]",
                    "mae: 3.0 mse: 4.0 time: 1.0 best mae: 3.0",
                    "Training time 0:10:00",
                ]
            )
        )
        self.assertEqual(parsed["epochs"], [(5, 0.2, 0.15, 0.05, 2.5)])
        self.assertEqual(parsed["evaluations"], [(5, 3.0, 4.0)])
        self.assertEqual(parsed["training_time"], "0:10:00")

    def test_replaces_only_requested_markdown_section(self):
        text = "# Title\n\n## Status\n\nPending.\n\n## Metrics\n\nPending.\n"
        updated = replace_markdown_section(text, "Status", "Completed.")
        self.assertIn("## Status\n\nCompleted.", updated)
        self.assertIn("## Metrics\n\nPending.", updated)

    def test_resolves_main_worktree_from_linked_worktree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repository"
            linked_worktree = Path(temporary_directory) / "linked"
            repository.mkdir()
            self._git(repository, "init", "-b", "wheat50-model")
            self._configure_git(repository)
            (repository / "tracked.txt").write_text("initial\n", encoding="utf-8")
            self._git(repository, "add", "tracked.txt")
            self._git(repository, "commit", "-m", "Initial commit")
            self._git(repository, "branch", "experiment/test")
            self._git(
                repository,
                "worktree",
                "add",
                str(linked_worktree),
                "experiment/test",
            )

            self.assertEqual(resolve_ledger_root(linked_worktree), repository)
            self.assertEqual(
                resolve_train_log(repository, Path("logs/test/train.log")),
                repository / "logs/test/train.log",
            )

    def test_finalizes_and_commits_only_central_ledger_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repository"
            repository.mkdir()
            self._git(repository, "init", "-b", "wheat50-model")
            self._configure_git(repository)
            experiment_file = Path("experiment-bank/experiments/test.md")
            todo_file = Path("experiment-bank/todo.md")
            (repository / experiment_file).parent.mkdir(parents=True)
            (repository / experiment_file).write_text(
                "# Test\n\n"
                "## Status\n\nPending.\n\n"
                "## Metrics\n\nPending.\n\n"
                "## Observations\n\nPending.\n\n"
                "## Outcome\n\nPending.\n",
                encoding="utf-8",
            )
            (repository / todo_file).write_text(
                "| Task | Status |\n| --- | --- |\n| Test task | pending |\n",
                encoding="utf-8",
            )
            (repository / "unrelated.txt").write_text("initial\n", encoding="utf-8")
            self._git(repository, "add", ".")
            self._git(repository, "commit", "-m", "Add experiment template")
            (repository / "unrelated.txt").write_text("user change\n", encoding="utf-8")
            self._git(repository, "add", "unrelated.txt")

            args = SimpleNamespace(
                ledger_root=repository,
                ledger_branch="wheat50-model",
                experiment_file=experiment_file,
                todo_file=todo_file,
                task_name="Test task",
                code_commit="abc1234",
                commit_message="Record test result",
                baseline_mae=5.0,
                baseline_mse=6.0,
            )
            finalize(args, True, self._completed_log())

            experiment_text = (repository / experiment_file).read_text(
                encoding="utf-8"
            )
            todo_text = (repository / todo_file).read_text(encoding="utf-8")
            self.assertIn("Best validation MAE: 3.0000", experiment_text)
            self.assertIn("| Test task | compeleted |", todo_text)
            self.assertEqual(
                self._git(repository, "show", "--format=", "--name-only", "HEAD"),
                "experiment-bank/experiments/test.md\nexperiment-bank/todo.md",
            )
            self.assertEqual(
                self._git(repository, "status", "--short"),
                "M  unrelated.txt",
            )

    @staticmethod
    def _completed_log():
        return "\n".join(
            [
                "Averaged stats: loss: 0.1 (0.2) loss_ce: 0.1 (0.15) "
                "loss_point: 0.01 (0.05) loss_point_unscaled: 2.0 (2.5)",
                "[ep 5][lr 0.0001000][1.00s]",
                "mae: 3.0 mse: 4.0 time: 1.0 best mae: 3.0",
                "Training time 0:10:00",
            ]
        )

    @staticmethod
    def _configure_git(repository):
        FinalizeExperimentRunTest._git(
            repository, "config", "user.name", "Test User"
        )
        FinalizeExperimentRunTest._git(
            repository, "config", "user.email", "test@example.com"
        )

    @staticmethod
    def _git(repository, *arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
