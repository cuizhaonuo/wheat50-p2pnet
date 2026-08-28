import subprocess
import tempfile
import unittest
from pathlib import Path

from util.repository import (
    get_shared_repository_root,
    normalize_experiment_name,
    resolve_shared_path,
)


class RepositoryPathsTest(unittest.TestCase):
    def test_resolves_artifacts_below_main_worktree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repository"
            linked_worktree = Path(temporary_directory) / "linked"
            repository.mkdir()
            self._git(repository, "init", "-b", "main")
            self._git(repository, "config", "user.name", "Test User")
            self._git(repository, "config", "user.email", "test@example.com")
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

            shared_root = get_shared_repository_root(linked_worktree)
            self.assertEqual(shared_root, repository)
            self.assertEqual(
                resolve_shared_path("logs", shared_root, "example-run"),
                str(repository / "logs/example-run"),
            )

    def test_rejects_experiment_name_with_path_components(self):
        with self.assertRaises(ValueError):
            normalize_experiment_name("../outside")

    @staticmethod
    def _git(repository, *arguments):
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
