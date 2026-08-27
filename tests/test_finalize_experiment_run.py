import unittest

from tools.finalize_experiment_run import (
    parse_training_log,
    replace_markdown_section,
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


if __name__ == "__main__":
    unittest.main()
