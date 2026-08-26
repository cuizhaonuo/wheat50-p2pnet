import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from models import p2pnet


class P2PNetLossTest(unittest.TestCase):
    @mock.patch("models.p2pnet.build_backbone", return_value=torch.nn.Identity())
    def test_point_loss_contributes_gradient(self, _build_backbone):
        args = SimpleNamespace(
            row=2,
            line=2,
            point_loss_coef=0.0002,
            set_cost_class=1.0,
            set_cost_point=0.05,
            eos_coef=0.5,
        )
        _, criterion = p2pnet.build(args, training=True)
        self.assertEqual(
            criterion.weight_dict,
            {"loss_ce": 1, "loss_point": args.point_loss_coef},
        )

        pred_logits = torch.tensor(
            [[[0.0, 2.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]],
            requires_grad=True,
        )
        pred_points = torch.tensor(
            [[[2.0, 3.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]],
            requires_grad=True,
        )
        targets = [
            {
                "labels": torch.ones(1, dtype=torch.int64),
                "point": torch.tensor([[5.0, 7.0]]),
            }
        ]

        loss_dict = criterion(
            {"pred_logits": pred_logits, "pred_points": pred_points},
            targets,
        )
        total_loss = sum(
            loss_dict[name] * criterion.weight_dict[name]
            for name in loss_dict
            if name in criterion.weight_dict
        )
        total_loss.backward()

        self.assertIn("loss_point", loss_dict)
        self.assertIsNotNone(pred_points.grad)
        self.assertGreater(pred_points.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
