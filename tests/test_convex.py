import unittest

import torch

from hard_thermodynamics.convex import ICNN, PICNN


class ConvexNetworkTests(unittest.TestCase):
    def test_icnn_is_convex(self):
        torch.manual_seed(7)
        model = ICNN(
            2,
            [8, 8],
            1,
            activation="softplus",
            dtype=torch.float64,
        )
        x = torch.randn(32, 2, dtype=torch.float64)
        y = torch.randn(32, 2, dtype=torch.float64)
        weight = torch.rand(32, 1, dtype=torch.float64)

        mixed_output = model(weight * x + (1.0 - weight) * y)
        mixed_bound = (
            weight * model(x) + (1.0 - weight) * model(y)
        )

        self.assertTrue(
            torch.all(mixed_output <= mixed_bound + 1.0e-10)
        )

    def test_picnn_is_convex_for_fixed_context(self):
        torch.manual_seed(11)
        model = PICNN(
            input_x_dim=2,
            input_y_dim=1,
            feature_dim=8,
            feature_y_dim=6,
            out_dim=1,
            num_layers=3,
            act="softplus",
            act_v="elu",
            dtype=torch.float64,
        )
        x = torch.randn(24, 2, dtype=torch.float64)
        y = torch.randn(24, 2, dtype=torch.float64)
        context = torch.randn(24, 1, dtype=torch.float64)
        weight = torch.rand(24, 1, dtype=torch.float64)

        mixed_output = model(
            weight * x + (1.0 - weight) * y,
            context,
        )
        mixed_bound = (
            weight * model(x, context)
            + (1.0 - weight) * model(y, context)
        )

        self.assertTrue(
            torch.all(mixed_output <= mixed_bound + 1.0e-10)
        )

if __name__ == "__main__":
    unittest.main()
