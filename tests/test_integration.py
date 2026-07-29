import unittest

import torch

from hard_thermodynamics import (
    BatchZOHRate,
    integrate_training,
)


class ConstantRateModel(torch.nn.Module):
    def __init__(self, rate=2.0):
        super().__init__()
        self.rate = torch.nn.Parameter(
            torch.tensor(rate, dtype=torch.float64)
        )
        self.register_buffer(
            "prm_strain",
            torch.tensor(
                [[1.0], [0.0]],
                dtype=torch.float64,
            ),
        )
        self.dim = 1
        self.dim_total = 1
        self.step_size = 0.1
        self.rate_interp = lambda time, indices: None

    def forward(self, time, state):
        del time
        return torch.ones_like(state) * self.rate

    def compute_stress(self, state):
        return state[..., :1]


class LoadingRateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "prm_strain",
            torch.tensor(
                [[1.0], [0.0]],
                dtype=torch.float64,
            ),
        )
        self.dim = 1
        self.dim_total = 1
        self.step_size = 1.0
        self.rate_interp = BatchZOHRate(
            [0.0, 0.5, 1.0],
            [[[1.0]], [[3.0]]],
            dtype=torch.float64,
        )

    def forward(self, time, state):
        del state
        return self.rate_interp(time, self.idx)

    def compute_stress(self, state):
        return state[..., :1]


class IntegrationTests(unittest.TestCase):
    def test_fixed_euler_integrates_constant_rate(self):
        model = ConstantRateModel(rate=2.0)
        initial = torch.zeros(1, 1, dtype=torch.float64)
        times = torch.tensor(
            [0.0, 0.25, 1.0],
            dtype=torch.float64,
        )

        states, stresses = integrate_training(
            model,
            initial,
            times,
        )
        expected = 2.0 * times[:, None, None]

        torch.testing.assert_close(states, expected)
        torch.testing.assert_close(stresses, expected)

    def test_training_gradient_reaches_model_parameter(self):
        model = ConstantRateModel(rate=2.0)
        initial = torch.zeros(1, 1, dtype=torch.float64)
        states, _ = integrate_training(
            model,
            initial,
            [0.0, 1.0],
        )

        states[-1].sum().backward()

        torch.testing.assert_close(
            model.rate.grad,
            torch.tensor(1.0, dtype=torch.float64),
        )

    def test_fixed_euler_aligns_zero_order_hold_knots(self):
        model = LoadingRateModel()
        initial = torch.zeros(1, 1, dtype=torch.float64)

        states, _ = integrate_training(
            model,
            initial,
            [0.0, 1.0],
        )

        torch.testing.assert_close(
            states[-1],
            torch.tensor([[2.0]], dtype=torch.float64),
        )

if __name__ == "__main__":
    unittest.main()
