import unittest

import torch

from hard_thermodynamics import (
    HardThermodynamicsOpNet,
    ICNN,
    PICNN,
)


def normalization(dim, dtype):
    return [
        torch.stack(
            (
                torch.linspace(1.0, 2.0, dim, dtype=dtype),
                torch.zeros(dim, dtype=dtype),
            )
        ),
        torch.stack(
            (
                torch.linspace(3.0, 4.0, dim, dtype=dtype),
                torch.zeros(dim, dtype=dtype),
            )
        ),
        torch.stack(
            (
                torch.linspace(0.1, 0.2, dim, dtype=dtype),
                torch.zeros(dim, dtype=dtype),
            )
        ),
    ]


def model_without_internal_variables(dtype=torch.float64):
    energy = ICNN(
        2,
        [8, 8],
        1,
        activation="softplus",
        dtype=dtype,
    )
    return HardThermodynamicsOpNet(
        [2, 4, [8], "elu"],
        energy,
        normalization(2, dtype),
        dim=2,
        dim_hidden=0,
        dtype=dtype,
    )


class ThermodynamicModelTests(unittest.TestCase):
    def test_reference_energy_and_stress_vanish(self):
        torch.manual_seed(3)
        model = model_without_internal_variables()
        zero = torch.zeros(5, 2, dtype=torch.float64)

        energy = model.energy_value(zero)
        stress = model.compute_stress(zero)

        torch.testing.assert_close(
            energy,
            torch.zeros_like(energy),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            stress,
            torch.zeros_like(stress),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_hidden_force_vanishes_at_reference(self):
        torch.manual_seed(5)
        dtype = torch.float64
        energy = PICNN(
            input_x_dim=1,
            input_y_dim=1,
            feature_dim=6,
            feature_y_dim=6,
            out_dim=1,
            num_layers=2,
            dtype=dtype,
        )
        model = HardThermodynamicsOpNet(
            [2, 4, [8], "relu"],
            energy,
            normalization(1, dtype),
            dim=1,
            dim_hidden=1,
            dtype=dtype,
        )
        zero = torch.zeros(4, 2, dtype=dtype)

        stress, force = model.compute_stress(
            zero,
            return_all=True,
        )

        torch.testing.assert_close(
            stress,
            torch.zeros_like(stress),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            force,
            torch.zeros_like(force),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_transport_is_positive_semidefinite_and_dissipative(self):
        torch.manual_seed(13)
        model = model_without_internal_variables()
        forces = torch.randn(16, 2, dtype=torch.float64)
        operator = model.transport_operator(forces)

        symmetric = 0.5 * (
            operator + operator.transpose(-1, -2)
        )
        skew = 0.5 * (
            operator - operator.transpose(-1, -2)
        )
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        dissipation = torch.einsum(
            "...i,...ij,...j->...",
            forces,
            operator,
            forces,
        )
        skew_power = torch.einsum(
            "...i,...ij,...j->...",
            forces,
            skew,
            forces,
        )

        self.assertGreaterEqual(eigenvalues.min().item(), -1.0e-10)
        self.assertGreaterEqual(dissipation.min().item(), -1.0e-10)
        torch.testing.assert_close(
            skew_power,
            torch.zeros_like(skew_power),
            atol=1.0e-10,
            rtol=0.0,
        )

if __name__ == "__main__":
    unittest.main()
