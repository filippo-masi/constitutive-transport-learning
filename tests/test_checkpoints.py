import pickle
import unittest
from pathlib import Path

import numpy as np
import torch

from hard_thermodynamics import (
    BatchZOHRate,
    HardThermodynamicsOpNet,
    ICNN,
    PICNN,
    get_params,
)


ROOT = Path(__file__).resolve().parents[1]


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_drucker_prager_checkpoint_loads_strictly(self):
        dtype = torch.float64
        with (
            ROOT / "data/Drucker_Prager/training_data.pkl"
        ).open("rb") as stream:
            (
                strain,
                strain_next,
                stress,
                _stress_next,
                n_snapshots,
                _n_protocols,
                dim,
                _stop,
            ) = pickle.load(stream)

        step = 1.0 / int(n_snapshots)
        strain_rate = (strain_next - strain) / step
        normalization = [
            get_params(strain),
            get_params(stress),
            get_params(strain_rate),
        ]
        energy = ICNN(
            2,
            [64, 64],
            1,
            activation="softplus",
            dtype=dtype,
        )
        model = HardThermodynamicsOpNet(
            [dim, dim**2, [64, 64, 64], "softplus"],
            energy,
            normalization,
            dim=dim,
            dim_hidden=0,
            dtype=dtype,
        )
        time_grid = torch.arange(0.0, 1.0, step)
        model.rate_interp = BatchZOHRate(
            time_grid,
            strain_rate,
            dtype=dtype,
        )
        model.norm_init_cond = torch.nn.Parameter(
            torch.zeros((6, dim), dtype=dtype)
        )

        state = torch.load(
            ROOT / "checkpoints/drucker_prager_2.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)

        stress_at_reference = model.compute_stress(
            torch.zeros(2, dim, dtype=dtype)
        )
        self.assertTrue(torch.isfinite(stress_at_reference).all())

    def test_isotropic_hardening_checkpoint_loads_strictly(self):
        dtype = torch.float32
        with (
            ROOT
            / "data/elasto_plastic_hardening/training_data.pkl"
        ).open("rb") as stream:
            (
                strain,
                _strain_next,
                stress,
                _stress_next,
                n_snapshots,
                _n_protocols,
                dim,
            ) = pickle.load(stream)

        step = np.float32(1.0 / int(n_snapshots))
        strain_rate = np.zeros_like(strain)
        for protocol in range(strain.shape[1]):
            strain_rate[:, protocol] = np.gradient(
                strain[:, protocol, 0],
                step,
                edge_order=1,
            )[:, None]

        normalization = [
            get_params(strain),
            get_params(stress),
            get_params(strain_rate),
        ]
        dim_hidden = 1
        dim_total = dim + dim_hidden
        energy = PICNN(
            input_x_dim=dim,
            input_y_dim=dim_hidden,
            feature_dim=32,
            feature_y_dim=32,
            out_dim=1,
            num_layers=3,
            act="elu",
            act_v="elu",
            dtype=dtype,
        )
        model = HardThermodynamicsOpNet(
            [
                dim_total,
                dim_total**2,
                [64, 64, 64, 64],
                "relu",
            ],
            energy,
            normalization,
            dim=dim,
            dim_hidden=dim_hidden,
            dtype=dtype,
        )
        time_grid = torch.arange(0.0, 1.0, step)
        model.rate_interp = BatchZOHRate(
            time_grid,
            strain_rate,
            dtype=dtype,
        )
        model.norm_init_cond = torch.nn.Parameter(
            torch.zeros((1, dim), dtype=dtype)
        )

        state = torch.load(
            ROOT
            / "checkpoints/elasto_plastic_hardening2.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)

        stress_at_reference, force_at_reference = (
            model.compute_stress(
                torch.zeros(2, dim_total, dtype=dtype),
                return_all=True,
            )
        )
        self.assertTrue(torch.isfinite(stress_at_reference).all())
        self.assertTrue(torch.isfinite(force_at_reference).all())


if __name__ == "__main__":
    unittest.main()
