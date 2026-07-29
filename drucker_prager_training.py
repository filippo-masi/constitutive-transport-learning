"""
Train the hard-thermodynamic operator network on Drucker–Prager data.
Author: Filippo Masi
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.utils.parametrize as P

from hard_thermodynamics import (
    BatchZOHRate,
    EarlyStopping,
    HardThermodynamicsOpNet,
    ICNN,
    get_params,
    integrate_training,
)


# ---------------------------------------------------------------------------
# Reproducibility and numerical configuration
# ---------------------------------------------------------------------------

np.random.seed(1)
torch.manual_seed(1)

# Restrict PyTorch to one thread for reproducible timing and CPU behavior.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

dtype = torch.float64
verbose_frequency = 10

data_path = "./data/Drucker_Prager/training_data.pkl"
results_path = "./checkpoints/drucker_prager_retrained.pt"


# ---------------------------------------------------------------------------
# Load and preprocess the training data
# ---------------------------------------------------------------------------

# The pickle file contains the strain/stress trajectories, their values at the
# next time step, dataset dimensions, and the valid length of each protocol.
with open(data_path, "rb") as file:
    data = pickle.load(file)

(
    strain,
    strain_tdt,
    stress,
    stress_tdt,
    N_snapshots,
    N_protocols,
    dim,
    STOP,
) = data

# All loading protocols are defined over the normalized interval [0, 1].
dt = 1 / N_snapshots

# Match the NumPy data type to the requested PyTorch precision.
if dtype == torch.float32:
    dt = np.float32(dt)
    strain = np.float32(strain)
    strain_tdt = np.float32(strain_tdt)
    stress = np.float32(stress)

# Approximate the strain rate from consecutive strain snapshots.
strain_rate = (strain_tdt - strain) / dt

# Store the normalization parameters used by the network.
prm_strain = get_params(strain)
prm_strain_rate = get_params(strain_rate)
prm_stress = get_params(stress)


# ---------------------------------------------------------------------------
# Construct the hard-thermodynamic operator network
# ---------------------------------------------------------------------------

# This benchmark does not introduce additional hidden state variables.
dim_hidden = 0
dim_total = dim + dim_hidden

norm_params = [prm_strain, prm_stress, prm_strain_rate]

# Architecture of the transport/evolution network:
# [input dimension, flattened operator dimension, hidden layers, activation].
evolution_architecture = [
    dim,
    dim_total**2,
    [64, 64, 64],
    "softplus",
]

# Convex energy potential. For the Drucker–Prager benchmark, the input consists
# of the two strain-invariant coordinates.
energy_net = ICNN(
    2,
    [64, 64],
    1,
    activation="softplus",
    dtype=dtype,
)

model = HardThermodynamicsOpNet(
    evolution_architecture,
    energy_net,
    norm_params,
    dim,
    dim_hidden,
    dtype=dtype,
)


# ---------------------------------------------------------------------------
# Configure time integration
# ---------------------------------------------------------------------------

time_grid = torch.arange(0, 1.0, dt)

model.step_size = dt

# The prescribed strain rate is treated as piecewise constant over each time
# interval (zero-order hold).
model.rate_interp = BatchZOHRate(
    time_grid,
    strain_rate,
    dtype=dtype,
)


# ---------------------------------------------------------------------------
# Prepare tensors and trainable initial conditions
# ---------------------------------------------------------------------------

stress_train = torch.from_numpy(stress)
strain_train = torch.from_numpy(strain)

# All available protocols are used for training; this script does not define a
# separate validation split.
id_protocol_train_val = np.arange(strain_train.shape[1])

# Learn six initial strain states in normalized coordinates. Below, these six
# states are repeated for three groups of loading protocols. This assumes that
# the dataset contains 18 protocols arranged in three groups of six.
model.norm_init_cond = torch.nn.Parameter(
    torch.zeros((6, dim), dtype=dtype),
    requires_grad=True,
)


# ---------------------------------------------------------------------------
# Optimization and early stopping
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
n_epochs = int(os.environ.get("HARD_THERMO_EPOCHS", "10000"))
if n_epochs < 2:
    raise ValueError("HARD_THERMO_EPOCHS must be at least 2")

training_loss_hist = []

early_stopping = EarlyStopping(
    patience=2_000,
    delta=1.0e-6,
    verbose=False,
    path=results_path,
)

# Normalize the targets once because the normalization parameters remain fixed.
norm_stress_target = model.Normalize(stress_train, prm_stress)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for epoch in range(1, n_epochs):
    optimizer.zero_grad(set_to_none=True)

    # Cache parametrized tensors during the trajectory integration to avoid
    # repeatedly reconstructing the constrained model parameters.
    with P.cached():
        # Update the energy-gradient reference using the first strain snapshot.
        model.ref_grads = model.reference_grads(strain_train[0])

        # Convert the learned initial conditions back to physical strain units.
        init_cond = model.DeNormalize(
            model.norm_init_cond,
            model.prm_strain,
        )

        e_strain0 = torch.zeros_like(
            strain_train[0, id_protocol_train_val]
        )

        # Apply the same six learned initial states to the three protocol groups.
        e_strain0[:6] = init_cond
        e_strain0[6:12] = init_cond
        e_strain0[12:18] = init_cond

        # Evaluate the initial stress separately and then integrate the complete
        # state and stress trajectories.
        initial_pred_stress = model.compute_stress(e_strain0)
        initial_conditions = e_strain0

        pred_state, pred_stress = integrate_training(
            model,
            initial_conditions,
            time_grid,
            id_protocol_train_val,
        )

    # Compute the loss in normalized stress coordinates so that stress
    # components with different physical scales contribute comparably.
    norm_pred_stress = model.Normalize(pred_stress, prm_stress)
    norm_initial_pred_stress = model.Normalize(
        initial_pred_stress,
        prm_stress,
    )

    # Match the stress at the initial time.
    loss_init = torch.mean(
        (norm_initial_pred_stress - norm_stress_target[0]) ** 2
    )

    # Match the predicted stress histories over the valid portion of the data.
    loss_traj = stress_train.new_zeros(())

    for i in range(N_protocols):
        loss_traj = loss_traj + torch.mean(
            (
                norm_pred_stress[1 : STOP[i], i]
                - norm_stress_target[1 : STOP[i], i]
            )
            ** 2
        ) / N_protocols

    loss = loss_init + loss_traj
    training_loss_value = loss.item()
    training_loss_hist.append(training_loss_value)

    # Save improved checkpoints and stop after a prolonged lack of improvement.
    if not np.isnan(training_loss_value):
        early_stopping(training_loss_value, model)

        if early_stopping.early_stop:
            print("Training interrupted because of early stopping.")
            break

    loss.backward()
    optimizer.step()

    if epoch % verbose_frequency == 0:
        print(
            f"epoch: {epoch}/{n_epochs}"
            f" | loss: {training_loss_value:.4e}"
            f" | initial cond: {loss_init.item():.4e}"
            f" | stress hist: {loss_traj.item():.4e}"
        )
