"""
Train the model on elasto-plasticity.
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

# Restrict PyTorch to a single CPU thread.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Use single precision consistently for the data, network, and integration.
dtype = torch.float32

# Report the training losses every `verbose_frequency` epochs.
verbose_frequency = 10

data_path = "./data/elasto_plastic/training_data.pkl"
results_path = "./checkpoints/elasto_plasticity_retrained.pt"


# ---------------------------------------------------------------------------
# Load and preprocess the training data
# ---------------------------------------------------------------------------

# The dataset contains the strain and stress trajectories, their values at the
# following time step, and the dimensions of the collection.
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
) = data
# The loading protocols are parameterized over the normalized interval [0, 1].
dt = 1 / N_snapshots

# Convert the arrays and time step to the precision used by the model.
if dtype == torch.float32:
    dt = np.float32(dt)
    strain = np.float32(strain)
    strain_tdt = np.float32(strain_tdt)
    stress = np.float32(stress)
elif dtype == torch.float64:
    dt = np.float64(dt)
    strain = np.float64(strain)
    strain_tdt = np.float64(strain_tdt)
    stress = np.float64(stress)

# Approximate the prescribed strain rate by a forward difference.
# strain_rate = (strain_tdt - strain) / dt
strain_rate = np.zeros_like(strain)
for i in range(strain[:,:,0].shape[1]):
    strain_rate[:,i] = np.gradient(strain[:,i,0],dt, edge_order=1)[:,None]
# Compute the fixed normalization parameters used for the strain, strain-rate,
# and stress quantities.
prm_strain = get_params(strain)
prm_strain_rate = get_params(strain_rate)
prm_stress = get_params(stress)


# ---------------------------------------------------------------------------
# Construct the hard-thermodynamic operator network
# ---------------------------------------------------------------------------

# Augment the observable strain with one internal coordinate representing the
# hidden hardening state.
dim_hidden = 0
dim_total = dim + dim_hidden

norm_params = [prm_strain, prm_stress, prm_strain_rate]

# The evolution network returns a flattened square transport operator acting on
# the complete state composed of strain and the internal hardening coordinate.
#
# Format:
# [input dimension, output dimension, hidden-layer widths, activation].
evolution_architecture = [
    dim_total,
    dim_total**2,
    [64, 64, 64, 64],
    "relu",
]

# The input-convex network represents the scalar free-energy
# potential as a function of the observable strain.
energy_net = ICNN(
    dim,
    [64, 64],
    1,
    activation="softplus",
    dtype=dtype,
)

# Combine the energy potential and constrained evolution operator into the
# hard-thermodynamic constitutive model.
model = HardThermodynamicsOpNet(
    evolution_architecture,
    energy_net,
    norm_params,
    dim,
    dim_hidden,
    dtype=dtype,
)
model.to(dtype)


# ---------------------------------------------------------------------------
# Configure time integration
# ---------------------------------------------------------------------------

time_grid = torch.arange(0, 1.0, dt)

model.step_size = dt

# Treat the prescribed strain rate as piecewise constant over every integration
# interval using a zero-order hold. BatchZOHRate is registered as part of the model.
model.rate_interp = BatchZOHRate(
    time_grid,
    strain_rate,
    dtype=dtype,
)


# ---------------------------------------------------------------------------
# Prepare tensors and initial-state parameters
# ---------------------------------------------------------------------------

stress_train = torch.from_numpy(stress)
strain_train = torch.from_numpy(strain)

# Use every available loading protocol for training.
id_protocol_train_val = np.arange(strain_train.shape[1])

# Register one normalized initial-strain condition.
model.norm_init_cond = torch.nn.Parameter(
    torch.zeros((1, dim), dtype=dtype),
    requires_grad=True,
)


# ---------------------------------------------------------------------------
# Optimization and early stopping
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
n_epochs = int(os.environ.get("HARD_THERMO_EPOCHS", "5000"))
if n_epochs < 2:
    raise ValueError("HARD_THERMO_EPOCHS must be at least 2")

# Retain the scalar loss history for later inspection or plotting.
training_loss_hist = []

early_stopping = EarlyStopping(
    patience=2_000,
    delta=1.0e-6,
    verbose=False,
    path=results_path,
)

# Normalize the target stresses once because the normalization parameters remain
# fixed throughout training.
norm_stress_target = model.Normalize(stress_train, prm_stress)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Initialize the hidden hardening coordinate to zero for every loading protocol.
# Using `zeros_like(strain_train[0])` assumes that the observable and hidden
# dimensions are compatible, as they are for this one-dimensional benchmark.
init_svars = torch.zeros(
    (strain_train.shape[1], dim_hidden),
    dtype=dtype,
)

# Because Python excludes the upper range limit, this performs at most 9,999
# optimization steps.
for epoch in range(1, n_epochs):
    optimizer.zero_grad(set_to_none=True)

    # Reuse constrained parametrizations throughout one trajectory integration.
    with P.cached():
        # Recompute the reference energy gradients using the initial strain
        # snapshot and the current energy-network parameters.
        model.ref_grads = model.reference_grads(strain_train[0])

        # Convert the trainable normalized initial strain to physical units.
        # This value is currently inactive because it is not assigned below.
        init_cond = model.DeNormalize(
            model.norm_init_cond,
            model.prm_strain,
        )

        # Initialize the elastic-strain component to zero for every protocol.
        e_strain0 = torch.zeros_like(
            strain_train[0, id_protocol_train_val]
        )

        # Optional assignments from earlier experiments. While these lines remain
        # commented, every protocol starts from zero elastic strain.
        # e_strain0[:1] = init_cond
        # e_strain0[1:2] = init_cond

        # Form the complete initial state by concatenating the elastic-strain and
        # hidden hardening components along the state dimension.
        initial_conditions = e_strain0
        initial_pred_stress = model.compute_stress(
            initial_conditions
        )

        # Integrate all loading protocols over the prescribed time grid.
        pred_state, pred_stress = integrate_training(
            model,
            initial_conditions,
            time_grid,
            id_protocol_train_val,
        )

    # Evaluate the objective in normalized stress coordinates so that stress
    # components with different magnitudes contribute on comparable scales.
    norm_pred_stress = model.Normalize(pred_stress, prm_stress)
    norm_initial_pred_stress = model.Normalize(
        initial_pred_stress,
        prm_stress,
    )

    # Penalize the stress mismatch at the initial snapshot.
    loss_init = torch.mean(
        (norm_initial_pred_stress - norm_stress_target[0]) ** 2
    )

    # Average the full stress-trajectory error over the loading protocols.
    loss_traj = torch.mean((norm_pred_stress[1:] - norm_stress_target[1:])** 2)
    # Combine the initial-state and trajectory contributions.
    loss = loss_init + loss_traj
    training_loss_value = loss.item()
    training_loss_hist.append(training_loss_value)

    # Update the best checkpoint and interrupt training after the configured
    # number of epochs without sufficient improvement.
    if not np.isnan(training_loss_value):
        early_stopping(training_loss_value, model)

        if early_stopping.early_stop:
            print("Training interrupted because of early stopping.")
            break

    # Differentiate through the complete trajectory and update all active trainable parameters.
    loss.backward()
    optimizer.step()

    if epoch % verbose_frequency == 0:
        print(
            f"epoch: {epoch}/{n_epochs}"
            f" | loss: {training_loss_value:.4e}"
            f" | initial cond: {loss_init.item():.4e}"
            f" | stress hist: {loss_traj.item():.4e}"
        )
