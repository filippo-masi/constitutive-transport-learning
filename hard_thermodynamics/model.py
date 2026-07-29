"""

State convention
----------------
For a batch of material points, the state has shape

    (..., dim + dim_hidden)

and is ordered as

    [elastic_strain, internal_variables].

When ``dim_hidden == 0``, the state contains only elastic strain.

External requirements
---------------------
The following attributes must be configured:

    model.rate_interp  # callable: rate_interp(t, protocol_indices)
    model.step_size    # fixed integration step

Author: Filippo Masi
"""

import torch

activations = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "elu": torch.nn.ELU(),
    "tanh": torch.nn.Tanh(),
    "gelu": torch.nn.GELU(),
    "silu": torch.nn.SiLU(),
    "softplus": torch.nn.Softplus(beta=2, threshold=20),
    "leaky_relu": torch.nn.LeakyReLU(),
}


class HardThermodynamicsOpNet(torch.nn.Module):
    """
    The model learns two constitutive ingredients:

    1. A free-energy potential ``psi`` represented by ``energy_net``.
       Stress and internal thermodynamic forces are obtained by automatic
       differentiation of this potential.
    2. A state-dependent transport operator represented by a feed-forward
       evolution network. Its symmetric part is positive semidefinite, while
       its skew-symmetric part describes nondissipative coupling.

    The evolution equations have the generic form

        z_dot = L(Y) @ Y,

    where ``Y`` collects the thermodynamic forces and ``z_dot`` contains the
    plastic-strain rate followed, when present, by internal-variable rates.
    """

    def __init__(
        self,
        evolution_architecture,
        energy_net,
        norm_params,
        dim=2,
        dim_hidden=0,
        dtype=torch.float64,
    ):
        """
        Initialize the thermodynamic operator network.

        Parameters
        ----------
        evolution_architecture : sequence
            Evolution-network specification

                [input_dimension, output_dimension,
                 hidden_dimensions, activation_name].

            For a state dimension ``D = dim + dim_hidden``, the output
            dimension must equal ``D**2``: ``D(D+1)/2`` coefficients build the
            lower-triangular factor and ``D(D-1)/2`` build the skew part.
        energy_net : torch.nn.Module
            Energy network. With no hidden variables it must accept
            ``energy_net(normalized_elastic_strain)``. With hidden variables it
            must accept
            ``energy_net(normalized_elastic_strain,
                         normalized_internal_variables)``.
        norm_params : tuple
            ``(prm_strain, prm_stress, prm_strain_rate)``. Each normalization
            tensor stores scale in row 0 and offset in row 1.
        dim : int, default=2
            Number of elastic-strain components.
        dim_hidden : int, default=0
            Number of internal state variables.
        dtype : torch.dtype, default=torch.float64
            Floating-point type used by the evolution network and buffers.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Dimensions and state layout
        # ------------------------------------------------------------------
        self.dtype = dtype
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.dim_total = dim + dim_hidden

        if dim < 1:
            raise ValueError("dim must be positive")
        if dim_hidden < 0:
            raise ValueError("dim_hidden cannot be negative")

        # Number of independent entries used to parameterize a D x D matrix.
        # Their sum equals D**2, which fixes the evolution-network output size.
        self.n_sym = (self.dim_total * (self.dim_total + 1)) // 2
        self.n_skew = (self.dim_total * (self.dim_total - 1)) // 2

        # Indices are stored as buffers so they automatically follow the model
        # when it is moved between devices.
        tril_idx = torch.tril_indices(
            self.dim_total,
            self.dim_total,
            offset=0,
        )
        tri_idx = torch.triu_indices(
            self.dim_total,
            self.dim_total,
            offset=1,
        )
        self.register_buffer("tril_idx", tril_idx)
        self.register_buffer("tri_idx", tri_idx)

        # ------------------------------------------------------------------
        # Normalization parameters
        # ------------------------------------------------------------------
        prm_strain, prm_stress, prm_strain_rate = norm_params

        # The characteristic energy scale is estimated from stress x strain.
        # The energy offset is set to zero because only energy derivatives
        # affect the constitutive response.
        #
        # NOTE: for device-independent construction, the zero can instead be
        # created with ``prm_strain.new_zeros(())``.
        prm_energy = torch.stack(
            (
                torch.mean(prm_strain[0] * prm_stress[0]),
                prm_strain.new_zeros(()),
            )
        )

        # Normalization constants are fixed model data, not trainable
        # parameters, so they are registered as buffers.
        self.register_buffer(
            "prm_strain",
            prm_strain.detach().clone().to(dtype=self.dtype),
        )
        self.register_buffer(
            "prm_stress",
            prm_stress.detach().clone().to(dtype=self.dtype),
        )
        self.register_buffer(
            "prm_strain_rate",
            prm_strain_rate.detach().clone().to(dtype=self.dtype),
        )
        self.register_buffer(
            "prm_energy",
            prm_energy.detach().clone().to(dtype=self.dtype),
        )

        # When internal variables are present, scalar normalization parameters
        # are estimated from the mechanical strain and stress scales. This is a
        # modelling assumption; an application may instead supply dedicated
        # internal-variable and conjugate-force scales.
        if self.dim_hidden > 0:
            self.hidden = True

            prm_svars = torch.stack(
                (
                    torch.mean(prm_strain[0]),
                    torch.mean(prm_strain[1]),
                )
            )
            prm_force = torch.stack(
                (
                    torch.mean(prm_stress[0]),
                    torch.mean(prm_stress[1]),
                )
            )

            self.register_buffer(
                "prm_svars",
                prm_svars.detach().clone().to(dtype=self.dtype),
            )
            self.register_buffer(
                "prm_force",
                prm_force.detach().clone().to(dtype=self.dtype),
            )
        else:
            self.hidden = False

        # Diagonal change-of-units factors used to map the dimensionless
        # transport operator back to physical units.
        d_inv = self.build_dinv()
        self.register_buffer(
            "d_inv",
            d_inv.detach().clone().to(dtype=self.dtype),
        )

        # ------------------------------------------------------------------
        # Constitutive neural networks
        # ------------------------------------------------------------------
        input_dim, output_dim, _, _ = evolution_architecture
        if input_dim != self.dim_total:
            raise ValueError(
                "The evolution-network input dimension must equal "
                f"dim + dim_hidden ({self.dim_total})"
            )
        if output_dim != self.dim_total**2:
            raise ValueError(
                "The evolution-network output dimension must equal "
                f"(dim + dim_hidden)**2 ({self.dim_total**2})"
            )

        self.NeuralNetEvolution = self.constructor(
            evolution_architecture
        )
        self.NeuralNetEnergy = energy_net

    def constructor(self, params):
        """
        Build the feed-forward evolution network.

        Parameters
        ----------
        params : sequence
            ``[input_dim, output_dim, hidden_dims, activation_name]``.

        Returns
        -------
        torch.nn.Sequential
            Fully connected network mapping normalized thermodynamic forces to
            the coefficients of the transport operator.
        """
        input_dim, output_dim, hidden_dims, activation_name = params
        if activation_name not in activations:
            raise ValueError(
                f"Unknown activation {activation_name!r}; choose from "
                f"{sorted(activations)}"
            )
        if any(width < 1 for width in hidden_dims):
            raise ValueError("Hidden-layer widths must be positive")

        current_dim = input_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(
                        current_dim,
                        hidden_dim,
                        dtype=self.dtype,
                    ),
                    activations[activation_name],
                ]
            )
            current_dim = hidden_dim

        layers.append(
            torch.nn.Linear(
                current_dim,
                output_dim,
                dtype=self.dtype,
            )
        )
        return torch.nn.Sequential(*layers)

    @staticmethod
    def Normalize(inputs, parameters):
        """
        Normalize physical values using ``(value - offset) / scale``.

        ``parameters[0]`` is the scale and ``parameters[1]`` is the offset.
        """
        return (inputs - parameters[1]) / parameters[0]

    @staticmethod
    def DeNormalize(outputs, parameters):
        """
        Convert normalized values to physical units.

        ``parameters[0]`` is the scale and ``parameters[1]`` is the offset.
        """
        return outputs * parameters[0] + parameters[1]

    def build_dinv(self):
        """
        Construct diagonal transport-operator scaling factors.

        The characteristic mechanical power is estimated as

            strain-rate scale x stress scale.

        The current implementation uses one averaged factor for every state
        component. Applications with differently scaled internal variables
        should supply componentwise rate/force normalization instead.
        """
        mechanical_power = torch.mean(
            self.prm_strain_rate[0] * self.prm_stress[0]
        )
        d_i = (
            torch.mean(self.prm_strain_rate[0])
            / torch.sqrt(mechanical_power)[None]
        )
        return torch.cat(
            [d_i for _ in range(self.dim_total)],
            dim=0,
        )

    def evolution_equations(self, state):
        """
        Evaluate the irreversible evolution law ``z_dot = L(Y) Y``.

        Parameters
        ----------
        state : torch.Tensor
            Tensor with shape ``(batch, dim_total)`` and ordering
            ``[elastic_strain, internal_variables]``.

        Returns
        -------
        torch.Tensor
            Physical evolution rates with shape ``(batch, dim_total)``.
            The first ``dim`` entries are plastic-strain rates; remaining
            entries are internal-variable rates.
        """
        if self.hidden:
            # Y = [stress, conjugate internal force].
            stress, force = self.compute_stress(
                state,
                return_all=True,
            )
            normalized_stress = self.Normalize(
                stress,
                self.prm_stress,
            )
            normalized_svars = self.Normalize(
                state[..., self.dim :],
                self.prm_svars,
            )
            network_inputs = torch.cat(
                (normalized_stress, normalized_svars),
                axis=-1,
            )
            thermodynamic_forces = torch.cat(
                (stress, force),
                axis=-1,
            )
        else:
            # Without internal variables, stress is the only driving force.
            stress = self.compute_stress(state)
            network_inputs = self.Normalize(
                stress,
                self.prm_stress,
            )
            thermodynamic_forces = stress

        # The network predicts a dimensionless transport operator.
        normalized_operator = self.transport_operator(network_inputs)

        # Equivalent to D_inv @ L_normalized @ D_inv, evaluated using
        # broadcasting rather than explicitly constructing diagonal matrices.
        d = self.d_inv
        physical_operator = (
            d[:, None]
            * normalized_operator
            * d[None, :]
        )

        return torch.einsum(
            "...ij,...j->...i",
            physical_operator,
            thermodynamic_forces,
        )

    def transport_operator(self, inputs):
        """
        Construct the thermodynamically structured transport operator.

        The evolution network returns ``D**2`` coefficients, split into:

        - ``D(D+1)/2`` coefficients for a lower-triangular matrix ``B``;
        - ``D(D-1)/2`` coefficients for a skew-symmetric matrix.

        The final operator is

            L = B B^T + L_skew.

        ``B B^T`` is positive semidefinite and therefore produces
        non-negative dissipation. ``L_skew`` does not contribute to the scalar
        dissipation because ``Y^T L_skew Y = 0``.

        Parameters
        ----------
        inputs : torch.Tensor
            Normalized thermodynamic forces with shape
            ``(batch, dim_total)``.

        Returns
        -------
        torch.Tensor
            Operator with shape ``(batch, dim_total, dim_total)``.
        """
        # Preserve the autograd graph: these coefficients depend on the
        # evolution-network parameters and the current thermodynamic forces.
        flat_operator = self.NeuralNetEvolution(inputs)

        lower_entries = flat_operator[..., : self.n_sym]
        skew_entries = flat_operator[..., self.n_sym :]

        # Assemble the lower-triangular factor B.
        lower_factor = inputs.new_zeros(
            (*inputs.shape[:-1], self.dim_total, self.dim_total),
        )
        lower_factor[
            ...,
            self.tril_idx[0],
            self.tril_idx[1],
        ] = lower_entries

        symmetric_operator = (
            lower_factor
            @ lower_factor.transpose(-1, -2)
        )

        # Assemble the skew-symmetric contribution.
        skew_operator = inputs.new_zeros(
            (*inputs.shape[:-1], self.dim_total, self.dim_total),
        )
        skew_operator[
            ...,
            self.tri_idx[0],
            self.tri_idx[1],
        ] = skew_entries
        skew_operator[
            ...,
            self.tri_idx[1],
            self.tri_idx[0],
        ] = -skew_entries

        return symmetric_operator + skew_operator

    def _raw_energy_value(self, state):
        """Evaluate the uncorrected neural energy in physical units."""
        elastic_strain = state[..., : self.dim]
        normalized_strain = self.Normalize(
            elastic_strain,
            self.prm_strain,
        )

        if self.hidden:
            internal_variables = state[..., self.dim :]
            normalized_internal_variables = self.Normalize(
                internal_variables,
                self.prm_svars,
            )
            normalized_energy = self.NeuralNetEnergy(
                normalized_strain,
                normalized_internal_variables,
            )
        else:
            normalized_energy = self.NeuralNetEnergy(
                normalized_strain
            )

        return self.DeNormalize(
            normalized_energy,
            self.prm_energy,
        )

    def reference_energy_terms(self, like):
        """Return raw energy and raw gradient at the zero state."""
        with torch.enable_grad():
            state_zero = like.new_zeros(
                (1, self.dim_total),
                requires_grad=True,
            )
            energy_zero = self._raw_energy_value(state_zero)
            gradient_zero = torch.autograd.grad(
                energy_zero,
                state_zero,
                grad_outputs=torch.ones_like(energy_zero),
                create_graph=True,
            )[0]
        return energy_zero, gradient_zero

    def energy_value(self, state):
        """
        Evaluate the learned free-energy potential in physical units.

        Parameters
        ----------
        state : torch.Tensor
            State with final dimension ``dim_total``.

        Returns
        -------
        torch.Tensor
            Free energy for every state sample. The energy network should
            return one scalar value per sample.
        """
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)

            raw_energy = self._raw_energy_value(state)
            energy_zero, gradient_zero = self.reference_energy_terms(
                state
            )
            affine_correction = energy_zero + (
                state * gradient_zero
            ).sum(dim=-1, keepdim=True)
            return raw_energy - affine_correction

    def reference_grads(self, like):
        """
        Evaluate energy gradients at the zero physical state.

        Subtracting these reference gradients makes stress and internal force
        vanish at the chosen reference state without constraining the energy
        network architecture.

        ``like`` is used only to inherit device and dtype.
        """
        return self.reference_energy_terms(like)[1]

    def compute_stress(self, state, return_all=False):
        """
        Compute thermodynamic forces as derivatives of the free energy.

        For ``state = [elastic_strain, internal_variables]``,

            stress = d(psi)/d(elastic_strain),
            force  = -d(psi)/d(internal_variables).

        The affine energy correction makes both quantities zero at the
        reference state.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with final dimension ``dim_total``.
        return_all : bool, default=False
            If true and hidden variables exist, return ``(stress, force)``.
            Otherwise return stress only.
        """
        # Stress evaluation requires autograd even during model inference.
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)

            energy = self.energy_value(state)
            gradients = torch.autograd.grad(
                energy,
                state,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]

            stress = gradients[..., : self.dim]

            if not self.hidden:
                return stress

            force = -gradients[..., self.dim :]

            return (stress, force) if return_all else stress

    def forward(self, t, state):
        """
        Evaluate the right-hand side of the constitutive ODE.

        The imposed total-strain rate is split additively:

            total_strain_rate
                = elastic_strain_rate + plastic_strain_rate.

        The transport operator predicts the plastic-strain rate and, when
        present, the internal-variable rates.

        Parameters
        ----------
        t : scalar torch.Tensor
            Current integration time.
        state : torch.Tensor
            Current state with shape ``(batch, dim_total)``.

        Returns
        -------
        torch.Tensor
            State derivative with the same shape and ordering as ``state``.
        """
        total_strain_rate = self.rate_interp(t, self.idx)

        # z = [plastic_strain_rate, internal_variable_rates].
        evolution_rates = self.evolution_equations(state)
        plastic_strain_rate = evolution_rates[..., : self.dim]

        elastic_strain_rate = (
            total_strain_rate - plastic_strain_rate
        )

        if self.hidden:
            internal_variable_rates = evolution_rates[..., self.dim :]
            return torch.cat(
                (elastic_strain_rate, internal_variable_rates),
                axis=-1,
            )

        return elastic_strain_rate
