"""
Input-convex neural-network architectures used by the energy model.

The module provides:

``ICNN``
    A network that is convex with respect to its complete input.

``PICNN``
    A partially input-convex network. It is convex with respect to ``in_x``
    while allowing an unrestricted dependence on the context input ``in_y``.

Convexity is enforced structurally through nonnegative weights and convex,
nondecreasing activation functions.
"""

import torch
from torch import nn
from torch.nn.utils import parametrize


# Shared activation functions. The activations used in a convex stream must be
# convex and nondecreasing to preserve input convexity.
activations = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(beta=2, threshold=20),
    "leaky_relu": nn.LeakyReLU(),
}


class NonNeg(nn.Module):
    """
    Smoothly map unconstrained values to nonnegative values.

    For an unconstrained value ``x``, the parametrization is

        sqrt(x**2 + eps**2) - eps >= 0.

    This is a smooth approximation of ``abs(x)`` and is used to enforce nonnegative linear-layer weights without
    manually projecting them after each optimizer step.
    """

    def __init__(self, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def forward(self, values):
        return (
            torch.sqrt(values.square() + self.epsilon**2)
            - self.epsilon
        )


def nonneg_linear(
    in_features,
    out_features,
    bias=True,
    dtype=torch.float64,
):
    """
    Construct a linear layer whose effective weights are nonnegative.

    PyTorch stores and optimizes an unconstrained original weight. Whenever
    ``layer.weight`` is used, ``NonNeg`` transforms that original parameter
    into a nonnegative effective weight. The bias remains unconstrained.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default=True
        Whether the layer includes an unconstrained bias.
    dtype : torch.dtype, default=torch.float64
        Parameter data type.
    """
    if in_features < 1 or out_features < 1:
        raise ValueError("Linear dimensions must be positive")

    layer = nn.Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=dtype,
    )
    parametrize.register_parametrization(
        layer,
        "weight",
        NonNeg(),
    )
    return layer


class ICNN(nn.Module):
    """
    Input-Convex Neural Network.

    The complete output is convex with respect to ``x``. Convexity follows
    from three structural conditions:

    1. Hidden-to-hidden weights are nonnegative.
    2. The final weights multiplying hidden convex features are nonnegative.
    3. The hidden activation is convex and nondecreasing.

    Direct input-to-hidden and input-to-output skip connections are affine in
    ``x`` and can therefore remain unconstrained.

    Parameters
    ----------
    input_dim : int
        Input-feature dimension.
    hidden_dims : sequence of int
        Width of every hidden layer. At least one hidden layer is required.
    output_dim : int, default=1
        Number of outputs. Each output is convex with respect to ``x``.
    activation : str, default="relu"
        Name of a convex, nondecreasing activation.
    activations : mapping or None
        Activation mapping. If ``None``, a local default mapping is created.
    dtype : torch.dtype, default=torch.float64
        Parameter data type.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int = 1,
        activation: str = "relu",
        activations=activations,
        dtype=torch.float64,
    ):
        super().__init__()

        self.dtype = dtype
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if not self.hidden_dims or any(
            width < 1 for width in self.hidden_dims
        ):
            raise ValueError(
                "hidden_dims must contain positive layer widths"
            )

        if activations is None:
            activations = {
                "relu": nn.ReLU(),
                "softplus": nn.Softplus(
                    beta=1,
                    threshold=10,
                ),
                "leaky_relu": nn.LeakyReLU(),
            }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from "
                f"{sorted(activations)}"
            )

        # Nonnegative hidden-to-hidden maps preserve convexity when combining
        # the convex features produced by the preceding layer.
        self.Wz = nn.ModuleList(
            [
                nonneg_linear(
                    hidden_dim,
                    next_hidden_dim,
                    bias=False,
                    dtype=dtype,
                )
                for hidden_dim, next_hidden_dim in zip(
                    self.hidden_dims[:-1],
                    self.hidden_dims[1:],
                )
            ]
        )

        # Each output is a nonnegative linear combination of the final hidden
        # convex features.
        self.Wy = nonneg_linear(
            self.hidden_dims[-1],
            output_dim,
            bias=False,
            dtype=dtype,
        )

        # Affine skip connections inject the original input at every layer.
        # Their weights do not require sign constraints.
        self.Wx = nn.ModuleList(
            [
                nn.Linear(
                    input_dim,
                    hidden_dim,
                    bias=True,
                    dtype=dtype,
                )
                for hidden_dim in self.hidden_dims
            ]
        )

        # Final affine skip connection from the original input.
        self.Wx_out = nn.Linear(
            input_dim,
            output_dim,
            bias=True,
            dtype=dtype,
        )

        self.act = activations[activation]

    def forward(self, x):
        """
        Evaluate the input-convex network.

        Parameters
        ----------
        x : torch.Tensor, shape (..., input_dim)
            Input with respect to which convexity is enforced.

        Returns
        -------
        torch.Tensor, shape (..., output_dim)
            Convex network output.
        """
        # The first hidden representation is an activated affine function of
        # x and is therefore convex when the activation is convex and
        # nondecreasing.
        z = self.act(self.Wx[0](x))

        # Each subsequent layer combines:
        #   - an affine function of x, and
        #   - a nonnegative linear combination of existing convex features.
        for layer_index, hidden_map in enumerate(self.Wz):
            z = self.act(
                self.Wx[layer_index + 1](x)
                + hidden_map(z)
            )

        # The nonnegative hidden contribution is convex; adding the affine
        # input skip connection preserves convexity.
        return self.Wy(z) + self.Wx_out(x)


class PICNN(nn.Module):
    """
    Partially Input-Convex Neural Network.

    The network is convex with respect to ``in_x`` for every fixed value of
    ``in_y``. Dependence on ``in_y`` is unrestricted, allowing ``in_y`` to act
    as a context variable.

    Two feature streams are propagated:

    ``w``
        The convex stream. Its hidden-to-hidden and final weights are
        constrained to be nonnegative.

    ``v``
        The context stream. It depends only on ``in_y`` and is unconstrained
        because it does not affect convexity with respect to ``in_x``.

    Context-dependent gates couple both streams. Gates multiplying the convex
    features ``w`` are made nonnegative using a rectifier.

    Parameters
    ----------
    input_x_dim : int
        Dimension of the input with respect to which convexity is enforced.
    input_y_dim : int
        Dimension of the unrestricted context input.
    feature_dim : int
        Width of the convex ``w`` stream.
    feature_y_dim : int
        Width of the context ``v`` stream.
    out_dim : int
        Output dimension. For an energy potential this is normally one.
    num_layers : int
        Number of feature transformations. It must be at least one.
    act : str, default="softplus"
        Activation used in the convex stream. It should be convex and
        nondecreasing.
    act_v : str, default="softplus"
        Activation used in the unrestricted context stream.
    activations : mapping
        Mapping from activation names to ``nn.Module`` instances.
    dtype : torch.dtype, default=torch.float64
        Parameter data type.

    Notes
    -----
    For fixed ``in_y``, all direct ``in_x`` terms are affine in ``in_x``.
    Affine terms do not need sign-constrained weights. Only weights and gates
    multiplying previously computed convex features must be nonnegative.
    """

    def __init__(
        self,
        input_x_dim: int,
        input_y_dim: int,
        feature_dim: int,
        feature_y_dim: int,
        out_dim: int,
        num_layers: int,
        act: str = "softplus",
        act_v: str = "softplus",
        activations=activations,
        dtype=torch.float64,
    ) -> None:
        super().__init__()

        self.input_x_dim = input_x_dim
        self.input_y_dim = input_y_dim
        self.feature_dim = feature_dim
        self.feature_y_dim = feature_y_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        dimensions = {
            "input_x_dim": input_x_dim,
            "input_y_dim": input_y_dim,
            "feature_dim": feature_dim,
            "feature_y_dim": feature_y_dim,
            "out_dim": out_dim,
            "num_layers": num_layers,
        }
        invalid = [
            name for name, value in dimensions.items() if value < 1
        ]
        if invalid:
            raise ValueError(
                "PICNN dimensions must be positive: "
                + ", ".join(invalid)
            )
        if act not in activations or act_v not in activations:
            raise ValueError(
                "Unknown activation; choose from "
                f"{sorted(activations)}"
            )

        # ------------------------------------------------------------------
        # Context stream: v_k = activation(Lv_k(v_{k-1}))
        # ------------------------------------------------------------------
        # This stream depends only on in_y, so its weights are unrestricted.
        context_layers = [
            nn.Linear(
                input_y_dim,
                feature_y_dim,
                bias=True,
                dtype=dtype,
            )
        ]
        for _ in range(num_layers - 1):
            context_layers.append(
                nn.Linear(
                    feature_y_dim,
                    feature_y_dim,
                    bias=True,
                    dtype=dtype,
                )
            )
        self.Lv = nn.ModuleList(context_layers)

        # ------------------------------------------------------------------
        # Additive context contribution to the convex stream
        # ------------------------------------------------------------------
        # These terms depend only on v and are therefore constant with respect
        # to in_x for fixed context. Their weights need not be constrained.
        context_to_convex = [
            nn.Linear(
                input_y_dim,
                feature_dim,
                bias=False,
                dtype=dtype,
            )
        ]
        for _ in range(num_layers - 1):
            context_to_convex.append(
                nn.Linear(
                    feature_y_dim,
                    feature_dim,
                    bias=False,
                    dtype=dtype,
                )
            )
        context_to_convex.append(
            nn.Linear(
                feature_y_dim,
                out_dim,
                bias=False,
                dtype=dtype,
            )
        )
        self.Lvw = nn.ModuleList(context_to_convex)

        # ------------------------------------------------------------------
        # Convex stream
        # ------------------------------------------------------------------
        # The first transformation acts on an affine function of in_x, so its
        # weights may be unrestricted. Subsequent layers combine previously
        # computed convex features and therefore require nonnegative weights.
        convex_layers = [
            nn.Linear(
                input_x_dim,
                feature_dim,
                bias=True,
                dtype=dtype,
            )
        ]

        for _ in range(num_layers - 1):
            convex_layers.append(
                nonneg_linear(
                    feature_dim,
                    feature_dim,
                    bias=True,
                    dtype=dtype,
                )
            )

        # A nonnegative final map preserves convexity of each output.
        convex_layers.append(
            nonneg_linear(
                feature_dim,
                out_dim,
                bias=True,
                dtype=dtype,
            )
        )
        self.Lw = nn.ModuleList(convex_layers)

        # ------------------------------------------------------------------
        # Context-dependent multipliers of the convex features
        # ------------------------------------------------------------------
        # Lwv[0] modulates the original input in_x. The remaining outputs pass
        # through ``self.gate`` before multiplying w, making those
        # multipliers nonnegative.
        convex_gates = [
            nn.Linear(
                input_y_dim,
                input_x_dim,
                bias=True,
                dtype=dtype,
            )
        ]
        for _ in range(num_layers):
            convex_gates.append(
                nn.Linear(
                    feature_y_dim,
                    feature_dim,
                    bias=True,
                    dtype=dtype,
                )
            )
        self.Lwv = nn.ModuleList(convex_gates)

        # ------------------------------------------------------------------
        # Context-dependent multipliers of the original convex input
        # ------------------------------------------------------------------
        # For fixed context, in_x * Lxv(v) remains affine in in_x. Consequently
        # these multipliers may have either sign.
        input_gates = []
        for _ in range(num_layers):
            input_gates.append(
                nn.Linear(
                    feature_y_dim,
                    input_x_dim,
                    bias=True,
                    dtype=dtype,
                )
            )
        self.Lxv = nn.ModuleList(input_gates)

        # Map each context-modulated copy of in_x into the convex stream.
        # These are affine skip connections in in_x, so their weights are
        # unrestricted.
        input_skip_layers = []
        for _ in range(num_layers - 1):
            input_skip_layers.append(
                nn.Linear(
                    input_x_dim,
                    feature_dim,
                    bias=False,
                    dtype=dtype,
                )
            )
        input_skip_layers.append(
            nn.Linear(
                input_x_dim,
                out_dim,
                bias=False,
                dtype=dtype,
            )
        )
        self.Lx = nn.ModuleList(input_skip_layers)

        self.act_w = activations[act]
        self.act_v = activations[act_v]

        # ReLU guarantees that the context-dependent coefficient multiplying
        # the convex stream is nonnegative.
        self.gate = nn.ReLU()

    def forward(self, in_x, in_y):
        """
        Evaluate the partially input-convex network.

        Parameters
        ----------
        in_x : torch.Tensor, shape (..., input_x_dim)
            Input with respect to which the output is convex.
        in_y : torch.Tensor, shape (..., input_y_dim)
            Unrestricted context input.

        Returns
        -------
        torch.Tensor, shape (..., out_dim)
            Network output, convex in ``in_x`` for each fixed ``in_y``.
        """
        # Local bindings keep the equations below compact.
        Lw = self.Lw
        Lv = self.Lv
        Lvw = self.Lvw
        Lwv = self.Lwv
        Lxv = self.Lxv
        Lx = self.Lx

        # First convex layer.
        #
        # For fixed in_y, ``in_x * Lwv[0](in_y)`` is affine in in_x. Applying
        # an unrestricted linear map followed by a convex, nondecreasing
        # activation creates the first convex feature representation.
        v = in_y
        modulated_x = in_x * Lwv[0](v)
        w = self.act_w(
            Lw[0](modulated_x) + Lvw[0](v)
        )

        # Intermediate layers.
        for layer_index in range(self.num_layers - 1):
            # Update the unrestricted context features.
            v = self.act_v(Lv[layer_index](v))

            # A nonnegative context gate multiplies the existing convex
            # features. The following Lw map also has nonnegative weights.
            gated_w = (
                w
                * self.gate(Lwv[layer_index + 1](v))
            )

            # This skip term remains affine in in_x for fixed context.
            gated_x = in_x * Lxv[layer_index](v)

            w = self.act_w(
                Lw[layer_index + 1](gated_w)
                + Lx[layer_index](gated_x)
                + Lvw[layer_index + 1](v)
            )

        # Final context transformation and output layer. No final activation
        # is required: a nonnegative linear combination of convex features
        # plus affine/context-only terms remains convex.
        final_v = self.act_v(Lv[-1](v))
        final_gated_w = w * self.gate(Lwv[-1](final_v))
        final_gated_x = in_x * Lxv[-1](final_v)

        return (
            Lw[-1](final_gated_w)
            + Lx[-1](final_gated_x)
            + Lvw[-1](final_v)
        )
