"""
Author: Filippo Masi
"""

from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after the last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Monitors the validation loss and performs early stopping if needed.

        Args:
            val_loss (float): Current validation loss.
            model: PyTorch model.

        Returns:
            None
        """
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    "EarlyStopping counter: "
                    f"{self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when the monitored loss decreases."""
        if self.verbose:
            self.trace_func(
                "Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                "Saving model ..."
            )
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def slice_data(x, ntrainval, ntest):
    """
    Slices the data into training and testing sets.

    Args:
        x: Input data.
        ntrainval (int): Number of samples for training and validation.
        ntest (int): Number of samples for testing.

    Returns:
        Tuple: Sliced training/validation data, sliced testing data.
    """
    return x[:, ntrainval], x[:, ntest]


def get_params(x, norm=False, vectorial_norm=False):
    """
    Compute normalization parameters:
        - normalize ([-1,1]) component by component (vectorial_norm = True)
        - normalize ([-1,1]) (vectorial_norm = False, norm = True)
        - standardize (vectorial_norm = False, norm = False)

    Args:
        x: Input data.
        norm (bool): Normalize data to [-1,1].
        vectorial_norm (bool): Normalize data component by component (along axis = 1).

    Returns:
        torch.Tensor: Normalization parameters.
    """
    values = np.asarray(x)
    if values.ndim < 1:
        raise ValueError("x must contain at least one dimension")

    if not vectorial_norm:
        if norm:
            # Normalize to [-1, 1]
            scale = 0.5 * (
                np.amax(values) - np.amin(values)
            )
            offset = 0.5 * (
                np.amax(values) + np.amin(values)
            )
        else:
            # Standardize (mean = 0, std = 1)
            reduction_axes = tuple(range(values.ndim - 1))
            scale = np.std(values, axis=reduction_axes)
            offset = np.mean(values, axis=reduction_axes)
    else:
        # Normalize each final-axis component to [-1, 1].
        reduction_axes = tuple(range(values.ndim - 1))
        maximum = np.amax(values, axis=reduction_axes)
        minimum = np.amin(values, axis=reduction_axes)
        scale = 0.5 * (maximum - minimum)
        offset = 0.5 * (maximum + minimum)

    scale = np.asarray(scale)
    scale = np.where(scale == 0, 1, scale)
    parameters = np.stack((scale, np.asarray(offset)))
    dtype = (
        torch.float64
        if values.dtype == np.float64
        else torch.float32
    )
    return torch.as_tensor(parameters, dtype=dtype)


class BatchZOHRate(torch.nn.Module):
    """
    Zero-order-hold interpolation of prescribed strain-rate histories.

    Parameters
    ----------
    t_knots : array-like, shape (n_time,)
        Strictly increasing time knots.
    strain_rate : array-like
        Prescribed strain rates with shape
        ``(n_rate, n_protocols, n_dim)``.

        Two conventions are accepted:

        - ``n_rate == n_time - 1``:
          ``strain_rate[i]`` applies on ``[t_i, t_{i+1})``.
        - ``n_rate == n_time``:
          ``strain_rate[i]`` applies from ``t_i`` until the next knot; the
          final value is returned at and beyond the last knot.
    dtype : torch.dtype, default=torch.float64
        Floating-point type used to store the time and rate histories.

    Notes
    -----
    This class does not differentiate a strain history. The strain rate is
    supplied directly and is held constant between consecutive time knots.
    """

    def __init__(
        self,
        t_knots,
        strain_rate,
        dtype=torch.float64,
    ):
        super().__init__()

        rates = torch.as_tensor(
            strain_rate,
            dtype=dtype,
        ).detach().clone()

        t = torch.as_tensor(
            t_knots,
            dtype=dtype,
            device=rates.device,
        ).detach().clone()

        if t.ndim != 1:
            raise ValueError(
                "t_knots must have shape (n_time,)"
            )

        if rates.ndim != 3:
            raise ValueError(
                "strain_rate must have shape "
                "(n_rate, n_protocols, n_dim)"
            )

        if t.numel() < 2:
            raise ValueError(
                "At least two time knots are required"
            )

        if rates.shape[0] not in {
            t.numel() - 1,
            t.numel(),
        }:
            raise ValueError(
                "strain_rate must contain either n_time - 1 "
                "interval values or n_time nodal values"
            )

        if torch.any(t[1:] <= t[:-1]).item():
            raise ValueError(
                "t_knots must be strictly increasing"
            )

        # Buffers follow the parent model when model.to(device) is called.
        self.register_buffer("t", t)
        self.register_buffer("rates", rates)

    def forward(self, t, idx=None):
        """
        Return the prescribed strain rate at scalar time ``t``.

        Parameters
        ----------
        t : scalar
            Evaluation time.
        idx : sequence or torch.Tensor, optional
            Protocol indices. If omitted, rates for all protocols are
            returned.

        Returns
        -------
        torch.Tensor
            Shape ``(n_protocols, n_dim)`` when ``idx`` is omitted, otherwise
            ``(len(idx), n_dim)``.
        """
        evaluation_time = torch.as_tensor(
            t,
            dtype=self.t.dtype,
            device=self.t.device,
        )

        if evaluation_time.numel() != 1:
            raise ValueError("t must be scalar")

        evaluation_time = evaluation_time.reshape(()).clamp(
            self.t[0],
            self.t[-1],
        )

        # right=True gives a right-continuous hold: at t_i, use rate[i].
        interval = torch.searchsorted(
            self.t,
            evaluation_time.unsqueeze(0),
            right=True,
        )[0] - 1

        interval = interval.clamp(
            min=0,
            max=self.rates.shape[0] - 1,
        )
        rate = self.rates[interval]

        if idx is None:
            return rate

        protocol_indices = torch.as_tensor(
            idx,
            dtype=torch.long,
            device=rate.device,
        ).reshape(-1)

        return rate.index_select(0, protocol_indices)
