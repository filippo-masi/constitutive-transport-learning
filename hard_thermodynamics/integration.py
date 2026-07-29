"""
Differentiable fixed-step explicit-Euler integration for training and adaptive explicit-Euler inference.

For the adapative time-stepping, the local error is estimated by step doubling:

1. take one Euler step of size ``h``;
2. take two Euler steps of size ``h / 2``;
3. compare the two results;
4. accept or reject the trial step and adapt ``h``.

The two-half-step result is retained after an accepted step. Trial steps are
also clipped at every strain-rate knot, which is essential when
``BatchZOHRate`` supplies a discontinuous, zero-order-held loading rate.

"""

import math

import torch


def _loading_breakpoints(model, start_time, end_time):
    """
    Return ZOH strain-rate knots strictly inside the integration interval.

    ``BatchZOHRate`` stores its time knots in ``model.rate_interp.t``. If a
    different rate callable has no ``t`` attribute, no extra breakpoints are
    added.
    """
    rate_interpolator = getattr(model, "rate_interp", None)
    knots = getattr(rate_interpolator, "t", None)

    if knots is None:
        return []

    knot_values = knots.detach().reshape(-1).cpu().tolist()
    return sorted(
        {
            float(knot)
            for knot in knot_values
            if start_time < float(knot) < end_time
        }
    )


def integrate_training(
    model,
    initial_conditions,
    t_eval,
    idx=None,
    *,
    step_size=None,
    align_loading_knots=True,
    maximum_steps=1_000_000,
    check_finite=False,
):
    """
    Integrate model using differentiable fixed-step explicit Euler.

    Parameters
    ----------
    model : HardThermodynamicsOpNet
        Constitutive model whose ``forward(t, state)`` method evaluates the
        state derivative.
    initial_conditions : torch.Tensor
        Initial state with shape ``(batch, model.dim_total)``. The tensor is
        not detached, so gradients can also propagate to trainable initial
        conditions when required.
    t_eval : array-like or torch.Tensor
        Strictly increasing output times. Initial conditions correspond to
        ``t_eval[0]``.
    idx : sequence or torch.Tensor, optional
        Protocol indices forwarded to ``model.rate_interp``.
    step_size : float, optional
        Fixed internal Euler step. If omitted, ``model.step_size`` is used.
        A shorter final step is taken when necessary to land exactly on an
        output time or a loading-rate knot.
    align_loading_knots : bool, default=True
        Prevent an Euler step from crossing a zero-order-hold loading
        discontinuity.
    maximum_steps : int, default=1000000
        Safety limit on the number of internal Euler steps.
    check_finite : bool, default=False
        If true, check the state after every step and stop on NaN or infinity.
        This is useful for debugging but introduces a device synchronization
        at every step.

    Returns
    -------
    state_solution : torch.Tensor
        State at all requested output times, with shape
        ``(n_time, batch, model.dim_total)``.
    stress_solution : torch.Tensor
        Stress evaluated along the trajectory, with shape
        ``(n_time, batch, model.dim)``.

    Notes
    -----
    The Euler update is

        state_(n+1) = state_n + dt * model(t_n, state_n).

    No state or state-rate tensor is detached, and neither
    ``torch.no_grad()`` nor an in-place state update is used. Time-grid tensors
    are detached only for Python loop control. PyTorch therefore retains the
    full unrolled state graph required by backpropagation through time.

    The nominal internal step is fixed. It is shortened only at requested
    output times and ZOH loading knots so the solver never integrates across a
    discontinuity and always returns values exactly at ``t_eval``.
    """
    if not hasattr(model, "rate_interp"):
        raise AttributeError(
            "Assign model.rate_interp before integration"
        )

    if step_size is None:
        if not hasattr(model, "step_size"):
            raise AttributeError(
                "Provide step_size or assign model.step_size"
            )
        step_size = model.step_size

    if isinstance(step_size, torch.Tensor):
        if step_size.numel() != 1:
            raise ValueError("step_size must be scalar")
        fixed_step = float(step_size.detach().item())
    else:
        fixed_step = float(step_size)

    if fixed_step <= 0.0:
        raise ValueError("step_size must be positive")

    if maximum_steps < 1:
        raise ValueError("maximum_steps must be positive")

    # Match the model's registered dtype and device without detaching a tensor
    # supplied by the caller. ``Tensor.to`` remains differentiable.
    reference_tensor = model.prm_strain
    if isinstance(initial_conditions, torch.Tensor):
        state = initial_conditions.to(
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )
    else:
        state = torch.as_tensor(
            initial_conditions,
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )

    if state.ndim != 2:
        raise ValueError(
            "initial_conditions must have shape "
            "(batch, model.dim_total)"
        )

    if state.shape[-1] != model.dim_total:
        raise ValueError(
            f"Expected state dimension {model.dim_total}, "
            f"received {state.shape[-1]}"
        )

    times = torch.as_tensor(
        t_eval,
        dtype=state.dtype,
        device=state.device,
    )

    if times.ndim != 1:
        raise ValueError(
            "t_eval must have shape (n_time,)"
        )

    if times.numel() == 0:
        raise ValueError(
            "t_eval must contain at least one time"
        )

    if times.numel() > 1:
        if torch.any(times[1:] <= times[:-1]).item():
            raise ValueError(
                "t_eval must be strictly increasing"
            )

    # Time is not a trainable quantity here. Python floats are used only for
    # loop control; all state operations remain tensor operations in the
    # autograd graph.
    output_times = (
        times.detach().reshape(-1).cpu().tolist()
    )
    start_time = float(output_times[0])
    end_time = float(output_times[-1])

    if align_loading_knots:
        loading_breakpoints = _loading_breakpoints(
            model,
            start_time,
            end_time,
        )
    else:
        loading_breakpoints = []

    model.idx = idx

    state_history = [state]
    current_time = start_time
    breakpoint_index = 0
    completed_steps = 0

    for target_time_value in output_times[1:]:
        target_time = float(target_time_value)

        while current_time < target_time:
            completed_steps += 1
            if completed_steps > maximum_steps:
                raise RuntimeError(
                    "Maximum number of fixed Euler steps exceeded"
                )

            # Remove loading knots that have already been reached.
            while (
                breakpoint_index < len(loading_breakpoints)
                and loading_breakpoints[breakpoint_index]
                <= current_time
            ):
                breakpoint_index += 1

            # A step may finish at an output time or loading knot, but may not
            # cross either one.
            step_boundary = target_time
            if (
                breakpoint_index < len(loading_breakpoints)
                and loading_breakpoints[breakpoint_index]
                < step_boundary
            ):
                step_boundary = loading_breakpoints[
                    breakpoint_index
                ]

            remaining = step_boundary - current_time
            dt_value = min(fixed_step, remaining)
            reaches_boundary = dt_value >= remaining

            if dt_value <= 0.0:
                raise RuntimeError(
                    "Fixed Euler produced a nonpositive step"
                )

            # Convert time and step size to tensors on the state's device.
            # They are constants; gradients are required only through the
            # state and model parameters.
            time_tensor = state.new_tensor(current_time)
            dt_tensor = state.new_tensor(dt_value)

            state_rate = model(time_tensor, state)

            if state_rate.shape != state.shape:
                raise RuntimeError(
                    "model.forward returned shape "
                    f"{tuple(state_rate.shape)}; expected "
                    f"{tuple(state.shape)}"
                )

            # Out-of-place update: this is the edge that connects consecutive
            # time steps in the training computation graph.
            state = state + dt_tensor * state_rate

            if check_finite:
                if not torch.isfinite(state).all().item():
                    raise FloatingPointError(
                        "Non-finite state encountered during "
                        "fixed Euler integration"
                    )

            # Assigning the exact boundary value avoids accumulated
            # floating-point drift in the Python loop controller.
            if reaches_boundary:
                current_time = step_boundary
            else:
                current_time += dt_value

        # Store only requested output states. Each tensor retains its autograd
        # history back through all preceding Euler updates.
        state_history.append(state)

    state_solution = torch.stack(
        state_history,
        dim=0,
    )

    # This stress evaluation also remains in the graph, so a stress-based loss
    # trains both the energy and evolution networks through the trajectory.
    stress_solution = model.compute_stress(
        state_solution
    )

    return state_solution, stress_solution
    
   

def _scaled_error_norm(
    error,
    state_old,
    state_new,
    absolute_tolerance,
    relative_tolerance,
):
    """
    Return the largest per-protocol RMS normalized error.

    A single adaptive step size is used for the full protocol batch. Taking the
    maximum over protocols prevents a large error in one protocol from being
    hidden by averaging over the entire batch.
    """
    scale = (
        absolute_tolerance
        + relative_tolerance
        * torch.maximum(state_old.abs(), state_new.abs())
    )

    # A positive absolute tolerance normally prevents a zero scale. The clamp
    # additionally protects componentwise tolerance inputs.
    scale = scale.clamp_min(
        torch.finfo(state_new.dtype).tiny
    )
    normalized_error = error / scale

    if normalized_error.ndim == 1:
        return normalized_error.square().mean().sqrt()

    batch_size = normalized_error.shape[0]
    error_by_protocol = (
        normalized_error
        .reshape(batch_size, -1)
        .square()
        .mean(dim=1)
        .sqrt()
    )
    return error_by_protocol.max()


def _controller_factor(
    error_norm,
    *,
    safety,
    minimum_factor,
    maximum_factor,
):
    """
    Compute the next-step multiplier for a first-order Euler method.

    Euler's local truncation error is proportional to ``h**2``. Therefore the
    error-controller exponent is ``-1/2``.
    """
    if error_norm == 0.0:
        return maximum_factor

    if not math.isfinite(error_norm):
        return minimum_factor

    factor = safety * error_norm ** (-0.5)
    return min(
        maximum_factor,
        max(minimum_factor, factor),
    )


def integrate_inference(
    model,
    initial_conditions,
    t_eval,
    idx=None,
    *,
    initial_step=None,
    minimum_step=None,
    maximum_step=None,
    relative_tolerance=1.0e-4,
    absolute_tolerance=1.0e-7,
    safety=0.9,
    minimum_factor=0.2,
    maximum_factor=5.0,
    maximum_trials=100_000,
    align_loading_knots=True,
):
    """
    Run batched inference with adaptive explicit Euler integration.

    Parameters
    ----------
    model : HardThermodynamicsOpNet
        Trained constitutive model. ``model.rate_interp`` must already contain
        the prescribed strain-rate interpolator.
    initial_conditions : array-like or torch.Tensor
        Initial state with shape ``(batch, model.dim_total)``.
    t_eval : array-like or torch.Tensor
        Strictly increasing output times. The initial conditions correspond to
        ``t_eval[0]``.
    idx : sequence or torch.Tensor, optional
        Protocol indices forwarded to ``model.rate_interp``.
    initial_step : float, optional
        First attempted internal step. By default, the first output interval
        is used, limited by ``maximum_step``.
    minimum_step : float, optional
        Smallest permitted adaptive step. The default is based on the total
        integration span and floating-point precision.
    maximum_step : float, optional
        Largest permitted adaptive step. By default, the largest interval in
        ``t_eval`` is used.
    relative_tolerance : float or torch.Tensor, default=1e-4
        Relative local-error tolerance.
    absolute_tolerance : float or torch.Tensor, default=1e-7
        Absolute local-error tolerance. A vector with ``model.dim_total``
        values can be used when state components have different scales.
    safety : float, default=0.9
        Safety factor applied by the step-size controller.
    minimum_factor : float, default=0.2
        Smallest factor by which a trial step may be changed.
    maximum_factor : float, default=5.0
        Largest factor by which an accepted step may grow.
    maximum_trials : int, default=100000
        Maximum number of accepted plus rejected trial steps.
    align_loading_knots : bool, default=True
        Prevent steps from crossing zero-order-hold strain-rate knots.

    Returns
    -------
    state_solution : torch.Tensor
        State at ``t_eval``, with shape
        ``(n_time, batch, model.dim_total)``.
    stress_solution : torch.Tensor
        Stress at ``t_eval``, with shape
        ``(n_time, batch, model.dim)``.
    diagnostics : dict
        Step counts, right-hand-side evaluation count, accepted internal times,
        accepted step sizes, and local error estimates.

    Notes
    -----
    This function is for inference. Every accepted state and every evaluated
    right-hand side is detached, so gradients are not propagated through time.
    The model may still use autograd internally to differentiate its energy
    potential when computing stress.
    """
    if not hasattr(model, "rate_interp"):
        raise AttributeError(
            "Assign model.rate_interp before inference"
        )

    # Use the model's registered normalization buffer to identify the intended
    # device and floating-point type.
    reference_tensor = model.prm_strain
    state = torch.as_tensor(
        initial_conditions,
        dtype=reference_tensor.dtype,
        device=reference_tensor.device,
    ).detach().clone()

    if state.ndim != 2:
        raise ValueError(
            "initial_conditions must have shape "
            "(batch, model.dim_total)"
        )

    if state.shape[-1] != model.dim_total:
        raise ValueError(
            f"Expected state dimension {model.dim_total}, "
            f"received {state.shape[-1]}"
        )

    times = torch.as_tensor(
        t_eval,
        dtype=state.dtype,
        device=state.device,
    ).detach()

    if times.ndim != 1:
        raise ValueError(
            "t_eval must have shape (n_time,)"
        )

    if times.numel() == 0:
        raise ValueError(
            "t_eval must contain at least one time"
        )

    if times.numel() > 1:
        time_differences = times[1:] - times[:-1]
        if torch.any(time_differences <= 0).item():
            raise ValueError(
                "t_eval must be strictly increasing"
            )
    else:
        time_differences = None

    absolute_tolerance = torch.as_tensor(
        absolute_tolerance,
        dtype=state.dtype,
        device=state.device,
    )
    relative_tolerance = torch.as_tensor(
        relative_tolerance,
        dtype=state.dtype,
        device=state.device,
    )

    if torch.any(absolute_tolerance < 0).item():
        raise ValueError(
            "absolute_tolerance must be nonnegative"
        )

    if torch.any(relative_tolerance < 0).item():
        raise ValueError(
            "relative_tolerance must be nonnegative"
        )

    if (
        torch.all(absolute_tolerance == 0).item()
        and torch.all(relative_tolerance == 0).item()
    ):
        raise ValueError(
            "At least one error tolerance must be positive"
        )

    if not 0.0 < safety <= 1.0:
        raise ValueError(
            "safety must lie in (0, 1]"
        )

    if not 0.0 < minimum_factor <= 1.0:
        raise ValueError(
            "minimum_factor must lie in (0, 1]"
        )

    if maximum_factor < 1.0:
        raise ValueError(
            "maximum_factor must be at least 1"
        )

    if maximum_trials < 1:
        raise ValueError(
            "maximum_trials must be positive"
        )

    # The initial state corresponds exactly to the first requested time.
    start_time = float(times[0].item())
    end_time = float(times[-1].item())
    output_states = [state.clone()]

    diagnostics = {
        "accepted_steps": 0,
        "rejected_steps": 0,
        "rhs_evaluations": 0,
        "accepted_times": [start_time],
        "accepted_step_sizes": [],
        "accepted_error_norms": [],
    }

    previous_training_mode = model.training
    model.eval()
    model.idx = idx

    def evaluate_rhs(time_value, state_value):
        """
        Evaluate and immediately detach the model's ODE right-hand side.
        """
        time_tensor = state_value.new_tensor(time_value)
        derivative = model(time_tensor, state_value)
        diagnostics["rhs_evaluations"] += 1

        if derivative.shape != state_value.shape:
            raise RuntimeError(
                "model.forward returned shape "
                f"{tuple(derivative.shape)}; expected "
                f"{tuple(state_value.shape)}"
            )

        return derivative.detach()

    try:
        # No integration is required when only the initial time is requested.
        if times.numel() > 1:
            span = end_time - start_time
            floating_epsilon = torch.finfo(state.dtype).eps

            default_minimum_step = max(
                span * 1.0e-12,
                16.0
                * floating_epsilon
                * max(1.0, abs(start_time), abs(end_time)),
            )

            if minimum_step is None:
                minimum_step = default_minimum_step
            else:
                minimum_step = float(minimum_step)

            if maximum_step is None:
                maximum_step = float(
                    time_differences.max().item()
                )
            else:
                maximum_step = float(maximum_step)

            if minimum_step <= 0.0:
                raise ValueError(
                    "minimum_step must be positive"
                )

            if maximum_step <= 0.0:
                raise ValueError(
                    "maximum_step must be positive"
                )

            if minimum_step > maximum_step:
                raise ValueError(
                    "minimum_step cannot exceed maximum_step"
                )

            if initial_step is None:
                step_size = min(
                    maximum_step,
                    float(time_differences[0].item()),
                )
            else:
                step_size = float(initial_step)

            if step_size <= 0.0:
                raise ValueError(
                    "initial_step must be positive"
                )

            step_size = min(
                maximum_step,
                max(minimum_step, step_size),
            )

            if align_loading_knots:
                loading_breakpoints = _loading_breakpoints(
                    model,
                    start_time,
                    end_time,
                )
            else:
                loading_breakpoints = []

            breakpoint_index = 0
            current_time = start_time
            trial_count = 0

            for output_index in range(1, times.numel()):
                target_time = float(
                    times[output_index].item()
                )

                while current_time < target_time:
                    trial_count += 1
                    if trial_count > maximum_trials:
                        raise RuntimeError(
                            "Maximum number of adaptive Euler "
                            "trials exceeded"
                        )

                    # Discard loading knots already reached.
                    while (
                        breakpoint_index
                        < len(loading_breakpoints)
                        and loading_breakpoints[breakpoint_index]
                        <= current_time
                    ):
                        breakpoint_index += 1

                    # The next accepted step may end at an output time or at a
                    # loading discontinuity, but it may never cross either.
                    step_boundary = target_time
                    if (
                        breakpoint_index
                        < len(loading_breakpoints)
                        and loading_breakpoints[breakpoint_index]
                        < step_boundary
                    ):
                        step_boundary = loading_breakpoints[
                            breakpoint_index
                        ]

                    remaining = step_boundary - current_time
                    trial_step = min(step_size, remaining)
                    reaches_boundary = trial_step >= remaining

                    if trial_step <= 0.0:
                        raise RuntimeError(
                            "Adaptive Euler produced a "
                            "nonpositive trial step"
                        )

                    # One full explicit-Euler step.
                    slope_start = evaluate_rhs(
                        current_time,
                        state,
                    )
                    state_full = (
                        state + trial_step * slope_start
                    ).detach()

                    # Two explicit-Euler half steps.
                    half_step = 0.5 * trial_step
                    state_half = (
                        state + half_step * slope_start
                    ).detach()
                    slope_half = evaluate_rhs(
                        current_time + half_step,
                        state_half,
                    )
                    state_two_half = (
                        state_half + half_step * slope_half
                    ).detach()

                    if (
                        torch.isfinite(state_full).all().item()
                        and torch.isfinite(state_two_half).all().item()
                    ):
                        error = state_two_half - state_full
                        error_norm = float(
                            _scaled_error_norm(
                                error,
                                state,
                                state_two_half,
                                absolute_tolerance,
                                relative_tolerance,
                            ).item()
                        )
                    else:
                        error_norm = math.inf

                    factor = _controller_factor(
                        error_norm,
                        safety=safety,
                        minimum_factor=minimum_factor,
                        maximum_factor=maximum_factor,
                    )

                    if error_norm <= 1.0:
                        # The two-half-step approximation is more accurate
                        # than the single full Euler step, so retain it.
                        state = state_two_half

                        if reaches_boundary:
                            current_time = step_boundary
                        else:
                            current_time += trial_step

                        diagnostics["accepted_steps"] += 1
                        diagnostics["accepted_times"].append(
                            current_time
                        )
                        diagnostics[
                            "accepted_step_sizes"
                        ].append(trial_step)
                        diagnostics[
                            "accepted_error_norms"
                        ].append(error_norm)

                        step_size = min(
                            maximum_step,
                            max(
                                minimum_step,
                                trial_step * factor,
                            ),
                        )
                    else:
                        diagnostics["rejected_steps"] += 1

                        # A rejected step must not grow.
                        factor = min(1.0, factor)
                        proposed_step = trial_step * factor

                        if proposed_step < minimum_step:
                            if trial_step <= minimum_step:
                                raise RuntimeError(
                                    "Required step size fell below "
                                    "minimum_step; relax the "
                                    "tolerances or reduce "
                                    "minimum_step"
                                )
                            proposed_step = minimum_step

                        step_size = proposed_step

                # Store only the states requested by the user, not every
                # accepted internal adaptive step.
                output_states.append(state.clone())

        state_solution = torch.stack(
            output_states,
            dim=0,
        )

        # Evaluate stress one output time at a time. Detaching each result
        # avoids retaining the energy-gradient graph for the full trajectory.
        stress_solution = torch.stack(
            [
                model.compute_stress(output_state).detach()
                for output_state in state_solution
            ],
            dim=0,
        )

    finally:
        model.train(previous_training_mode)

    diagnostics["accepted_times"] = torch.tensor(
        diagnostics["accepted_times"],
        dtype=times.dtype,
    )
    diagnostics["accepted_step_sizes"] = torch.tensor(
        diagnostics["accepted_step_sizes"],
        dtype=times.dtype,
    )
    diagnostics["accepted_error_norms"] = torch.tensor(
        diagnostics["accepted_error_norms"],
        dtype=times.dtype,
    )

    return state_solution, stress_solution, diagnostics
