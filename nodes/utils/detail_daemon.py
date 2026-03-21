import torch
import numpy as np


# Default values for Detail Daemon parameters (overridable via DetailDaemonOptions node)
DD_DEFAULTS = {
    "dd_start": 0.20,
    "dd_end": 0.80,
    "dd_bias": 0.50,
    "dd_exponent": 0.99,
    "dd_start_offset": 0.00,
    "dd_end_offset": 0.00,
    "dd_fade": 0.00,
}


def make_detail_daemon_schedule(
    steps: int,
    start: float,
    end: float,
    bias: float,
    detail_amount: float,
    exponent: float,
    start_offset: float = 0.0,
    end_offset: float = 0.0,
    fade: float = 0.0,
    smooth: bool = True,
) -> np.ndarray:
    """Create per-step multiplier schedule for Detail Daemon sigma manipulation.

    Returns a numpy array of shape (steps,) with multiplier values.
    """
    if steps <= 0 or detail_amount == 0:
        return np.ones(steps, dtype=np.float64)

    schedule = np.ones(steps, dtype=np.float64)

    # Apply offsets to start/end
    actual_start = max(0.0, start + start_offset)
    actual_end = min(1.0, end + end_offset)

    if actual_start >= actual_end:
        return schedule

    for i in range(steps):
        progress = i / max(steps - 1, 1)

        if progress < actual_start or progress > actual_end:
            continue

        # Normalized position within the active range
        range_width = actual_end - actual_start
        if range_width <= 0:
            continue
        local_progress = (progress - actual_start) / range_width

        # Bias: shift the curve
        biased = local_progress ** (1.0 / max(bias, 0.001))

        # Exponent for curve shape
        modulation = biased ** exponent

        # Fade at range boundaries
        if fade > 0:
            fade_in = min(local_progress / fade, 1.0)
            fade_out = min((1.0 - local_progress) / fade, 1.0)
            modulation *= fade_in * fade_out

        schedule[i] = 1.0 + detail_amount * modulation

    if smooth and steps > 2:
        smoothed = schedule.copy()
        for i in range(1, steps - 1):
            smoothed[i] = (schedule[i - 1] + schedule[i] * 2 + schedule[i + 1]) / 4
        schedule = smoothed

    return schedule


def apply_detail_daemon_to_sigmas(
    sigmas: torch.Tensor,
    detail_amount: float,
    start: float = 0.20,
    end: float = 0.80,
    bias: float = 0.50,
    exponent: float = 0.99,
    start_offset: float = 0.00,
    end_offset: float = 0.00,
    fade: float = 0.00,
    smooth: bool = True,
) -> torch.Tensor:
    """Apply Detail Daemon modulation directly to a sigma tensor.

    Args:
        sigmas: 1D tensor of sigma values (last value is typically 0)

    Returns:
        Modified sigma tensor with Detail Daemon applied.
    """
    n = len(sigmas) - 1  # last sigma is 0
    if n <= 0 or detail_amount == 0:
        return sigmas

    schedule = make_detail_daemon_schedule(
        n, start, end, bias, detail_amount, exponent,
        start_offset, end_offset, fade, smooth
    )

    modified = sigmas.clone()
    for i in range(n):
        modified[i] = sigmas[i] * schedule[i]

    return modified
