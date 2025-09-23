"""Helper utilities for the simplified ptychography package."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def create_circular_probe(
    shape: Tuple[int, int],
    radius: float,
    *,
    defocus: float = 0.0,
    wavelength: float = 1.0,
    pixel_size: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a simple complex probe with optional quadratic phase."""

    ny, nx = shape
    y = torch.linspace(-ny // 2, ny // 2 - 1, ny, device=device)
    x = torch.linspace(-nx // 2, nx // 2 - 1, nx, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rr = torch.sqrt(yy ** 2 + xx ** 2)

    aperture = (rr <= radius).to(torch.float32)

    if defocus == 0.0:
        phase = torch.zeros_like(aperture)
    else:
        k = 2 * math.pi / wavelength
        quad = (yy ** 2 + xx ** 2) * (pixel_size ** 2)
        phase = 0.5 * k * quad / max(defocus, 1e-9)

    probe = aperture * torch.exp(1j * phase)
    norm = torch.sqrt(torch.sum(torch.abs(probe) ** 2))
    if norm > 0:
        probe = probe / norm
    return probe.to(torch.complex64)


def complex_to_numpy(data: torch.Tensor) -> np.ndarray:
    """Convert a complex PyTorch tensor to a NumPy array."""

    array = data.detach().cpu().numpy()
    return array.astype(np.complex64, copy=False)


def save_complex_field(path: str, data: torch.Tensor) -> None:
    """Persist a complex tensor as ``.npy`` file."""

    np.save(path, complex_to_numpy(data))
