"""Utilities for loading experimental ptychography data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PtychographyData:
    """Container for the measured diffraction data and scan geometry.

    Attributes
    ----------
    intensities:
        Array of far-field diffraction intensities with shape
        ``(n_scan, detector_y, detector_x)``.
    positions:
        Array of (y, x) scan positions given in pixel units with shape
        ``(n_scan, 2)``.
    """

    intensities: np.ndarray
    positions: np.ndarray

    def amplitudes(self) -> np.ndarray:
        """Return the measured amplitudes derived from the intensities."""

        return np.sqrt(np.clip(self.intensities, a_min=0.0, a_max=None))


def _ensure_scan_dimensions(array: np.ndarray) -> np.ndarray:
    """Flatten leading dimensions so we end up with ``(n_scan, ...)``.

    The Siemens-star data set used in the demos stores a leading dimension
    for the rotation angle.  For a 2-D reconstruction there is typically
    only a single angle, so we squeeze it away here.
    """

    if array.ndim == 4:
        # Layout: (n_angle, n_scan, ny, nx)
        array = array.reshape(-1, array.shape[-2], array.shape[-1])
    elif array.ndim == 3:
        # Already in the desired shape.
        pass
    else:
        raise ValueError(
            "Expected the diffraction data to have 3 or 4 dimensions, "
            f"but received shape {array.shape}."
        )
    return array


def _ensure_position_dimensions(array: np.ndarray) -> np.ndarray:
    """Reshape the scan position array to ``(n_scan, 2)``."""

    array = np.asarray(array)
    if array.ndim == 3:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[-1] != 2:
        raise ValueError(
            "Scan positions are expected to be pairs of (y, x) coordinates "
            f"but have trailing dimension {array.shape[-1]}."
        )
    return array


def load_ptycho_hdf5(
    path: str,
    *,
    data_key: str = "exchange/data",
    position_key: str = "exchange/positions",
    dtype: np.dtype = np.float32,
    position_scale: float = 1.0,
    position_offset: Optional[Tuple[float, float]] = None,
) -> PtychographyData:
    """Load diffraction data and scan positions from an HDF5 file.

    Parameters
    ----------
    path:
        Path to the HDF5 file.
    data_key:
        HDF5 key of the diffraction data.  The default matches the format
        used in the Adorym demos.
    position_key:
        HDF5 key of the probe scan positions.
    dtype:
        Data type of the returned intensities.
    position_scale:
        Optional scaling factor applied to the scan positions.  The Siemens
        star measurements store real-space positions in meters.  When a
        pixel-based object grid is used we typically want integer pixel
        offsets, so this parameter lets the caller convert from meters to
        pixels.
    position_offset:
        Optional (y, x) tuple added to the scaled scan positions.  This is
        handy when the coordinate origin in the file does not match the
        object grid.
    """

    try:
        import h5py  # Imported lazily so the module works without h5py.
    except ImportError as exc:  # pragma: no cover - import is environment specific.
        raise ImportError(
            "The simplified ptychography loader requires the 'h5py' package "
            "to read HDF5 files. Please install it in your Python environment"
        ) from exc

    with h5py.File(path, "r") as f:
        raw_intensity = np.asarray(f[data_key], dtype=dtype)
        raw_positions = np.asarray(f[position_key], dtype=np.float32)

    intensities = _ensure_scan_dimensions(raw_intensity)
    positions = _ensure_position_dimensions(raw_positions)

    if position_scale != 1.0:
        positions = positions * position_scale
    if position_offset is not None:
        positions = positions + np.asarray(position_offset, dtype=np.float32)

    return PtychographyData(intensities=intensities, positions=positions)
