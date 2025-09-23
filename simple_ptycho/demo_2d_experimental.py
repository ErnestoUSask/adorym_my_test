"""Run a simple 2-D ptychographic reconstruction using the lightweight code."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_ptycho_hdf5
from .reconstruction import ReconstructionConfig, run_reconstruction
from .utils import create_circular_probe, save_complex_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../demos/siemens_star_aps_2idd/data.h5"),
        help="Path to the experimental data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simple_ptycho_output"),
        help="Directory where reconstruction results are saved.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Number of ePIE iterations to perform.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of scan positions processed per batch.",
    )
    parser.add_argument(
        "--object-shape",
        type=int,
        nargs=2,
        default=(618, 606),
        metavar=("NY", "NX"),
        help="Size of the reconstruction grid in pixels.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device identifier (e.g. 'cpu' or 'cuda:0').",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=24.0,
        help="Radius of the circular probe aperture in pixels.",
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar=("NY", "NX"),
        help="Size of the probe grid in pixels.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the scan positions while loading the data. "
            "Use this to convert from metres to pixels when necessary."
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=2,
        default=None,
        metavar=("DY", "DX"),
        help="Optional offset added to the scan positions after scaling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_ptycho_hdf5(
        str(args.data_path),
        position_scale=args.position_scale,
        position_offset=tuple(args.position_offset) if args.position_offset else None,
    )

    probe = create_circular_probe(
        tuple(args.probe_size),
        radius=args.probe_radius,
        device=args.device,
    )

    config = ReconstructionConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    result = run_reconstruction(
        data,
        object_shape=tuple(args.object_shape),
        config=config,
        device=args.device,
        probe=probe,
    )

    magnitude = np.abs(result.object_estimate.cpu().numpy())
    phase = np.angle(result.object_estimate.cpu().numpy())
    np.save(args.output_dir / "object_magnitude.npy", magnitude.astype(np.float32))
    np.save(args.output_dir / "object_phase.npy", phase.astype(np.float32))
    save_complex_field(args.output_dir / "probe.npy", result.probe_estimate)
    np.save(args.output_dir / "history.npy", np.asarray(result.history, dtype=np.float32))

    print("Reconstruction finished.")
    print(f"Magnitude saved to {args.output_dir / 'object_magnitude.npy'}")
    print(f"Phase saved to {args.output_dir / 'object_phase.npy'}")
    print(f"Probe saved to {args.output_dir / 'probe.npy'}")


if __name__ == "__main__":
    main()
