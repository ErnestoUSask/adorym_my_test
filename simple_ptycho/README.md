# Simple Ptychography

This directory contains a self-contained, minimal implementation of a
2-D ptychographic reconstruction pipeline.  The goal is to expose the core
ePIE algorithm in a compact form that is easy to study and adapt without
navigating the full Adorym code base.

## Contents

- `data.py` – Loading helper that understands the experimental HDF5 layout
  used in the demos.
- `reconstruction.py` – Implementation of a lightweight ePIE solver.
- `utils.py` – Small utility helpers for probe generation and saving output.
- `demo_2d_experimental.py` – Command line entry point mirroring the original
  Adorym demo but using the simplified modules.

## Usage

1. Install the required Python packages in your environment.  Only NumPy,
   PyTorch and h5py are needed.  The latter is used to read the
   experimental HDF5 file.
2. Run the demo script:

   ```bash
   python -m simple_ptycho.demo_2d_experimental --data-path path/to/data.h5 \
       --output-dir my_reconstruction
   ```

   The default arguments target the Siemens star data set bundled with the
   repository.
3. Inspect the generated `*.npy` files in the output directory to examine the
   reconstructed object magnitude, phase and probe estimate.

Feel free to modify `reconstruction.py` to experiment with alternative update
rules or constraints.  The code is deliberately concise to make such
experiments straightforward.
