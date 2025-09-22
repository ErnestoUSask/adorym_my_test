"""Example reconstruction script that enables background refinement.

The input data file is expected to contain the usual ADORYM datasets,
plus an additional dataset called ``exchange/background`` that stores the
measured detector background pattern.  Set ``background_dataset_path`` to
``None`` if you want to start the background estimate from zeros instead.
"""

from adorym.ptychography import reconstruct_ptychography


def main():
    reconstruct_ptychography(
        fname='path/to/ptychography_dataset.h5',
        obj_size=[512, 512, 1],
        psize_cm=1.3e-4,
        free_prop_cm=13.5,
        energy_ev=520,
        n_epochs=10,
        minibatch_size=8,
        background_type='per_detector',
        background_dataset_path='exchange/background',
        optimize_background=True,
        background_learning_rate=5e-3,
        output_folder='recon_with_background'
    )


if __name__ == '__main__':
    main()
