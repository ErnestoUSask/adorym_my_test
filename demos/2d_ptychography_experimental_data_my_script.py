import h5py
import tifffile

from adorym.ptychography import reconstruct_ptychography
import adorym
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

DEFAULT_BACKGROUND_PATH = (
    r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data"
    r"\Joseph - PtychoRec\Reconstruction Data\A230127060bg_1_1.tif"
)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--save_path', default='cone_256_foam_ptycho')
parser.add_argument('--output_folder', default='test')  # Will create epoch folders under this

parser.add_argument('--background-type', default='per_detector',

                    choices=['none', 'per_detector', 'per_angle', 'per_pattern'])
parser.add_argument('--background-path', default=None,
                    help='Optional path to an external background file (TIFF/NumPy).')
parser.add_argument('--background-dataset', default=None,
                    help='Optional HDF5 dataset path inside the projection file.')
parser.add_argument('--background-mean-axis', type=int, default=0,
                    help='Axis over which to compute the mean when reducing a background stack.')
parser.add_argument('--skip-background-mean', action='store_true',
                    help='Keep the loaded background stack without averaging along background-mean-axis.')
parser.add_argument('--background-npz-key', default=None,
                    help='Array key to use when loading background data from a .npz archive.')
parser.add_argument('--optimize-background', action='store_true',
                    help='Enable background optimisation during reconstruction.')
parser.add_argument('--background-learning-rate', type=float, default=1e-3,
                    help='Learning rate for the background optimiser when enabled.')
args = parser.parse_args()
epoch = args.epoch
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]

output_folder = r'D:\Joseph Reconstruction\Simulation 20250914\Reconstructions'
distribution_mode = None
optimizer_obj = adorym.AdamOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                     options_dict={'step_size': 1e-3})
optimizer_probe = adorym.AdamOptimizer('probe', output_folder=output_folder, distribution_mode=distribution_mode,
                                        options_dict={'step_size': 1e-3, 'eps': 1e-7})
optimizer_all_probe_pos = adorym.AdamOptimizer('probe_pos_correction', output_folder=output_folder, distribution_mode=distribution_mode,
                                               options_dict={'step_size': 1e-2})

# probe_h5_path = r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Simulated Probe\sim_probe_1.h5"
# with h5py.File(probe_h5_path, 'r') as f:
#     probe = f['cropped_probe'][()]
#
# mag0  = np.abs(probe).astype(np.float32)
# ph0   = np.angle(probe).astype(np.float32)
#
# # --- Choose radii (in pixels): 0 < cs_r < probe_r ---
# probe_r = 187   # example: outer radius in px
# cs_r    = 55   # example: central stop radius in px (must be < probe_r)
#
# # --- Build donut mask: keep where cs_r <= r <= probe_r ---
# H, W = mag0.shape
# cy = (H - 1) / 2.0
# cx = (W - 1) / 2.0
#
# yy, xx = np.ogrid[:H, :W]
# rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
#
# keep = (rr >= cs_r) & (rr <= probe_r)   # True where we keep values
#
# # --- Apply to mag and phase (zero elsewhere) ---
# mag0 = mag0.copy()
# ph0  = ph0.copy()
#
# mag0[~keep] = mag0[~keep] / 100
# ph0[~keep]  = ph0[~keep] / 100
#
#
# H, W  = mag0.shape
n_probe_modes = 1
#
# # pixel size (meters) from your reconstruction settings/files (adjust if needed)
# px = 13e-9  # 13 nm from your filename
# # wavelength from your experiment files/settings (adjust as needed)
# lam = 1.42e-9  # example: 1.42 nm
#
# yy, xx = np.meshgrid(np.arange(H) - H//2, np.arange(W) - W//2, indexing='ij')
# r2 = ((xx*px)**2 + (yy*px)**2)
#
# # small defocus offsets around 0 (meters); tune span to your experiment
# dz_list = np.array([0.0])#, -0.5e-3, +0.5e-3, -1.0e-3, +1.0e-3])  # ±0.5–1.0 mm
#
# mag_stack  = np.empty((n_probe_modes, H, W), np.float32)
# phase_stack = np.empty((n_probe_modes, H, W), np.float32)
#
# for m, dz in enumerate(dz_list):
#     quad_phase = np.pi * r2 / (lam * (dz + 1e-30)) if dz != 0 else 0.0
#     mag_stack[m]   = (mag0 / np.sqrt(n_probe_modes)).astype(np.float32)
#     phase_stack[m] = (ph0 + (quad_phase if isinstance(quad_phase, np.ndarray) else 0.0)).astype(np.float32)
#
# probe_mag_phase = [mag0, ph0]
# probe_mag_phase = np.array(probe_mag_phase)

H = W = 1920

mag0 = np.ones((H, W), dtype=np.float32)   # uniform amplitude = 1
ph0  = np.zeros((H, W), dtype=np.float32)  # uniform phase   = 0

mag0 = tifffile.imread(r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\probe_abs.tif")
ph0 = tifffile.imread(r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\probe_angle.tif")

probe_mag_phase = [mag0, ph0]              # what ADORYM expects for 'supplied'
probe_mag_phase = np.array(probe_mag_phase)  # optional; list is fine

def load_background_from_file(file_path, npz_key=None):
    file_path = os.path.expanduser(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        data = tifffile.imread(file_path)
    elif ext == '.npy':
        data = np.load(file_path)
    elif ext == '.npz':
        with np.load(file_path) as npz_file:
            if npz_key is not None:
                if npz_key not in npz_file:
                    raise KeyError(f'Key "{npz_key}" not found in {file_path}.')
                data = npz_file[npz_key]
            elif len(npz_file.files) == 1:
                data = npz_file[npz_file.files[0]]
            elif 'arr_0' in npz_file:
                data = npz_file['arr_0']
            else:
                raise ValueError(
                    f'Multiple arrays found in {file_path}. Provide --background-npz-key to select one.')
    else:
        raise ValueError(f'Unsupported background file extension: {ext}')
    return np.array(data, dtype=np.float32)

background_initial = None
background_path = args.background_path if args.background_path is not None else DEFAULT_BACKGROUND_PATH
if background_path is not None:
    background_file = os.path.expanduser(background_path)
    if os.path.isfile(background_file):
        background_data = load_background_from_file(background_file, args.background_npz_key)
        if not args.skip_background_mean and background_data.ndim > 2:
            background_initial = np.mean(background_data, axis=args.background_mean_axis)
        else:
            background_initial = background_data
    else:
        if args.background_path is not None:
            raise FileNotFoundError(f'Background file not found: {args.background_path}')
        else:
            print('Warning: default background file not found. Continuing without background_initial.')
elif args.background_type != 'none' and args.background_dataset is None:
    print('Warning: background_type is set but no background data provided.')


if background_initial is not None:
    background_initial = np.array(background_initial, dtype=np.float32, copy=False)

params_2idd_gpu = {'fname': r"D:\Joseph Reconstruction\h5 files\data_3.h5",
                    'theta_st': 0,
                    'theta_end': 0,
                    'n_epochs': 2000,
                    'obj_size': (2408, 2408, 1),
                    'two_d_mode': True,
                    'energy_ev': 571,
                    'psize_cm': 1.3365e-06,
                    'minibatch_size': 5,
                    'output_folder': 'data_3_bgsub_20250915_supplied_2',
                    'cpu_only': True,
                    'save_path': r'D:\Joseph Reconstruction\Reconstructions',
                    'use_checkpoint': False,
                    'n_epoch_final_pass': None,
                    'save_intermediate': False,
                    'full_intermediate': True,
                    'initial_guess': None,
                    'random_guess_means_sigmas': (1., 0., 0.001, 0.002),
                    'n_dp_batch': 350,
                    # ===============================plane
                    'n_probe_modes': 5,
                    # 'probe_type': 'ifft',
                    # 'probe_type': 'plane',
                    'probe_type': 'aperture_defocus',
                    'aperture_radius': 1,
                    # 'beamstop_radius': 5,
                    'probe_defocus_cm': 0.0046,
                    # 'probe_type': 'gaussian',
                    # 'probe_mag_sigma': 187,
                    # 'probe_phase_sigma': 187,
                    # 'probe_phase_max': 1,
                    # 'probe_type': 'supplied',
                    # 'probe_initial': probe_mag_phase,
                    # ===============================
                    'rescale_probe_intensity': True,
                    'free_prop_cm': 'inf',
                    'backend': 'pytorch',
                    'raw_data_type': 'intensity',
                    'beamstop': None,
                    'randomize_probe_pos': True,
                    'optimizer': optimizer_obj,
                    'optimize_probe': True,
                    'optimizer_probe': optimizer_probe,
                    'optimize_all_probe_pos': False,
                    'optimizer_all_probe_pos': optimizer_all_probe_pos,
                    'save_history': True,
                    'update_scheme': 'immediate',
                    'unknown_type': 'real_imag',
                    'save_stdout': True,
                    'loss_function_type': 'lsq',
                    # 'normalize_fft': False
                    }

params_2idd_gpu['background_type'] = args.background_type
if args.background_dataset is not None:
    params_2idd_gpu['background_dataset_path'] = args.background_dataset
if background_initial is not None:
    params_2idd_gpu['background_initial'] = background_initial
if args.optimize_background:
    params_2idd_gpu['optimize_background'] = True
    params_2idd_gpu['background_learning_rate'] = args.background_learning_rate

params = params_2idd_gpu

# h5 = r"D:\Joseph Reconstruction\h5 files\data_3.h5"
# with h5py.File(h5, "r") as f:
#     ny, nx = f["exchange/data"].shape[-2:]   # detector frame size

# # ---- Option 1: fixed circular mask (simple & robust) ----
# beamstop_px = 120   # <-- set this to your central-stop radius on the DETECTOR in pixels
# yy, xx = np.ogrid[:ny, :nx]
# cy, cx = ny//2, nx//2
# rr = np.hypot(yy - cy, xx - cx)
# beamstop_mask = rr >= beamstop_px   # True=use pixel, False=ignore central stop
#
# # (If the stop is shifted, set cy,cx accordingly)
#
# params['beamstop'] = beamstop_mask.astype(bool)

reconstruct_ptychography(**params)