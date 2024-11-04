# MEDiCINe: Motion Estimation by Distributional Contrastive Inference for Neurophysioloy

## Introduction

MEDiCINe is a method for estimating motion in neurophysiology data for spike
sorting. See our [publication](https://) for a complete description of the
method and results. The general idea of MEDiCINe is to decompose neural activity
data into two components:
* The **motion** of the brain relative to a probe in depth.
* An **activity distribution** of the brain along the probe.

These two components are jointly optimized via gradient descent to maximize the
likelihood of a dataset of detected spikes extracted from a neural recording
session. Here is a video of this optimization process in action:

<img src="graphics/model_fitting.gif" width="100%">

The red curves on the left show the motion learned by the model, the heatmap on
the right show the activity distribution learned by the model and the
scatterplots show detected spikes (colored by amplitude). Below is a schematic
of the model:

<img src="graphics/model_schematic.jpg" width="100%">

## Usage

### Getting Started

We recomend using a virtual environment (e.g. conda or pipenv) to manage
dependencies. Once in a virtual environment with python version at least 3.9.6,
install MEDiCINe with:
```
pip install medicine-neuro
```
This will also install the necessary dependencies. Then you can run the demo
script with
```
python -m medicine_demos.run_demo
```
This will run the [demo
script](https://github.com/jazlab/medicine/blob/main/medicine_demos/run_demo.py)
and display several figures showing the results. See
[medicine_demos/run_demo.py](https://github.com/jazlab/medicine/blob/main/medicine_demos/run_demo.py)
for more details.

### SpikeInterface Integration

Once you have installed the `medicine-neuro` package, you can use it to do
motion correction in a SpikeInterface data processing pipeline. SpikeInterface
peak detection methods require the `numba` package (`$ pip install numba`).
Using the currently most recent SpikeInterface version 0.101.2, here is an
example SpikeInterface pipeline with peak extraction and motion correction using
MEDiCINe motion estimation:
```
from pathlib import Path
from medicine import run_medicine, get_spikeinterface_motion
import numpy as np

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording

# SpikeInterface recording object you would like to do motion correction for
recording = ...

# Detect, extract, and localize peaks, such as with the following pipeline
peaks = detect_peaks(recording, method="locally_exclusive")
peak_locations = localize_peaks(recording, peaks, method="monopolar_triangulation")

# Create directory to store MEDiCINe outputs for this recording
medicine_output_dir = Path('path/to/medicine/output/directory')
medicine_output_dir.mkdir(parents=True, exist_ok=True)

# Run MEDiCINe to estimate motion
run_medicine(
    peak_amplitudes=peaks['amplitude'],
    peak_depths=peak_locations['y'],
    peak_times=peaks['sample_index'] / recording.get_sampling_frequency(),
    output_dir=medicine_output_dir,
)

# Load motion estimated by MEDiCINe
motion_object = get_spikeinterface_motion(medicine_output_dir)

# Interpolate motion into recording object
recording_motion_corrected = InterpolateMotionRecording(
    recording,
    motion_object,
    border_mode='force_extrapolate',
)

# Continue with spike sorting or other processing on recording_motion_corrected.
# If you run a spike sorter with built-in motion correction, you may want to
# turn off that motion correction. If you use the SpikeInterface sorter module,
# this would entail `sorters.run_sorter(do_correction=False, ...)`.
```

### Kilosort4 Integration

If you are using Kilosort4 for spike-sorting, we recommend using a
SpikeInterface pipeline to run MEDiCINe as shown above. However, if you prefer
to use Kilosort4 directly without SpikeInterface, you may still use MEDiCINe for
motion correction. The easiest way to do this is to modify Kilosort4's
[datashift.py](https://github.com/MouseLand/Kilosort/blob/main/kilosort/datashift.py)
file directly. Using the currently most recent Kilsort4 version 4.0.19, this
entails overriding the `run()` function in `datashift.py` as follows:

```
from medicine import run as run_medicine

def run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
        clear_cache=False):
    # Extract spikes
    st, _, ops  = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache,
    )

    # Run MEDiCINe to estimate motion
    medicine_output_dir = ops['data_dir'] / 'medicine_output'
    run_medicine.run_medicine(
        peak_amplitudes=st[:, 2],
        peak_depths=st[:, 1],
        peak_times=st[:, 0],
        output_dir=medicine_output_dir,
        training_steps=2000,
    )
    motion = np.mean(np.load(medicine_output_dir / 'motion.npy'), axis=1)
    dshift_indices = np.linspace(0, len(motion), ops['Nbatches'] + 1)
    dshift_indices = np.floor(dshift_indices).astype(int)[:-1]
    dshift = motion[dshift_indices]

    # Continue Kilosort processing
    ops['yblk'] = np.array([-1])
    ops['dshift'] = dshift
    xp = np.vstack((ops['xc'],ops['yc'])).T
    Kxx = torch.from_numpy(kernel2D(xp, xp, ops['sig_interp']))
    iKxx = torch.linalg.inv(Kxx + 0.01 * torch.eye(Kxx.shape[0]))
    ops['iKxx'] = iKxx.to(device)

    return ops, st
```

### Hyperparameters

Here are descriptions of all of the hyperparameters in the [MEDiCINe
method](https://github.com/jazlab/medicine/blob/main/medicine/run.py#L20):


* `motion_bound`: Float, default 800. A bound on the maximum allowed absolute
  motion, namely the spread of the motion function output. Units are the same as
  units of spike depth (typically microns). The motion function is smoothed
  after application of this bound, so we recommend being liberal with the bound.
  In all of our recording sessions, the value 800 worked well. Adjusting this
  parameter is unlikely to improve results unless you have very large-magnitude
  motion.


* `time_bin_size`: Float, default 1. Temporal resolution of motion estimation,
  in the same units as the spike times (typically seconds). The motion array
  discretizes the total session duration into temporal bins of this size. Note
  that the actual motion is smoothed and interpolated, so while this temporal
  resolution limits the computational complexity of the motion function, it does
  not make the motion function itself a discrete or step-wise function. We find
  that 1 second is sufficient for all of our datasets. Increasing this improves
  runtime slightly. Adjusting this parameter is unlikely to improve results.


* `time_kernel_width`: Float, default 30. Width of the triangular smoothing
  kernel applied to the motion function, in the same units as the spike times
  (typically seconds). If you have very high-frequency motion, you may want to
  reduce this. If you observe over-fitting or have very low firing rates, you
  may want to increase this.


* `activity_network_hidden_features`: Tuple, default (256, 256). Hidden layer
  sizes for the activity network. Adjusting this is unlikely to improve results.


* `num_depth_bins`: Int, default 2. Number of depth bins for motion estimation.
  A value of 1 will enforce uniform (rigid) motion at all depths. Higher numbers
  allow more complex dependencies of motion on depth. We find that a value of 2
  to be sufficient for all of our datasets. A value of 2 allows linear
  interpolation of motion between two independently fit motion functions at the
  extremes of the array.


* `amplitude_threshold_quantile`: Float in [-1, 1], default 0. Cutoff quantile
  for peak amplitudes. If 0, no cutoff is applied and all peaks are used. If >
  0, then the smallest amplitude_threshold_quantile fraction of amplitudes are
  ignored. If < 0, then the largest amplitude_threshold_quantile fraction of
  amplitudes are ignored. See "raw_raster_and_amplitudes.png" output figure for
  a histogram of all amplitudes used by the model. Typically 0 is a good value.
  However, if you find that there are a lot of low amplitude spikes and they may
  be causing the model to go astray, you may want to consider making this
  non-zero. Making it positive will remove low-amplitude spikes if the spike
  amplitudes are positive. Making it negative will remove low-amplitude spikes
  if the spike amplitudes are negative.


* `batch_size`: Int, default 4096. Batch size for optimization. Reducing this
  too low can cause unstable convergence (we have observed this with batch size
  1024 and below). Adjusting this parameter is unlikely to improve results.


* `training_steps`: Int, default 10,000. Number of optimization steps to take.
  Reducing this improves runtime, but be careful to keep it greater than
  `motion_noise_steps`. Adjusting this parameter is unlikely to improve results.


* `initial_motion_noise`: Float, default 0.1. Magnitude of noise to add to the
  motion function output at the start of training. This is annealed to 0 in the
  first `motion_noise_steps` of training steps. Adding this noise avoids getting
  caught in local minima early in training. Adjusting this parameter is unlikely
  to improve results.


* `motion_noise_steps`: Int, default 2,000. Number of steps at the beginning of
  training to anneal the motion noise to 0. Adjusting this parameter is unlikely
  to improve results unless you reduce `training_steps`, in which case this
  should be reduced as well.


* `optimizer`: Torch optimizer, default `torch.optim.Adam`. Adjusting this
  parameter is unlikely to improve results.


* `learning_rate`: Float, default 0.0005. Adjusting this parameter is unlikely
  to improve results.


* `epsilon`: Float, default 0.001. Prevents divide-by-zero instabilities.
  Adjusting this parameter is unlikely to improve results.


* `plot_figures`: Bool, default `True`. Whether to plot and save figures
  summarizing the model results.

The MEDiCINe model is not sensitive to most of these hyperparameters. We have
never needed to tune hyperparameters for any of dozens of our NHP
neurophysiology datasets. The only parameters we can imagine may be necessary to
tune are `motion_bound`, `time_kernel_width`, and
`amplitude_threshold_quantile`.

### Troubleshooting Results

After running MEDiCINe, check the figures it produces (figures will be written
to the output directory, printed to the console as the model runs). The most
useful figure to look at is `corrected_motion_raster.png`. If you see
unsatisfactory results in this figure, here are some potential problems and
resolutions:

1. Overfitting. If you have very few neurons or low firing rates, you may find
   that the motion estimation overfits and looks excessively wiggly. In this
   case, you may want to try setting `num_depth_bins = 1`, which will enforce
   uniform (rigit) motion and reduce overfitting. You may also want to increase
   the motion smoothing kernel size `time_kernel_width`.
2. Underfitting. If your data has high-frequency motion and it looks like the
   motion is not capturing this, you may want to reduce `time_kernel_width`. If
   your motion has non-linear dependencies on depth, you may want to increase
   `num_depth_bins`.
3. Insufficient motion amplitude. If it looks like you have high-amplitude
   motion that MEDiCINe is not capturing that, try increasing `motion_bound`.
4. Bad spikes. If spike detection is poor, you may have a lot of bad spikes,
   such as low-amplitude spikes that are specific to particular channels and do
   not show motion. In this case increase `amplitude_threshold_quantile`. For
   example, setting `amplitude_threshold_quantile = 0.25` will ignore the
   lowest-amplitude 25% of spikes. If high-amplitude spikes are the issue, set
   `amplitude_threshold_quantile < 0`.

## Reproducing Our Results

To reproduce the results in our paper, please see
[https://github.com/jazlab/medicine_paper](https://github.com/jazlab/medicine_paper).
This has all code and instructions for replicating our results.

## Contact and Support

Please see
[CONTRIBUTING.md](https://github.com/jazlab/medicine/blob/main/CONTRIBUTING.md)
for information about support. Please email Nick Watters at
[nwatters@mit.edu](mailto:nwatters@mit.edu) with questions and feedback.

## Reference

If you use MEDiCINe or a derivative of it in your work, please cite it as
follows:

```
@article{watters2024,
author = {Nick Watters and Alessio Buccino and Mehrdad Jazayeri},
title = {MEDiCINe: Motion Estimation by DIstributional Contrastive Inference for NEurophysiology},
url = {https://arxiv.org/},
journal = {arXiv preprint arXiv:},
year = {2024}
}
```

## MEDiCINe Website

The [MEDiCINe website](https://jazlab.github.io/medicine/) is a [GitHub
Pages](https://pages.github.com/) website with a [Slate
theme](https://github.com/pages-themes/slate). The website is generated from
this [`README.md`](https://github.com/jazlab/medicine/blob/main/README.md) with
the settings in
[`_config.yml`](https://github.com/jazlab/medicine/blob/main/_config.yml) and
the Ruby dependencies in
[`Gemfile`](https://github.com/jazlab/medicine/blob/main/Gemfile).

If you would like to modify the website, first make sure you can test deploying
it locally by following the [GitHub Pages testing
instructions](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll).
Then modify this
[`README.md`](https://github.com/jazlab/medicine/blob/main/README.md) and test
deploy to view the changes before committing.

## MEDiCINe Package

For packaging and registering releases of the `medicine-neuro` package on PyPI,
consult the [Python user guide for packaging
projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
