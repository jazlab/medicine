"""Run MEDICINE motion estimation.

This file contains a function run_medicine() that is the main entry point for
running MEDICINE motion estimation. It takes in peak times, depths, and
amplitudes. It creates and runs MEDICINE motion estimation on the data and saves
the results to an output directory.

Usage: See ../medicine_demos/run_demo.py for an example of usage.
"""

import json
from pathlib import Path

import numpy as np
import torch

from medicine import model, plotting
from medicine.logger import logger

def run_medicine(
    peak_times: np.ndarray,
    peak_depths: np.ndarray,
    peak_amplitudes: np.ndarray,
    output_dir: str | None,
    motion_bound: float = 800,
    time_bin_size: float = 1,
    time_kernel_width: float = 30,
    activity_network_hidden_features: tuple = (256, 256),
    num_depth_bins: int = 2,
    amplitude_threshold_quantile: float = 0.0,
    batch_size: int = 4096,
    training_steps: int = 10000,
    initial_motion_noise: float = 0.1,
    motion_noise_steps: int = 2000,
    optimizer: torch.optim.Optimizer = torch.optim.Adam,
    learning_rate: float = 0.0005,
    epsilon: float = 1e-3,
    plot_figures: bool = True,
    return_motion: bool = True,
) -> model.Trainer:
    """Run MEDICINE motion estimation.

    Args:
        peak_times: Array of shape [num_peaks] containing peak times.
        peak_depths: Array of shape [num_peaks] containing peak depths.
        peak_amplitudes: Array of shape [num_peaks] containing peak amplitudes.
        output_dir: Directory to save output. If None then nothing is saved.
        motion_bound: Bound on maximum absolute motion, namely the difference
            between the max and min depth.
        time_bin_size: Temporal resolution of motion estimation.
        time_kernel_width: Width of temporal kernel for motion estimation.
        activity_network_hidden_features: Tuple of hidden features for the
            activity network.
        num_depth_bins: Number of depth bins for motion estimation.
        amplitude_threshold_quantile: FLoat in [-1, 1]. Cutoff quantile for peak
            amplitudes. If 0, no cutoff is applied and all peaks are used. If >
            0, then the smallest amplitude_threshold_quantile fraction of
            amplitudes are ignored. If < 0, then the largest
            amplitude_threshold_quantile fraction of amplitudes are ignored. See
            "raw_raster_and_amplitudes.png" output figure for a histogram of all
            amplitudes used by the model.
        batch_size: Batch size for training.
        training_steps: Number of training steps.
        initial_motion_noise: Initial motion noise.
        motion_noise_steps: Number of training steps to reduce motion noise from
            initial_motion_noise to 0.
        optimizer: Optimizer for training.
        learning_rate: Learning rate for training.
        epsilon: Small value to prevent instabilities.
        plot_figures: Whether to plot figures.
        return_motion: also return the spikeinterface Motion object

    Returns:
        trainer: Trainer object after running motion estimation.
        if return_motion=True also return the Motion object
    """
    # Create and clear output_dir
    if output_dir is not None:
        logger.info(f"Creating output_dir {output_dir}")
        output_dir = Path(output_dir)
        if output_dir.exists():
            logger.info(f"Warning: {output_dir} already exists")
        output_dir.mkdir(exist_ok=True, parents=True)

    # Save parameters
    parameters = dict(
        output_dir=str(output_dir) if output_dir is not None else None,
        motion_bound=motion_bound,
        time_bin_size=time_bin_size,
        time_kernel_width=time_kernel_width,
        activity_network_hidden_features=activity_network_hidden_features,
        num_depth_bins=num_depth_bins,
        batch_size=batch_size,
        training_steps=training_steps,
        initial_motion_noise=initial_motion_noise,
        motion_noise_steps=motion_noise_steps,
        optimizer=f"{optimizer.__module__}.{optimizer.__name__}",
        learning_rate=learning_rate,
        epsilon=epsilon,
        plot_figures=plot_figures,
    )
    if output_dir is not None:
        parameters_path = output_dir / "medicine_parameters.json"
        logger.info(f"Saving parameters to {output_dir}")
        json.dump(parameters, open(parameters_path, "w"))

    # Plot raster and amplitudes if necessary
    if plot_figures:
        if output_dir is None:
            raise ValueError("run_medicine(): when plot_figures=True, output_dir must be not None")
        plotting.plot_raster_and_amplitudes(
            peak_times, peak_depths, peak_amplitudes, figure_dir=output_dir
        )

    # Create dataset
    dataset = model.Dataset(
        times=peak_times,
        depths=peak_depths,
        amplitudes=peak_amplitudes,
        amplitude_threshold_quantile=amplitude_threshold_quantile,
    )

    # Create motion_function
    motion_bound_normalized = (
        0.5 * motion_bound / (dataset.depth_range[1] - dataset.depth_range[0])
    )
    time_range = (
        np.min(peak_times) - epsilon,
        np.max(peak_times) + epsilon,
    )
    motion_function = model.MotionFunction(
        bound_normalized=motion_bound_normalized,
        time_range=time_range,
        time_bin_size=time_bin_size,
        time_kernel_width=time_kernel_width,
        num_depth_bins=num_depth_bins,
    )

    # Create medicine model
    medicine_model = model.Medicine(
        motion_function=motion_function,
        activity_network=model.ActivityNetwork(
            hidden_features=activity_network_hidden_features,
        ),
    )

    # Create trainer
    trainer = model.Trainer(
        dataset,
        medicine_model=medicine_model,
        batch_size=batch_size,
        training_steps=training_steps,
        initial_motion_noise=initial_motion_noise,
        motion_noise_steps=motion_noise_steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    # Run trainer
    trainer()

    # Get motion estimation results
    motion_function = trainer.medicine_model.motion_function
    num_time_bins = motion_function.num_time_bins
    time_bins = motion_function.time_range[
        0
    ] + motion_function.time_bin_size * np.arange(num_time_bins)
    depth_range = trainer.dataset.depth_range
    depth_bins = np.linspace(0, 1, num_depth_bins)
    times = np.repeat(time_bins[:, None], num_depth_bins, axis=1)
    depths = np.repeat(depth_bins[None, :], num_time_bins, axis=0)
    times_torch = torch.from_numpy(times.astype(np.float32)).flatten()
    depths_torch = torch.from_numpy(depths.astype(np.float32)).flatten()
    times_torch = times_torch.to(trainer.device)
    depths_torch = depths_torch.to(trainer.device)
    pred_motion_flat = motion_function(times_torch, depths_torch).cpu()
    pred_motion = pred_motion_flat.reshape(num_time_bins, num_depth_bins)
    pred_motion = -1 * pred_motion.detach().numpy()
    pred_motion *= depth_range[1] - depth_range[0]
    depth_bins = depth_range[0] + depth_bins * (
        depth_range[1] - depth_range[0]
    )




    # Save motion estimation results
    if output_dir is not None:
        logger.info(f"Saving outputs to {output_dir}")
        np.save(output_dir / "time_bins.npy", time_bins)
        np.save(output_dir / "depth_bins.npy", depth_bins)
        np.save(output_dir / "motion.npy", pred_motion)

    # Plot motion estimation results if necessary
    if plot_figures:
        plotting.run_post_motion_estimation_plots(
            figure_dir=output_dir,
            trainer=trainer,
        )


    if return_motion:
        from spikeinterface.sortingcomponents.motion import Motion
        motion = Motion(
            displacement=[pred_motion],
            temporal_bins_s=[time_bins],
            spatial_bins_um=depth_bins,
        )
        return trainer, motion
    else:
        return trainer
