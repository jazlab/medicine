"""
Helper functions for the MEDiCINe package.
"""
import numpy as np
from spikeinterface.sortingcomponents.motion import Motion


def get_spikeinterface_motion(medicine_dir):
    """
    Get the motion of the spikeinterface from the folder path.

    Args:
        medicine_dir: The path to the folder.

    Returns:
        The motion of the spikeinterface.
    """
    motion = np.load(medicine_dir / 'motion.npy')
    time_bins = np.load(medicine_dir / 'time_bins.npy')
    depth_bins = np.load(medicine_dir / 'depth_bins.npy')

    # Use interpolation to correct for motion estimated by MEDiCINe
    motion_object = Motion(
        displacement=motion,
        temporal_bins_s=time_bins,
        spatial_bins_um=depth_bins,
    )
    return motion_object
