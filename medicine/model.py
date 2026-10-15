"""MEDICINE motion estimation algorithm.

The general idea of this algorithm is to fit two things to a recording:
    (i) The motion of the brain relative to a probe in depth.
    (ii) An activity distribution of the brain along the probe.
These are fit based on peaks (essentially threshold crossings) extracted from
neural recording. The peaks have two parameters: depth and amplitude.

The algorithm fits motion and activity simultaneously. They are parameterized as
follows:
    Motion: The motion is parameterized as a discretized timeseries across the
        entire session. The discretization has bin size, e.g. 1 second, in which
        case motion is a vector of length number of seconds in the session,
        where values are the motion in microns at that point in time.
        Importantly, we smooth this motion vector with a triangular kernel
        (~30-second support tends to work well), to prevent the motion from
        being too jumpy and noise-sensitive. In the future, a Gaussian process
        prior might be more principled.
    Activity distribution: The activity distribution in [depth, amplitude] space
        is parameterized by a neural network taking a (depth, amplitude) point
        and returning the probability of receiving a spike with that depth and
        amplitude. In other words, the density is parameterized implicitly. This
        leads to much better fitting than an explicitly parameterized density.

The fitting is done by gradient descent to maximize the fit of the activity
to data across the entire session, specifically to classify with a logistic loss
real datapoints from points randomly uniformly sampled in [depth, amplitude]
space. Since the activity does not depend on time, the motion is pressured
to facilitate this classification, i.e. to stabilize the activity over time.
"""

import math
from typing import Optional

import numpy as np
import torch
import tqdm

from medicine.logger import logger


class MLP(torch.nn.Module):
    """MLP model."""

    def __init__(
        self,
        in_features: int,
        layer_features: list[int],
        activation: Optional[torch.nn.Module] = None,
    ):
        """Create MLP module.

        Args:
            in_features: Number of features of the input.
            layer_features: Iterable of ints. Output sizes of the layers.
            activation: Activation function. If None, defaults to ReLU.
            bias: Bool. Whether to use bias.
        """
        super(MLP, self).__init__()

        self._in_features = in_features
        self._layer_features = layer_features
        if activation is None:
            activation = torch.nn.ReLU()
        self.activation = activation

        features_list = [in_features] + list(layer_features)
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(activation)
            layer = torch.nn.Linear(
                in_features=features_list[i], out_features=features_list[i + 1]
            )
            module_list.append(layer)

        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input.

        Args:
            x: Tensor of shape [batch_size, ..., in_features].

        Returns:
            Output of shape [batch_size, ..., self.out_features].
        """
        return self.net(x)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def layer_features(self) -> list[int]:
        return self._layer_features

    @property
    def out_features(self) -> int:
        return self._layer_features[-1]


class Dataset:
    """Dataset that gives samples of peaks."""

    def __init__(
        self,
        times: np.ndarray,
        depths: np.ndarray,
        amplitudes: np.ndarray,
        amplitude_threshold_quantile: float = 0.0,
    ):
        """Constructor.

        Args:
            times: Array of times for each peak (in seconds).
            depths: Array of depths for each peak (in microns).
            amplitudes: Array of amplitudes for each peak.
            amplitude_threshold_quantile: Float in [0, 1). Fraction of peaks
                with the smallest absolute amplitude to ignore. If 0, no
                cutoff is applied and all peaks are used. Because the cutoff
                is applied to absolute amplitude, this removes peaks with
                amplitude near zero regardless of whether amplitudes are
                positive, negative, or both. See
                "raw_raster_and_amplitudes.png" output figure for a histogram
                of all amplitudes.
        """
        logger.info("Constructing Dataset instance")

        # Validate amplitude threshold
        if amplitude_threshold_quantile < 0:
            raise ValueError(
                "amplitude_threshold_quantile must be in [0, 1), but got "
                f"{amplitude_threshold_quantile}. Negative values are no "
                "longer supported: the threshold is now applied to absolute "
                "amplitude, so a positive value removes peaks with amplitude "
                "near zero whether amplitudes are positive or negative."
            )
        if amplitude_threshold_quantile >= 1:
            raise ValueError(
                "amplitude_threshold_quantile must be in [0, 1), but got "
                f"{amplitude_threshold_quantile}. A value of 1 or greater "
                "would remove all peaks."
            )

        # Apply amplitude threshold to remove peaks with near-zero amplitude
        if amplitude_threshold_quantile > 0:
            abs_amplitudes = np.abs(amplitudes)
            threshold = np.quantile(
                abs_amplitudes, amplitude_threshold_quantile
            )
            mask = abs_amplitudes > threshold
        else:
            mask = np.ones_like(amplitudes, dtype=bool)
        times = times[mask]
        depths = depths[mask]
        amplitudes = amplitudes[mask]

        # Register data variables
        self._times = torch.from_numpy(times.astype(np.float32))
        self._num_samples = len(times)
        self._depths_raw = depths
        self._amplitudes_raw = amplitudes

        # Normalize depths and amplitudes to [0, 1]
        depths_normalized, self._depth_range = self._normalize(depths)
        amplitudes_normalized, self._amplitude_range = self._normalize(
            amplitudes
        )
        self._depths_normalized = torch.from_numpy(
            depths_normalized.astype(np.float32)
        )
        self._amplitudes_normalized = torch.from_numpy(
            amplitudes_normalized.astype(np.float32)
        )

    def _normalize(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Normalize input x to [0, 1]."""
        x_range = (np.min(x), np.max(x))
        x_normalized = (x - x_range[0]) / (x_range[1] - x_range[0])
        return x_normalized, x_range

    def sample_real(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch of real data."""
        indices = np.random.randint(self._num_samples, size=batch_size)
        data = {
            "times": self._times[indices],
            "depths": self._depths_normalized[indices],
            "amplitudes": self._amplitudes_normalized[indices],
        }
        return data

    def sample_fake(
        self,
        batch_size: int,
        motion_bound: float,
    ) -> dict[str, torch.Tensor]:
        """Sample batch of fake data.

        Args:
            batch_size: Number of samples to generate.
            motion_bound: Maximum absolute motion to allow.

        Returns:
            data: Dictionary with keys 'times', 'depths', and 'amplitudes'.
        """
        indices = np.random.randint(self._num_samples, size=batch_size)
        data = {
            "times": self._times[indices],
            "depths": torch.FloatTensor(batch_size).uniform_(
                -motion_bound, 1 + motion_bound
            ),
            "amplitudes": torch.FloatTensor(batch_size).uniform_(0, 1),
        }
        return data

    def sample_grid(
        self,
        time: float,
        grid_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample grid of data."""
        amplitudes, depths = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size),
        )
        data = {
            "times": time * np.ones(grid_size * grid_size),
            "depths": np.ravel(depths),
            "amplitudes": np.ravel(amplitudes),
        }
        data = {
            key: torch.from_numpy(value.astype(np.float32))
            for key, value in data.items()
        }
        return data

    @property
    def times(self) -> torch.Tensor:
        return self._times

    @property
    def times_numpy(self) -> np.ndarray:
        return self.times.numpy()

    @property
    def depths_raw(self) -> np.ndarray:
        return self._depths_raw

    @property
    def amplitudes_raw(self) -> np.ndarray:
        return self._amplitudes_raw

    @property
    def depths_normalized(self) -> torch.Tensor:
        return self._depths_normalized

    @property
    def amplitudes_normalized(self) -> torch.Tensor:
        return self._amplitudes_normalized

    @property
    def depth_range(self) -> tuple[float, float]:
        return self._depth_range

    @property
    def amplitude_range(self) -> tuple[float, float]:
        return self._amplitude_range


class MotionFunction(torch.nn.Module):
    """MotionFunction class.

    This class takes in a batch of timestamps and returns a batch of predicted
    motions.
    """

    def __init__(
        self,
        bound_normalized: float,
        time_range: tuple[float, float],
        time_bin_size: float,
        time_kernel_width: float,
        num_depth_bins: int = 2,
        epsilon: float = 1e-3,
    ):
        """Constructor.

        Args:
            bound_normalized: Scalar. Bound on maximum absolute motion, after
                normalization of depth to [0, 1]. So this should be less than 1.
            time_range: Tuple (min_time, max_time) for data.
            time_bin_size: Scalar. Discretization of the motion function in
                units of time.
            time_kernel_width: Scalar. Width of the smoothing kernel in units of
                time.
            num_depth_bins: Int. Number of depth bins.
            epsilon: Small value to prevent division by zero.
        """
        super(MotionFunction, self).__init__()
        logger.info(
            "Constructing MotionFunction instance with parameters:\n"
            f"    bound_normalized = {bound_normalized}\n"
            f"    time_range = {time_range}\n"
            f"    time_bin_size = {time_bin_size}\n"
            f"    time_kernel_width = {time_kernel_width}\n"
            f"    num_depth_bins = {num_depth_bins}"
        )
        self._bound_normalized = bound_normalized
        self._time_range = time_range
        self._time_bin_size = time_bin_size
        self._num_depth_bins = num_depth_bins
        self._epsilon = epsilon * epsilon
        self._depth_smoothing = 1.0 / max(1, num_depth_bins - 1) + epsilon

        # Construct motion matrix time kernel
        self._num_time_bins = math.ceil(
            (time_range[1] - time_range[0]) / time_bin_size
        )
        self._motion = torch.nn.Parameter(
            torch.zeros(self._num_depth_bins, self._num_time_bins),
            requires_grad=True,
        )

        # Construct depth levels
        self._depth_levels = torch.linspace(0.0, 1.0, num_depth_bins)

        # Construct time kernel
        self._time_kernel = self._get_kernel(time_kernel_width)

        # Compute kernel applied to ones, which will be used to normalize kernel
        # convolution applications to remove edge effects.
        ones_input = torch.ones((1, 1, self._num_time_bins))
        self._conv_ones = torch.nn.functional.conv1d(
            ones_input, self._time_kernel, padding="same"
        )[0]
        self._tanh = torch.nn.Tanh()

    def to(self, device: torch.device) -> None:
        """Move model to device."""
        self._depth_levels = self._depth_levels.to(device)
        self._time_kernel = self._time_kernel.to(device)
        self._conv_ones = self._conv_ones.to(device)
        super().to(device)

    def _get_kernel(self, time_kernel_width: float) -> torch.Tensor:
        """Get triangular kernel discretized by time_bin_size.

        Args:
            time_kernel_width: Scalar. Width of the kernel in units of time.

        Returns:
            kernel: Torch array of size [1, 1, bins_in_kernel]. Triangular
                kernel, normalized to have unit area.
        """
        if time_kernel_width < self._time_bin_size:
            logger.info(
                f"time_kernel_width {time_kernel_width} is smaller than "
                f"time_bin_size {self._time_bin_size}, so rounding up "
                "smoothing kernel to one bin.\n"
            )
        kernel_slope = 0.5 * time_kernel_width / self._time_bin_size
        half_kernel = np.arange(1.0, 0.0, -1 / kernel_slope)
        kernel = np.concatenate([half_kernel[::-1], half_kernel[1:]])
        kernel /= np.sum(kernel)
        kernel = torch.from_numpy(kernel.astype(np.float32))
        kernel = kernel[None, None]
        return kernel

    def forward(
        self,
        times: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply motion function to times and depths."""
        time_bins = torch.floor(
            (times - self._time_range[0]) / self._time_bin_size
        )
        time_bins = time_bins.type(torch.int64)

        # Shape [num_depth_bins, batch_size]
        pred_motions = self.smooth_motion[:, time_bins]

        # Get coefficients for depth smoothing
        diffs = torch.abs(depths[None] - self._depth_levels[:, None])
        coeffs = torch.nn.functional.relu(self._depth_smoothing - diffs)
        coeffs /= self._epsilon + torch.sum(coeffs, dim=0, keepdim=True)

        # Apply depth smoothing to get shape [batch_size]
        pred_motions = torch.sum(pred_motions * coeffs, dim=0)

        return pred_motions

    @property
    def bound_normalized(self) -> float:
        return self._bound_normalized

    @property
    def time_range(self) -> tuple[float, float]:
        return self._time_range

    @property
    def time_bin_size(self) -> float:
        return self._time_bin_size

    @property
    def num_time_bins(self) -> int:
        return self._num_time_bins

    @property
    def num_depth_bins(self) -> int:
        return self._num_depth_bins

    @property
    def smooth_motion(self) -> torch.Tensor:
        """Normalize and smooth self._motion by kernel.

        Returns:
            smooth_motion_normalized: Tensor of shape
                [self.num_time_bins, self.num_depth_bins].
        """
        # Apply nonlinearity and smoothing
        motion = self._bound_normalized * self._tanh(self._motion)
        smooth_motion = torch.nn.functional.conv1d(
            motion[:, None], self._time_kernel, padding="same"
        )[:, 0]
        smooth_motion_normalized = smooth_motion / self._conv_ones

        # Center each motion vector. We detach the mean in order to prevent the
        # model intentionally expanding dense timepoints because of the motion
        # noise.
        smooth_motion_normalized -= torch.mean(
            smooth_motion_normalized, axis=1, keepdims=True
        ).detach()

        return smooth_motion_normalized


class ActivityNetwork(torch.nn.Module):
    """ActivityNetwork class.

    This class takes in a batch of depths and amplitudes and returns an
    un-normalized probability for each of them occuring in the real dataset.
    """

    def __init__(
        self,
        hidden_features: tuple = (256, 256),
        activation: Optional[torch.nn.Module] = None,
        feature_frequencies: tuple = (1, 2, 4, 8, 16, 32),
    ):
        """Construct activity network.

        Args:
            hidden_features: Tuple of ints. Number of hidden features in each
                layer.
            activation: Activation function. If None, defaults to ReLU.
            feature_frequencies: Tuple of ints. Frequencies of sine features to
                include in the input to the network.
        """
        super(ActivityNetwork, self).__init__()
        logger.info(
            "Constructing ActivityNetwork instance with parameters:\n"
            f"    hidden_features = {hidden_features}\n"
            f"    activation = {activation}"
        )
        self._feature_frequencies = feature_frequencies
        in_features = 2 * (1 + len(self._feature_frequencies))
        self._net = MLP(
            in_features=in_features,
            layer_features=list(hidden_features) + [1],
            activation=activation,
        )
        self._sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        depths: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spike probabilities in [0, 1] for depths and amplitudes."""
        features_depths = [depths] + [
            torch.sin(f * depths) for f in self._feature_frequencies
        ]
        features_amplitudes = [amplitudes] + [
            torch.sin(f * amplitudes) for f in self._feature_frequencies
        ]
        net_inputs = torch.stack(features_depths + features_amplitudes, axis=1)
        net_outputs = self._net(net_inputs)[:, 0]
        spike_probability = self._sigmoid(net_outputs)
        return spike_probability


class Medicine(torch.nn.Module):
    """Medicine class."""

    def __init__(
        self,
        motion_function: MotionFunction,
        activity_network: ActivityNetwork,
        epsilon: float = 1e-4,
    ):
        """Constructor.

        Args:
            motion_function: MotionFunction instance. Callable, takes in a batch
                of times and returns a batch of predicted motions.
            activity_network: ActivityNetwork instance. Callable,
                takes in a batch of depths and a batch of amplitudes and
                returns a batch of likelihoods.
            epsilon: Small value to prevent logarithms from exploding.
        """
        super(Medicine, self).__init__()
        logger.info("Constructing Medicine instance")

        self.add_module("motion_function", motion_function)
        self.add_module("activity_network", activity_network)
        self._epsilon = epsilon

    def forward(
        self,
        data_batch: dict,
        motion_noise: float = 0.0,
    ) -> torch.Tensor:
        """Predict likelihood for each sample in data_batch.

        Args:
            data_batch: Dictionary with keys 'times', 'depths', and
                'amplitudes'.
            motion_noise: Scalar. Standard deviation of noise to add to motion.

        Returns:
            spike_probability: Torch array of shape [batch_size] with values in
                [0, 1].
        """
        times = data_batch["times"]
        depths = data_batch["depths"]
        amplitudes = data_batch["amplitudes"]
        pred_motion = self.motion_function(times, depths)
        pred_motion += motion_noise * torch.randn_like(pred_motion)
        pred_depths = depths + pred_motion
        spike_probability = self.activity_network(pred_depths, amplitudes)

        return spike_probability

    def loss(
        self,
        data_real: dict,
        data_fake: dict,
        motion_noise: float = 0.0,
    ) -> torch.Tensor:
        """Compute loss on a batch of real and fake data.

        Args:
            data_real: Dictionary with keys 'times', 'depths', and
                'amplitudes'.
            data_fake: Dictionary with keys 'time16s', 'depths', and
                'amplitudes'.

        Returns:
            loss: Torch scalar. Loss on the batch.
        """
        spike_probability_real = self.forward(
            data_real, motion_noise=motion_noise
        )
        spike_probability_fake = self.forward(
            data_fake, motion_noise=motion_noise
        )

        # Logistic loss to pressure spike_probability_real towards 1 and
        # spike_probability_fake towards 0
        loss = -1 * (
            torch.mean(torch.log(self._epsilon + spike_probability_real))
            + torch.mean(torch.log(self._epsilon + 1 - spike_probability_fake))
        )

        return loss

    def to(self, device: torch.device) -> None:
        """Move model to device."""
        self.motion_function.to(device)
        self.activity_network.to(device)
        super().to(device)


class Trainer:

    def __init__(
        self,
        dataset: Dataset,
        medicine_model: Medicine,
        batch_size: int,
        training_steps: int,
        initial_motion_noise=0.1,
        motion_noise_steps=2000,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        grad_clip=1,
    ):
        """Constructor.

        Args:
            dataset: Dataset object.
            medicine_model: Medicine object.
            bar_size: Int. Batch size.
            training_steps: Int. Number of training steps.
            initial_motion_noise: Float. Initial motion noise.
            motion_noise_steps: Int. Number of steps to decrease motion noise.
            optimizer: Optimizer for training.
            learning_rate: Float. Learning rate for training.
            grad_clip: Float. Gradient clipping value.
        """
        self._dataset = dataset
        self._medicine_model = medicine_model
        self._batch_size = batch_size
        self._training_steps = training_steps
        self._initial_motion_noise = initial_motion_noise
        self._motion_noise_steps = motion_noise_steps
        self._learning_rate = learning_rate
        self._grad_clip = grad_clip

        self._optimizer = optimizer(
            self._medicine_model.parameters(),
            lr=learning_rate,
        )
        self._losses = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._medicine_model.to(self._device)

    def __call__(self):
        """Evaluate model on dataset."""
        logger.info("Fitting motion estimation")

        motion_bound = self._medicine_model.motion_function.bound_normalized
        training_losses = []
        for step in tqdm.tqdm(range(self._training_steps)):

            # Compute motion noise
            remaining = (
                self._motion_noise_steps - step
            ) / self._motion_noise_steps
            motion_noise = max(remaining * self._initial_motion_noise, 0)

            # Sample real and fake data
            self._optimizer.zero_grad()
            data_real = self._dataset.sample_real(batch_size=self._batch_size)
            data_fake = self._dataset.sample_fake(
                batch_size=self._batch_size, motion_bound=motion_bound
            )

            # Move data to GPU if necessary
            data_real = {
                key: value.to(self._device) for key, value in data_real.items()
            }
            data_fake = {
                key: value.to(self._device) for key, value in data_fake.items()
            }

            # Compute loss and backpropagate
            loss = self._medicine_model.loss(
                data_real, data_fake, motion_noise=motion_noise
            )
            loss.backward()

            # Clip gradients and step optimizer
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._medicine_model.parameters(), self._grad_clip
                )
            self._optimizer.step()
            training_losses.append(float(loss.cpu().detach()))

        self._losses = training_losses
        logger.info("Finished fitting motion estimation")

    @property
    def medicine_model(self) -> Medicine:
        return self._medicine_model

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def losses(self) -> list[float]:
        return self._losses

    @property
    def device(self) -> torch.device:
        return self._device
