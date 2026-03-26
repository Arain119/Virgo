"""Online normalization utilities for streaming observations.

Used by trading environments to normalize features without lookahead bias.
"""

import numpy as np


class OnlineNormalizer:
    """
    Enhanced online normalizer with exponential moving averages and outlier handling.
    Implements Welford's online algorithm with additional stability improvements.
    """

    def __init__(self, shape, momentum=0.01, clip_range=5.0):
        """
        Initializes the normalizer.
        Args:
            shape: The shape of the features to be normalized.
            momentum: EMA momentum for smoother statistics (0.01-0.1)
            clip_range: Range for outlier clipping (typically 3-5)
        """
        self.shape = shape
        self.momentum = momentum
        self.clip_range = clip_range
        self.count = 0

        # Welford's algorithm variables
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)

        # Exponential moving average variables for smoother statistics
        self.ema_mean = np.zeros(shape)
        self.ema_var = np.ones(shape)

        # Track initialization state
        self.is_initialized = False

    def update(self, x: np.ndarray):
        """
        Updates the running mean and variance with a new data point.
        Args:
            x: A new data point (must match the shape).
        """
        # Handle potential NaN or infinite values
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        self.count += 1

        # Welford's online algorithm for precise mean and variance
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

        # Exponential moving averages for smoother statistics
        if not self.is_initialized:
            self.ema_mean = x.copy()
            self.ema_var = np.ones(self.shape)
            self.is_initialized = True
        else:
            # Update EMA mean
            self.ema_mean = (1 - self.momentum) * self.ema_mean + self.momentum * x

            # Update EMA variance
            squared_diff = (x - self.ema_mean) ** 2
            self.ema_var = (1 - self.momentum) * self.ema_var + self.momentum * squared_diff

    @property
    def variance(self) -> np.ndarray:
        """Calculates the current variance using Welford's algorithm."""
        if self.count > 1:
            return self.M2 / (self.count - 1)  # Sample variance
        else:
            return np.ones(self.shape)

    @property
    def std(self) -> np.ndarray:
        """Calculates the current standard deviation."""
        return np.sqrt(self.variance)

    @property
    def ema_std(self) -> np.ndarray:
        """Calculates EMA-based standard deviation for smoother normalization."""
        return np.sqrt(np.maximum(self.ema_var, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes a data point using enhanced statistics with outlier handling.
        Args:
            x: The data point to normalize.
        Returns:
            The normalized and clipped data point.
        """
        # Handle potential NaN or infinite values
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if not self.is_initialized:
            return x  # Return unchanged if not initialized

        # Use EMA statistics for smoother normalization
        # Fall back to Welford statistics if EMA is not stable
        if self.count > 10:  # Use EMA after sufficient samples
            mean_to_use = self.ema_mean
            std_to_use = self.ema_std
        else:
            mean_to_use = self.mean
            std_to_use = np.sqrt(np.maximum(self.variance, 1e-8))

        # Normalize
        normalized = (x - mean_to_use) / std_to_use

        # Clip outliers to prevent extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def normalize_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized normalization for a batch of observations.

        Args:
            x: Array with trailing dimension matching `self.shape` (e.g. (T, D)).

        Returns:
            Normalized array with the same shape as input.
        """
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if not self.is_initialized:
            return x

        if self.count > 10:
            mean_to_use = self.ema_mean
            std_to_use = self.ema_std
        else:
            mean_to_use = self.mean
            std_to_use = np.sqrt(np.maximum(self.variance, 1e-8))

        normalized = (x - mean_to_use) / std_to_use
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def get_stats(self) -> dict:
        """Returns current normalization statistics for debugging."""
        return {
            "count": self.count,
            "mean": self.mean.copy(),
            "std": self.std.copy(),
            "ema_mean": self.ema_mean.copy(),
            "ema_std": self.ema_std.copy(),
            "is_initialized": self.is_initialized,
        }
