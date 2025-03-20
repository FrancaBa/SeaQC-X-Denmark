import numpy as np
import pywt
import pandas as pd
from typing import Tuple, List, Optional, Dict
from scipy import interpolate
from dataclasses import dataclass
#from models import WaveletResult

@dataclass
class WaveletResult:
    """Container for wavelet analysis results"""
    # Wavelet transform coefficients matrix
    coefficients: np.ndarray
    # Array of frequencies for each scale
    frequencies: np.ndarray
    # Time points array
    times: np.ndarray
    # Power spectrum matrix (squared magnitude of coefficients)
    power_spectrum: np.ndarray
    # Dictionary mapping tidal component names to their relative powers
    tidal_components: Dict[str, float]
    # Matrix of confidence intervals for statistical significance
    confidence_intervals: np.ndarray
    # Cone of influence array
    coi: Optional[np.ndarray] = None


class TidalWaveletAnalysis:
    """Class for computing wavelet coefficients optimized for tidal data using PyWavelets"""

    # Define known tidal periods (in hours)
    TIDAL_PERIODS = {
        'M2': 12.42,  # Principal lunar semidiurnal
        'S2': 12.00,  # Principal solar semidiurnal
        'N2': 12.66,  # Larger lunar elliptic
        'K1': 23.93,  # Lunar-solar diurnal
        'O1': 25.82,  # Principal lunar diurnal
        'P1': 24.07,  # Principal solar diurnal
        'M4': 6.21,  # Shallow water overtides
        'MS4': 6.10  # Shallow water compound tide
    }

    def __init__(self,
                 target_sampling_rate_hours: float = 1 / 60,  # 5 minutes in hours
                 min_period: float = 4.0,
                 max_period: float = 64,
                 n_periods: int = 200):
        """Initialize the tidal wavelet analysis.

        Args:
            target_sampling_rate_hours: Desired sampling interval in hours.
                Default is 5 minutes (5/60 hours) to match common tidal gauge data.
                This determines the spacing of the regular grid used for wavelet analysis.
            min_period: Minimum period to analyze in hours
            max_period: Maximum period to analyze in hours
            n_periods: Number of periods to analyze between min and max
        """
        self.target_sampling_rate = target_sampling_rate_hours
        self.min_period = min_period
        self.max_period = max_period
        self.n_periods = n_periods

        # Validate sampling rate
        nyquist_period = 2 * target_sampling_rate_hours
        if min_period < nyquist_period:
            raise ValueError(
                f"Minimum period ({min_period} hours) must be greater than twice the "
                f"sampling rate ({nyquist_period} hours) to satisfy Nyquist criterion"
            )

        # Generate logarithmically spaced periods
        self.periods = np.logspace(
            np.log10(min_period),
            np.log10(max_period),
            n_periods
        )

        # Convert periods to scales for 'cmor1.5-1.0' wavelet
        self.scales = self.periods / (4 * np.pi)

    def _handle_duplicates(self, times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle duplicate timestamps by averaging values at same timestamp.

        Args:
            times: Array of timestamps in hours from start
            values: Array of observation values

        Returns:
            Tuple of (unique_times, averaged_values)
        """
        # Create a DataFrame for easy grouping
        df = pd.DataFrame({'value': values}, index=times)

        # Group by index (times) and take mean of duplicates
        df = df.groupby(level=0).mean()

        return df.index.values, df['value'].values

    def _create_regular_grid(self,
                             times: np.ndarray,
                             values: np.ndarray,
                             max_gap_hours: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create a regular time grid and interpolate values onto it."""
        # Handle duplicates first
        times, values = self._handle_duplicates(times, values)

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        valid_times = times[valid_mask]
        valid_values = values[valid_mask]

        if len(valid_times) < 2:
            raise ValueError("Not enough valid data points for interpolation")

        # Create regular time grid
        start_time = np.min(times)
        end_time = np.max(times)
        regular_times = np.arange(
            start_time,
            end_time + self.target_sampling_rate,
            self.target_sampling_rate
        )

        # Create interpolation function
        f = interpolate.interp1d(
            valid_times,
            valid_values,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate values
        regular_values = f(regular_times)

        # Handle gaps if max_gap_hours is specified
        if max_gap_hours is not None:
            max_points = int(max_gap_hours / self.target_sampling_rate)

            # Find gaps in original data
            gaps = np.diff(valid_times)
            big_gaps = np.where(gaps > max_gap_hours)[0]

            # Mask out interpolated values in big gaps
            for gap_idx in big_gaps:
                gap_start = valid_times[gap_idx]
                gap_end = valid_times[gap_idx + 1]

                # Find corresponding indices in regular grid
                mask = (regular_times > gap_start) & (regular_times < gap_end)
                regular_values[mask] = np.nan

        return regular_times, regular_values

    def _compute_significance(self, power_spectrum: np.ndarray,
                              significance_level: float = 0.95) -> np.ndarray:
        """Compute significance levels for the wavelet spectrum."""
        from scipy import stats

        background = np.mean(power_spectrum, axis=1)
        dof = 2  # degrees of freedom for complex wavelet
        chi2_value = stats.chi2.ppf(significance_level, dof)
        significance = (background[:, np.newaxis] * chi2_value / dof)

        return significance

    def _identify_tidal_components(self,
                                   power_spectrum: np.ndarray,
                                   threshold_factor: float = 0.1) -> Dict[str, float]:
        """Identify known tidal components in the wavelet spectrum."""
        mean_power = np.mean(power_spectrum, axis=1)
        max_power = np.max(mean_power)
        threshold = max_power * threshold_factor

        components = {}
        for name, period in self.TIDAL_PERIODS.items():
            idx = np.abs(self.periods - period).argmin()
            if mean_power[idx] > threshold:
                components[name] = mean_power[idx] / max_power

        return components

    def _handle_time_index(self, data: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame index to hours from start."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # Convert to hours from start
        start_time = data.index.min()
        times = (data.index - start_time).total_seconds() / 3600
        return times

    def _cone_of_influence(self, regular_times):
        """
        :param regular_times:
        :return:cone of influence in hours
        """
        # Calculate cone of influence manually
        dt = pd.Timedelta(self.target_sampling_rate, unit='hours').total_seconds()

        # For complex Morlet wavelet, COI is sqrt(2) * period at each time
        n = len(regular_times)
        coi = np.zeros(n)

        # COI increases linearly from edges
        for i in range(n):
            edge_dist = min(i, n - 1 - i)
            # Convert scale to period and apply sqrt(2) factor
            coi[i] = np.sqrt(2) * edge_dist * dt

        return coi

    def analyze(self,
                data: pd.DataFrame,
                column: str = 'tide_level',
                detrend: bool = True,
                significance_level: float = 0.95,
                max_gap_hours: Optional[float] = None) -> WaveletResult:
        """Perform wavelet analysis optimized for irregular tidal data."""
        # Create regular grid and interpolate
        original_times = self._handle_time_index(data)
        values = data[column].values

        # Create a mask for gaps
        gap_mask = np.isnan(values)

        # Create regular grid and interpolate
        regular_times, regular_values = self._create_regular_grid(
            original_times, values, max_gap_hours
        )

        # Create a mask for the regular grid
        regular_gap_mask = np.isnan(regular_values)

        # Replace NaN with zeros for wavelet transform
        regular_values = np.nan_to_num(regular_values, 0)

        if detrend:
            from scipy import signal as sp_signal
            regular_values = sp_signal.detrend(regular_values)

        # Remove mean and normalize
        regular_values = regular_values - np.mean(regular_values)
        regular_values = regular_values / np.std(regular_values)

        # Compute continuous wavelet transform
        wavelet = 'cmor1.5-1.0'
        coefficients, freqs = pywt.cwt(regular_values, self.scales, wavelet)

        # Calculate cone of influence
        period_coi = self._cone_of_influence(regular_times)

        # Apply gap mask to coefficients
        # Extend the gap mask to match the coefficient dimensions
        extended_mask = np.tile(regular_gap_mask, (coefficients.shape[0], 1))
        coefficients[extended_mask] = np.nan

        # Compute power spectrum
        power_spectrum = np.abs(coefficients) ** 2

        # Compute significance levels
        confidence = self._compute_significance(power_spectrum, significance_level)

        # Identify tidal components
        components = self._identify_tidal_components(power_spectrum)

        # Convert scales to frequencies
        frequencies = 1 / self.periods

        return WaveletResult(
            coefficients=coefficients,
            frequencies=frequencies,
            times=regular_times,
            power_spectrum=power_spectrum,
            tidal_components=components,
            confidence_intervals=confidence,
            coi=period_coi
        )
