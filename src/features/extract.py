"""Feature extraction for EEG signals."""

import numpy as np


def line_length(signal):
    """Sum of absolute differences between consecutive samples."""
    return np.sum(np.abs(np.diff(signal)))


def rms(signal):
    """Root mean square amplitude."""
    return np.sqrt(np.mean(signal**2))


def variance(signal):
    """Signal variance."""
    return np.var(signal)


def zero_crossing_rate(signal):
    """Number of times the signal crosses zero."""
    return np.sum(np.diff(np.sign(signal)) != 0)


def band_power(signal, fs, low, high):
    """Power in a given frequency band using FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_vals = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_vals) ** 2
    band_mask = (freqs >= low) & (freqs <= high)
    return np.sum(power_spectrum[band_mask])


def extract_features(signal, fs=173.61):
    """Extract a feature dict from a single EEG signal."""
    return {
        "line_length": line_length(signal),
        "rms": rms(signal),
        "variance": variance(signal),
        "zero_crossing_rate": zero_crossing_rate(signal),
        "delta_power": band_power(signal, fs, 0.5, 4),
        "theta_power": band_power(signal, fs, 4, 8),
        "alpha_power": band_power(signal, fs, 8, 13),
        "beta_power": band_power(signal, fs, 13, 30),
        "gamma_power": band_power(signal, fs, 30, 40),
    }


if __name__ == "__main__":
    dummy_signal = np.random.randn(4097)
    features = extract_features(dummy_signal)
    for name, value in features.items():
        print(f"{name}: {value:.2f}")
