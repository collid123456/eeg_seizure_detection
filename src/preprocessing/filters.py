"""Filtering functions for EEG signals."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=173.61, order=4):
    """Apply a zero-phase Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, freq=50.0, fs=173.61, quality=30.0):
    """Apply a notch filter to remove mains interference (50 Hz in EU/Ireland)."""
    nyquist = 0.5 * fs
    freq_norm = freq / nyquist
    b, a = iirnotch(freq_norm, quality)
    return filtfilt(b, a, signal)


def preprocess_signal(signal, fs=173.61):
    """Apply band-pass then notch filtering to a raw signal."""
    filtered = bandpass_filter(signal, fs=fs)
    filtered = notch_filter(filtered, fs=fs)
    return filtered


if __name__ == "__main__":
    # Quick sanity check with a dummy signal
    dummy_signal = np.random.randn(4097)
    result = preprocess_signal(dummy_signal)
    print(f"Input shape: {dummy_signal.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output looks sane: {not np.isnan(result).any()}")
