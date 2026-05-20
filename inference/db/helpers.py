import numpy as np
import pyedflib
import wfdb
from scipy.signal import butter, filtfilt, resample
import os


def _bandpass(signal_data, fs, low=1.0, high=100.0, order=3):
    """Apply bandpass filter to signal(s). Works on 1D or 2D (channels × samples)."""
    nyq = 0.5 * fs
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype='band')
    if signal_data.ndim == 1:
        return filtfilt(b, a, signal_data)
    return np.array([filtfilt(b, a, ch) for ch in signal_data])


def _standardise(signal_data):
    """Z‑score standardise per channel."""
    if signal_data.ndim == 1:
        s = np.std(signal_data)
        if s < 1e-10:
            return signal_data
        return (signal_data - np.mean(signal_data)) / s
    out = np.zeros_like(signal_data)
    for i in range(signal_data.shape[0]):
        s = np.std(signal_data[i])
        if s < 1e-10:
            out[i] = signal_data[i]
        else:
            out[i] = (signal_data[i] - np.mean(signal_data[i])) / s
    return out
