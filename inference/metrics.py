# ============================================================================
# Shared Evaluation Metrics for fECG Extraction Comparison
# ============================================================================
# R-peak detection, F1/Precision/Recall, BPM estimation
# ============================================================================

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_fetal_r_peaks(signal_data, fs, min_bpm=100, max_bpm=240):
    """
    Detect fetal R-peaks using an adaptive prominence threshold on bandpass-filtered signal.
    """
    # Use narrower bandpass tuned for fetal complexes
    nyq = 0.5 * fs
    low = max(5.0 / nyq, 0.01)
    high = min(40.0 / nyq, 0.99)
    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, signal_data)
    
    # Adaptive prominence based on Median Absolute Deviation (MAD)
    mad = np.median(np.abs(filtered - np.median(filtered)))
    prominence = 0.7 * mad
    
    # Physical constraints for fetal heartbeats
    min_distance = int(60.0 / max_bpm * fs)
    
    peaks, _ = find_peaks(filtered, height=None, distance=min_distance, prominence=prominence)
    return peaks


def calculate_f1_precision_recall(detected_peaks, true_peaks, tolerance_ms=50, fs=1000):
    """
    Calculate F1 Score, Precision, and Recall/Sensitivity.

    Args:
        detected_peaks: Array of detected R-peak sample indices
        true_peaks: Array of ground truth R-peak sample indices
        tolerance_ms: Matching tolerance in milliseconds
        fs: Sampling frequency (Hz)

    Returns:
        f1, precision, recall (sensitivity)
    """
    if len(true_peaks) == 0 or len(detected_peaks) == 0:
        return 0.0, 0.0, 0.0

    tolerance_samples = int(tolerance_ms / 1000.0 * fs)
    tp = 0
    matched_truth = set()

    for det_p in detected_peaks:
        distances = np.abs(true_peaks.astype(int) - int(det_p))
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance_samples and closest_idx not in matched_truth:
            tp += 1
            matched_truth.add(closest_idx)

    fp = len(detected_peaks) - tp
    fn = len(true_peaks) - len(matched_truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def calculate_peak_detection_metrics(detected_peaks, true_peaks, tolerance_ms=50, fs=1000):
    """
    Compute peak-level confusion statistics and derived metrics.

    Notes:
        - True negatives are not well-defined in event detection, so specificity is NaN.
        - "Accuracy" is reported as event-level accuracy: TP / (TP + FP + FN).
    """
    detected_peaks = np.asarray(detected_peaks, dtype=int)
    true_peaks = np.asarray(true_peaks, dtype=int)

    if len(true_peaks) == 0 and len(detected_peaks) == 0:
        return {
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'TN': np.nan,
            'Precision': 0.0,
            'Sensitivity': 0.0,
            'Recall': 0.0,
            'Specificity': np.nan,
            'F1': 0.0,
            'Accuracy': 0.0,
        }

    tolerance_samples = int(tolerance_ms / 1000.0 * fs)
    tp = 0
    matched_truth = set()

    for det_p in detected_peaks:
        if len(true_peaks) == 0:
            break
        distances = np.abs(true_peaks - int(det_p))
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] <= tolerance_samples and closest_idx not in matched_truth:
            tp += 1
            matched_truth.add(closest_idx)

    fp = int(len(detected_peaks) - tp)
    fn = int(len(true_peaks) - len(matched_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': np.nan,
        'Precision': precision,
        'Sensitivity': recall,
        'Recall': recall,
        'Specificity': np.nan,
        'F1': f1,
        'Accuracy': accuracy,
    }


def calculate_bpm(signal_data, fs, min_bpm=100, max_bpm=220):
    """
    Detect R-peaks and calculate Beats Per Minute (BPM).

    Args:
        signal_data: 1D fECG signal
        fs: Sampling frequency (Hz)
        min_bpm: Minimum expected fetal heart rate
        max_bpm: Maximum expected fetal heart rate

    Returns:
        avg_bpm, std_bpm, detected_peaks
    """
    peaks = detect_fetal_r_peaks(signal_data, fs, min_bpm, max_bpm)

    if len(peaks) < 2:
        return 0.0, 0.0, peaks

    rr_intervals = np.diff(peaks) / fs
    bpms = 60.0 / rr_intervals

    # Filter out physiologically impossible values
    valid = (bpms >= min_bpm * 0.8) & (bpms <= max_bpm * 1.2)
    if np.sum(valid) < 1:
        return 0.0, 0.0, peaks

    avg_bpm = np.mean(bpms[valid])
    std_bpm = np.std(bpms[valid])

    return avg_bpm, std_bpm, peaks
