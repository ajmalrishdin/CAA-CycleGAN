"""
ICA-SVD based fECG Extraction Module
Classical, non-ML approach for extracting fetal ECG from abdominal recordings.

Method:
1. FastICA to separate maternal ECG component
2. SVD-based enhancement of fetal component
"""

import numpy as np
import wfdb
from sklearn.decomposition import FastICA
from scipy.linalg import svd
from scipy.signal import butter, filtfilt, resample
import os

# Default database path (can be overridden)
DEFAULT_NIFEA_PATH = "./Databases/non-invasive-fetal-ecg-arrhythmia-database-1.0.0"


def highpass_filter(X, fs, cutoff=0.5):
    """Remove baseline wander with high-pass filter."""
    b, a = butter(2, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, X, axis=1)


def run_fastica(X, n_components=4):
    """
    Run FastICA to separate independent components.
    Returns source signals and mixing matrix.
    """
    ica = FastICA(
        n_components=min(n_components, X.shape[0]),
        whiten='unit-variance',
        max_iter=2000,
        random_state=0
    )
    S = ica.fit_transform(X.T)    # (samples, components)
    A = ica.mixing_               # (channels, components)
    return S.T, A


def remove_maternal_ecg(X, S, A):
    """
    Identify and remove maternal ECG component (highest variance).
    Returns cleaned signal with maternal ECG subtracted.
    """
    variances = np.var(S, axis=1)
    maternal_idx = np.argmax(variances)
    
    # Reconstruct maternal ECG in sensor space
    maternal_source = S[maternal_idx][None, :]      # (1, samples)
    maternal_mixing = A[:, maternal_idx][:, None]   # (channels, 1)
    maternal_recon = maternal_mixing @ maternal_source
    
    # Remove maternal ECG
    return X - maternal_recon, maternal_idx


def create_svd_matrix(signal, window_size, step):
    """Create overlapping segments for SVD analysis."""
    segments = []
    for start in range(0, len(signal) - window_size, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments)


def svd_reconstruct(U, S, Vt, k=2):
    """Reconstruct signal using top-k singular values."""
    S_new = np.zeros_like(S)
    S_new[:k] = S[:k]
    return (U * S_new) @ Vt


def overlap_add(segments, signal_length, step):
    """Reconstruct signal from overlapping segments."""
    recon = np.zeros(signal_length)
    weight = np.zeros(signal_length)

    for i, seg in enumerate(segments):
        start = i * step
        end = start + len(seg)
        if end <= signal_length:
            recon[start:end] += seg
            weight[start:end] += 1

    return recon / np.maximum(weight, 1)


def svd_enhance(signal, fs, k=2):
    """
    SVD-based fECG enhancement.
    Uses rank-k approximation to extract dominant periodic component.
    """
    window_size = int(0.6 * fs)   # 600 ms
    step = int(0.05 * fs)         # 50 ms overlap
    
    if len(signal) < window_size:
        return signal
    
    svd_matrix = create_svd_matrix(signal, window_size, step)
    
    if len(svd_matrix) == 0:
        return signal
    
    U, S_vals, Vt = svd(svd_matrix, full_matrices=False)
    
    # Rank-k reconstruction
    svd_segments = svd_reconstruct(U, S_vals, Vt, k=k)
    
    return overlap_add(svd_segments, len(signal), step)


def extract_fecg(record_path, target_fs=200, channel_idx=0):
    """
    Main extraction function.
    
    Args:
        record_path: Path to WFDB record (without extension)
        target_fs: Target sampling frequency for output (default 200Hz)
        channel_idx: Which channel to use for final SVD enhancement (default 0)
    
    Returns:
        fecg: Extracted fetal ECG signal at target_fs
        fs_original: Original sampling frequency
    """
    # Load record
    record = wfdb.rdrecord(record_path)
    X = record.p_signal.T        # shape: (channels, samples)
    fs = record.fs               # typically 1000 Hz
    
    # Handle NaN values
    if np.isnan(X).any():
        channel_medians = np.nanmedian(X, axis=1, keepdims=True)
        X = np.where(np.isnan(X), channel_medians, X)
    
    # 1. High-pass filter to remove baseline wander
    X = highpass_filter(X, fs)
    
    # 2. FastICA to separate maternal component
    S, A = run_fastica(X)
    
    # 3. Remove maternal ECG
    X_clean, maternal_idx = remove_maternal_ecg(X, S, A)
    
    # 4. SVD enhancement on selected channel
    signal_for_svd = X_clean[channel_idx]
    fecg = svd_enhance(signal_for_svd, fs, k=2)
    
    # 5. Resample to target frequency
    if fs != target_fs:
        num_samples = int(len(fecg) * target_fs / fs)
        fecg = resample(fecg, num_samples)
    
    return fecg, fs


def extract_fecg_from_nifea(record_name, db_path=None, target_fs=200):
    """
    Convenience function for NIFEA database records.
    
    Args:
        record_name: Record name (e.g., "ARR_01" or "NR_05")
        db_path: Path to NIFEA database (uses default if None)
        target_fs: Target sampling frequency
    
    Returns:
        fecg: Extracted fetal ECG signal
    """
    if db_path is None:
        db_path = DEFAULT_NIFEA_PATH
    
    record_path = os.path.join(db_path, record_name)
    fecg, _ = extract_fecg(record_path, target_fs=target_fs)
    return fecg


if __name__ == "__main__":
    # Test extraction
    import sys
    
    if len(sys.argv) > 1:
        record_name = sys.argv[1]
    else:
        record_name = "ARR_01"
    
    print(f"Extracting fECG from {record_name}...")
    
    try:
        fecg = extract_fecg_from_nifea(record_name)
        print(f"Extracted signal length: {len(fecg)} samples at 200Hz")
        print(f"Duration: {len(fecg)/200:.1f} seconds")
        
        # Save result
        output_path = f"outputs/{record_name}_fecg_ica.npy"
        os.makedirs("outputs", exist_ok=True)
        np.save(output_path, fecg)
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
