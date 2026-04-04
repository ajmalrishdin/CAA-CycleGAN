"""
Generate QRS annotation files for EDF records.
Detects R-peaks in fetal ECG signals and creates .qrs files for use with WFDB.
"""

import os
import numpy as np
import pyedflib
import wfdb
from scipy import signal
from scipy.signal import find_peaks
from sklearn.preprocessing import scale


def get_edf_files(db_path):
    """Get all EDF files from the database directory."""
    return sorted(
        [f for f in os.listdir(db_path) if f.lower().endswith(".edf")]
    )


def extract_fetal_ecg(edf_file_path):
    """Extract fetal ECG signal from EDF file."""
    f = pyedflib.EdfReader(edf_file_path)
    n = f.signals_in_file
    
    # Fetal ECG is the first signal
    fetal_ecg = f.readSignal(0)
    
    f.close()
    return fetal_ecg


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply bandpass filter to signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y


def suppress_abnormal_amplitudes(sig, z_thresh=8.0):
    """Suppress extreme amplitude outliers using robust z-score and interpolation."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size == 0:
        return sig

    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad == 0:
        return sig

    robust_z = 0.6745 * (sig - med) / mad
    outlier_mask = np.abs(robust_z) > z_thresh
    if not np.any(outlier_mask):
        return sig

    valid_idx = np.where(~outlier_mask)[0]
    outlier_idx = np.where(outlier_mask)[0]

    if valid_idx.size < 2:
        return np.clip(sig, med - 6 * mad, med + 6 * mad)

    cleaned = sig.copy()
    cleaned[outlier_idx] = np.interp(outlier_idx, valid_idx, sig[valid_idx])
    return cleaned


def detect_qrs_peaks(fecg, fs=1000, prominence_factor=0.3):
    """
    Detect QRS peaks in fetal ECG signal.
    
    Args:
        fecg: Fetal ECG signal
        fs: Sampling frequency (default 1000 Hz)
        prominence_factor: Peak prominence threshold as fraction of signal range
    
    Returns:
        Array of sample indices where QRS peaks occur
    """
    # Apply bandpass filter to enhance QRS complexes (1-100 Hz)
    filtered = butter_bandpass_filter(fecg, 1, 100, fs)

    # Remove large transient spikes that can hide true QRS peaks.
    filtered = suppress_abnormal_amplitudes(filtered)
    
    # Normalize
    filtered = scale(filtered)
    
    # Calculate prominence threshold
    signal_range = np.max(filtered) - np.min(filtered)
    prominence = signal_range * prominence_factor
    
    # Find peaks with adaptive parameters
    # Min distance between peaks: ~0.4s at 1000 Hz = 400 samples
    min_distance = int(0.4 * fs)
    
    peaks, properties = find_peaks(
        filtered,
        prominence=prominence,
        distance=min_distance,
        height=0
    )

    return peaks.astype(np.int64)


def create_qrs_file(edf_file_path, db_path, output_db_path=None):
    """
    Generate QRS annotation file for an EDF record.
    
    Args:
        edf_file_path: Path to the EDF file
        db_path: Path to the database directory
        output_db_path: Optional output path (defaults to db_path)
    """
    if output_db_path is None:
        output_db_path = db_path
    
    file_base = os.path.splitext(os.path.basename(edf_file_path))[0]
    
    print(f"Processing {file_base}...", end=" ", flush=True)
    
    try:
        # Extract fetal ECG
        fecg = extract_fetal_ecg(edf_file_path)
        
        # Detect QRS peaks
        peaks = detect_qrs_peaks(fecg, fs=1000)
        
        if len(peaks) == 0:
            print("WARNING: No peaks detected")
            return False

        output_annotation_path = os.path.join(output_db_path, f"{file_base}.edf.qrs")
        if os.path.exists(output_annotation_path):
            print("SKIPPED: .edf.qrs already exists")
            return True
        
        # Write annotation file
        wfdb.wrann(
            record_name=f"{file_base}",
            extension='qrs',
            sample=peaks,
            symbol=['N'] * len(peaks),  # 'N' = normal heartbeat
            aux_note=[''] * len(peaks),
            write_dir=output_db_path
        )
        
        print(f"✓ {len(peaks)} peaks detected")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Main function to generate QRS files for all EDFs in database."""
    # Database path
    db_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "CAA-CycleGAN",
            "Databases",
            "ADFECGDB",
        )
    )
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database path not found: {db_path}")
        return
    
    print(f"Generating QRS annotations for: {db_path}\n")
    
    edf_files = get_edf_files(db_path)
    
    if not edf_files:
        print("ERROR: No EDF files found in database")
        return
    
    print(f"Found {len(edf_files)} EDF file(s):\n")
    
    successful = 0
    failed = 0
    
    for edf_file in edf_files:
        edf_path = os.path.join(db_path, edf_file)
        if create_qrs_file(edf_path, db_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
