"""
Arrhythmia Detection from fECG signals
Supports both CycleGAN and ICA-SVD extraction methods

IMPROVED: Uses HRV-based irregularity detection, not just heart rate thresholds
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import glob

# Import ICA extraction module
try:
    from extract_fecg_ica import extract_fecg_from_nifea, DEFAULT_NIFEA_PATH
    ICA_AVAILABLE = True
except ImportError:
    ICA_AVAILABLE = False
    print("Warning: extract_fecg_ica module not found. ICA extraction disabled.")


def preprocess_fecg(fecg_signal, fs=200):
    """Savitzky-Golay filter for smoothing and detrending."""
    detrended_signal = signal.detrend(fecg_signal)
    
    window_length = min(21, len(detrended_signal) - 1)  # Increased window
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 5:
        return detrended_signal
    
    return signal.savgol_filter(detrended_signal, window_length=window_length, polyorder=3)


def detect_r_peaks_adaptive(cleaned_signal, fs=200):
    """
    Improved Adaptive R-Peak Detection.
    Uses lower threshold and wider frequency band for better sensitivity.
    """
    nyq = fs / 2
    
    # Wider bandpass (3-25 Hz) to capture more of QRS complex
    low = 3 / nyq
    high = min(25 / nyq, 0.99)
    b, a = signal.butter(2, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, cleaned_signal)
    
    # Square and integrate
    squared = filtered ** 2
    integration_window = int(0.10 * fs)  # 100ms window (reduced)
    integrated = np.convolve(squared, np.ones(integration_window)/integration_window, mode='same')
    
    # Lower threshold for better sensitivity
    threshold = np.mean(integrated) + 0.15 * np.std(integrated)
    
    # Minimum distance: 250ms (fetal HR up to 240 BPM max)
    min_distance = int(0.25 * fs)
    
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
    
    # Refine peaks in original signal
    refined_peaks = []
    search_window = int(0.04 * fs)  # 40ms window
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(cleaned_signal), peak + search_window)
        # Find max absolute value (handles inverted R-peaks)
        local_max_idx = start + np.argmax(np.abs(cleaned_signal[start:end]))
        refined_peaks.append(local_max_idx)
    
    return np.array(refined_peaks)


def analyze_heart_metrics(peaks, fs=200):
    """
    Calculate comprehensive HRV metrics including irregularity indicators.
    """
    if len(peaks) < 3:
        return None

    rr_intervals_samples = np.diff(peaks)
    rr_intervals_sec = rr_intervals_samples / fs
    rr_intervals_ms = rr_intervals_sec * 1000
    
    # Wider physiological range (fetal HR 80-220 BPM)
    valid_mask = (rr_intervals_sec > 0.27) & (rr_intervals_sec < 0.75)
    valid_rr_sec = rr_intervals_sec[valid_mask]
    valid_rr_ms = valid_rr_sec * 1000
    
    if len(valid_rr_sec) < 3:
        return None
    
    # BPM
    bpm_instantaneous = 60 / valid_rr_sec
    avg_bpm = np.mean(bpm_instantaneous)
    bpm_std = np.std(bpm_instantaneous)
    
    # SDNN: Standard Deviation of NN intervals
    sdnn = np.std(valid_rr_ms)
    
    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(valid_rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0
    
    # pNN50: Percentage of successive RR differences > 50ms
    # HIGH pNN50 = irregular rhythm = possible arrhythmia
    pnn50 = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr) if len(diff_rr) > 0 else 0
    
    # Coefficient of Variation of RR intervals
    cv_rr = (np.std(valid_rr_ms) / np.mean(valid_rr_ms)) * 100
    
    return {
        "avg_bpm": avg_bpm,
        "bpm_std": bpm_std,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "cv_rr": cv_rr,
        "rr_intervals": valid_rr_sec,
        "num_valid_beats": len(valid_rr_sec) + 1
    }


def diagnose_arrhythmia(metrics):
    """
    Multi-criteria classification with severity scoring.
    Thresholds tuned for fetal ECG from NIFEA database.
    """
    if metrics is None:
        return "Insufficient Data", "unknown"
    
    bpm = metrics["avg_bpm"]
    pnn50 = metrics["pnn50"]
    cv_rr = metrics["cv_rr"]
    rmssd = metrics["rmssd_ms"]
    
    abnormalities = []
    severity = 0
    
    # 1. Rate-based (strong indicators)
    if bpm < 100:
        abnormalities.append("Bradycardia")
        severity += 3
    elif bpm > 175:
        abnormalities.append("Tachycardia")
        severity += 3
    elif bpm < 110 or bpm > 165:
        severity += 1  # Borderline
    
    # 2. Rhythm irregularity (pNN50 > 55% is abnormal for fetal ECG)
    if pnn50 > 65:
        abnormalities.append("Irregular Rhythm")
        severity += 2
    elif pnn50 > 55:
        severity += 1
    
    # 3. High CV indicates variable rhythm
    if cv_rr > 32:
        abnormalities.append("High Variability")
        severity += 1
    
    # 4. Very high RMSSD indicates ectopic activity
    if rmssd > 200:
        abnormalities.append("Ectopic Activity")
        severity += 2
    
    # Classification
    if severity >= 3:
        label = f"Arrhythmia ({', '.join(abnormalities[:2])})" if abnormalities else "Arrhythmia"
        return label, "abnormal"
    elif severity >= 2 and len(abnormalities) > 0:
        return f"Abnormal ({abnormalities[0]})", "abnormal"
    else:
        return "Normal Sinus Rhythm", "normal"


def analyze_signal(fecg_signal, fs=200):
    """Analyze a single fECG signal."""
    if fecg_signal.ndim > 1:
        fecg_signal = fecg_signal.flatten()
    
    cleaned = preprocess_fecg(fecg_signal, fs)
    peaks = detect_r_peaks_adaptive(cleaned, fs)
    metrics = analyze_heart_metrics(peaks, fs)
    diagnosis, category = diagnose_arrhythmia(metrics)
    
    return {
        "metrics": metrics,
        "diagnosis": diagnosis,
        "category": category,
        "peaks": peaks,
        "cleaned_signal": cleaned
    }


def batch_analyze_nifea(method="ica", db_path=None, gan_results_path="outputs/nifeadb_results"):
    """
    Batch analyze NIFEA database using specified extraction method.
    """
    fs = 200
    
    print("=" * 80)
    print(f"NI-FECG Holter Analysis Report (Method: {method.upper()})")
    print("=" * 80)
    print(f"{'File':<15} {'BPM':>6} {'SDNN':>6} {'RMSSD':>6} {'pNN50':>6} {'CV%':>5} {'Diagnosis':<30}")
    print("-" * 80)
    
    correct_nr, correct_arr = 0, 0
    total_nr, total_arr = 0, 0
    
    if method == "ica":
        if not ICA_AVAILABLE:
            print("Error: ICA extraction module not available.")
            return
        
        if db_path is None:
            db_path = DEFAULT_NIFEA_PATH
        
        # Get all records
        records_file = os.path.join(db_path, "RECORDS")
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                record_names = [line.strip() for line in f if line.strip()]
        else:
            record_names = [os.path.splitext(os.path.basename(f))[0] 
                          for f in glob.glob(os.path.join(db_path, "*.hea"))]
        
        for record_name in sorted(record_names):
            try:
                fecg = extract_fecg_from_nifea(record_name, db_path=db_path, target_fs=fs)
                result = analyze_signal(fecg, fs)
                
                _print_result(record_name, result)
                
                # Track accuracy
                is_normal_gt = record_name.startswith("NR_")
                is_arr_gt = record_name.startswith("ARR_")
                
                if is_normal_gt:
                    total_nr += 1
                    if result["category"] == "normal":
                        correct_nr += 1
                elif is_arr_gt:
                    total_arr += 1
                    if result["category"] == "abnormal":
                        correct_arr += 1
                        
            except Exception as e:
                print(f"{record_name:<15} Error: {str(e)[:50]}")
    
    else:  # GAN method
        files = sorted(glob.glob(os.path.join(gan_results_path, "*.npy")))
        
        for filepath in files:
            try:
                fecg = np.load(filepath)
                result = analyze_signal(fecg, fs)
                
                basename = os.path.basename(filepath).replace("_fecg.npy", "")
                _print_result(basename, result)
                
                # Track accuracy
                is_normal_gt = basename.startswith("NR_")
                is_arr_gt = basename.startswith("ARR_")
                
                if is_normal_gt:
                    total_nr += 1
                    if result["category"] == "normal":
                        correct_nr += 1
                elif is_arr_gt:
                    total_arr += 1
                    if result["category"] == "abnormal":
                        correct_arr += 1
                        
            except Exception as e:
                print(f"{os.path.basename(filepath):<15} Error: {e}")
    
    # Print summary
    print("-" * 80)
    if total_nr > 0:
        print(f"Normal (NR) Accuracy:     {correct_nr}/{total_nr} ({100*correct_nr/total_nr:.1f}%)")
    if total_arr > 0:
        print(f"Arrhythmia (ARR) Accuracy: {correct_arr}/{total_arr} ({100*correct_arr/total_arr:.1f}%)")
    if total_nr > 0 and total_arr > 0:
        total_correct = correct_nr + correct_arr
        total = total_nr + total_arr
        print(f"Overall Accuracy:          {total_correct}/{total} ({100*total_correct/total:.1f}%)")


def _print_result(name, result):
    m = result["metrics"]
    if m:
        print(f"{name:<15} {m['avg_bpm']:>6.1f} {m['sdnn_ms']:>6.1f} {m['rmssd_ms']:>6.1f} "
              f"{m['pnn50']:>6.1f} {m['cv_rr']:>5.1f} {result['diagnosis']:<30}")
    else:
        print(f"{name:<15} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>5} {result['diagnosis']:<30}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arrhythmia Detection from fECG")
    parser.add_argument("--method", choices=["ica", "gan"], default="ica",
                       help="Extraction method: 'ica' (ICA-SVD) or 'gan' (CycleGAN)")
    parser.add_argument("--db_path", type=str, default=None,
                       help="Path to NIFEA database (for ICA method)")
    parser.add_argument("--gan_path", type=str, default="outputs/nifeadb_results",
                       help="Path to GAN results (for GAN method)")
    
    args = parser.parse_args()
    
    batch_analyze_nifea(method=args.method, db_path=args.db_path, 
                        gan_results_path=args.gan_path)