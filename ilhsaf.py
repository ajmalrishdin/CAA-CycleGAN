import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import wfdb
import os
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import scale

LOCAL_DATABASE_PATH = 'Databases/ADFECGDB'
OUTPUT_DIR = 'Output_ilhsaf'


class ILHSAF:
    """Improved Logarithmic Hyperbolic Secant Adaptive Filter"""
    def __init__(self, L, mu, H=0.1):
        self.w = np.zeros(L)
        self.mu = mu
        self.factor = 0.35 / (H + 1e-8)
    
    def update(self, x_vec, error):
        val = self.factor * error
        if abs(val) > 50:
            psi = np.sign(val)
        else:
            sech_val = 1.0 / np.cosh(val)
            tanh_val = np.tanh(val)
            psi = (tanh_val * sech_val) / (1.0 + sech_val)
        self.w = self.w + self.mu * psi * x_vec

class LMS:
    """Standard LMS"""
    def __init__(self, L, mu):
        self.w = np.zeros(L)
        self.mu = mu
    
    def update(self, x_vec, error):
        self.w = self.w + self.mu * error * x_vec

# ============================================================================
# 3. METRICS & HEART RATE CALCULATION
# ============================================================================
def calculate_bpm(signal_data, fs, algo_name="Signal"):
    """
    Detects R-peaks and calculates Beats Per Minute (BPM).
    Tuned for Fetal ECG (Expected ~140 BPM).
    """
    enhanced = signal_data ** 2
    threshold = np.mean(enhanced) * 1.5
    peaks, _ = find_peaks(enhanced, distance=int(fs*0.3), height=threshold)
    
    if len(peaks) < 2:
        return 0, 0, peaks
        
    rr_intervals = np.diff(peaks) / fs
    bpms = 60.0 / rr_intervals
    avg_bpm = np.mean(bpms)
    std_bpm = np.std(bpms)
    
    return avg_bpm, std_bpm, peaks

def calculate_f1_score(detected_peaks, true_peaks, tolerance_ms=50, fs=1000):
    """
    Calculates F1 Score, Precision, Recall/Sensitivity given
    detected peaks and ground truth R-peak locations.
    
    Args:
        detected_peaks: Array of detected peak sample indices
        true_peaks: Array of ground truth peak sample indices
        tolerance_ms: Matching tolerance in milliseconds
        fs: Sampling frequency in Hz
    
    Returns:
        f1, precision, recall (sensitivity)
    """
    if len(true_peaks) == 0 or len(detected_peaks) == 0:
        return 0.0, 0.0, 0.0
        
    tolerance = int(tolerance_ms / 1000 * fs)
    tp = 0
    fp = 0
    
    matched_truth = set()
    for det_p in detected_peaks:
        match_found = False
        for i, true_p in enumerate(true_peaks):
            if abs(det_p - true_p) <= tolerance:
                tp += 1
                matched_truth.add(i)
                match_found = True
                break
        if not match_found:
            fp += 1
            
    fn = len(true_peaks) - len(matched_truth)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall

# ============================================================================
# 4. DATA LOADING & PREPROCESSING (ADFECGDB - EDF format)
# ============================================================================
def get_real_data_edf(record_name, folder_path, abdominal_ch=1):
    """
    Load an ADFECGDB EDF record and its QRS annotations.
    
    Channel layout for ADFECGDB:
        0: Direct_1   -> direct fetal ECG (ground truth reference)
        1: Abdomen_1  -> abdominal signal
        2: Abdomen_2
        3: Abdomen_3
        4: Abdomen_4
    
    Args:
        record_name: Record filename (e.g. 'r01.edf')
        folder_path: Path to ADFECGDB directory
        abdominal_ch: Which abdominal channel to use (1-4, default 1)
    
    Returns:
        aecg: Abdominal ECG signal
        direct_fecg: Direct fetal ECG (ground truth signal)
        fs: Sampling frequency
        fqrs_rpeaks: Ground truth fetal QRS R-peak sample indices
    """
    file_path = os.path.join(folder_path, record_name)
    try:
        # Read EDF file
        f = pyedflib.EdfReader(file_path)
        n_channels = f.signals_in_file
        fs = f.getSampleFrequency(0)
        n_samples = f.getNSamples()[0]
        
        # Read channels
        direct_fecg = f.readSignal(0)          # Channel 0: Direct fetal ECG
        aecg = f.readSignal(abdominal_ch)      # Channel 1-4: Abdominal ECG
        f.close()
        
        # Read QRS annotations
        ann_path = os.path.join(folder_path, record_name)
        annotation = wfdb.rdann(ann_path, 'qrs')
        fqrs_rpeaks = annotation.sample  # sample indices at original fs (1000 Hz)
        
        return aecg, direct_fecg, int(fs), fqrs_rpeaks
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def preprocess_signal(signal_data, fs):
    if np.std(signal_data) == 0: return signal_data
    sig_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    nyquist = 0.5 * fs
    b, a = butter(3, [1.0/nyquist, 60.0/nyquist], btype='band')
    return filtfilt(b, a, sig_norm)

def extract_maternal_reference(signal_data, fs):
    """
    Constructs a synthetic maternal reference signal from the abdominal ECG.
    Detects large Maternal QRS peaks and replaces them with an average beat template.
    """
    # 1. Detect Maternal Peaks (they are dominant in abdominal signal)
    # Use squared signal + moving average or simple threshold
    enhanced = signal_data ** 2
    # Maternal peaks are usually much larger than fetal
    threshold = np.mean(enhanced) * 3.0  # Conservative high threshold
    # Distance: Maternal HR ~60-100 BPM => ~600-1000ms. Fetal ~140 BPM => ~400ms.
    # Set distance to ~500ms (500 samples at 1000Hz) to skip fetal beats if possible
    # but still catch valid maternal beats.
    peaks, _ = find_peaks(enhanced, distance=int(fs*0.5), height=threshold)
    
    if len(peaks) < 5:
        # Fallback: if very few peaks found, return original signal as reference
        # (better than nothing, though not ideal)
        return signal_data
        
    # 2. Create Average Template
    # Window size: 100ms before, 150ms after? (Total ~250ms)
    window_pre = int(0.10 * fs)
    window_post = int(0.15 * fs)
    
    templates = []
    for p in peaks:
        if p - window_pre >= 0 and p + window_post < len(signal_data):
            templates.append(signal_data[p - window_pre : p + window_post])
            
    if not templates:
        return signal_data
        
    avg_template = np.mean(templates, axis=0)
    
    # 3. Construct Synthetic Reference
    synth_ref = np.zeros_like(signal_data)
    for p in peaks:
        start = p - window_pre
        end = p + window_post
        if start >= 0 and end < len(signal_data):
            synth_ref[start:end] += avg_template
            
    return synth_ref

def run_ilhsaf(record_name):
    """
    Run ILHSAF adaptive filter on one ADFECGDB record.
    Returns a dict of metrics, or None on failure.
    """
    # Load Data
    aecg_raw, direct_fecg_raw, fs, fqrs_rpeaks = get_real_data_edf(record_name, LOCAL_DATABASE_PATH)
    
    if aecg_raw is None:
        return None

    # Preprocess Inputs
    d_n = preprocess_signal(aecg_raw, fs)        # abdominal (desired = M + F + N)
    
    # --- KEY CHANGE: Generate Synthetic Maternal Reference ---
    # We want x_ref to be the MATERNAL signal, so the filter estimates M and cancels it from d_n.
    x_ref = extract_maternal_reference(d_n, fs) 
    x_ref = preprocess_signal(x_ref, fs) # Normalize it too
    
    N = len(d_n)
    L = 25
    
    # 1. LMS
    lms = LMS(L, 0.005)
    e_lms = np.zeros(N)
    for n in range(L, N):
        x_vec = x_ref[n:n-L:-1]
        y = np.dot(lms.w, x_vec)
        e_lms[n] = d_n[n] - y  # Error = Abdominal - Est_Maternal ~= Fetal
        lms.update(x_vec, e_lms[n])
        
    # 2. ILHSAF
    ilhsaf = ILHSAF(L, 0.005, H=0.5)
    e_ilhsaf = np.zeros(N)
    for n in range(L, N):
        x_vec = x_ref[n:n-L:-1]
        y = np.dot(ilhsaf.w, x_vec)
        e_ilhsaf[n] = d_n[n] - y
        ilhsaf.update(x_vec, e_ilhsaf[n])
        
    # --- METRICS CALCULATION ---
    
    # Calculate BPM for both
    bpm_lms, std_lms, peaks_lms = calculate_bpm(e_lms, fs, "LMS")
    bpm_ilhsaf, std_ilhsaf, peaks_ilhsaf = calculate_bpm(e_ilhsaf, fs, "ILHSAF")
    
    # Calculate F1, Precision, Recall/Sensitivity against ground truth
    f1_lms, prec_lms, recall_lms = calculate_f1_score(peaks_lms, fqrs_rpeaks, tolerance_ms=50, fs=fs)
    f1_ilhsaf, prec_ilhsaf, recall_ilhsaf = calculate_f1_score(peaks_ilhsaf, fqrs_rpeaks, tolerance_ms=50, fs=fs)
    
    # Save output wave
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rec_basename = record_name.replace('.edf', '')
    data_to_save = {
        'signal': e_ilhsaf,
        'reference_used': x_ref,
        'abdomen_input': d_n,
        'fs': fs,
        'record_name': record_name,
        'detected_peaks': peaks_ilhsaf,
        'gt_peaks': fqrs_rpeaks,
    }
    np.save(os.path.join(OUTPUT_DIR, f"{rec_basename}.npy"), data_to_save)
    
    # Print per-record summary
    print(f"  {record_name:<10} "
          f"F1={f1_ilhsaf:.4f}  Prec={prec_ilhsaf:.4f}  "
          f"Recall={recall_ilhsaf:.4f}  "
          f"BPM={bpm_ilhsaf:.1f}  "
          f"Det={len(peaks_ilhsaf)}  GT={len(fqrs_rpeaks)}")
    
    return {
        'Record': record_name,
        'F1_LMS': f1_lms,
        'Precision_LMS': prec_lms,
        'Recall_LMS': recall_lms,
        'BPM_LMS': bpm_lms,
        'F1_ILHSAF': f1_ilhsaf,
        'Precision_ILHSAF': prec_ilhsaf,
        'Recall_ILHSAF': recall_ilhsaf,
        'Sensitivity_ILHSAF': recall_ilhsaf,  # sensitivity = recall
        'BPM_ILHSAF': bpm_ilhsaf,
        'Detected_Peaks': len(peaks_ilhsaf),
        'GT_Peaks': len(fqrs_rpeaks),
    }


# ============================================================================
# MAIN - Process all ADFECGDB records
# ============================================================================
if __name__ == '__main__':
    # Read records list
    records_file = os.path.join(LOCAL_DATABASE_PATH, 'RECORDS')
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            record_names = [line.strip() for line in f if line.strip()]
    else:
        record_names = sorted([f for f in os.listdir(LOCAL_DATABASE_PATH) if f.endswith('.edf')])
    
    print(f"ADFECGDB Evaluation (Synthetic Maternal Reference) — {len(record_names)} records")
    print(f"{'='*70}")
    print(f"  {'Record':<10} {'F1':>6}  {'Prec':>6}  {'Recall':>6}  "
          f"{'BPM':>6}  {'Det':>4}  {'GT':>4}")
    print(f"  {'-'*60}")
    
    all_results = []
    for rec_name in record_names:
        result = run_ilhsaf(rec_name)
        if result is not None:
            all_results.append(result)
    
    # Aggregate summary
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS (ILHSAF)")
        print(f"{'='*70}")
        avg_f1 = df['F1_ILHSAF'].mean()
        avg_prec = df['Precision_ILHSAF'].mean()
        avg_recall = df['Recall_ILHSAF'].mean()
        avg_bpm = df['BPM_ILHSAF'].mean()
        
        print(f"  Avg F1 Score:     {avg_f1:.4f}")
        print(f"  Avg Precision:    {avg_prec:.4f}")
        print(f"  Avg Recall/Sens:  {avg_recall:.4f}")
        print(f"  Avg BPM:          {avg_bpm:.1f}")
        
        print(f"\nAGGREGATE RESULTS (LMS)")
        print(f"{'='*70}")
        avg_f1_lms = df['F1_LMS'].mean()
        avg_prec_lms = df['Precision_LMS'].mean()
        avg_recall_lms = df['Recall_LMS'].mean()
        avg_bpm_lms = df['BPM_LMS'].mean()
        
        print(f"  Avg F1 Score:     {avg_f1_lms:.4f}")
        print(f"  Avg Precision:    {avg_prec_lms:.4f}")
        print(f"  Avg Recall/Sens:  {avg_recall_lms:.4f}")
        print(f"  Avg BPM:          {avg_bpm_lms:.1f}")
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, 'adfecgdb_metrics_optimized.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(f"Output waves saved to {OUTPUT_DIR}/")
    else:
        print("No records processed successfully.")
