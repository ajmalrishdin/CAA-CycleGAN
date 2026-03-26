import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def preprocess_fecg(fecg_signal, fs=500):

    detrended_signal = signal.detrend(fecg_signal)
    smoothed_signal = signal.savgol_filter(detrended_signal, window_length=11, polyorder=3)
    
    return smoothed_signal

def detect_r_peaks(cleaned_signal, fs=500):
    # FIX: Ensure signal is 1-D
    cleaned_signal = np.squeeze(cleaned_signal) 
    if cleaned_signal.ndim > 1:
        cleaned_signal = cleaned_signal.flatten()

    # Dynamic thresholding: Look for peaks that are at least 50% of the max signal height
    height_threshold = np.max(np.abs(cleaned_signal)) * 0.5
    
    # distance=fs*0.2 ensures we don't pick up two peaks closer than 200ms
    peaks, _ = signal.find_peaks(cleaned_signal, height=height_threshold, distance=int(fs*0.2))
    
    return peaks

def analyze_heart_metrics(peaks, fs=500):
    """
    Phase 3: Feature Extraction (Insights).
    Calculates BPM and HRV metrics (SDNN, RMSSD) as per Paper 1.
    """
    if len(peaks) < 2:
        return None

    # Calculate RR intervals in seconds
    rr_intervals_samples = np.diff(peaks)
    rr_intervals_sec = rr_intervals_samples / fs
    
    # 1. Instantaneous Heart Rate (BPM)
    bpm_instantaneous = 60 / rr_intervals_sec
    avg_bpm = np.mean(bpm_instantaneous)
    
    # 2. Heart Rate Variability (HRV) Metrics 
    sdnn = np.std(rr_intervals_sec) * 1000  # Convert to ms
    
    # RMSSD: Root Mean Square of Successive Differences (Short-term/Vagal tone)
    diff_rr = np.diff(rr_intervals_sec)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000 # Convert to ms
    
    return {
        "avg_bpm": avg_bpm,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "rr_intervals": rr_intervals_sec
    }

def diagnose_arrhythmia(metrics):
    """
    Phase 4: Classification based on Paper 2 Thresholds.
    """
    if metrics is None:
        return "Insufficient Data"
        
    bpm = metrics["avg_bpm"]
    
    # Thresholds defined in Paper 2 
    if bpm < 120:
        return "Bradycardia (Abnormal: Too Slow)"
    elif bpm > 160:
        return "Tachycardia (Abnormal: Too Fast)"
    else:
        return "Normal Sinus Rhythm"

import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Import your existing functions
# from detect_arrhythmia import preprocess_fecg, detect_r_peaks, analyze_heart_metrics

def load_physionet_record(record_path, channel_index=0):

    # rdsamp returns: (signals, fields_dictionary)
    # signals is a 2D numpy array: [samples, channels]
    signals, fields = wfdb.rdsamp(record_path)
    
    # Extract the sampling frequency (Crucial for your filters!)
    fs = fields['fs']
    
    # Extract the specific channel you want (e.g., Abdominal Lead 1)
    # The paper mentions using single channel abdominal signals [cite: 804]
    raw_signal = signals[:, channel_index]
    
    return raw_signal, fs

record_name = 'Databases/non-invasive-fetal-ecg-arrhythmia-database-1.0.0/NR_02' 

try:
    # Load channel 0 (usually the first abdominal lead)
    fecg_input, fs = load_physionet_record(record_name, channel_index=0)
    
    print(f"Successfully loaded {record_name}")
    print(f"Sampling Frequency: {fs} Hz")
    print(f"Signal Shape: {fecg_input.shape}")

except Exception as e:
    print(f"Error loading file: {e}")
    print("Ensure both .hea and .dat files are in the same folder and the path is correct.")

fs = 500
duration = 50
t = np.linspace(0, duration, duration*fs)

cleaned_fecg = preprocess_fecg(fecg_input, fs)
r_peaks = detect_r_peaks(cleaned_fecg, fs)
metrics = analyze_heart_metrics(r_peaks, fs)
diagnosis = diagnose_arrhythmia(metrics)

# Results

print(f"--- NI-FECG Holter Analysis Report ---")
if metrics:
    print(f"Average Fetal HR: {metrics['avg_bpm']:.2f} BPM")
    print(f"HRV (SDNN):       {metrics['sdnn_ms']:.2f} ms")
    print(f"HRV (RMSSD):      {metrics['rmssd_ms']:.2f} ms")
    print(f"Diagnosis:        {diagnosis}")
    print("-" * 30)
else:
    print("Error: Not enough peaks detected.")

# Plotting for your Project Report
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:1000], fecg_input[:1000], label='Raw GAN Output (Noisy)')
plt.title("Raw Extracted fECG (Simulated)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t[:1000], cleaned_fecg[:1000], 'g', label='Savitzky-Golay Filtered')
plt.plot(r_peaks[r_peaks < 1000]/fs, cleaned_fecg[r_peaks[r_peaks < 1000]], "rx", label="Detected R-Peaks")
plt.title(f"Processed fECG & Detection (Diagnosis: {diagnosis})")
plt.legend()
plt.tight_layout()
plt.show()