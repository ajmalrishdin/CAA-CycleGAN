"""
Investigate R-peak detection sensitivity. 
High Precision + Low Recall = the detector is too conservative (threshold too high).
Let's try different detection approaches.
"""
import numpy as np
import pyedflib
import wfdb
import torch
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import scale

# Copy exact preprocessing from test_exact_preproc.py
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def signal_filter(data):
    A = np.array([1,0,0,0,0,0,0,0,0,0,-0.854])
    B = np.array([0.927,0,0,0,0,0,0,0,0,0,-0.927])
    data = sp_signal.filtfilt(B, A, data)
    B1 = np.array([0.995,-1.8504,0.995])
    A1 = np.array([1,-1.8505,0.99])
    data = sp_signal.filtfilt(B1, A1, data)
    B2 = np.array([0.388,0.388])
    A2 = np.array([1,-0.42578])
    data = sp_signal.filtfilt(B2, A2, data)
    return data

def load_exact(record_name, folder):
    fpath = folder + "/" + record_name
    f = pyedflib.EdfReader(fpath)
    n = f.signals_in_file
    abdECG = np.zeros((n - 1, f.getNSamples()[0]))
    fetalECG = np.zeros((1, f.getNSamples()[0]))
    fetalECG[0, :] = f.readSignal(0)
    fetalECG[0, :] = scale(signal_filter(butter_bandpass_filter(fetalECG, 1, 100, 1000)), axis=1)
    for i in np.arange(1, n):
        abdECG[i - 1, :] = f.readSignal(i)
    abdECG = scale(signal_filter(butter_bandpass_filter(abdECG, 1, 100, 1000)), axis=1)
    f.close()
    abdECG = sp_signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
    fetalECG = sp_signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
    ann = wfdb.rdann(fpath, "qrs", sampfrom=0, sampto=60000*5)
    fqrs = np.asarray(np.floor_divide(ann.sample, 5), 'int64')
    return abdECG, fetalECG, fqrs

def ola_infer(engine, sig_200, window_size=128, step=64):
    out_length = len(sig_200)
    output = np.zeros(out_length)
    weight = np.zeros(out_length)
    window = np.hanning(window_size)
    device = next(engine.engine.model.parameters()).device
    for start in range(0, out_length - window_size + 1, step):
        chunk = sig_200[start:start+window_size]
        s_min, s_max = chunk.min(), chunk.max()
        if s_max - s_min < 1e-6:
            c_norm = np.zeros_like(chunk)
        else:
            c_norm = 2 * (chunk - s_min) / (s_max - s_min) - 1
        with torch.no_grad():
            t_in = torch.FloatTensor(c_norm).view(1, 1, window_size).to(device)
            t_out = engine.engine.model(t_in).cpu().numpy().flatten()
        if s_max - s_min >= 1e-6:
            t_out = (t_out + 1) / 2.0 * (s_max - s_min) + s_min
        output[start:start+window_size] += t_out * window
        weight[start:start+window_size] += window
    valid = weight > 0
    output[valid] /= weight[valid]
    return output

def calc_f1(detected, true, tol_samples):
    if len(true) == 0 or len(detected) == 0:
        return 0, 0, 0
    tp = 0
    matched = set()
    for d in detected:
        dists = np.abs(true.astype(int) - int(d))
        ci = np.argmin(dists)
        if dists[ci] <= tol_samples and ci not in matched:
            tp += 1
            matched.add(ci)
    fp = len(detected) - tp
    fn = len(true) - len(matched)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    return f1, prec, rec

# ---------- Different R-peak detectors ----------

def detect_v1_original(signal_data, fs, min_bpm=100, max_bpm=220):
    """Our current detector (too conservative)"""
    nyq = 0.5 * fs
    lo = max(3.0 / nyq, 0.01)
    hi = min(45.0 / nyq, 0.99)
    b, a = butter(3, [lo, hi], btype='band')
    filtered = filtfilt(b, a, signal_data)
    enhanced = filtered ** 2
    win_size = max(int(0.15 * fs), 1)
    kernel = np.ones(win_size) / win_size
    smoothed = np.convolve(enhanced, kernel, mode='same')
    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
    min_distance = int(60.0 / max_bpm * fs)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)
    return peaks

def detect_v2_lower_thresh(signal_data, fs, min_bpm=100, max_bpm=220):
    """Lower threshold: mean + 0.2*std instead of 0.5*std"""
    nyq = 0.5 * fs
    lo = max(3.0 / nyq, 0.01)
    hi = min(45.0 / nyq, 0.99)
    b, a = butter(3, [lo, hi], btype='band')
    filtered = filtfilt(b, a, signal_data)
    enhanced = filtered ** 2
    win_size = max(int(0.15 * fs), 1)
    kernel = np.ones(win_size) / win_size
    smoothed = np.convolve(enhanced, kernel, mode='same')
    threshold = np.mean(smoothed) + 0.2 * np.std(smoothed)
    min_distance = int(60.0 / max_bpm * fs)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)
    return peaks

def detect_v3_adaptive(signal_data, fs, min_bpm=100, max_bpm=220):
    """Adaptive windowed threshold"""
    nyq = 0.5 * fs
    lo = max(3.0 / nyq, 0.01)
    hi = min(45.0 / nyq, 0.99)
    b, a = butter(3, [lo, hi], btype='band')
    filtered = filtfilt(b, a, signal_data)
    enhanced = filtered ** 2
    # Larger smoothing window
    win_size = max(int(0.08 * fs), 1)
    kernel = np.ones(win_size) / win_size
    smoothed = np.convolve(enhanced, kernel, mode='same')
    
    # Moving window adaptive threshold
    tw = int(2.0 * fs)  # 2-second windows
    threshold = np.zeros_like(smoothed)
    for i in range(len(smoothed)):
        start = max(0, i - tw)
        end = min(len(smoothed), i + tw)
        local = smoothed[start:end]
        threshold[i] = np.mean(local) + 0.15 * np.std(local)
    
    min_distance = int(60.0 / max_bpm * fs)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)
    return peaks

def detect_v4_raw_peaks(signal_data, fs, min_bpm=100, max_bpm=220):
    """Direct peak detection on the signal (no squaring)"""
    nyq = 0.5 * fs
    lo = max(3.0 / nyq, 0.01)
    hi = min(45.0 / nyq, 0.99)
    b, a = butter(3, [lo, hi], btype='band')
    filtered = filtfilt(b, a, signal_data)
    
    # Use prominence-based detection
    min_distance = int(60.0 / max_bpm * fs)
    prominence = 0.3 * np.std(filtered)
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=prominence)
    return peaks

# ---------- Main ----------

from extraction_engines import CycleGANEngine

FOLDER = "../../Datasets/abdominal-and-direct-fetal-ecg-database-1.0.0"
import os
if not os.path.isdir(FOLDER):
    FOLDER = "Databases/ADFECGDB"

records = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]
engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

detectors = {
    "v1_original (mean+0.5*std)": detect_v1_original,
    "v2_lower   (mean+0.2*std)": detect_v2_lower_thresh,
    "v3_adaptive (local window)": detect_v3_adaptive,
    "v4_prominence (no squaring)": detect_v4_raw_peaks,
}

tol = 10  # 50ms at 200Hz

for det_name, det_fn in detectors.items():
    print(f"\n{'='*60}")
    print(f"Detector: {det_name}")
    print(f"{'='*60}")
    f1s = []
    for rec in records:
        abd, fecg_gt, fqrs = load_exact(rec, FOLDER)
        fecg_extracted = ola_infer(engine, abd[0])
        peaks = det_fn(fecg_extracted, fs=200)
        f1, prec, recall = calc_f1(peaks, fqrs, tol)
        f1s.append(f1)
        print(f"  {rec}: F1={f1:.3f}  P={prec:.3f}  R={recall:.3f}  "
              f"(Det={len(peaks)}, GT={len(fqrs)})")
    print(f"  MEAN F1: {np.mean(f1s):.3f}")
