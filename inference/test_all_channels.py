import numpy as np
import pyedflib
import wfdb
import torch
import os
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import scale
from extraction_engines import CycleGANEngine

# Preprocessing from training pipeline
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
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
    for i in np.arange(1, n):
        abdECG[i - 1, :] = f.readSignal(i)
    abdECG = scale(signal_filter(butter_bandpass_filter(abdECG, 1, 100, 1000)), axis=1)
    f.close()
    abdECG = sp_signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
    ann = wfdb.rdann(fpath, "qrs", sampfrom=0, sampto=60000*5)
    fqrs = np.asarray(np.floor_divide(ann.sample, 5), 'int64')
    return abdECG, fqrs

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

def detect_fetal_adaptive(signal_data, fs):
    nyq = 0.5 * fs
    b, a = butter(3, [5.0/nyq, 40.0/nyq], btype='band')
    filtered = filtfilt(b, a, signal_data)
    min_dist = int(0.25 * fs) # 240 BPM max
    mad = np.median(np.abs(filtered - np.median(filtered)))
    prominence = 0.7 * mad
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=prominence)
    return peaks

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
    return f1

FOLDER = "../../Datasets/abdominal-and-direct-fetal-ecg-database-1.0.0"
if not os.path.isdir(FOLDER):
    FOLDER = "Databases/ADFECGDB"

records = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]
engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

print("=" * 60)
print("Evaluating across ALL Abdominal channels to find the Best-F1")
print("=" * 60)

best_f1s = []

for rec in records:
    abd, fqrs = load_exact(rec, FOLDER)
    f1s_for_rec = []
    
    for ch_idx in range(abd.shape[0]):
        sig_200 = abd[ch_idx]
        fecg_ext = ola_infer(engine, sig_200)
        peaks = detect_fetal_adaptive(fecg_ext, 200)
        f1 = calc_f1(peaks, fqrs, 10) # 50ms at 200Hz
        f1s_for_rec.append(f1)
        
    best_ch_f1 = max(f1s_for_rec)
    best_f1s.append(best_ch_f1)
    
    # Also calculate the F1 if we randomly picked Abdomen_1 
    ch1_f1 = f1s_for_rec[0]
    
    print(f"{rec}: Abdomen_1 F1 = {ch1_f1:.3f} | Best Channel F1 = {best_ch_f1:.3f} (Ch {np.argmax(f1s_for_rec)})")

print(f"\nMean Abdomen_1 F1: {np.mean([f1s_for_rec[0] for rec in records]):.3f}")  
print(f"Mean Best-Channel F1: {np.mean(best_f1s):.3f}")
