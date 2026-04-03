"""
Test CycleGAN with EXACT training preprocessing from DataUtils.readData.
This replicates the full pipeline: bandpass -> SignalFilter -> scale -> resample to 200Hz.
Annotations are also divided by 5 to match 200Hz.
"""
import numpy as np
import pyedflib
import wfdb
import torch
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import scale
from extraction_engines import CycleGANEngine
from metrics import calculate_f1_precision_recall, detect_fetal_r_peaks

# ---------- Exact filters from DataUtils ----------

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def signal_filter(data):
    """Exact copy of DataUtils.SignalFilter"""
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

# ---------- Load data EXACTLY like training ----------

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
    
    # Resample to 200 Hz (divide by 5)
    abdECG = sp_signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
    fetalECG = sp_signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
    
    # Read and resample annotations
    ann = wfdb.rdann(fpath, "qrs", sampfrom=0, sampto=60000*5)
    fqrs = np.asarray(np.floor_divide(ann.sample, 5), 'int64')
    
    return abdECG, fetalECG, fqrs

# ---------- OLA inference at 200Hz ----------

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

# ---------- Main ----------

FOLDER = "../../Datasets/abdominal-and-direct-fetal-ecg-database-1.0.0"
import os
if not os.path.isdir(FOLDER):
    FOLDER = "Databases/ADFECGDB"

records = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]

engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

print("=" * 60)
print("Test with EXACT training preprocessing (200Hz domain)")
print("=" * 60)

for rec in records:
    abd, fecg_gt, fqrs = load_exact(rec, FOLDER)
    
    # Use channel 0 (Abdomen_1), process at 200Hz directly
    sig_200 = abd[0]
    
    fecg_extracted = ola_infer(engine, sig_200)
    
    # Detect peaks in extracted fECG at 200Hz
    det_peaks = detect_fetal_r_peaks(fecg_extracted, fs=200)
    
    # Calculate F1 at 200Hz (50ms tolerance = 10 samples at 200Hz)
    f1, prec, recall = calculate_f1_precision_recall(det_peaks, fqrs, tolerance_ms=50, fs=200)
    
    print(f"{rec}: F1={f1:.3f}  Prec={prec:.3f}  Recall={recall:.3f}  "
          f"(Det={len(det_peaks)}, GT={len(fqrs)})")
