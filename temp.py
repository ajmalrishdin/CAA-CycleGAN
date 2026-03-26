# ============================================================
# PhysioNet 2013 fECG extraction using FastICA + SVD
# Classical, non-ML, non-generative
# ============================================================

import numpy as np
import wfdb
from sklearn.decomposition import FastICA
from scipy.linalg import svd
from scipy.signal import butter, filtfilt

# -----------------------------
# 1. LOAD PHYSIONET RECORD
# -----------------------------

record_name = '/Users/ajmalrishdin/Documents/Projects/Pre-term Birth Detector/CAA-CycleGAN/Databases/non-invasive-fetal-ecg-arrhythmia-database-1.0.0/ARR_01'   # change to your record name (no extension)

record = wfdb.rdrecord(record_name)
X = record.p_signal.T        # shape: (4, samples)
fs = record.fs               # typically 1000 Hz

print("Loaded signal shape:", X.shape)
print("Sampling frequency:", fs)

# FastICA cannot handle NaNs; replace any missing samples per-channel.
# Use median per channel to avoid biasing with zeros.
if np.isnan(X).any():
    channel_medians = np.nanmedian(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), channel_medians, X)

# Load fetal QRS annotations (for evaluation only)
# fqrs_ann = wfdb.rdann(record_name, "fqrs")
# fetal_qrs_ref = fqrs_ann.sample


# -----------------------------
# 2. BASIC PREPROCESSING
# -----------------------------

def highpass_filter(X, fs, cutoff=0.5):
    b, a = butter(2, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, X, axis=1)

X = highpass_filter(X, fs)


# -----------------------------
# 3. FASTICA (MATERNAL SEPARATION)
# -----------------------------

def run_fastica(X, n_components=4):
    ica = FastICA(
        n_components=n_components,
        whiten='unit-variance',
        max_iter=2000,
        random_state=0
    )
    S = ica.fit_transform(X.T)    # (samples, components)
    A = ica.mixing_               # (channels, components)
    return S.T, A

S, A = run_fastica(X)

# Identify maternal component (highest variance)
variances = np.var(S, axis=1)
maternal_idx = np.argmax(variances)

print("Maternal ICA component index:", maternal_idx)

# Reconstruct maternal ECG in sensor space
maternal_source = S[maternal_idx][None, :]      # (1, samples)
maternal_mixing = A[:, maternal_idx][:, None]   # (channels, 1)
maternal_recon = maternal_mixing @ maternal_source

# Remove maternal ECG
X_clean = X - maternal_recon


# -----------------------------
# 4. SVD-BASED fECG ENHANCEMENT
# -----------------------------

def create_svd_matrix(signal, window_size, step):
    segments = []
    for start in range(0, len(signal) - window_size, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

def svd_reconstruct(U, S, Vt, k=2):
    S_new = np.zeros_like(S)
    S_new[:k] = S[:k]
    return (U * S_new) @ Vt

def overlap_add(segments, signal_length, step):
    recon = np.zeros(signal_length)
    weight = np.zeros(signal_length)

    for i, seg in enumerate(segments):
        start = i * step
        recon[start:start + len(seg)] += seg
        weight[start:start + len(seg)] += 1

    return recon / np.maximum(weight, 1)

# Choose one cleaned channel for SVD
channel_idx = 0
signal_for_svd = X_clean[channel_idx]

window_size = int(0.6 * fs)   # 600 ms
step = int(0.05 * fs)         # 50 ms overlap

svd_matrix = create_svd_matrix(signal_for_svd, window_size, step)

U, S_vals, Vt = svd(svd_matrix, full_matrices=False)

# Rank-2 reconstruction (paper-consistent)
svd_segments = svd_reconstruct(U, S_vals, Vt, k=2)

fecg_svd = overlap_add(svd_segments, len(signal_for_svd), step)


# -----------------------------
# 5. OUTPUT SUMMARY
# -----------------------------

print("Finished fECG extraction.")
print("fECG signal length:", len(fecg_svd))
# print("Reference fetal QRS count:", len(fetal_qrs_ref))

# Save extracted fECG as NumPy array
output_path = f"{record_name}_fecg_svd.npy"
np.save(output_path, fecg_svd)
print("Saved fECG to:", output_path)
