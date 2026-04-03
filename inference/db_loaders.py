# ============================================================================
# Unified Database Loaders
# ============================================================================
# Each loader returns a standardised dict:
#   {
#       'abdominal':  np.array  — shape (n_channels, n_samples),
#       'chest':      np.array or None — shape (n_channels, n_samples),
#       'direct_fecg': np.array or None — ground‑truth fECG waveform,
#       'fqrs':       np.array or None — ground‑truth R‑peak sample indices,
#       'fs':         int,
#       'record':     str,
#   }
# ============================================================================

import os
import numpy as np
import pyedflib
import wfdb
from scipy.signal import butter, filtfilt, resample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bandpass(signal_data, fs, low=1.0, high=100.0, order=3):
    """Apply bandpass filter to signal(s). Works on 1D or 2D (channels × samples)."""
    nyq = 0.5 * fs
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype='band')
    if signal_data.ndim == 1:
        return filtfilt(b, a, signal_data)
    return np.array([filtfilt(b, a, ch) for ch in signal_data])


def _standardise(signal_data):
    """Z‑score standardise per channel."""
    if signal_data.ndim == 1:
        s = np.std(signal_data)
        if s < 1e-10:
            return signal_data
        return (signal_data - np.mean(signal_data)) / s
    out = np.zeros_like(signal_data)
    for i in range(signal_data.shape[0]):
        s = np.std(signal_data[i])
        if s < 1e-10:
            out[i] = signal_data[i]
        else:
            out[i] = (signal_data[i] - np.mean(signal_data[i])) / s
    return out


# ============================================================================
# 1.  ADFECGDB  —  Abdominal and Direct Fetal ECG Database
#     Format: EDF  |  Channels: Direct_1, Abdomen_1‑4  |  Fs: 1000 Hz
#     Annotations: .qrs (wfdb)
# ============================================================================

ADFECGDB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Datasets',
    'abdominal-and-direct-fetal-ecg-database-1.0.0')

ADFECGDB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), 'Databases', 'ADFECGDB')


def list_adfecgdb_records(folder=None):
    folder = folder or (ADFECGDB_PATH if os.path.isdir(ADFECGDB_PATH) else ADFECGDB_PATH_LOCAL)
    recs_file = os.path.join(folder, 'RECORDS')
    if os.path.exists(recs_file):
        with open(recs_file) as f:
            return [l.strip() for l in f if l.strip()], folder
    return sorted([f for f in os.listdir(folder) if f.endswith('.edf') and '_ARR_' not in f]), folder


def load_adfecgdb_record(record_name, folder=None):
    if folder is None:
        _, folder = list_adfecgdb_records()
    fpath = os.path.join(folder, record_name)

    f = pyedflib.EdfReader(fpath)
    fs = int(f.getSampleFrequency(0))
    direct_fecg = f.readSignal(0)  # Direct_1 (scalp electrode)
    abd = np.array([f.readSignal(i) for i in range(1, f.signals_in_file)])
    f.close()

    # Read annotations
    fqrs = None
    try:
        ann = wfdb.rdann(fpath, 'qrs')
        fqrs = ann.sample
    except Exception:
        pass

    return {
        'abdominal': _standardise(_bandpass(abd, fs)),
        'chest': None,
        'direct_fecg': _standardise(_bandpass(direct_fecg, fs)),
        'fqrs': fqrs,
        'fs': fs,
        'record': record_name,
    }


# ============================================================================
# 2.  NIFEADB  —  Non‑Invasive Fetal ECG Arrhythmia Database
#     Format: WFDB .dat/.hea  |  Channels: ECG (chest), Abdomen_1‑5
#     Fs: 500 or 1000 Hz  |  No fetal QRS annotations
# ============================================================================

NIFEADB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Datasets',
    'non-invasive-fetal-ecg-arrhythmia-database-1.0.0')

NIFEADB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), 'Databases',
                                   'non-invasive-fetal-ecg-arrhythmia-database-1.0.0')


def list_nifeadb_records(folder=None):
    folder = folder or (NIFEADB_PATH if os.path.isdir(NIFEADB_PATH) else NIFEADB_PATH_LOCAL)
    recs_file = os.path.join(folder, 'RECORDS')
    if os.path.exists(recs_file):
        with open(recs_file) as f:
            return [l.strip() for l in f if l.strip()], folder
    names = sorted(set(
        os.path.splitext(fn)[0] for fn in os.listdir(folder) if fn.endswith('.dat')
    ))
    return names, folder


def load_nifeadb_record(record_name, folder=None):
    if folder is None:
        _, folder = list_nifeadb_records()
    rec = wfdb.rdrecord(os.path.join(folder, record_name))
    fs = int(rec.fs)
    signals = rec.p_signal.T  # shape (n_channels, n_samples)
    names = rec.sig_name

    # Channel 0 = ECG (chest), Channels 1‑5 = Abdomen
    chest_idx = [i for i, n in enumerate(names) if 'ECG' in n.upper() and 'ABD' not in n.upper()]
    abd_idx = [i for i, n in enumerate(names) if 'ABD' in n.upper()]

    chest = _standardise(_bandpass(signals[chest_idx], fs)) if chest_idx else None
    abd = _standardise(_bandpass(signals[abd_idx], fs))

    return {
        'abdominal': abd,
        'chest': chest,
        'direct_fecg': None,
        'fqrs': None,  # No fetal annotations
        'fs': fs,
        'record': record_name,
    }


# ============================================================================
# 3.  NI‑FECG DB  —  Non‑Invasive Fetal ECG Database (ecgca*.edf)
#     Format: EDF  |  Channels: Thorax_1‑2, Abdomen_1‑3  |  Fs: 1000 Hz
#     Annotations: .qrs (wfdb)
# ============================================================================

NIFECG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Datasets',
    'non-invasive-fetal-ecg-database-1.0.0')


def list_nifecg_records(folder=None):
    folder = folder or NIFECG_PATH
    recs_file = os.path.join(folder, 'RECORDS')
    if os.path.exists(recs_file):
        with open(recs_file) as f:
            return [l.strip() for l in f if l.strip()], folder
    return sorted([f for f in os.listdir(folder) if f.endswith('.edf')]), folder


def load_nifecg_record(record_name, folder=None):
    if folder is None:
        _, folder = list_nifecg_records()
    fpath = os.path.join(folder, record_name)

    f = pyedflib.EdfReader(fpath)
    fs = int(f.getSampleFrequency(0))
    labels = f.getSignalLabels()
    n = f.signals_in_file

    thorax_idx = [i for i in range(n) if 'thorax' in labels[i].lower()]
    abd_idx = [i for i in range(n) if 'abdomen' in labels[i].lower()]

    thorax = np.array([f.readSignal(i) for i in thorax_idx])
    abd = np.array([f.readSignal(i) for i in abd_idx])
    f.close()

    chest = _standardise(_bandpass(thorax, fs)) if len(thorax_idx) > 0 else None
    abd = _standardise(_bandpass(abd, fs))

    # QRS annotations
    fqrs = None
    try:
        ann = wfdb.rdann(fpath, 'qrs')
        fqrs = ann.sample
    except Exception:
        pass

    return {
        'abdominal': abd,
        'chest': chest,
        'direct_fecg': None,
        'fqrs': fqrs,
        'fs': fs,
        'record': record_name,
    }


# ============================================================================
# 4.  CinC 2013  —  PhysioNet Computing in Cardiology Challenge 2013 (set‑a)
#     Format: WFDB  |  Channels: AECG1‑4  |  Fs: 1000 Hz
#     Annotations: .fqrs
# ============================================================================

CINC2013_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Datasets',
    'noninvasive-fetal-ecg-the-physionet-computing-in-cardiology-challenge-2013-1.0.0',
    'set-a')


def list_cinc2013_records(folder=None):
    folder = folder or CINC2013_PATH
    names = sorted(set(
        os.path.splitext(fn)[0] for fn in os.listdir(folder) if fn.endswith('.dat')
    ))
    return names, folder


def load_cinc2013_record(record_name, folder=None):
    if folder is None:
        _, folder = list_cinc2013_records()
    rec = wfdb.rdrecord(os.path.join(folder, record_name))
    fs = int(rec.fs)
    abd = rec.p_signal.T  # shape (4, n_samples)
    abd = _standardise(_bandpass(abd, fs))

    fqrs = None
    try:
        ann = wfdb.rdann(os.path.join(folder, record_name), 'fqrs')
        fqrs = ann.sample
    except Exception:
        pass

    return {
        'abdominal': abd,
        'chest': None,
        'direct_fecg': None,
        'fqrs': fqrs,
        'fs': fs,
        'record': record_name,
    }


# ============================================================================
# 5.  NINFEA  —  Non‑Invasive Multimodal Foetal ECG‑Doppler Dataset
#     Format: WFDB  |  Channels: uni_abd1‑24, bi_tho1‑3, dc, matrsp…
#     Fs: 2048 Hz  |  No fetal QRS annotations
# ============================================================================

NINFEA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'Datasets',
    'ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0',
    'wfdb_format_ecg_and_respiration')


def list_ninfea_records(folder=None):
    folder = folder or NINFEA_PATH
    names = sorted(
        set(os.path.splitext(fn)[0] for fn in os.listdir(folder) if fn.endswith('.dat')),
        key=lambda x: int(x)
    )
    return names, folder


def load_ninfea_record(record_name, folder=None):
    if folder is None:
        _, folder = list_ninfea_records()
    rec = wfdb.rdrecord(os.path.join(folder, record_name))
    fs = int(rec.fs)
    signals = rec.p_signal.T
    names = rec.sig_name

    abd_idx = [i for i, n in enumerate(names) if n.startswith('uni_abd')]
    tho_idx = [i for i, n in enumerate(names) if n.startswith('bi_tho')]

    abd = _standardise(_bandpass(signals[abd_idx], fs))
    chest = _standardise(_bandpass(signals[tho_idx], fs)) if tho_idx else None

    return {
        'abdominal': abd,
        'chest': chest,
        'direct_fecg': None,
        'fqrs': None,
        'fs': fs,
        'record': record_name,
    }


# ============================================================================
# Registry for easy iteration
# ============================================================================

DATABASE_REGISTRY = {
    'ADFECGDB': {
        'list_fn': list_adfecgdb_records,
        'load_fn': load_adfecgdb_record,
        'has_chest': False,
        'has_annotations': True,
        'description': 'Abdominal & Direct Fetal ECG DB (5 records)',
    },
    'NIFEADB': {
        'list_fn': list_nifeadb_records,
        'load_fn': load_nifeadb_record,
        'has_chest': True,
        'has_annotations': False,
        'description': 'NI-FECG Arrhythmia DB (26 records)',
    },
    'NI-FECG': {
        'list_fn': list_nifecg_records,
        'load_fn': load_nifecg_record,
        'has_chest': True,
        'has_annotations': True,
        'description': 'Non-Invasive Fetal ECG DB (55 records)',
    },
    'CinC2013': {
        'list_fn': list_cinc2013_records,
        'load_fn': load_cinc2013_record,
        'has_chest': False,
        'has_annotations': True,
        'description': 'PhysioNet CinC 2013 Challenge (25 records)',
    },
    'NINFEA': {
        'list_fn': list_ninfea_records,
        'load_fn': load_ninfea_record,
        'has_chest': True,
        'has_annotations': False,
        'description': 'NINFEA Multimodal Foetal ECG (60 records)',
    },
}
