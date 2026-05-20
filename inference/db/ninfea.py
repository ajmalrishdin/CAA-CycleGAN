import os
import numpy as np
import wfdb
from .helpers import _bandpass, _standardise


NINFEA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'Datasets',
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
