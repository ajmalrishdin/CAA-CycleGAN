import os
import numpy as np
import wfdb
from .helpers import _bandpass, _standardise


NIFEADB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'Datasets',
    'non-invasive-fetal-ecg-arrhythmia-database-1.0.0')

NIFEADB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), '..', 'Databases',
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
    signals = rec.p_signal.T
    names = rec.sig_name

    chest_idx = [i for i, n in enumerate(names) if 'ECG' in n.upper() and 'ABD' not in n.upper()]
    abd_idx = [i for i, n in enumerate(names) if 'ABD' in n.upper()]

    chest = _standardise(_bandpass(signals[chest_idx], fs)) if chest_idx else None
    abd = _standardise(_bandpass(signals[abd_idx], fs))

    return {
        'abdominal': abd,
        'chest': chest,
        'direct_fecg': None,
        'fqrs': None,
        'fs': fs,
        'record': record_name,
    }
