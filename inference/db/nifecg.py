import os
import numpy as np
import pyedflib
import wfdb
from .helpers import _bandpass, _standardise


NIFECG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'Datasets',
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
