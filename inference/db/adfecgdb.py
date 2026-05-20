import os
import numpy as np
import pyedflib
import wfdb
from .helpers import _bandpass, _standardise


ADFECGDB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'Datasets',
    'abdominal-and-direct-fetal-ecg-database-1.0.0')

ADFECGDB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), '..', 'Databases', 'ADFECGDB')


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
    direct_fecg = f.readSignal(0)
    abd = np.array([f.readSignal(i) for i in range(1, f.signals_in_file)])
    f.close()

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
