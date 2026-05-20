import os
import wfdb
from .helpers import _bandpass, _standardise


CINC2013_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'Datasets',
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
    abd = rec.p_signal.T
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
