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
        # wfdb.rdann expects the record name without the file extension
        ann = wfdb.rdann(os.path.splitext(fpath)[0], 'qrs')
        fqrs = ann.sample
    except Exception:
        pass

    # Validate wfdb output — some WFDB readers misinterpret the project's
    # binary .qrs format and return large, implausible sample values.
    # If the samples look wrong, try a local parser for our .qrs format.
    def _parse_qrs_binary(qrs_path):
        import struct
        samples = []
        with open(qrs_path, 'rb') as fh:
            data = fh.read()
        if not data:
            return np.array([], dtype=int)

        # Helper to find the close NOTE marker (same two-byte pattern as open)
        pack_word = lambda v: struct.pack('<H', v)
        i = 0
        try:
            first_word = struct.unpack_from('<H', data, i)[0]
        except struct.error:
            return np.array([], dtype=int)
        i += 2

        # Find the next occurrence of the NOTE word which closes the ASCII note
        note_bytes = pack_word(first_word)
        close_pos = data.find(note_bytes, i)
        if close_pos == -1:
            return np.array([], dtype=int)
        i = close_pos + 2

        # Skip three 0xFFFF words and one 0x0001 preamble (8 bytes)
        i += 8

        prev = 0
        L = len(data)
        while i + 2 <= L:
            word = struct.unpack_from('<H', data, i)[0]
            i += 2
            anntype = (word >> 10) & 0x3F
            dt = word & 0x3FF

            # SKIP/NOTE escape: next 4 bytes contain 32-bit chunk to add
            if anntype == 59:
                if i + 4 > L:
                    break
                chunk = struct.unpack_from('<i', data, i)[0]
                i += 4
                prev += int(chunk)
                continue

            # EOF marker
            if anntype == 0:
                break

            # Normal beat: delta encoded
            prev += int(dt)
            samples.append(prev)

        return np.asarray(samples, dtype=int)

    # If fqrs looks invalid (very large numbers) try parsing the .qrs file directly
    if fqrs is not None and len(fqrs) > 0:
        # direct_fecg length gives an expectation for max sample index
        max_reasonable = len(direct_fecg) * 10
        try:
            if np.max(fqrs) > max_reasonable:
                qrs_path = os.path.splitext(fpath)[0] + '.qrs'
                if os.path.exists(qrs_path):
                    parsed = _parse_qrs_binary(qrs_path)
                    if parsed.size > 0 and np.max(parsed) < max_reasonable:
                        fqrs = parsed
        except Exception:
            # On any parsing error, keep original fqrs (or None)
            pass

    return {
        'abdominal': _standardise(_bandpass(abd, fs)),
        'chest': None,
        'direct_fecg': _standardise(_bandpass(direct_fecg, fs)),
        'fqrs': fqrs,
        'fs': fs,
        'record': record_name,
    }
