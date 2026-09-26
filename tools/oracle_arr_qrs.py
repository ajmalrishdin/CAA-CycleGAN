#!/usr/bin/env python3
"""
Oracle / ceiling check for ADFECGDB (healthy) and synthetic ARR records.

Runs the production peak detector (`inference.metrics.detect_fetal_r_peaks`) on
channel 0 -- the direct (healthy) or synthesised (ARR) fetal ECG -- and scores
it against the shipped `.qrs` annotations.

Channel 0 *is* the ground-truth waveform, so this measures the ceiling of the
whole evaluation pipeline. A low score here means the annotations or the peak
detector are the bottleneck, not the extraction model.

Usage:
    python tools/oracle_arr_qrs.py
    python tools/oracle_arr_qrs.py --max-bpm 300 --csv outputs/oracle.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'inference'))

from db import load_adfecgdb_record  # noqa: E402
from metrics import calculate_peak_detection_metrics, detect_fetal_r_peaks  # noqa: E402


DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, 'Databases', 'ADFECGDB')


def list_records(folder, kind):
    edfs = sorted(f for f in os.listdir(folder) if f.endswith('.edf'))
    if kind == 'arr':
        edfs = [f for f in edfs if '_ARR_' in f]
    elif kind == 'healthy':
        edfs = [f for f in edfs if '_ARR_' not in f]
    return edfs


def _sort_key(name):
    stem = os.path.splitext(name)[0]
    if '_ARR_' not in stem:
        return (stem, -1)
    subject, case = stem.split('_ARR_')
    return (subject, int(case))


def evaluate(folder, records, tolerance_ms, min_bpm, max_bpm, min_distance_ms):
    rows = []
    for rec in records:
        try:
            data = load_adfecgdb_record(rec, folder)
        except Exception as ex:
            print(f'{rec:<20} LOAD FAILED: {ex}')
            continue

        fs = int(data['fs'])
        direct = np.asarray(data['direct_fecg'], dtype=np.float64)
        gt = data['fqrs']
        if gt is None:
            print(f'{rec:<20} no .qrs annotations found -- skipped')
            continue
        gt = np.asarray(gt, dtype=int)
        gt = gt[(gt >= 0) & (gt < len(direct))]

        if min_distance_ms is None:
            peaks = detect_fetal_r_peaks(direct, fs, min_bpm=min_bpm, max_bpm=max_bpm)
        else:
            peaks = _detect_with_min_distance(direct, fs, min_distance_ms)

        m = calculate_peak_detection_metrics(peaks, gt, tolerance_ms=tolerance_ms, fs=fs)

        stem = os.path.splitext(rec)[0]
        is_arr = '_ARR_' in stem
        rows.append({
            'Record': stem,
            'Subject': stem.split('_ARR_')[0],
            'Group': 'ARR' if is_arr else 'healthy',
            'Case': stem.split('_ARR_')[1] if is_arr else '',
            'fs': fs,
            'Detected': len(peaks),
            'GT': len(gt),
            'Det_over_GT': len(peaks) / len(gt) if len(gt) else np.nan,
            'TP': m['TP'],
            'FP': m['FP'],
            'FN': m['FN'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'F1': m['F1'],
            'MedianRR_det_ms': _median_rr_ms(peaks, fs),
            'MedianRR_gt_ms': _median_rr_ms(gt, fs),
        })

        print(
            f"{stem:<20} F1={m['F1']:.3f}  P={m['Precision']:.3f}  R={m['Recall']:.3f}  "
            f"det={len(peaks):<5} gt={len(gt):<5} "
            f"RRdet={rows[-1]['MedianRR_det_ms']:.0f}ms RRgt={rows[-1]['MedianRR_gt_ms']:.0f}ms"
        )

    return pd.DataFrame(rows)


def _detect_with_min_distance(signal_data, fs, min_distance_ms):
    """Same detector as `detect_fetal_r_peaks` but with an explicit refractory period."""
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = 0.5 * fs
    b, a = butter(3, [max(5.0 / nyq, 0.01), min(40.0 / nyq, 0.99)], btype='band')
    filtered = filtfilt(b, a, signal_data)
    mad = np.median(np.abs(filtered - np.median(filtered)))
    distance = max(1, int(min_distance_ms / 1000.0 * fs))
    peaks, _ = find_peaks(filtered, distance=distance, prominence=0.7 * mad)
    return peaks


def _median_rr_ms(peaks, fs):
    peaks = np.asarray(peaks)
    if len(peaks) < 2:
        return float('nan')
    return float(np.median(np.diff(peaks)) / fs * 1000.0)


def summarise(df):
    out = []
    for group, sub in df.groupby('Group'):
        tp, fp, fn = int(sub['TP'].sum()), int(sub['FP'].sum()), int(sub['FN'].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append({
            'Group': group,
            'Records': len(sub),
            'F1_macro': sub['F1'].mean(),
            'Precision_macro': sub['Precision'].mean(),
            'Recall_macro': sub['Recall'].mean(),
            'F1_micro': f1,
            'Precision_micro': precision,
            'Recall_micro': recall,
            'Det_over_GT_mean': sub['Det_over_GT'].mean(),
        })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', default=DEFAULT_FOLDER, help='Folder containing the .edf/.qrs pairs.')
    p.add_argument('--kind', choices=['all', 'arr', 'healthy'], default='all')
    p.add_argument('--tolerance-ms', type=float, default=50.0)
    p.add_argument('--min-bpm', type=float, default=100.0)
    p.add_argument('--max-bpm', type=float, default=240.0)
    p.add_argument('--min-distance-ms', type=float, default=None,
                   help='Override the refractory period (default: 60000/max_bpm ms).')
    p.add_argument('--csv', default=None, help='Optional path for the per-record CSV.')
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    records = sorted(list_records(folder, args.kind), key=_sort_key)
    if not records:
        print(f'No matching .edf files in {folder}')
        return 1

    print('=' * 96)
    print('Oracle peak-detection check: channel 0 (direct/synthetic fECG) vs .qrs annotations')
    print('=' * 96)
    print(f'Folder      : {folder}')
    print(f'Records     : {len(records)} ({args.kind})')
    print(f'Tolerance   : {args.tolerance_ms:.0f} ms')
    print(f'Detector    : min_bpm={args.min_bpm:.0f} max_bpm={args.max_bpm:.0f} '
          f'min_distance={args.min_distance_ms if args.min_distance_ms else 60000.0 / args.max_bpm:.0f} ms')
    print('-' * 96)

    df = evaluate(folder, records, args.tolerance_ms, args.min_bpm, args.max_bpm, args.min_distance_ms)
    if df.empty:
        print('No records evaluated.')
        return 1

    print('-' * 96)
    print(summarise(df).to_string(index=False, float_format='%.3f'))

    if args.csv:
        csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(PROJECT_ROOT, args.csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f'\nPer-record CSV: {csv_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
