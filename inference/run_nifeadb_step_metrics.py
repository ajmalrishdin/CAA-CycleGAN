#!/usr/bin/env python3
"""
Evaluate one CycleGAN checkpoint on NIFEADB and report peak-based metrics.

Important:
- The arrhythmia database does not provide fetal QRS annotations.
- This script therefore computes metrics against a proxy reference signal.
  Priority:
    1) Use <record>_fecg_svd.npy if present in the database folder.
    2) Otherwise build an SVD-based proxy from abdominal channels.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from db_loaders import DATABASE_REGISTRY
from extraction_engines import CycleGANEngine
from metrics import (
    calculate_bpm,
    calculate_peak_detection_metrics,
    detect_fetal_r_peaks,
)


def _standardize(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def _bandpass_1d(signal_1d, fs, low=1.0, high=80.0, order=3):
    nyq = 0.5 * fs
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, signal_1d)


def _safe_first_channel(signal_data):
    if signal_data is None:
        return None
    arr = np.asarray(signal_data)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr[0]
    return None


def _build_proxy_fecg_from_svd(abdominal, chest, fs):
    """
    Build a pseudo fECG reference from multi-channel abdominal signals.

    Steps:
    1) Remove chest-correlated component from each abdominal channel.
    2) SVD decomposition on channels x samples.
    3) Pick temporal component with strongest fetal-band dominance.
    """
    abd = np.asarray(abdominal, dtype=np.float64)
    if abd.ndim == 1:
        abd = abd[None, :]

    chest_1d = _safe_first_channel(chest)
    if chest_1d is not None:
        chest_1d = _standardize(_bandpass_1d(chest_1d, fs, low=1.0, high=80.0))
        denom = float(np.dot(chest_1d, chest_1d)) + 1e-12
        abd_clean = []
        for ch in abd:
            ch_f = _standardize(_bandpass_1d(ch, fs, low=1.0, high=80.0))
            alpha = float(np.dot(ch_f, chest_1d) / denom)
            abd_clean.append(ch_f - alpha * chest_1d)
        abd = np.asarray(abd_clean, dtype=np.float64)
    else:
        abd = np.asarray([_standardize(_bandpass_1d(ch, fs, low=1.0, high=80.0)) for ch in abd])

    # X: channels x samples -> right singular vectors encode temporal patterns.
    x = abd
    _, _, v_t = np.linalg.svd(x, full_matrices=False)

    kmax = min(5, v_t.shape[0])
    candidates = [v_t[k] for k in range(kmax)]

    freqs = np.fft.rfftfreq(candidates[0].size, d=1.0 / fs)

    def _band_power(sig, f_lo, f_hi):
        spec = np.abs(np.fft.rfft(sig)) ** 2
        m = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(m):
            return 0.0
        return float(np.mean(spec[m]))

    best = candidates[0]
    best_score = -np.inf
    for c in candidates:
        c_std = _standardize(c)
        fetal = _band_power(c_std, 2.0, 4.5)
        maternal = _band_power(c_std, 0.8, 2.0)
        broad = _band_power(c_std, 0.5, 20.0)
        score = (fetal + 1e-8) / (maternal + 1e-8) + 0.5 * (fetal + 1e-8) / (broad + 1e-8)
        if score > best_score:
            best = c_std
            best_score = score

    return _standardize(best)


def _resolve_nifeadb_folder(project_root, override_folder=None):
    if override_folder:
        folder = os.path.abspath(override_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'NIFEADB folder not found: {folder}')
        return folder

    candidates = [
        os.path.join(project_root, 'Databases', 'non-invasive-fetal-ecg-arrhythmia-database-1.0.0'),
        os.path.join(project_root, 'inference', 'Databases', 'non-invasive-fetal-ecg-arrhythmia-database-1.0.0'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError('Could not locate NIFEADB folder. Pass --db-folder explicitly.')


def _resolve_model_dir(project_root, model_dir):
    out = model_dir if os.path.isabs(model_dir) else os.path.join(project_root, model_dir)
    out = os.path.abspath(out)
    if not os.path.isdir(out):
        raise FileNotFoundError(f'Model directory not found: {out}')
    return out


def _compute_macro_micro(df):
    macro = {
        'F1_macro': float(df['F1'].mean()),
        'Accuracy_macro': float(df['Accuracy'].mean()),
        'Precision_macro': float(df['Precision'].mean()),
        'Recall_macro': float(df['Recall'].mean()),
        'Sensitivity_macro': float(df['Sensitivity'].mean()),
        'BPM_mean_macro': float(df['BPM_mean'].mean()),
    }

    tp = int(df['TP'].sum())
    fp = int(df['FP'].sum())
    fn = int(df['FN'].sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    micro = {
        'TP_total': tp,
        'FP_total': fp,
        'FN_total': fn,
        'Precision_micro': float(precision),
        'Recall_micro': float(recall),
        'Sensitivity_micro': float(recall),
        'F1_micro': float(f1),
        'Accuracy_micro': float(acc),
    }
    return macro, micro


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate one CycleGAN checkpoint on NIFEADB with proxy peak metrics.'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/sagan_1_SynDB1_bs128_1n8',
        help='Model directory (relative to project root or absolute).',
    )
    parser.add_argument('--step', type=int, default=113, help='Checkpoint step number.')
    parser.add_argument(
        '--db-folder',
        type=str,
        default=None,
        help='Optional path to non-invasive-fetal-ecg-arrhythmia-database-1.0.0',
    )
    parser.add_argument('--output-dir', type=str, default='inference/Output_selected_models')
    parser.add_argument('--max-records', type=int, default=0, help='Optional cap; 0 means all records.')
    parser.add_argument('--tolerance-ms', type=float, default=50.0)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = _resolve_model_dir(project_root, args.model_dir)
    db_folder = _resolve_nifeadb_folder(project_root, args.db_folder)
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    db_name = 'NIFEADB'
    db_info = DATABASE_REGISTRY[db_name]
    records, _ = db_info['list_fn'](db_folder)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]

    engine_name = f'{os.path.basename(model_dir)}_step_{args.step}'
    engine = CycleGANEngine(model_dir, version_name=engine_name, step=args.step)

    print('=' * 88)
    print('NIFEADB checkpoint evaluation (proxy-reference metrics)')
    print('=' * 88)
    print(f'Database folder : {db_folder}')
    print(f'Record count    : {len(records)}')
    print(f'Model folder    : {model_dir}')
    print(f'Checkpoint step : {args.step}')

    rows = []
    for rec_name in records:
        data = db_info['load_fn'](rec_name, db_folder)
        fs = int(data['fs'])

        abd = data['abdominal']
        if abd.ndim == 2:
            abd_1d = abd[0]
        else:
            abd_1d = abd

        pred_fecg = engine.extract_fecg(abd_1d, chest=_safe_first_channel(data['chest']), fs=fs)
        pred_peaks = detect_fetal_r_peaks(pred_fecg, fs=fs)

        svd_ref_path = os.path.join(db_folder, f'{rec_name}_fecg_svd.npy')
        if os.path.exists(svd_ref_path):
            proxy_fecg = np.load(svd_ref_path)
            proxy_source = 'precomputed_svd'
        else:
            proxy_fecg = _build_proxy_fecg_from_svd(data['abdominal'], data['chest'], fs)
            proxy_source = 'generated_svd'

        # Ensure equal length for robust peak comparison.
        n = min(len(pred_fecg), len(proxy_fecg))
        pred_fecg = np.asarray(pred_fecg[:n])
        proxy_fecg = np.asarray(proxy_fecg[:n])

        pred_peaks = detect_fetal_r_peaks(pred_fecg, fs=fs)
        ref_peaks = detect_fetal_r_peaks(proxy_fecg, fs=fs)
        m = calculate_peak_detection_metrics(
            pred_peaks,
            ref_peaks,
            tolerance_ms=float(args.tolerance_ms),
            fs=fs,
        )

        bpm_pred, bpm_std, _ = calculate_bpm(pred_fecg, fs=fs)
        bpm_ref, bpm_ref_std, _ = calculate_bpm(proxy_fecg, fs=fs)

        row = {
            'Database': db_name,
            'Record': rec_name,
            'Technique': engine_name,
            'ReferenceType': proxy_source,
            'F1': m['F1'],
            'Accuracy': m['Accuracy'],
            'Precision': m['Precision'],
            'Sensitivity': m['Sensitivity'],
            'Recall': m['Recall'],
            'TP': m['TP'],
            'FP': m['FP'],
            'FN': m['FN'],
            'BPM_mean': bpm_pred,
            'BPM_std': bpm_std,
            'BPM_ref_mean': bpm_ref,
            'BPM_ref_std': bpm_ref_std,
            'Detected_Peaks': len(pred_peaks),
            'Reference_Peaks': len(ref_peaks),
            'fs': fs,
        }
        rows.append(row)

        print(
            f"{rec_name:<12} F1={row['F1']:.3f} "
            f"Acc={row['Accuracy']:.3f} "
            f"Prec={row['Precision']:.3f} "
            f"Rec={row['Recall']:.3f} "
            f"BPM={row['BPM_mean']:.1f} "
            f"ref={proxy_source}"
        )

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_csv = os.path.join(out_dir, f'nifeadb_step_{args.step}_detailed_{ts}.csv')
    df.to_csv(detailed_csv, index=False)

    macro, micro = _compute_macro_micro(df)
    summary = {
        'Database': db_name,
        'ModelFolder': os.path.basename(model_dir),
        'Step': int(args.step),
        'NumRecords': int(len(df)),
        **macro,
        **micro,
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(out_dir, f'nifeadb_step_{args.step}_summary_{ts}.csv')
    summary_df.to_csv(summary_csv, index=False)

    print('\n' + '=' * 88)
    print('Summary (proxy-reference metrics)')
    print('=' * 88)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print(f'\nDetailed CSV: {detailed_csv}')
    print(f'Summary CSV : {summary_csv}')


if __name__ == '__main__':
    main()
