#!/usr/bin/env python3
"""
Rebuild oracle .qrs annotations for already-generated synthetic ARR records.

The shipped ARR annotations were produced by re-running a QRS detector on the
synthesised fetal ECG, which scores ~0.25 F1 against its own signal (see
`tools/oracle_arr_qrs.py`). This tool instead recovers the *exact* beat
placements the synthesiser used:

  1. Deterministic replay -- re-run `generate_table2_arrhythmia` on the source
     healthy record. All cases except ARR_12 (randomised RR) reproduce the
     stored waveform bit-for-bit, so their beat placements are exact.
  2. Beam-search alignment -- for ARR_12 the RR multipliers were random, so the
     beat templates are re-aligned against the stored waveform, keeping the best
     `--beam` running hypotheses.

A rebuilt annotation is only written when the reconstruction correlates with
the stored channel 0 above `--corr-threshold`, so a bad recovery can never
silently overwrite good data.

Usage:
    python tools/rebuild_arr_oracle_qrs.py --dry-run
    python tools/rebuild_arr_oracle_qrs.py
    python tools/rebuild_arr_oracle_qrs.py --folder "Databases/ADFECGDB/Synthetic Database"
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyedflib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADFECGDB = os.path.join(PROJECT_ROOT, 'Databases', 'ADFECGDB')


def _load_synthesizer():
    """Import FINAL-MAINCODE.py (dashed filename, so not importable directly)."""
    path = os.path.join(ADFECGDB, 'FINAL-MAINCODE.py')
    sys.path.insert(0, ADFECGDB)
    spec = importlib.util.spec_from_file_location('arr_synth', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synth = _load_synthesizer()


def read_channel0(edf_path):
    f = pyedflib.EdfReader(str(edf_path))
    signal = np.asarray(f.readSignal(0), dtype=np.float64)
    fs = float(f.getSampleFrequency(0))
    f.close()
    return signal, fs


def correlation(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def replay(clean_fetal, fs, case, seed):
    """Re-run the synthesiser; exact for every case with deterministic RR rules."""
    rng = np.random.default_rng(seed)
    signal, peaks = synth.generate_table2_arrhythmia(clean_fetal, fs, case=case, rng=rng)
    return np.asarray(signal, dtype=np.float64), np.asarray(peaks, dtype=int)


def beam_align(stored, clean_fetal, fs, beam=8, lo_mult=0.65, hi_mult=1.45):
    """
    Recover beat placements when the RR sequence was random.

    Beats are placed left to right; for each step the next template is matched
    against the stored waveform over the plausible RR range and the best `beam`
    running hypotheses are kept, which avoids the drift a greedy search suffers.
    """
    peaks, beats, r_offsets, pre = synth.extract_beat_templates(clean_fetal, fs)
    if len(peaks) < 4:
        return np.asarray([], dtype=int), np.zeros_like(stored)

    states = [(0.0, int(peaks[0]), [int(peaks[0])])]
    for i in range(len(peaks) - 1):
        base_rr = int(peaks[i + 1] - peaks[i])
        lo, hi = int(base_rr * lo_mult), int(base_rr * hi_mult)
        template = beats[i + 1]

        candidates = {}
        for score, current_idx, path in states:
            win_lo = current_idx + lo - pre
            win_hi = current_idx + hi - pre
            if win_lo < 0 or win_hi + len(template) >= len(stored):
                continue
            match = np.correlate(stored[win_lo:win_hi + len(template)], template, mode='valid')
            for k in np.argsort(match)[::-1][:beam]:
                next_idx = current_idx + lo + int(k)
                total = score + float(match[k])
                if next_idx not in candidates or candidates[next_idx][0] < total:
                    candidates[next_idx] = (total, path + [next_idx])
        if not candidates:
            break
        best = sorted(candidates.items(), key=lambda kv: -kv[1][0])[:beam]
        states = [(v[0], k, v[1]) for k, v in best]

    path = max(states, key=lambda s: s[0])[2]

    recon = np.zeros_like(stored)
    oracle = []
    for i, current_idx in enumerate(path):
        start = current_idx - pre
        beat = beats[i]
        if start >= 0 and start + len(beat) < len(recon):
            recon[start:start + len(beat)] += beat
            oracle.append(start + r_offsets[i])
    return np.asarray(oracle, dtype=int), recon


def backup_existing(paths, enabled):
    for p in paths:
        if enabled and os.path.exists(p) and not os.path.exists(p + '.bak'):
            shutil.copy2(p, p + '.bak')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', default=ADFECGDB, help='Folder holding the *_ARR_*.edf files.')
    p.add_argument('--source-folder', default=ADFECGDB,
                   help='Folder holding the healthy source records (r01.edf, ...).')
    p.add_argument('--corr-threshold', type=float, default=0.99,
                   help='Minimum reconstruction correlation required to write annotations.')
    p.add_argument('--beam', type=int, default=8, help='Beam width for randomised-RR recovery.')
    p.add_argument('--seed', type=int, default=getattr(synth, 'ARR_SEED', 1234),
                   help='Seed tried first for randomised-RR cases.')
    p.add_argument('--dry-run', action='store_true', help='Report only; write nothing.')
    p.add_argument('--no-backup', action='store_true', help='Do not keep .qrs.bak copies.')
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    source_folder = os.path.abspath(args.source_folder)
    records = sorted(f for f in os.listdir(folder) if f.endswith('.edf') and '_ARR_' in f)
    if not records:
        print(f'No *_ARR_*.edf files in {folder}')
        return 1

    print('=' * 96)
    print('Rebuilding oracle .qrs annotations from synthesiser beat placements')
    print('=' * 96)
    print(f'ARR folder    : {folder}')
    print(f'Source folder : {source_folder}')
    print(f'Records       : {len(records)}')
    print(f'Threshold     : corr >= {args.corr_threshold}')
    print(f'Mode          : {"dry run (no writes)" if args.dry_run else "writing"}')
    print('-' * 96)

    clean_cache = {}
    written, skipped = 0, []

    for rec in records:
        stem = os.path.splitext(rec)[0]
        subject, case = stem.split('_ARR_')
        case = f'ARR_{case}'

        source_edf = os.path.join(source_folder, f'{subject}.edf')
        if not os.path.exists(source_edf):
            print(f'{stem:<20} SKIP: source record {subject}.edf not found')
            skipped.append((stem, 'missing source'))
            continue

        if subject not in clean_cache:
            signals, _, fs_src = synth.read_edf(source_edf)
            clean_cache[subject] = (synth.butter_highpass_filter(signals[0], 2.0, fs_src), fs_src)
        clean_fetal, fs_src = clean_cache[subject]

        stored, fs = read_channel0(os.path.join(folder, rec))

        recon, peaks = replay(clean_fetal, fs_src, case, args.seed)
        corr = correlation(stored, recon)
        method = 'replay'

        if corr < args.corr_threshold:
            peaks_b, recon_b = beam_align(stored, clean_fetal, fs_src, beam=args.beam)
            corr_b = correlation(stored, recon_b)
            if corr_b > corr:
                peaks, corr, method = peaks_b, corr_b, 'beam-align'

        if corr < args.corr_threshold or len(peaks) == 0:
            print(f'{stem:<20} SKIP: corr={corr:.4f} < {args.corr_threshold} ({method})')
            skipped.append((stem, f'corr={corr:.4f}'))
            continue

        median_rr = np.median(np.diff(peaks)) / fs * 1000.0 if len(peaks) > 1 else float('nan')
        if args.dry_run:
            print(f'{stem:<20} OK   corr={corr:.4f} beats={len(peaks):<5} '
                  f'medRR={median_rr:.0f}ms via {method} (dry run)')
            written += 1
            continue

        edf_path = os.path.join(folder, rec)
        backup_existing([f'{edf_path}.qrs', os.path.join(folder, f'{stem}.qrs')],
                        enabled=not args.no_backup)
        synth.write_oracle_qrs(edf_path, peaks, fs)
        print(f'{stem:<20} OK   corr={corr:.4f} beats={len(peaks):<5} '
              f'medRR={median_rr:.0f}ms via {method}')
        written += 1

    print('-' * 96)
    print(f'{"Would rebuild" if args.dry_run else "Rebuilt"}: {written}/{len(records)}')
    if skipped:
        print(f'Skipped ({len(skipped)}):')
        for name, reason in skipped:
            print(f'  - {name}: {reason}')
    if not args.dry_run and written:
        print('\nVerify with: python tools/oracle_arr_qrs.py')
    return 0 if not skipped else 2


if __name__ == '__main__':
    sys.exit(main())
