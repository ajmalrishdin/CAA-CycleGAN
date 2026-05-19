#!/usr/bin/env python3
"""
edf_to_qrs.py  –  Batch QRS detector for EDF recordings
=========================================================
Reads every .edf in a folder, detects R-peaks from the ECG channel,
and writes a .qrs file next to each EDF (or into a separate output folder).

Tested against the Synthetic Database naming scheme:
    r01_ARR_1.edf, r04_ARR_3.edf, …

Dependencies (install once):
    pip install pyedflib neurokit2 numpy pandas

Usage examples
--------------
# Process all EDFs in a folder (saves .qrs alongside each .edf):
    python edf_to_qrs.py "C:/Users/obsid/Downloads/Synthetic Database"

# Save QRS files to a separate folder:
    python edf_to_qrs.py "C:/Users/obsid/Downloads/Synthetic Database" ^
                         --outdir "C:/Users/obsid/Downloads/QRS_Output"

# Choose a specific detection algorithm:
    python edf_to_qrs.py "..." --method hamilton

# Force a specific ECG channel name (if auto-detect fails):
    python edf_to_qrs.py "..." --channel "ECG"

Output .qrs format (CSV)
-------------------------
    sample_index,time_sec,rr_ms
    510,1.000,—
    1096,2.149,1148.6
    1671,3.279,1129.3

    sample_index : R-peak location in samples (0-based)
    time_sec     : R-peak time in seconds from recording start
    rr_ms        : R-R interval to the PREVIOUS beat (ms); "—" for first beat
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# ECG channel auto-detection keywords (case-insensitive substring match)
# ─────────────────────────────────────────────────────────────────────────────
ECG_KEYWORDS = [
    "ecg", "ekg", "lead", "ii", "avf", "avr", "avl",
    "v1", "v2", "v3", "v4", "v5", "v6",
    "chest", "limb", "cardiac", "heart",
]


def find_ecg_channel(labels):
    """Return the first channel label that looks like an ECG lead."""
    for label in labels:
        if any(kw in label.lower() for kw in ECG_KEYWORDS):
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EDF reader
# ─────────────────────────────────────────────────────────────────────────────
def read_edf(path, channel=None):
    """
    Read a single EDF file.
    Returns (signal_1d_array, sampling_rate_hz, channel_name_used).
    """
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        labels = f.getSignalLabels()

        if channel:
            if channel not in labels:
                raise ValueError(
                    f"Channel '{channel}' not found in {path.name}.\n"
                    f"  Available: {labels}"
                )
            idx = labels.index(channel)
        else:
            detected = find_ecg_channel(labels)
            if detected is None:
                raise ValueError(
                    f"Could not auto-detect an ECG channel in {path.name}.\n"
                    f"  Available channels: {labels}\n"
                    f"  Re-run with --channel <name> to specify one."
                )
            idx = labels.index(detected)
            channel = detected

        fs = float(f.getSampleFrequency(idx))
        signal = f.readSignal(idx)

    return np.asarray(signal, dtype=float), fs, channel


# ─────────────────────────────────────────────────────────────────────────────
# QRS detector
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_METHODS = [
    "pantompkins",  # Pan & Tompkins (1985) – good all-rounder
    "hamilton",     # Hamilton (2002)
    "elgendi",      # Elgendi et al. (2010)
    "engzee",       # Engelse & Zeelenberg (1979)
    "christov",     # Christov (2004)
    "kalidas",      # Kalidas & Tamil (2017)
    "neurokit",     # NeuroKit2 default pipeline
    "rodrigues",    # Rodrigues et al. (2021)
    "wqrs",         # Wavelet-based
    "ssf",          # Slope Sum Function
]


def detect_qrs(signal, fs, method="pantompkins"):
    """
    Detect R-peak sample indices using NeuroKit2.
    Returns a 1-D integer array of 0-based sample indices.
    """
    import neurokit2 as nk

    _, info = nk.ecg_peaks(signal, sampling_rate=int(fs), method=method)
    peaks = info.get("ECG_R_Peaks", np.array([], dtype=int))
    return np.asarray(peaks, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation builder
# ─────────────────────────────────────────────────────────────────────────────
def build_df(peaks, fs):
    """Build an annotation DataFrame from R-peak sample indices."""
    times   = peaks / fs
    rr      = np.diff(peaks) / fs * 1000.0
    rr_col  = np.concatenate([[np.nan], rr])   # NaN for first beat

    return pd.DataFrame({
        "sample_index": peaks,
        "time_sec":     np.round(times, 6),
        "rr_ms":        np.round(rr_col, 2),
    })


def save_qrs(df, out_path, fs):
    """
    Write annotation DataFrame to a WFDB-compatible binary .qrs file.

    Format:
      - 2-byte little-endian words throughout
      - Text note header:  anntype=59 (NOTE), then ASCII bytes of
        "## time resolution: <fs>" terminated by anntype=59 again
      - Three 0xFFFF sentinel words  +  one 0x0001 word (standard WFDB preamble)
      - One word per R-peak: upper 6 bits = anntype 1 (Normal beat),
        lower 10 bits = time delta from previous peak (in samples).
        Deltas ≥ 1024 are split via SKIP words (anntype=59, dt=0 + 32-bit value).
      - EOF: two zero bytes (anntype=0, dt=0)
    """
    import struct

    NORMAL   = 1   # 'N' beat annotation type
    NOTE     = 59  # used for header note and SKIP escape
    SKIP     = 59

    def encode_word(anntype, dt):
        return struct.pack("<H", ((anntype & 0x3F) << 10) | (dt & 0x3FF))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = df["sample_index"].dropna().astype(int).tolist()

    with open(out_path, "wb") as f:
        # ── Header note: "## time resolution: <fs>" ──────────────────────
        note_text = f"## time resolution: {int(fs)}"
        note_bytes = note_text.encode("ascii")
        # Pad to even length
        if len(note_bytes) % 2:
            note_bytes += b" "
        f.write(encode_word(NOTE, 0))           # open note (anntype=59, dt=0)
        f.write(note_bytes)                     # raw ASCII (even length)
        f.write(encode_word(NOTE, 0))           # close note

        # ── Preamble: three 0xFFFF + one 0x0001 ──────────────────────────
        f.write(b"\xff\xff" * 3)
        f.write(b"\x01\x00")

        # ── Beat annotations ─────────────────────────────────────────────
        prev = 0
        for s in samples:
            delta = s - prev
            # Large delta → SKIP word (anntype=59, dt=0) + 32-bit skip value
            while delta >= 1024:
                chunk = min(delta, 0x3FFFFFFF)  # 30-bit max per SKIP
                f.write(encode_word(SKIP, 0))
                f.write(struct.pack("<i", chunk))
                delta -= chunk
            f.write(encode_word(NORMAL, delta))
            prev = s

        # ── EOF ───────────────────────────────────────────────────────────
        f.write(b"\x00\x00")


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_folder(edf_dir, outdir, channel, method):
    edf_files = sorted(edf_dir.glob("*.edf"))
    if not edf_files:
        print(f"[!] No .edf files found in: {edf_dir}")
        sys.exit(1)

    print(f"\n{'─'*62}")
    print(f"  EDF folder  : {edf_dir}")
    print(f"  Output dir  : {outdir or '(same folder as each EDF)'}")
    print(f"  Algorithm   : {method}")
    print(f"  Channel     : {channel or '(auto-detect)'}")
    print(f"  Files found : {len(edf_files)}")
    print(f"{'─'*62}\n")

    ok, failed = 0, []

    for i, edf_path in enumerate(edf_files, 1):
        tag = f"[{i:>3}/{len(edf_files)}]"
        print(f"{tag}  {edf_path.name}", end="  …  ", flush=True)

        try:
            signal, fs, ch_used = read_edf(edf_path, channel)
            peaks               = detect_qrs(signal, fs, method)
            df                  = build_df(peaks, fs)
            dest_dir            = outdir if outdir else edf_path.parent
            out_path            = dest_dir / edf_path.with_suffix(".edf.qrs").name
            save_qrs(df, out_path, fs)

            print(f"OK  |  {len(peaks)} beats  |  ch='{ch_used}'  |  {out_path.name}")
            ok += 1

        except Exception as e:
            print(f"FAILED")
            print(f"          {e}")
            failed.append((edf_path.name, str(e)))

    print(f"\n{'─'*62}")
    print(f"  Finished: {ok}/{len(edf_files)} succeeded.")
    if failed:
        print(f"\n  Failed files ({len(failed)}):")
        for name, err in failed:
            print(f"    • {name}: {err}")
    print(f"{'─'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    for pkg in ("pyedflib", "neurokit2", "numpy", "pandas"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("[!] Missing packages. Install with:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect QRS complexes in EDF files and write .qrs annotation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "edf_dir",
        type=Path,
        help='Folder containing the .edf files',
    )
    p.add_argument(
        "--outdir", "-o",
        type=Path,
        default=None,
        help="Destination folder for .qrs files (default: same folder as each EDF).",
    )
    p.add_argument(
        "--channel", "-c",
        default=None,
        help='ECG channel name (auto-detected if omitted). E.g. --channel "ECG II"',
    )
    p.add_argument(
        "--method", "-m",
        default="pantompkins",
        choices=SUPPORTED_METHODS,
        help="QRS detection algorithm (default: pantompkins).",
    )
    p.add_argument(
        "--list-channels",
        action="store_true",
        help="Print channel names from the first EDF and exit (useful for debugging).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_dependencies()

    if not args.edf_dir.is_dir():
        print(f"[!] Not a directory: {args.edf_dir}")
        sys.exit(1)

    # ── Diagnostic mode: just list channels ──────────────────────────────────
    if args.list_channels:
        import pyedflib
        first = sorted(args.edf_dir.glob("*.edf"))
        if not first:
            print("[!] No EDF files found.")
            sys.exit(1)
        with pyedflib.EdfReader(str(first[0])) as f:
            labels = f.getSignalLabels()
        print(f"\nChannels in {first[0].name}:")
        for i, lbl in enumerate(labels):
            marker = "  ← likely ECG" if find_ecg_channel([lbl]) else ""
            print(f"  [{i}] '{lbl}'{marker}")
        sys.exit(0)

    # ── Normal processing ─────────────────────────────────────────────────────
    process_folder(
        edf_dir=args.edf_dir,
        outdir=args.outdir,
        channel=args.channel,
        method=args.method,
    )