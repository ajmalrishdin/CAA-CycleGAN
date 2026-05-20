#!/usr/bin/env python3
"""Plot EDF or DAT ECG recordings with interactive matplotlib viewer.

The script supports:
- EDF files via pyedflib
- WFDB-style DAT files with an accompanying .hea file via wfdb
- Raw binary DAT files as a fallback when the record is not WFDB-formatted

Features:
- Interactive time-window navigation (arrow keys)
- Up to 5 channels displayed stacked vertically
- Keyboard and button controls for browsing
- Visual style matching the reference ECG viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pyedflib
import wfdb


def load_edf(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load EDF file and return signals, sample rate, and channel labels."""
    reader = pyedflib.EdfReader(str(path))
    try:
        channel_count = reader.signals_in_file
        signals = np.array([reader.readSignal(index) for index in range(channel_count)], dtype=np.float32)
        fs = float(reader.getSampleFrequency(0))
        labels = list(reader.getSignalLabels())
        if len(labels) < channel_count:
            labels.extend(f"ch_{index}" for index in range(len(labels), channel_count))
        return signals, fs, labels
    finally:
        reader.close()


def load_wfdb_dat(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load WFDB DAT file and return signals, sample rate, and channel labels."""
    record = wfdb.rdrecord(str(path.with_suffix("")))
    signals = np.asarray(record.p_signal.T, dtype=np.float32)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    fs = float(record.fs)
    labels = list(record.sig_name) if getattr(record, "sig_name", None) else []
    if len(labels) < signals.shape[0]:
        labels.extend(f"ch_{index}" for index in range(len(labels), signals.shape[0]))
    return signals, fs, labels


def load_raw_dat(path: Path, dtype: str, n_channels: int, fs: float, gain: float) -> tuple[np.ndarray, float, list[str]]:
    """Load raw binary DAT file and return signals, sample rate, and channel labels."""
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    if raw.size == 0:
        raise ValueError(f"No samples found in raw DAT file: {path}")

    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")

    sample_count = raw.size // n_channels
    if sample_count == 0:
        raise ValueError(
            f"DAT file {path} does not contain enough samples for {n_channels} channels"
        )

    trimmed = raw[: sample_count * n_channels]
    signals = trimmed.reshape(sample_count, n_channels).T.astype(np.float32)
    if gain != 0:
        signals = signals / float(gain)
    labels = [f"ch_{index}" for index in range(n_channels)]
    return signals, float(fs), labels


def load_dat(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, float, list[str]]:
    """Load DAT file (WFDB or raw binary)."""
    wfdb_header = path.with_suffix(".hea")
    if not args.raw and wfdb_header.exists():
        try:
            return load_wfdb_dat(path)
        except Exception:
            if not args.auto_fallback:
                raise

    dtype = "int16" if args.raw_int16 else "float32"
    return load_raw_dat(path, dtype=dtype, n_channels=args.n_channels, fs=args.fs, gain=args.gain)


def load_file(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, float, list[str], str]:
    """Load signal file and return signals, sample rate, labels, and filename."""
    suffix = path.suffix.lower()
    if suffix == ".edf":
        signals, fs, labels = load_edf(path)
    elif suffix == ".dat":
        signals, fs, labels = load_dat(path, args)
    else:
        raise ValueError("Unsupported file type. Please provide an .edf or .dat file.")
    
    return signals, fs, labels, path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive EDF/DAT ECG viewer")
    parser.add_argument("input_file", help="Path to the .edf or .dat file")
    parser.add_argument("--window-size", type=float, default=4.0, help="Visible window width in seconds (default 4.0)")
    parser.add_argument("--step-small", type=float, default=0.5, help="Arrow key nudge in seconds (default 0.5)")
    parser.add_argument("--max-channels", type=int, default=5, help="Maximum channels to display (default 5)")
    parser.add_argument("--raw", action="store_true", help="Treat .dat input as raw binary instead of WFDB")
    parser.add_argument("--auto-fallback", action="store_true", help="Fall back to raw DAT loading when WFDB parsing fails")
    parser.add_argument("--raw-int16", action="store_true", help="Read raw DAT data as int16 instead of float32")
    parser.add_argument("--n-channels", type=int, default=4, help="Channel count for raw DAT fallback")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate for raw DAT fallback")
    parser.add_argument("--gain", type=float, default=1000.0, help="Gain used to scale raw int16 DAT values")
    parser.add_argument("--save-dir", default="outputs/matplotlib", help="Directory to save PNG snapshots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the file
    print(f"Loading {input_path.name}...")
    signals, fs, labels, stem = load_file(input_path, args)
    
    # Limit to max_channels
    n_channels_to_show = min(signals.shape[0], args.max_channels)
    signals = signals[:n_channels_to_show]
    labels = labels[:n_channels_to_show]
    
    duration = signals.shape[1] / fs if fs > 0 else signals.shape[1]
    
    print(f"  Channels: {n_channels_to_show}")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {duration:.2f} s")
    
    # State for interactive viewer
    state = {
        't_start': 0.0,
        'window_size': args.window_size,
        'step_small': args.step_small,
        'step_big': args.window_size,
    }
    
    # Create figure and axes
    fig, axes = plt.subplots(n_channels_to_show, 1, figsize=(14, 2.5 * n_channels_to_show), sharex=True)
    if n_channels_to_show == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.15, top=0.90, hspace=0.2)
    
    # Setup button controls at bottom
    ax_left = plt.axes([0.15, 0.04, 0.08, 0.05])
    ax_right = plt.axes([0.68, 0.04, 0.08, 0.05])
    ax_save = plt.axes([0.38, 0.04, 0.08, 0.05])
    
    btn_left = Button(ax_left, '◀ ' + f'{args.step_small}s')
    btn_right = Button(ax_right, f'{args.step_small}s' + ' ▶')
    btn_save = Button(ax_save, 'Save PNG')
    
    # Status text
    ax_status = plt.axes([0.38, 0.58, 0.25, 0.04])
    ax_status.axis('off')
    status_text = ax_status.text(0.5, 0.5, '', ha='center', va='center', 
                                  fontsize=9, transform=ax_status.transAxes)
    
    def shift_time(delta: float) -> None:
        """Shift the time window."""
        new_t = state['t_start'] + delta
        new_t = max(0.0, min(new_t, max(0.0, duration - state['window_size'])))
        state['t_start'] = round(new_t, 2)
        redraw()
    
    def save_png() -> None:
        """Save current view as PNG."""
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        t0 = state['t_start']
        t1 = t0 + state['window_size']
        out_path = save_dir / f"{stem}_t{t0:06.1f}-{t1:06.1f}s.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        status_text.set_text(f"✓ Saved to {out_path.name}")
        fig.canvas.draw_idle()
    
    def redraw() -> None:
        """Redraw the signal plots."""
        t0 = state['t_start']
        t1 = t0 + state['window_size']
        
        for ax in axes:
            ax.cla()
        
        s0 = int(t0 * fs) if fs > 0 else int(t0)
        s1 = int(min(t1, duration) * fs) if fs > 0 else int(min(t1, duration))
        
        for i, ax in enumerate(axes):
            if i < n_channels_to_show:
                segment = signals[i, s0:s1]
                time_pts = np.linspace(t0, t0 + len(segment) / fs if fs > 0 else len(segment), len(segment), endpoint=False)
                
                ax.plot(time_pts, segment, linewidth=0.9, color='navy')
                ax.set_xlim(t0, t1)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel(f"{labels[i]}\n(μV)", fontsize=9)
                
                if i < n_channels_to_show - 1:
                    ax.set_xticklabels([])
        
        axes[-1].set_xlabel("Time (seconds)", fontsize=10)
        
        # Title with time window info
        step_info = f"← {args.step_small}s | {args.step_small}s →"
        fig.suptitle(
            f"File: {stem}\n"
            f"Window: {t0:.2f}–{t1:.2f} s   [{step_info}]",
            fontsize=11, fontweight='bold'
        )
        
        status_text.set_text(f"Time: {t0:.2f}s – {t1:.2f}s / {duration:.2f}s total")
        fig.canvas.draw_idle()
    
    # Wire up callbacks
    btn_left.on_clicked(lambda e: shift_time(-state['step_small']))
    btn_right.on_clicked(lambda e: shift_time(state['step_small']))
    btn_save.on_clicked(lambda e: save_png())
    
    def on_key(event):
        if not event.key:
            return
        
        ctrl = event.key.startswith('ctrl+')
        key = event.key.replace('ctrl+', '')
        
        if ctrl and key == 'left':
            shift_time(-state['step_big'])
        elif ctrl and key == 'right':
            shift_time(state['step_big'])
        elif key == 'left':
            shift_time(-state['step_small'])
        elif key == 'right':
            shift_time(state['step_small'])
        elif key == 's':
            save_png()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    redraw()
    plt.show()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())