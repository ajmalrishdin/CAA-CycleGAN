#!/usr/bin/env python3
"""
Final inference pipeline: Load AECG → Extract FECG → Interactive viewer

Combines:
  - ECG file loading (plot_signal.py)
  - FECG extraction using trained generator (temp.py)
  - Interactive visualization of both signals
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import pyedflib
import wfdb
from scipy import signal as scipy_signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils.sagan_models import Generator
from Utils.device_utils import get_device_backend, resolve_device
from npz import default_session_path, save_view_session, show_interactive_viewer


# ============================================================================
# FILE LOADING (from plot_signal.py)
# ============================================================================

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


# ============================================================================
# FECG EXTRACTION (from temp.py)
# ============================================================================

def infer_checkpoint_role(state_dict):
    """Infer whether a .pth checkpoint is for a generator or discriminator."""
    if not state_dict:
        return "unknown"
    
    first_key = next(iter(state_dict))
    if first_key.startswith("conv1DWithSINE0") or first_key.startswith("attn1"):
        return "discriminator"
    if first_key.startswith("conv1DWithSINE_l0") or first_key.startswith("transformer"):
        return "generator"
    return "unknown"


def load_generator_from_pth(pth_path, device='cpu'):
    """Load pretrained generator from .pth file."""
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    print(f"[1/3] Loading generator from: {pth_path}")
    
    # Initialize generator architecture
    generator = Generator(
        batch_size=1,
        image_size=64,
        z_dim=128,
        conv_dim=64
    )
    
    # Load .pth weights
    state_dict = torch.load(pth_path, map_location=device)
    
    # Validate it's a generator, not discriminator
    checkpoint_role = infer_checkpoint_role(state_dict)
    if checkpoint_role == "discriminator":
        raise ValueError(
            f"{pth_path} appears to be a discriminator checkpoint. "
            f"Use a file named like '*_G_AECG2FECG.pth' instead."
        )
    
    # Remove 'module.' prefix if model was saved with DataParallel
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights into generator
    try:
        generator.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not load weights from {pth_path}. "
            f"Architecture mismatch or corrupted file. Original error: {e}"
        ) from e
    
    generator = generator.to(device)
    generator.eval()
    
    print(f"✓ Generator loaded on device: {device}\n")
    return generator


def _bandpass(signal_1d: np.ndarray, fs: float, low_hz: float = 1.0, high_hz: float = 100.0, order: int = 3) -> np.ndarray:
    """Bandpass a signal with safe cutoffs for the current sample rate."""
    if fs <= 0:
        return signal_1d

    nyquist = 0.5 * fs
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    if low >= high:
        return signal_1d

    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, signal_1d).astype(np.float32)


def _robust_standardize(signal_1d: np.ndarray) -> np.ndarray:
    """Robustly standardize using median and MAD before model inference."""
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    median = float(np.median(signal_1d))
    mad = float(np.median(np.abs(signal_1d - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        std = float(np.std(signal_1d))
        scale = std if std > 1e-8 else 1.0
    return ((signal_1d - median) / scale).astype(np.float32)


def _minmax_to_minus1_plus1(window: np.ndarray) -> np.ndarray:
    """Normalize a window to [-1, 1], matching the training data scaling."""
    w_min = float(window.min())
    w_max = float(window.max())
    w_range = w_max - w_min
    if w_range < 1e-8:
        return np.zeros_like(window, dtype=np.float32)
    return (2.0 * (window - w_min) / w_range - 1.0).astype(np.float32)


def extract_fecg_segment(
    aecg_signal,
    generator,
    device='cpu',
    input_fs=1000.0,
    target_fs=200.0,
    window_size=128,
    overlap=96,
    apply_input_bandpass=True,
    apply_output_bandpass=True,
):
    """
    Extract FECG from AECG signal by processing overlapping windows.
    
    Args:
        aecg_signal: 1D numpy array of AECG signal
        generator: Loaded generator model
        device: torch device
        window_size: Size of each window (must match training, typically 128)
        overlap: Number of samples to overlap between windows
    
    Returns:
        fecg_signal: Extracted FECG signal (same length as input)
    """
    print(f"[2/3] Extracting FECG from AECG signal...")

    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if input_fs <= 0 or target_fs <= 0:
        raise ValueError("input_fs and target_fs must be > 0")

    raw = np.asarray(aecg_signal, dtype=np.float32).reshape(-1)
    original_length = raw.shape[0]

    # Match model training distribution: denoise/band-limit then robust standardization.
    prepared = _bandpass(raw, fs=input_fs) if apply_input_bandpass else raw
    prepared = _robust_standardize(prepared)

    # Important: training windows were created after downsampling 1000 Hz -> 200 Hz.
    if abs(input_fs - target_fs) > 1e-6:
        target_length = int(round(original_length * target_fs / input_fs))
        target_length = max(target_length, window_size)
        work = scipy_signal.resample(prepared, target_length).astype(np.float32)
    else:
        work = prepared
        target_length = work.shape[0]

    if target_length < window_size:
        pad = window_size - target_length
        mode = 'reflect' if target_length > 1 else 'edge'
        work = np.pad(work, (0, pad), mode=mode)
        target_length = work.shape[0]

    stride = window_size - overlap
    starts = list(range(0, target_length - window_size + 1, stride))
    if not starts or starts[-1] + window_size < target_length:
        starts.append(target_length - window_size)

    # Hann-weighted overlap-add removes hard stitching artifacts between windows.
    fecg_ola = np.zeros(target_length, dtype=np.float32)
    weight = np.zeros(target_length, dtype=np.float32)
    blend = np.hanning(window_size).astype(np.float32)
    if np.all(blend == 0):
        blend = np.ones(window_size, dtype=np.float32)

    with torch.no_grad():
        for start in starts:
            end = start + window_size
            window = work[start:end]
            norm_window = _minmax_to_minus1_plus1(window)
            tensor_in = torch.from_numpy(norm_window[None, None, :]).to(device)

            pred = generator(tensor_in).squeeze().detach().cpu().numpy().astype(np.float32)
            pred = np.clip(pred, -3.0, 3.0)

            fecg_ola[start:end] += pred * blend
            weight[start:end] += blend

    weight = np.where(weight < 1e-8, 1.0, weight)
    fecg_200 = fecg_ola / weight

    if apply_output_bandpass:
        fecg_200 = _bandpass(fecg_200, fs=target_fs, low_hz=2.0, high_hz=80.0, order=3)

    # Keep output in a stable normalized scale rather than mapping to AECG amplitude.
    fecg_200 = _robust_standardize(fecg_200)

    if abs(input_fs - target_fs) > 1e-6:
        fecg_signal = scipy_signal.resample(fecg_200, original_length).astype(np.float32)
    else:
        fecg_signal = fecg_200.astype(np.float32)

    print(f"✓ FECG extracted (signal length: {original_length} samples, model fs: {target_fs:.1f} Hz)\n")
    return fecg_signal


# Note: Using the shared `show_interactive_viewer` from `npz.py` instead of a
# local copy to avoid duplication and keep viewer behavior consistent.


# ============================================================================
# INTERACTIVE CHANNEL SELECTION
# ============================================================================

def select_channel(signals, labels):
    """Display available channels and get user selection."""
    print("\n" + "="*60)
    print("AVAILABLE CHANNELS")
    print("="*60)
    
    for i, label in enumerate(labels):
        print(f"  [{i}] {label}")
    
    print()
    while True:
        try:
            choice = input(f"Select channel to extract FECG from (0-{len(labels)-1}): ").strip()
            channel_idx = int(choice)
            if 0 <= channel_idx < len(labels):
                return channel_idx, labels[channel_idx]
            else:
                print(f"Please enter a number between 0 and {len(labels)-1}")
        except ValueError:
            print("Invalid input. Please enter a valid channel number.")


def select_model_path():
    """Get .pth model path from user."""
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    while True:
        pth_path = input("Enter path to .pth generator model: ").strip()
        if os.path.exists(pth_path):
            return pth_path
        else:
            print(f"File not found: {pth_path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract FECG from AECG and display in interactive viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  1. Load .edf or .dat ECG file
  2. Select which channel to extract FECG from
  3. Select trained generator .pth model
  4. Extract FECG using the model
  5. View both signals in interactive viewer

EXAMPLE:
  python inference/final_inference.py \\
    --input Databases/ADFECGDB/r01.edf \\
    --pth_path models/sagan_1/100_G_AECG2FECG.pth \\
    --device cpu

INTERACTIVE CONTROLS:
  - Left/Right arrows: Nudge ±0.5s
  - Ctrl+Left/Ctrl+Right: Jump ±4s
  - S key: Save current view as PNG
  - Button controls at bottom for navigation
        """
    )
    
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to input .edf or .dat file'
    )
    parser.add_argument(
        '--pth_path', type=str,
        help='Path to trained generator .pth file (interactive prompt if not provided)'
    )
    parser.add_argument(
        '--device', type=str, default=get_device_backend(), choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: DEVICE_BACKEND from .env)'
    )
    parser.add_argument(
        '--window-size', type=float, default=4.0,
        help='Viewer window size in seconds (default: 4.0)'
    )
    parser.add_argument(
        '--model-fs', type=float, default=200.0,
        help='Sampling rate used by model training windows (default: 200 Hz)'
    )
    parser.add_argument(
        '--segment-size', type=int, default=128,
        help='Inference segment size in samples at --model-fs (default: 128)'
    )
    parser.add_argument(
        '--overlap', type=int, default=96,
        help='Overlap between inference segments (default: 96)'
    )
    parser.add_argument(
        '--no-input-bandpass', action='store_true',
        help='Disable input bandpass preprocessing before inference'
    )
    parser.add_argument(
        '--no-output-bandpass', action='store_true',
        help='Disable output bandpass postprocessing after inference'
    )
    parser.add_argument(
        '--save-dir', type=str, default='outputs/fecg_extraction',
        help='Directory to save PNG snapshots (default: outputs/fecg_extraction)'
    )
    parser.add_argument(
        '--save-session', type=str, default=None,
        help='Optional .npz file to save the extracted AECG/FECG session. If omitted, a session is saved automatically.'
    )
    parser.add_argument(
        '--no-save-session', action='store_true',
        help='Disable automatic session saving.'
    )
    # DAT file specific args
    parser.add_argument(
        '--raw', action='store_true',
        help='Treat .dat input as raw binary instead of WFDB'
    )
    parser.add_argument(
        '--auto-fallback', action='store_true',
        help='Fall back to raw DAT loading when WFDB parsing fails'
    )
    parser.add_argument(
        '--raw-int16', action='store_true',
        help='Read raw DAT data as int16 instead of float32'
    )
    parser.add_argument(
        '--n-channels', type=int, default=4,
        help='Channel count for raw DAT fallback (default: 4)'
    )
    parser.add_argument(
        '--fs', type=float, default=1000.0,
        help='Sampling rate for raw DAT fallback (default: 1000 Hz)'
    )
    parser.add_argument(
        '--gain', type=float, default=1000.0,
        help='Gain for raw int16 DAT values (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Load input file
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print("\n" + "="*60)
    print("FECG EXTRACTION & VISUALIZATION PIPELINE")
    print("="*60)
    
    print(f"\nLoading {input_path.name}...")
    signals, fs, labels, stem = load_file(input_path, args)
    duration = signals.shape[1] / fs if fs > 0 else signals.shape[1]
    
    print(f"  Channels: {signals.shape[0]}")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {duration:.2f} s")
    
    # Select channel
    channel_idx, channel_label = select_channel(signals, labels)
    aecg_signal = signals[channel_idx]
    print(f"\n✓ Selected channel: {channel_label}")
    
    # Get model path
    if args.pth_path:
        pth_path = args.pth_path
    else:
        pth_path = select_model_path()
    
    print(f"✓ Model path: {pth_path}\n")
    
    # Setup device
    device = resolve_device(args.device)
    print(f"Using device: {device}\n")
    
    # Load generator
    generator = load_generator_from_pth(pth_path, device=device)
    
    # Extract FECG
    fecg_signal = extract_fecg_segment(
        aecg_signal,
        generator,
        device=device,
        input_fs=fs,
        target_fs=args.model_fs,
        window_size=args.segment_size,
        overlap=args.overlap,
        apply_input_bandpass=not args.no_input_bandpass,
        apply_output_bandpass=not args.no_output_bandpass,
    )

    # Optionally save extracted session (AECG + FECG)
    save_sessions = not args.no_save_session
    if save_sessions:
        session_path = Path(args.save_session).expanduser().resolve() if args.save_session else default_session_path(channel_label, args.save_dir)
        saved_path = save_view_session(
            session_path,
            aecg_signal,
            fecg_signal,
            channel_label,
            fs,
            duration,
            args.save_dir,
        )
        print(f"Saved session: {saved_path}")
    
    # Show results
    print("="*60)
    print("FECG EXTRACTION COMPLETE")
    print("="*60)
    print(f"Channel:        {channel_label}")
    print(f"AECG samples:   {len(aecg_signal)}")
    print(f"FECG samples:   {len(fecg_signal)}")
    print(f"Duration:       {duration:.2f} s")
    print("="*60 + "\n")
    
    # Show interactive viewer
    print("Opening interactive viewer...")
    print("Controls: Arrow keys to navigate, 'S' to save, Ctrl+Arrow for big jumps\n")
    show_interactive_viewer(
        aecg_signal,
        fecg_signal,
        channel_label,
        fs,
        duration,
        args.save_dir,
        initial_window_size=args.window_size,
    )


if __name__ == '__main__':
    main()
