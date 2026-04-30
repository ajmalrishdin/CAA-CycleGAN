#!/usr/bin/env python3
"""
Convert EDF and DAT files to NPY format for FECG generation.

Supports:
- EDF files (using pyedflib)
- DAT files (binary ECG data from PhysioNet)
- Signal filtering and normalization
- Windowing into fixed-size segments

Usage:
    python convert_ecg_to_npy.py --input_file signal.edf --output_dir ./npy_output/
    python convert_ecg_to_npy.py --input_dir ./Databases/ADFECGDB/ --output_dir ./npy_signals/
    python convert_ecg_to_npy.py --input_file signal.dat --sampling_rate 1000 --channel 0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False
    print("Warning: pyedflib not installed. EDF support disabled. Install with: pip install pyedflib")


# ============================================================================
# Signal Processing Functions
# ============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter to signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def normalize_signal(sig, method='minmax'):
    """
    Normalize signal to specified range.
    
    Args:
        sig: Input signal (numpy array)
        method: 'minmax' (0-1) or 'zscore' (mean=0, std=1)
    
    Returns:
        Normalized signal
    """
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
            normalized = scaler.fit_transform(sig).flatten()
        else:
            normalized = scaler.fit_transform(sig)
        return normalized
    elif method == 'zscore':
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    else:
        return sig


def window_signal(signal_data, window_size=128, overlap=0):
    """
    Split signal into overlapping or non-overlapping windows.
    
    Args:
        signal_data: Input signal (1D or 2D array)
        window_size: Size of each window
        overlap: Number of overlapping samples between windows (0 = no overlap)
    
    Returns:
        Array of windows with shape (num_windows, num_channels, window_size)
    """
    if signal_data.ndim == 1:
        signal_data = signal_data[np.newaxis, :]  # Add channel dimension
    
    num_channels, signal_length = signal_data.shape
    step = window_size - overlap
    
    windows = []
    for start in range(0, signal_length - window_size + 1, step):
        window = signal_data[:, start:start + window_size]
        windows.append(window)
    
    if len(windows) == 0:
        raise ValueError(f"Signal too short ({signal_length}) for window size {window_size}")
    
    return np.array(windows)  # Shape: (num_windows, num_channels, window_size)


# ============================================================================
# EDF File Handling
# ============================================================================

def read_edf_file(edf_path, channels=None, segment_start=None, segment_duration=None):
    """
    Read EDF file and extract signals.
    
    Args:
        edf_path: Path to EDF file
        channels: List of channel indices to extract (None = all)
        segment_start: Start time in seconds (None = from beginning)
        segment_duration: Duration in seconds (None = entire file)
    
    Returns:
        signals: Extracted signals (num_channels, num_samples)
        info: Dictionary with file info
    """
    if not HAS_PYEDFLIB:
        raise ImportError("pyedflib is required for EDF support. Install with: pip install pyedflib")
    
    edf_path = str(edf_path)  # Convert Path object to string if needed
    print(f"Reading EDF file: {edf_path}")
    
    reader = pyedflib.EdfReader(edf_path)
    n_signals = reader.signals_in_file
    signal_labels = reader.getSignalLabels()
    signal_freqs = reader.getSampleFrequencies()
    
    print(f"  Number of signals: {n_signals}")
    print(f"  Signal labels: {signal_labels}")
    print(f"  Sampling frequencies: {signal_freqs}")
    
    # Determine which channels to read
    if channels is None:
        channels = list(range(n_signals))
    
    channels = [c for c in channels if c < n_signals]
    if not channels:
        channels = [0]
    
    # Read signals
    signals = []
    for ch in channels:
        full_signal = reader.readSignal(ch)
        fs = signal_freqs[ch]
        
        # Extract segment if specified
        if segment_start is not None and segment_duration is not None:
            start_sample = int(segment_start * fs)
            end_sample = int((segment_start + segment_duration) * fs)
            segment = full_signal[start_sample:end_sample]
            signals.append(segment)
            print(f"  Channel {ch}: Extracted {end_sample-start_sample} samples ({segment_duration}s)")
        else:
            signals.append(full_signal)
            print(f"  Channel {ch}: Read {len(full_signal)} samples")
    
    reader.close()
    
    # Stack into single array (num_channels, num_samples)
    signals = np.array(signals)
    
    info = {
        'file': edf_path,
        'n_signals': n_signals,
        'channels': channels,
        'signal_labels': [signal_labels[c] for c in channels],
        'sampling_frequencies': [signal_freqs[c] for c in channels],
        'shape': signals.shape
    }
    
    return signals, info


# ============================================================================
# DAT File Handling
# ============================================================================

def read_dat_file(dat_path, num_channels, sampling_rate, data_type=np.int16):
    """
    Read binary DAT file (PhysioNet format).
    
    Args:
        dat_path: Path to DAT file
        num_channels: Number of channels in the file
        sampling_rate: Sampling rate in Hz
        data_type: Data type (default: int16)
    
    Returns:
        signals: Extracted signals (num_channels, num_samples)
        info: Dictionary with file info
    """
    dat_path = str(dat_path)  # Convert Path object to string if needed
    print(f"Reading DAT file: {dat_path}")
    
    # Read binary data
    data = np.fromfile(dat_path, dtype=data_type)
    
    num_samples = len(data) // num_channels
    if len(data) % num_channels != 0:
        print(f"  Warning: File size not multiple of num_channels. Trimming...")
        data = data[:num_samples * num_channels]
    
    # Reshape to (num_samples, num_channels) then transpose
    data = data.reshape(num_samples, num_channels).T
    
    print(f"  Channels: {num_channels}")
    print(f"  Samples per channel: {num_samples}")
    print(f"  Duration: {num_samples / sampling_rate:.2f}s")
    print(f"  Shape: {data.shape}")
    
    info = {
        'file': dat_path,
        'n_channels': num_channels,
        'n_samples': num_samples,
        'sampling_rate': sampling_rate,
        'duration_seconds': num_samples / sampling_rate,
        'shape': data.shape
    }
    
    return data, info


# ============================================================================
# File Processing
# ============================================================================

def process_signal(signals, info, filtering=True, window_size=None):
    """
    Process signals: filter, normalize, and optionally window.
    
    Args:
        signals: Input signals (num_channels, num_samples)
        info: File info dictionary
        filtering: Apply bandpass filter
        window_size: If not None, split into windows of this size
    
    Returns:
        Processed signals
    """
    fs = info.get('sampling_frequencies', [1000])[0] if isinstance(info.get('sampling_frequencies', [1000]), list) else info.get('sampling_rate', 1000)
    
    # Apply filtering if requested
    if filtering:
        print("Applying bandpass filter (1-100 Hz)...")
        signals = butter_bandpass_filter(signals, lowcut=1, highcut=100, fs=fs)
    
    # Normalize
    print("Normalizing signals...")
    if signals.ndim == 1:
        signals = normalize_signal(signals, method='minmax')
    else:
        for i in range(signals.shape[0]):
            signals[i] = normalize_signal(signals[i], method='minmax')
    
    # Window signals if requested
    if window_size is not None:
        print(f"Windowing signals into {window_size}-sample windows...")
        signals = window_signal(signals, window_size=window_size)
        print(f"  Generated {signals.shape[0]} windows")
    
    return signals


def convert_file_to_npy(input_path, output_dir, file_type=None, **kwargs):
    """
    Convert EDF or DAT file to NPY format.
    
    Args:
        input_path: Path to input file
        output_dir: Output directory
        file_type: 'edf' or 'dat' (auto-detect if None)
        **kwargs: Additional arguments (channels, window_size, etc.)
    
    Returns:
        Path to output NPY file
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect file type
    if file_type is None:
        if input_path.suffix.lower() == '.edf':
            file_type = 'edf'
        elif input_path.suffix.lower() == '.dat':
            file_type = 'dat'
        else:
            raise ValueError(f"Unknown file type: {input_path.suffix}. Specify with --file_type")
    
    print(f"\n{'='*60}")
    print(f"Converting: {input_path.name}")
    print(f"Type: {file_type.upper()}")
    print(f"{'='*60}")
    
    try:
        # Read file
        if file_type.lower() == 'edf':
            signals, info = read_edf_file(
                input_path,
                channels=kwargs.get('channels'),
                segment_start=kwargs.get('segment_start'),
                segment_duration=kwargs.get('segment_duration')
            )
        elif file_type.lower() == 'dat':
            signals, info = read_dat_file(
                input_path,
                num_channels=kwargs.get('num_channels', 1),
                sampling_rate=kwargs.get('sampling_rate', 1000),
                data_type=np.int16
            )
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        # Process signals
        window_size = kwargs.get('window_size', None)
        filtering = kwargs.get('filtering', True)
        signals = process_signal(signals, info, filtering=filtering, window_size=window_size)
        
        # Save to NPY
        output_file = output_dir / f"{input_path.stem}.npy"
        np.save(output_file, signals.astype(np.float32))
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Output shape: {signals.shape}")
        print(f"  Output dtype: {signals.dtype}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        return output_file
    
    except Exception as e:
        print(f"\n✗ Error processing {input_path.name}: {e}")
        raise


def batch_convert(input_dir, output_dir, pattern='*.edf', **kwargs):
    """
    Convert all matching files in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        pattern: File pattern to match (e.g., '*.edf', '*.dat')
        **kwargs: Additional arguments
    
    Returns:
        List of output file paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return []
    
    print(f"\nFound {len(files)} files to convert")
    
    output_files = []
    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", end=" ")
        try:
            output_file = convert_file_to_npy(file, output_dir, **kwargs)
            output_files.append(output_file)
        except Exception as e:
            print(f"Skipped due to error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Batch conversion complete: {len(output_files)}/{len(files)} files processed")
    print(f"{'='*60}")
    
    return output_files


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert EDF and DAT files to NPY format for FECG generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single EDF file
  python convert_ecg_to_npy.py --input_file Databases/ADFECGDB/r01.edf --output_dir ./npy_output/
  
  # Convert single EDF file with windowing (128-sample windows for FECG model)
  python convert_ecg_to_npy.py --input_file Databases/ADFECGDB/r01.edf \\
    --output_dir ./npy_output/ --window_size 128
  
  # Convert single EDF file, extract specific channel
  python convert_ecg_to_npy.py --input_file Databases/ADFECGDB/r01.edf \\
    --output_dir ./npy_output/ --channels 0
  
  # Convert all EDF files in a directory
  python convert_ecg_to_npy.py --input_dir Databases/ADFECGDB/ \\
    --output_dir ./npy_output/ --batch
  
  # Convert DAT file
  python convert_ecg_to_npy.py --input_file signal.dat --output_dir ./npy_output/ \\
    --file_type dat --num_channels 1 --sampling_rate 1000
  
  # Extract 60-second segment from EDF
  python convert_ecg_to_npy.py --input_file record.edf --output_dir ./npy_output/ \\
    --segment_start 30 --segment_duration 60
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_file', type=str,
                            help='Path to input EDF or DAT file')
    input_group.add_argument('--input_dir', type=str,
                            help='Input directory for batch conversion')
    
    parser.add_argument('--output_dir', type=str, default='./npy_ecg_signals/',
                       help='Output directory (default: ./npy_ecg_signals/)')
    
    # File type arguments
    parser.add_argument('--file_type', type=str, choices=['edf', 'dat'],
                       help='File type (auto-detect if not specified)')
    
    # EDF arguments
    parser.add_argument('--channels', type=int, nargs='+', default=None,
                       help='Channel indices to extract (default: all)')
    parser.add_argument('--segment_start', type=float, default=None,
                       help='Start time in seconds for EDF segment extraction')
    parser.add_argument('--segment_duration', type=float, default=None,
                       help='Duration in seconds for EDF segment extraction')
    
    # DAT arguments
    parser.add_argument('--num_channels', type=int, default=1,
                       help='Number of channels for DAT files (default: 1)')
    parser.add_argument('--sampling_rate', type=int, default=1000,
                       help='Sampling rate in Hz for DAT files (default: 1000)')
    
    # Processing arguments
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size for signal segmentation (e.g., 128 for FECG model)')
    parser.add_argument('--no_filtering', action='store_true',
                       help='Skip bandpass filtering (1-100 Hz)')
    
    # Batch arguments
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process all matching files in directory')
    parser.add_argument('--pattern', type=str, default='*.edf',
                       help='File pattern for batch mode (default: *.edf)')
    
    args = parser.parse_args()
    
    # Prepare kwargs
    kwargs = {
        'channels': args.channels,
        'segment_start': args.segment_start,
        'segment_duration': args.segment_duration,
        'num_channels': args.num_channels,
        'sampling_rate': args.sampling_rate,
        'window_size': args.window_size,
        'filtering': not args.no_filtering,
    }
    
    # Process files
    try:
        if args.batch:
            output_files = batch_convert(
                args.input_dir,
                args.output_dir,
                pattern=args.pattern,
                file_type=args.file_type,
                **kwargs
            )
            
            print(f"\nOutput files:")
            for f in output_files:
                print(f"  - {f}")
        else:
            output_file = convert_file_to_npy(
                args.input_file,
                args.output_dir,
                file_type=args.file_type,
                **kwargs
            )
            
            print(f"\nOutput file: {output_file}")
            print("\nYou can now use this file with FECG generation:")
            print(f"  python generate_fecg_inference.py \\")
            print(f"    --model_path models/sagan_1_SynDB1_bs128_1n8/68_G_AECG2FECG.pth \\")
            print(f"    --aecg_file {output_file} \\")
            print(f"    --visualize")
    
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
