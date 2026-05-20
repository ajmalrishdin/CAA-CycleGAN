#!/usr/bin/env python3
"""Unified viewer launcher for NPZ sessions, EDF files, and DAT files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ECG viewer launcher")
    parser.add_argument("input_file", help="Path to a .npz, .edf, or .dat file")

    parser.add_argument(
        "--shared-y-axis",
        action="store_true",
        help="For .npz sessions, force the AECG and FECG plots to share y-axis limits.",
    )

    parser.add_argument(
        "--window-size",
        type=float,
        default=4.0,
        help="Visible window width in seconds for EDF/DAT viewing (default 4.0)",
    )
    parser.add_argument(
        "--step-small",
        type=float,
        default=0.5,
        help="Arrow key nudge in seconds for EDF/DAT viewing (default 0.5)",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=5,
        help="Maximum channels to display for EDF/DAT viewing (default 5)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat .dat input as raw binary instead of WFDB",
    )
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        help="Fall back to raw DAT loading when WFDB parsing fails",
    )
    parser.add_argument(
        "--raw-int16",
        action="store_true",
        help="Read raw DAT data as int16 instead of float32",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=4,
        help="Channel count for raw DAT fallback",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Sampling rate for raw DAT fallback",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1000.0,
        help="Gain used to scale raw int16 DAT values",
    )
    parser.add_argument(
        "--save-dir",
        default="outputs/matplotlib",
        help="Directory to save PNG snapshots for EDF/DAT viewing",
    )
    return parser


def dispatch_viewer(input_file: Path, args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent.parent
    suffix = input_file.suffix.lower()

    if suffix == ".npz":
        script = Path(__file__).with_name("npz.py")
        command = [sys.executable, str(script), "--session", str(input_file)]
        if args.shared_y_axis:
            command.append("--shared-y-axis")
    elif suffix in {".edf", ".dat"}:
        script = Path(__file__).with_name("edfndat.py")
        command = [sys.executable, str(script), str(input_file)]
        if args.window_size != 4.0:
            command.extend(["--window-size", str(args.window_size)])
        if args.step_small != 0.5:
            command.extend(["--step-small", str(args.step_small)])
        if args.max_channels != 5:
            command.extend(["--max-channels", str(args.max_channels)])
        if args.raw:
            command.append("--raw")
        if args.auto_fallback:
            command.append("--auto-fallback")
        if args.raw_int16:
            command.append("--raw-int16")
        if args.n_channels != 4:
            command.extend(["--n-channels", str(args.n_channels)])
        if args.fs != 1000.0:
            command.extend(["--fs", str(args.fs)])
        if args.gain != 1000.0:
            command.extend(["--gain", str(args.gain)])
        if args.save_dir != "outputs/matplotlib":
            command.extend(["--save-dir", args.save_dir])
    else:
        raise ValueError("Unsupported file type. Please provide a .npz, .edf, or .dat file.")

    completed = subprocess.run(command, cwd=str(project_root), check=False)
    return int(completed.returncode)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input_file).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Launching viewer for {input_path.name}...")
    return dispatch_viewer(input_path, args)


if __name__ == "__main__":
    raise SystemExit(main())