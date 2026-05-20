#!/usr/bin/env python3
"""Standalone interactive viewer for extracted fetal ECG sessions."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


def sanitize_filename_part(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    return result or "signal"


def default_session_path(channel_label: str, save_dir: str) -> Path:
    session_dir = Path(save_dir).expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = sanitize_filename_part(channel_label)
    return session_dir / f"{safe_label}_session_{stamp}.npz"


def save_view_session(
    session_path: str | Path,
    aecg_signal: np.ndarray,
    fecg_signal: np.ndarray,
    channel_label: str,
    fs: float,
    duration: float,
    save_dir: str,
) -> Path:
    session_path = Path(session_path).expanduser().resolve()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        session_path,
        aecg_signal=np.asarray(aecg_signal, dtype=np.float32),
        fecg_signal=np.asarray(fecg_signal, dtype=np.float32),
        channel_label=np.asarray(channel_label),
        fs=np.float32(fs),
        duration=np.float32(duration),
        save_dir=np.asarray(save_dir),
    )
    return session_path


def load_view_session(session_path: str | Path) -> dict:
    session_path = Path(session_path).expanduser().resolve()
    data = np.load(session_path, allow_pickle=False)
    keys = set(data.files)

    if {"aecg_signal", "fecg_signal"}.issubset(keys):
        channel_label = str(data["channel_label"]) if "channel_label" in keys else session_path.stem
        save_dir = str(data["save_dir"]) if "save_dir" in keys else str(session_path.parent)
        return {
            "aecg_signal": data["aecg_signal"],
            "fecg_signal": data["fecg_signal"],
            "channel_label": channel_label,
            "fs": float(data["fs"]) if "fs" in keys else 1.0,
            "duration": float(data["duration"]) if "duration" in keys else float(len(data["aecg_signal"])),
            "save_dir": save_dir,
        }

    if {"aecg_windows", "fecg_ext_windows"}.issubset(keys):
        aecg_signal = np.asarray(data["aecg_windows"], dtype=np.float32).reshape(-1)
        fecg_signal = np.asarray(data["fecg_ext_windows"], dtype=np.float32).reshape(-1)
        if "fecg_gt_windows" in keys:
            _ = data["fecg_gt_windows"]

        fs = float(data["fs"]) if "fs" in keys else 1.0
        if "duration" in keys:
            duration = float(data["duration"])
        elif fs > 0:
            duration = float(len(aecg_signal) / fs)
        else:
            duration = float(len(aecg_signal))

        if "record_name" in keys:
            channel_label = f"{str(data['record_name'])}_ch{int(data['channel_idx'])}" if "channel_idx" in keys else str(data["record_name"])
        else:
            channel_label = session_path.stem

        save_dir = str(session_path.parent)
        if "save_dir" in keys:
            save_dir = str(data["save_dir"])

        return {
            "aecg_signal": aecg_signal,
            "fecg_signal": fecg_signal,
            "channel_label": channel_label,
            "fs": fs,
            "duration": duration,
            "save_dir": save_dir,
        }

    raise KeyError(
        f"Unsupported session format in {session_path}. Available keys: {sorted(keys)}"
    )


def show_interactive_viewer(
    aecg_signal: np.ndarray,
    fecg_signal: np.ndarray,
    channel_label: str,
    fs: float,
    duration: float,
    save_dir: str,
    shared_y_axis: bool = False,
    initial_window_size: float = 4.0,
):
    """Display interactive viewer with AECG and FECG signals side by side."""

    state = {
        "t_start": 0.0,
        "window_size": float(initial_window_size),
        "step_small": 0.5,
        "step_big": 4.0,
    }

    fig, (ax_aecg, ax_fecg) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(bottom=0.15, top=0.90, hspace=0.3)

    ax_left = plt.axes([0.15, 0.04, 0.08, 0.05])
    ax_right = plt.axes([0.68, 0.04, 0.08, 0.05])
    ax_save = plt.axes([0.38, 0.04, 0.08, 0.05])

    btn_left = Button(ax_left, "◀ 0.5s")
    btn_right = Button(ax_right, "0.5s ▶")
    btn_save = Button(ax_save, "Save PNG")

    ax_status = plt.axes([0.38, 0.58, 0.25, 0.04])
    ax_status.axis("off")
    status_text = ax_status.text(0.5, 0.5, "", ha="center", va="center", fontsize=9, transform=ax_status.transAxes)

    def shift_time(delta: float) -> None:
        new_t = state["t_start"] + delta
        new_t = max(0.0, min(new_t, max(0.0, duration - state["window_size"])))
        state["t_start"] = round(new_t, 2)
        redraw()

    def save_png() -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        t0 = state["t_start"]
        t1 = t0 + state["window_size"]
        out_path = Path(save_dir) / f"{sanitize_filename_part(channel_label)}_fecg_extraction_t{t0:06.1f}-{t1:06.1f}s.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        status_text.set_text(f"✓ Saved {out_path.name}")
        fig.canvas.draw_idle()

    def redraw() -> None:
        t0 = state["t_start"]
        t1 = t0 + state["window_size"]

        ax_aecg.cla()
        ax_fecg.cla()

        s0 = int(t0 * fs)
        s1 = int(min(t1, duration) * fs)

        time_pts = np.linspace(t0, t0 + (s1 - s0) / fs, s1 - s0, endpoint=False)

        ax_aecg.plot(time_pts, aecg_signal[s0:s1], linewidth=1.0, color="navy", label="AECG (Input)")
        ax_aecg.set_xlim(t0, t1)
        ax_aecg.set_ylabel("AECG (μV)", fontsize=10, fontweight="bold")
        ax_aecg.grid(True, alpha=0.3)
        ax_aecg.legend(loc="upper right")
        ax_aecg.set_xticklabels([])

        ax_fecg.plot(time_pts, fecg_signal[s0:s1], linewidth=1.0, color="darkred", label="FECG (Extracted)")
        ax_fecg.set_xlim(t0, t1)
        ax_fecg.set_ylabel("FECG (μV)", fontsize=10, fontweight="bold")
        ax_fecg.set_xlabel("Time (seconds)", fontsize=10)
        ax_fecg.grid(True, alpha=0.3)
        ax_fecg.legend(loc="upper right")

        visible_aecg = aecg_signal[s0:s1]
        visible_fecg = fecg_signal[s0:s1]
        if visible_aecg.size > 0:
            if shared_y_axis and visible_fecg.size > 0:
                y_min = float(min(np.min(visible_aecg), np.min(visible_fecg)))
                y_max = float(max(np.max(visible_aecg), np.max(visible_fecg)))
                if y_max <= y_min:
                    center = y_min
                    y_min = center - 1.0
                    y_max = center + 1.0
                else:
                    pad = 0.05 * (y_max - y_min)
                    y_min -= pad
                    y_max += pad
                ax_aecg.set_ylim(y_min, y_max)
                ax_fecg.set_ylim(y_min, y_max)
            else:
                aecg_min = float(np.min(visible_aecg))
                aecg_max = float(np.max(visible_aecg))
                if aecg_max <= aecg_min:
                    center = aecg_min
                    aecg_min = center - 1.0
                    aecg_max = center + 1.0
                else:
                    pad = 0.05 * (aecg_max - aecg_min)
                    aecg_min -= pad
                    aecg_max += pad
                ax_aecg.set_ylim(aecg_min, aecg_max)

        if visible_fecg.size > 0 and not shared_y_axis:
            fecg_min = float(np.min(visible_fecg))
            fecg_max = float(np.max(visible_fecg))
            if fecg_max <= fecg_min:
                center = fecg_min
                fecg_min = center - 1.0
                fecg_max = center + 1.0
            else:
                pad = 0.05 * (fecg_max - fecg_min)
                fecg_min -= pad
                fecg_max += pad
            ax_fecg.set_ylim(fecg_min, fecg_max)

        fig.suptitle(
            f"FECG Extraction Results — Channel: {channel_label}\n"
            f"Window: {t0:.2f}–{t1:.2f} s   [← 0.5s | 0.5s →]",
            fontsize=12,
            fontweight="bold",
        )

        status_text.set_text(f"Time: {t0:.2f}s – {t1:.2f}s / {duration:.2f}s total")
        fig.canvas.draw_idle()

    btn_left.on_clicked(lambda _event: shift_time(-state["step_small"]))
    btn_right.on_clicked(lambda _event: shift_time(state["step_small"]))
    btn_save.on_clicked(lambda _event: save_png())

    def on_key(event):
        if not event.key:
            return

        ctrl = event.key.startswith("ctrl+")
        key = event.key.replace("ctrl+", "")

        if ctrl and key == "left":
            shift_time(-state["step_big"])
        elif ctrl and key == "right":
            shift_time(state["step_big"])
        elif key == "left":
            shift_time(-state["step_small"])
        elif key == "right":
            shift_time(state["step_small"])
        elif key == "s":
            save_png()

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open and view a saved fECG interactive session")
    parser.add_argument("--session", required=True, help="Path to a saved .npz session")
    parser.add_argument(
        "--shared-y-axis",
        action="store_true",
        help="Force the AECG and FECG plots to use the same y-axis range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = load_view_session(args.session)
    print(f"Opening session: {Path(args.session).expanduser().resolve()}")
    show_interactive_viewer(
        session["aecg_signal"],
        session["fecg_signal"],
        session["channel_label"],
        session["fs"],
        session["duration"],
        session["save_dir"],
        shared_y_axis=args.shared_y_axis,
    )


if __name__ == "__main__":
    main()