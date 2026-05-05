#!/usr/bin/env python3
"""
extract_and_view.py — Extract fECG from aECG using CAA-CycleGAN V1.1 models
                       and visualise results in an interactive viewer.

Supports:
  .edf  — ADFECGDB (with ground-truth fECG)
  .dat  — NIFEADB / WFDB format (no ground truth)

Usage:
    python3 extract_and_view.py --record r01.edf
    python3 extract_and_view.py --record ARR_01.dat --db-dir Databases/non-invasive-fetal-ecg-arrhythmia-database-1.0.0
    python3 extract_and_view.py --epoch 7948 --device cpu
"""

import os, sys, argparse, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from Utils.sagan_models import Generator
from Utils.device_utils import resolve_device

MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "V1.1")
DB_DIR_EDF = os.path.join(SCRIPT_DIR, "Databases", "ADFECGDB")
DB_DIR_DAT = os.path.join(SCRIPT_DIR, "Databases", "non-invasive-fetal-ecg-arrhythmia-database-1.0.0")

# ── Signal preprocessing (mirrors training DataUtils exactly) ────────────

def butter_bandpass(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)

def signal_filter(data):
    """Comb + notch + lowpass filters matching DataUtils.SignalFilter."""
    A = np.array([1,0,0,0,0,0,0,0,0,0,-0.854])
    B = np.array([0.927,0,0,0,0,0,0,0,0,0,-0.927])
    data = scipy_signal.filtfilt(B, A, data)
    B1 = np.array([0.995,-1.8504,0.995]); A1 = np.array([1,-1.8505,0.99])
    data = scipy_signal.filtfilt(B1, A1, data)
    B2 = np.array([0.388,0.388]); A2 = np.array([1,-0.42578])
    data = scipy_signal.filtfilt(B2, A2, data)
    return data

def preprocess_signal(raw_signal, fs=1000):
    from sklearn.preprocessing import scale
    filtered = butter_bandpass(raw_signal, 1, 100, fs)
    filtered = signal_filter(filtered)
    if filtered.ndim == 1:
        filtered = scale(filtered.reshape(1, -1), axis=1).squeeze()
    else:
        filtered = scale(filtered, axis=1)
    n_out = int(filtered.shape[-1] / 5)
    if filtered.ndim == 1:
        filtered = resample(filtered, n_out)
    else:
        filtered = resample(filtered, n_out, axis=1)
    return filtered

def window_signal(sig_1d, window_size=128):
    return np.array([sig_1d[i:i+window_size] for i in range(0, len(sig_1d)-window_size+1, window_size)])

def normalise_window(w):
    w = w.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    return scaler.fit_transform(w.reshape(-1, 1)).flatten()

# ── Data loading ─────────────────────────────────────────────────────────

def load_edf_record(record_path, start_sec=30, duration_sec=60):
    """Load ADFECGDB .edf → (abd, fecg_gt_or_None, fs_out=200)."""
    import pyedflib, wfdb
    fs_orig = 1000
    f = pyedflib.EdfReader(record_path)
    sig_len = f.getNSamples()[0]
    s0 = max(0, min(int(start_sec * fs_orig), sig_len))
    s1 = max(s0, min(int((start_sec + duration_sec) * fs_orig), sig_len))
    direct_fecg_raw = f.readSignal(0)[s0:s1]
    abd_raw = np.array([f.readSignal(i)[s0:s1] for i in range(1, f.signals_in_file)])
    f.close()
    return preprocess_signal(abd_raw, fs_orig), preprocess_signal(direct_fecg_raw, fs_orig), 200

def load_dat_record(record_path, start_sec=0, duration_sec=60):
    """Load WFDB .dat/.hea → (abd, None, fs_out=200). No ground truth."""
    import wfdb
    base = record_path.replace('.dat', '').replace('.hea', '')
    rec = wfdb.rdrecord(base)
    fs_orig = int(rec.fs)
    signals = rec.p_signal.T  # (n_channels, n_samples)
    names = rec.sig_name

    s0 = max(0, min(int(start_sec * fs_orig), signals.shape[1]))
    s1 = max(s0, min(int((start_sec + duration_sec) * fs_orig), signals.shape[1]))
    signals = signals[:, s0:s1]

    # Separate chest (ECG) vs abdominal channels
    abd_idx = [i for i, n in enumerate(names) if 'abd' in n.lower()]
    if not abd_idx:
        # Fallback: skip first channel (chest), use rest as abdominal
        abd_idx = list(range(1, len(names)))
    abd_raw = signals[abd_idx]

    return preprocess_signal(abd_raw, fs_orig), None, 200

def load_record(record_path, start_sec, duration_sec):
    """Auto-detect format and load."""
    ext = os.path.splitext(record_path)[1].lower()
    if ext == '.edf':
        return load_edf_record(record_path, start_sec, duration_sec)
    elif ext in ('.dat', '.hea', ''):
        return load_dat_record(record_path, start_sec, duration_sec)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def list_records(db_dir):
    """List available .edf and .dat records."""
    files = os.listdir(db_dir)
    edfs = sorted(f for f in files if f.lower().endswith('.edf') and '_ARR_' not in f)
    dats = sorted(set(os.path.splitext(f)[0] + '.dat' for f in files if f.endswith('.dat')))
    return edfs + dats

# ── Model loading & inference ────────────────────────────────────────────

def list_checkpoints(model_dir=None):
    model_dir = model_dir or MODEL_DIR
    files = glob.glob(os.path.join(model_dir, "*_G_AECG2FECG.pth"))
    return sorted(int(os.path.basename(f).split("_")[0]) for f in files)

def load_generator(epoch, model_dir=None, device=None):
    model_dir = model_dir or MODEL_DIR
    pth = os.path.join(model_dir, f"{epoch}_G_AECG2FECG.pth")
    gen = Generator(batch_size=1, image_size=64, z_dim=128, conv_dim=64)
    gen.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
    gen.to(device); gen.eval()
    return gen

def run_inference(generator, windows, device, batch_size=64):
    outputs = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        batch_norm = np.array([normalise_window(w) for w in batch])
        tensor_in = torch.FloatTensor(batch_norm).unsqueeze(1).to(device)
        with torch.no_grad():
            tensor_out = generator(tensor_in)
        outputs.append(tensor_out.cpu().numpy().squeeze(1))
    return np.concatenate(outputs, axis=0)

# ── Interactive Viewer ───────────────────────────────────────────────────

class SignalViewer:
    VISIBLE_WINDOWS = 20
    C = {
        "aecg": "#3b82f6", "fecg_ext": "#ef4444", "fecg_gt": "#22c55e",
        "bg": "#0f172a", "panel": "#1e293b", "grid": "#334155",
        "text": "#e2e8f0", "accent": "#8b5cf6", "slider_bg": "#334155",
    }

    def __init__(self, aecg_windows, fecg_ext_windows, fecg_gt_windows,
                 epochs, current_epoch_idx, generator_loader, device,
                 record_name, channel_idx, fs=200):
        self.aecg = aecg_windows
        self.fecg_ext = fecg_ext_windows
        self.fecg_gt = fecg_gt_windows          # None when no ground truth
        self.has_gt = fecg_gt_windows is not None
        self.epochs = epochs
        self.epoch_idx = current_epoch_idx
        self.gen_loader = generator_loader
        self.device = device
        self.record_name = record_name
        self.channel_idx = channel_idx
        self.fs = fs
        self.n_windows = len(self.aecg)
        self.pos = 0
        self.visible = min(self.VISIBLE_WINDOWS, self.n_windows)
        self._build()

    def _build(self):
        c = self.C
        n_rows = 3 if self.has_gt else 2
        ratios = [1]*n_rows + [0.15]
        self.fig = plt.figure(figsize=(16, 4*n_rows + 1), facecolor=c["bg"])
        self.fig.canvas.manager.set_window_title(
            f"fECG Viewer — {self.record_name} ch{self.channel_idx}")
        gs = self.fig.add_gridspec(n_rows+1, 1, height_ratios=ratios,
                                   left=0.06, right=0.94, top=0.92, bottom=0.08, hspace=0.35)

        self.ax_aecg = self.fig.add_subplot(gs[0])
        self.ax_fecg = self.fig.add_subplot(gs[1], sharex=self.ax_aecg)
        self.ax_gt = self.fig.add_subplot(gs[2], sharex=self.ax_aecg) if self.has_gt else None
        self.all_axes = [self.ax_aecg, self.ax_fecg] + ([self.ax_gt] if self.has_gt else [])

        for ax in self.all_axes:
            ax.set_facecolor(c["panel"])
            ax.tick_params(colors=c["text"], labelsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color(c["grid"])
            ax.grid(True, color=c["grid"], alpha=0.3, linewidth=0.5)

        # Slider
        ax_sl = self.fig.add_subplot(gs[n_rows])
        ax_sl.set_facecolor(c["slider_bg"])
        mx = max(0, self.n_windows - self.visible)
        self.slider = Slider(ax_sl, "Position", 0, max(mx,1), valinit=0, valstep=1,
                             color=c["accent"], initcolor="none")
        self.slider.label.set_color(c["text"]); self.slider.valtext.set_color(c["text"])
        self.slider.on_changed(self._on_slider)

        # Buttons
        ax_p = self.fig.add_axes([0.30, 0.95, 0.08, 0.035])
        ax_n = self.fig.add_axes([0.62, 0.95, 0.08, 0.035])
        self.btn_p = Button(ax_p, "◀  Prev Epoch", color=c["panel"], hovercolor=c["accent"])
        self.btn_n = Button(ax_n, "Next Epoch  ▶", color=c["panel"], hovercolor=c["accent"])
        for b in [self.btn_p, self.btn_n]:
            b.label.set_color(c["text"]); b.label.set_fontsize(9)
        self.btn_p.on_clicked(self._prev_epoch)
        self.btn_n.on_clicked(self._next_epoch)

        self.epoch_text = self.fig.text(0.50, 0.965, self._elabel(),
                                        ha="center", va="center", color=c["accent"],
                                        fontsize=12, fontweight="bold")
        gt_tag = "  •  has ground truth" if self.has_gt else "  •  no ground truth"
        self.fig.text(0.06, 0.965,
                      f"{self.record_name}  •  ch {self.channel_idx}  •  {self.fs} Hz{gt_tag}",
                      ha="left", va="center", color=c["text"], fontsize=10)
        self._draw()

    def _elabel(self):
        return f"Epoch {self.epochs[self.epoch_idx]}  ({self.epoch_idx+1}/{len(self.epochs)})"

    def _draw(self):
        c = self.C
        s, e = self.pos, self.pos + self.visible
        aecg = self.aecg[s:e].flatten()
        fecg = self.fecg_ext[s:e].flatten()
        t = np.arange(len(aecg)) / self.fs + (s * 128) / self.fs

        for ax in self.all_axes:
            ax.clear(); ax.set_facecolor(c["panel"])
            ax.grid(True, color=c["grid"], alpha=0.3, linewidth=0.5)
            ax.tick_params(colors=c["text"], labelsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color(c["grid"])

        self.ax_aecg.plot(t, aecg, color=c["aecg"], lw=0.8)
        self.ax_aecg.set_title("Input aECG (abdominal)", color=c["text"], fontsize=11,
                               fontweight="bold", loc="left")
        self.ax_aecg.set_ylabel("Amplitude", color=c["text"], fontsize=9)

        self.ax_fecg.plot(t, fecg, color=c["fecg_ext"], lw=0.8)
        self.ax_fecg.set_title("Extracted fECG (CycleGAN)", color=c["text"], fontsize=11,
                               fontweight="bold", loc="left")
        self.ax_fecg.set_ylabel("Amplitude", color=c["text"], fontsize=9)

        if self.has_gt:
            gt = self.fecg_gt[s:e].flatten()
            self.ax_gt.plot(t, gt, color=c["fecg_gt"], lw=0.8)
            self.ax_gt.set_title("Ground-Truth fECG (direct scalp)", color=c["text"],
                                 fontsize=11, fontweight="bold", loc="left")
            self.ax_gt.set_ylabel("Amplitude", color=c["text"], fontsize=9)
            self.ax_gt.set_xlabel("Time (s)", color=c["text"], fontsize=9)
        else:
            self.ax_fecg.set_xlabel("Time (s)", color=c["text"], fontsize=9)

        self.fig.canvas.draw_idle()

    def _on_slider(self, val): self.pos = int(val); self._draw()

    def _reload_epoch(self):
        ep = self.epochs[self.epoch_idx]
        print(f"\n⏳ Loading epoch {ep}...")
        gen = self.gen_loader(ep)
        self.fecg_ext = run_inference(gen, self.aecg, self.device)
        self.epoch_text.set_text(self._elabel())
        self._draw(); print(f"   ✅ Done.")

    def _prev_epoch(self, _):
        if self.epoch_idx > 0: self.epoch_idx -= 1; self._reload_epoch()
    def _next_epoch(self, _):
        if self.epoch_idx < len(self.epochs)-1: self.epoch_idx += 1; self._reload_epoch()
    def show(self): plt.show()

# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract fECG from aECG using CAA-CycleGAN V1.1 and view interactively.")
    p.add_argument("--record", type=str, default=None,
                   help="Record file (e.g. r01.edf or ARR_01.dat). Default: first available.")
    p.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch. Default: latest.")
    p.add_argument("--channel", type=int, default=0, help="Abdominal channel index. Default: 0.")
    p.add_argument("--all-channels", action="store_true", help="Process all abdominal channels.")
    p.add_argument("--device", type=str, default="mps", choices=["cpu","mps","cuda"])
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--db-dir", type=str, default=None,
                   help="Database directory. Auto-detected from record extension if omitted.")
    p.add_argument("--start-sec", type=float, default=None,
                   help="Start second. Default: 30 for .edf, 0 for .dat.")
    p.add_argument("--duration-sec", type=float, default=60, help="Duration in seconds. Default: 60.")
    return p.parse_args()

def main():
    args = parse_args()
    model_dir = args.model_dir or MODEL_DIR
    device = resolve_device(args.device)

    # Resolve db_dir and record
    record = args.record
    db_dir = args.db_dir

    if record and not db_dir:
        # Auto-detect db_dir from extension
        ext = os.path.splitext(record)[1].lower()
        if ext == '.edf':
            db_dir = DB_DIR_EDF
        elif ext == '.dat':
            db_dir = DB_DIR_DAT
        else:
            db_dir = DB_DIR_EDF
    elif not db_dir:
        db_dir = DB_DIR_EDF

    # Make db_dir absolute
    if not os.path.isabs(db_dir):
        db_dir = os.path.join(SCRIPT_DIR, db_dir)

    # Default start_sec
    start_sec = args.start_sec
    if start_sec is None:
        start_sec = 30 if (record and record.lower().endswith('.edf')) else 0

    print("=" * 60)
    print("  CAA-CycleGAN fECG Extraction & Interactive Viewer")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Model dir : {model_dir}")
    print(f"  DB dir    : {db_dir}")

    epochs = list_checkpoints(model_dir)
    if not epochs:
        print(f"\n❌ No checkpoints found in {model_dir}"); sys.exit(1)
    print(f"  Epochs    : {len(epochs)} checkpoints ({epochs[0]}..{epochs[-1]})")

    if args.epoch is not None:
        if args.epoch not in epochs:
            print(f"\n❌ Epoch {args.epoch} not found."); sys.exit(1)
        start_epoch_idx = epochs.index(args.epoch)
    else:
        start_epoch_idx = len(epochs) - 1
    print(f"  Start     : epoch {epochs[start_epoch_idx]}")

    records = list_records(db_dir)
    if not records:
        print(f"\n❌ No records found in {db_dir}"); sys.exit(1)
    print(f"  Records   : {records}")

    record = record or records[0]
    # Allow specifying just the base name for .dat
    if record not in records and not record.endswith('.dat'):
        record = record + '.dat'
    if record not in records:
        base = os.path.splitext(record)[0]
        matches = [r for r in records if os.path.splitext(r)[0] == base]
        if matches:
            record = matches[0]
        else:
            print(f"\n❌ Record '{record}' not found. Available: {records}"); sys.exit(1)
    print(f"  Record    : {record}")
    print("=" * 60)

    # Load data
    record_path = os.path.join(db_dir, record)
    print(f"\n📂 Loading {record} ({start_sec}s + {args.duration_sec}s)...")
    abd, fecg_gt, fs = load_record(record_path, start_sec, args.duration_sec)
    print(f"   Abdominal shape : {abd.shape}  ({abd.shape[0]} channels)")
    print(f"   Ground-truth    : {'available' if fecg_gt is not None else 'not available'}")
    print(f"   Fs              : {fs} Hz")

    channels = list(range(abd.shape[0])) if args.all_channels else [args.channel]
    if max(channels) >= abd.shape[0]:
        print(f"\n❌ Channel {max(channels)} out of range (0-{abd.shape[0]-1})"); sys.exit(1)

    fecg_gt_windows = window_signal(fecg_gt, 128) if fecg_gt is not None else None

    gen_cache = {}
    def load_gen(epoch):
        if epoch not in gen_cache:
            gen_cache[epoch] = load_generator(epoch, model_dir, device)
        return gen_cache[epoch]

    for ch_idx in channels:
        print(f"\n🔬 Processing channel {ch_idx}...")
        aecg_windows = window_signal(abd[ch_idx], 128)
        print(f"   Windows: {len(aecg_windows)}")

        if fecg_gt_windows is not None:
            n_win = min(len(aecg_windows), len(fecg_gt_windows))
            aecg_w = aecg_windows[:n_win]
            gt_w = fecg_gt_windows[:n_win]
        else:
            aecg_w = aecg_windows
            gt_w = None

        epoch = epochs[start_epoch_idx]
        print(f"   Running inference with epoch {epoch}...")
        fecg_ext = run_inference(load_gen(epoch), aecg_w, device)
        print(f"   ✅ Inference complete. Launching viewer...")

        SignalViewer(aecg_w, fecg_ext, gt_w, epochs, start_epoch_idx,
                     load_gen, device, record, ch_idx, fs).show()

    print("\n👋 Done.")

if __name__ == "__main__":
    main()
