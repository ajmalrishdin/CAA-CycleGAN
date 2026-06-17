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
from datetime import datetime
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
from Utils.device_utils import get_device_backend, resolve_device

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

def save_view_session(session_path, aecg_windows, fecg_ext_windows, fecg_gt_windows,
                      epochs, current_epoch_idx, record_name, channel_idx, fs,
                      model_dir, device, start_sec, duration_sec, source_record):
    payload = {
        "aecg_windows": np.asarray(aecg_windows, dtype=np.float32),
        "fecg_ext_windows": np.asarray(fecg_ext_windows, dtype=np.float32),
        "epochs": np.asarray(epochs, dtype=np.int64),
        "current_epoch_idx": np.int64(current_epoch_idx),
        "record_name": np.asarray(record_name),
        "channel_idx": np.int64(channel_idx),
        "fs": np.float32(fs),
        "model_dir": np.asarray(model_dir),
        "device": np.asarray(str(device)),
        "start_sec": np.float32(start_sec),
        "duration_sec": np.float32(duration_sec),
        "source_record": np.asarray(source_record),
        "has_gt": np.bool_(fecg_gt_windows is not None),
    }
    if fecg_gt_windows is not None:
        payload["fecg_gt_windows"] = np.asarray(fecg_gt_windows, dtype=np.float32)
    np.savez_compressed(session_path, **payload)


def default_session_path(record_name, channel_idx, epoch, start_sec, duration_sec):
    session_dir = os.path.join(SCRIPT_DIR, "outputs", "view_sessions")
    os.makedirs(session_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record_base = os.path.splitext(os.path.basename(record_name))[0]
    filename = f"{record_base}_ch{channel_idx}_epoch{epoch}_start{int(start_sec)}_dur{int(duration_sec)}_{stamp}.npz"
    return os.path.join(session_dir, filename)


def load_view_session(session_path):
    data = np.load(session_path, allow_pickle=False)
    fecg_gt_windows = data["fecg_gt_windows"] if "fecg_gt_windows" in data.files else None
    return {
        "aecg_windows": data["aecg_windows"],
        "fecg_ext_windows": data["fecg_ext_windows"],
        "fecg_gt_windows": fecg_gt_windows,
        "epochs": data["epochs"].astype(int).tolist(),
        "current_epoch_idx": int(data["current_epoch_idx"]),
        "record_name": str(data["record_name"]),
        "channel_idx": int(data["channel_idx"]),
        "fs": float(data["fs"]),
        "model_dir": str(data["model_dir"]),
        "device": str(data["device"]),
        "start_sec": float(data["start_sec"]),
        "duration_sec": float(data["duration_sec"]),
        "source_record": str(data["source_record"]),
    }

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
        "aecg": "#2563eb", "fecg_ext": "#dc2626", "fecg_gt": "#16a34a",
        "bg": "#f8fafc", "panel": "#ffffff", "grid": "#cbd5e1",
        "text": "#1e293b", "accent": "#7c3aed", "slider_bg": "#e2e8f0",
        "zoom": "#f59e0b",
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
        ratios = [1]*n_rows + [0.15, 0.15]
        self.fig = plt.figure(figsize=(16, 4*n_rows + 1.5), facecolor=c["bg"])
        self.fig.canvas.manager.set_window_title(
            f"fECG Viewer — {self.record_name} ch{self.channel_idx}")
        gs = self.fig.add_gridspec(n_rows+2, 1, height_ratios=ratios,
                                   left=0.06, right=0.94, top=0.92, bottom=0.06, hspace=0.35)

        self.ax_aecg = self.fig.add_subplot(gs[0])
        self.ax_fecg = self.fig.add_subplot(gs[1], sharex=self.ax_aecg)
        self.ax_gt = self.fig.add_subplot(gs[2], sharex=self.ax_aecg) if self.has_gt else None
        self.all_axes = [self.ax_aecg, self.ax_fecg] + ([self.ax_gt] if self.has_gt else [])

        for ax in self.all_axes:
            ax.set_facecolor(c["panel"])
            ax.tick_params(colors=c["text"], labelsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color(c["grid"])
            ax.grid(True, color=c["grid"], alpha=0.5, linewidth=0.5)

        # Position slider
        ax_sl = self.fig.add_subplot(gs[n_rows])
        ax_sl.set_facecolor(c["slider_bg"])
        mx = max(0, self.n_windows - self.visible)
        self.slider = Slider(ax_sl, "Position", 0, max(mx,1), valinit=0, valstep=1,
                             color=c["accent"], initcolor="none")
        self.slider.label.set_color(c["text"]); self.slider.valtext.set_color(c["text"])
        self.slider.on_changed(self._on_slider)

        # Zoom slider (x-axis range: how many windows visible)
        ax_zoom = self.fig.add_subplot(gs[n_rows+1])
        ax_zoom.set_facecolor(c["slider_bg"])
        self.zoom_slider = Slider(ax_zoom, "Zoom", 1, self.n_windows, valinit=self.visible,
                                  valstep=1, color=c["zoom"], initcolor="none")
        self.zoom_slider.label.set_color(c["text"]); self.zoom_slider.valtext.set_color(c["text"])
        self.zoom_slider.on_changed(self._on_zoom)

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
            ax.grid(True, color=c["grid"], alpha=0.5, linewidth=0.5)
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

    def _on_slider(self, val): 
        self.pos = int(val)
        self._draw()

    def _on_zoom(self, val):
        self.visible = int(val)
        mx = max(0, self.n_windows - self.visible)
        self.slider.valmax = max(mx, 1)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        if self.pos > mx:
            self.pos = mx
            self.slider.set_val(mx)
        else:
            self._draw()
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
    p.add_argument("--device", type=str, default=get_device_backend(), choices=["cpu","mps","cuda"])
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--db-dir", type=str, default=None,
                   help="Database directory. Auto-detected from record extension if omitted.")
    p.add_argument("--start-sec", type=float, default=None,
                   help="Start second. Default: 30 for .edf, 0 for .dat.")
    p.add_argument("--duration-sec", type=float, default=60, help="Duration in seconds. Default: 60.")
    p.add_argument("--save-session", type=str, default=None,
                   help="Optional .npz file to save the current viewer session for later reopening. If omitted, a session is saved automatically.")
    p.add_argument("--no-save-session", action="store_true",
                   help="Disable automatic session saving.")
    p.add_argument("--load-session", type=str, default=None,
                   help="Load a previously saved .npz viewer session instead of extracting again.")
    return p.parse_args()

def main():
    args = parse_args()
    model_dir = args.model_dir or MODEL_DIR
    device = resolve_device(args.device)

    if args.save_session and args.load_session:
        print("\n❌ Use only one of --save-session or --load-session.")
        sys.exit(1)

    save_sessions = not args.no_save_session

    if args.load_session:
        session_path = os.path.abspath(args.load_session)
        if not os.path.isfile(session_path):
            print(f"\n❌ Session file not found: {session_path}")
            sys.exit(1)

        session = load_view_session(session_path)
        epochs = session["epochs"]
        start_epoch_idx = session["current_epoch_idx"]
        print("=" * 60)
        print("  CAA-CycleGAN fECG Extraction & Interactive Viewer")
        print("=" * 60)
        print(f"  Session   : {session_path}")
        print(f"  Record    : {session['record_name']}")
        print(f"  Channel   : {session['channel_idx']}")
        print(f"  Device    : {session['device']}")
        print(f"  Model dir : {session['model_dir']}")
        print(f"  Epochs    : {len(epochs)} checkpoints ({epochs[0]}..{epochs[-1]})")
        print(f"  Start     : epoch {epochs[start_epoch_idx]}")
        print("=" * 60)

        gen_cache = {}
        def load_gen(epoch):
            if epoch not in gen_cache:
                gen_cache[epoch] = load_generator(epoch, session["model_dir"], device)
            return gen_cache[epoch]

        SignalViewer(
            session["aecg_windows"],
            session["fecg_ext_windows"],
            session["fecg_gt_windows"],
            epochs,
            start_epoch_idx,
            load_gen,
            device,
            session["record_name"],
            session["channel_idx"],
            session["fs"],
        ).show()
        print("\n👋 Done.")
        return

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

        session_path = None
        if save_sessions:
            session_path = os.path.abspath(args.save_session) if args.save_session else default_session_path(
                record,
                ch_idx,
                epoch,
                start_sec,
                args.duration_sec,
            )
            if len(channels) > 1 and args.save_session:
                base, ext = os.path.splitext(session_path)
                session_path = f"{base}_ch{ch_idx}{ext or '.npz'}"
            elif args.save_session and not session_path.lower().endswith(".npz"):
                session_path += ".npz"
            save_view_session(
                session_path,
                aecg_w,
                fecg_ext,
                gt_w,
                epochs,
                start_epoch_idx,
                record,
                ch_idx,
                fs,
                model_dir,
                device,
                start_sec,
                args.duration_sec,
                record,
            )
            print(f"   💾 Session saved to {session_path}")

        SignalViewer(aecg_w, fecg_ext, gt_w, epochs, start_epoch_idx,
                     load_gen, device, record, ch_idx, fs).show()

    print("\n👋 Done.")

if __name__ == "__main__":
    main()
