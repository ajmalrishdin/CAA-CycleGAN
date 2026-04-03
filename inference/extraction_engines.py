# ============================================================================
# Extraction Engine Wrappers
# ============================================================================
# Unified interface for all fECG extraction techniques.
# Each engine exposes:
#   extract_fecg(abdominal, chest=None, fs=1000) -> np.ndarray (1‑D fECG)
# ============================================================================

import os
import numpy as np
from scipy.signal import resample


# ============================================================================
# 1.  CycleGAN  (V1 = sagan_1, V2 = CygleGAN V2 Models)
# ============================================================================

class CycleGANEngine:
    """
    Wraps the existing InferenceEngine from inference.py.
    Takes a 1‑D abdominal signal → outputs 1‑D extracted fECG.

    The CycleGAN was trained on ADFECGDB at an effective 200 Hz
    (1000 Hz → resampled ÷5 in DataUtils).  So for other sampling
    rates we resample to 200 Hz, run inference, then resample back.
    """

    TRAINED_FS = 200  # effective Fs the model was trained on

    def __init__(self, model_dir, version_name='CycleGAN', step=None):
        from inference import InferenceEngine
        self.version_name = version_name
        self.model_dir = model_dir

        # Determine checkpoint step
        if step is None:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_G_AECG2FECG.pth')]
            steps = [int(f.split('_')[0]) for f in model_files]
            step = max(steps) if steps else None
        if step is None:
            raise ValueError(f"No FECG model files found in {model_dir}")

        fecg_path = os.path.join(model_dir, f'{step}_G_AECG2FECG.pth')
        self.engine = InferenceEngine(fecg_path)
        self.step = step

    def extract_fecg(self, abdominal, chest=None, fs=1000):
        """
        Extract fECG from a single abdominal channel using Overlap-Add.

        Args:
            abdominal: 1‑D np.array (single channel)
            chest: ignored (CycleGAN doesn't use a reference)
            fs: sampling frequency of the input

        Returns:
            fecg: 1‑D np.array at the original fs
        """
        import torch

        original_len = len(abdominal)

        # Resample to training Fs if needed
        if fs != self.TRAINED_FS:
            n_target = int(original_len * self.TRAINED_FS / fs)
            from scipy.signal import resample
            sig = resample(abdominal, n_target)
        else:
            sig = abdominal.copy()

        # OLA processing
        import numpy as np
        window_size = 128
        step = 64
        out_length = len(sig)
        
        # Pad slightly to ensure we hit the end
        if out_length < window_size:
            sig = np.pad(sig, (0, window_size - out_length))
            out_length = len(sig)
            
        fecg_200 = np.zeros(out_length)
        weight = np.zeros(out_length)
        window = np.hanning(window_size)
        
        # We need the torch model
        device = self.engine.model.device if hasattr(self.engine.model, 'device') else next(self.engine.model.parameters()).device
        
        for start in range(0, out_length - window_size + 1, step):
            chunk = sig[start:start+window_size]
            
            s_min, s_max = chunk.min(), chunk.max()
            if s_max - s_min < 1e-6:
                c_norm = np.zeros_like(chunk)
            else:
                c_norm = 2 * (chunk - s_min) / (s_max - s_min) - 1
                
            with torch.no_grad():
                tensor_in = torch.FloatTensor(c_norm).view(1, 1, window_size).to(device)
                tensor_out = self.engine.model(tensor_in).cpu().numpy().flatten()
                
            if s_max - s_min >= 1e-6:
                tensor_out = (tensor_out + 1) / 2.0 * (s_max - s_min) + s_min
                
            fecg_200[start:start+window_size] += tensor_out * window
            weight[start:start+window_size] += window
            
        valid = weight > 0
        fecg_200[valid] /= weight[valid]
        
        # Resample back to original Fs
        if fs != self.TRAINED_FS:
            from scipy.signal import resample
            fecg = resample(fecg_200, original_len)
        else:
            fecg = fecg_200

        # Trim / pad to match original length exactly
        if len(fecg) > original_len:
            fecg = fecg[:original_len]
        elif len(fecg) < original_len:
            fecg = np.pad(fecg, (0, original_len - len(fecg)))

        return fecg

    @property
    def name(self):
        return self.version_name

    @property
    def needs_chest(self):
        return False


# ============================================================================
# 2.  ILHSAF  (Improved Logarithmic Hyperbolic Secant Adaptive Filter)
# ============================================================================

class _ILHSAF_Core:
    """Core ILHSAF algorithm (copied from ilhsaf.py for self‑containment)."""

    def __init__(self, L, mu, H=0.1):
        self.w = np.zeros(L)
        self.mu = mu
        self.factor = 0.35 / (H + 1e-8)
        self.L = L

    def run(self, d_n, x_ref):
        N = len(d_n)
        e = np.zeros(N)
        for n in range(self.L, N):
            x_vec = x_ref[n:n - self.L:-1]
            y = np.dot(self.w, x_vec)
            e[n] = d_n[n] - y
            val = self.factor * e[n]
            if abs(val) > 50:
                psi = np.sign(val)
            else:
                sech_val = 1.0 / np.cosh(val)
                tanh_val = np.tanh(val)
                psi = (tanh_val * sech_val) / (1.0 + sech_val)
            self.w = self.w + self.mu * psi * x_vec
        return e


def _extract_maternal_reference(signal_data, fs):
    """
    Constructs a synthetic maternal reference from the abdominal ECG
    by detecting large maternal peaks and creating an average‑beat template.
    Used when no chest lead is available.
    """
    from scipy.signal import find_peaks

    enhanced = signal_data ** 2
    threshold = np.mean(enhanced) * 3.0
    peaks, _ = find_peaks(enhanced, distance=int(fs * 0.5), height=threshold)

    if len(peaks) < 5:
        return signal_data

    window_pre = int(0.10 * fs)
    window_post = int(0.15 * fs)

    templates = []
    for p in peaks:
        if p - window_pre >= 0 and p + window_post < len(signal_data):
            templates.append(signal_data[p - window_pre: p + window_post])

    if not templates:
        return signal_data

    avg_template = np.mean(templates, axis=0)

    synth_ref = np.zeros_like(signal_data)
    for p in peaks:
        start = p - window_pre
        end = p + window_post
        if start >= 0 and end < len(signal_data):
            synth_ref[start:end] += avg_template

    return synth_ref


def _preprocess(signal_data, fs):
    """Normalise and bandpass a 1‑D signal."""
    from scipy.signal import butter, filtfilt
    s = np.std(signal_data)
    if s < 1e-10:
        return signal_data
    sig = (signal_data - np.mean(signal_data)) / s
    nyq = 0.5 * fs
    lo = max(1.0 / nyq, 0.001)
    hi = min(60.0 / nyq, 0.999)
    b, a = butter(3, [lo, hi], btype='band')
    return filtfilt(b, a, sig)


class ILHSAFEngine:
    """
    ILHSAF adaptive‑filter based fECG extraction.

    If a chest (maternal) signal is provided, it is used as the reference.
    Otherwise a synthetic maternal reference is constructed from the
    abdominal signal itself.
    """

    def __init__(self, L=25, mu=0.005, H=0.5, use_synthetic_if_no_chest=True):
        self.L = L
        self.mu = mu
        self.H = H
        self.use_synthetic = use_synthetic_if_no_chest

    def extract_fecg(self, abdominal, chest=None, fs=1000):
        """
        Args:
            abdominal: 1‑D np.array — one abdominal channel
            chest: 1‑D np.array or None — maternal chest ECG
            fs: sampling frequency

        Returns:
            fecg: 1‑D np.array
        """
        d_n = _preprocess(abdominal, fs)

        if chest is not None:
            x_ref = _preprocess(chest, fs)
        elif self.use_synthetic:
            x_ref = _extract_maternal_reference(d_n, fs)
            x_ref = _preprocess(x_ref, fs)
        else:
            raise ValueError("ILHSAF requires a chest reference signal "
                             "and use_synthetic_if_no_chest is False")

        filt = _ILHSAF_Core(self.L, self.mu, self.H)
        fecg = filt.run(d_n, x_ref)
        return fecg

    @property
    def name(self):
        return 'ILHSAF'

    @property
    def needs_chest(self):
        return False  # can work with synthetic reference


# ============================================================================
# Engine registry
# ============================================================================

def get_all_engines(cyclegan_base_dir=None):
    """
    Build and return all extraction engines.

    Returns a list of engine objects.
    """
    if cyclegan_base_dir is None:
        cyclegan_base_dir = os.path.dirname(__file__)

    engines = []

    # CycleGAN V1
    v1_dir = os.path.join(cyclegan_base_dir, 'models', 'sagan_1')
    if os.path.isdir(v1_dir):
        try:
            engines.append(CycleGANEngine(v1_dir, version_name='CycleGAN_V1'))
        except Exception as e:
            print(f"[WARN] Could not load CycleGAN V1: {e}")

    # CycleGAN V2
    v2_dir = os.path.join(cyclegan_base_dir, 'models', 'CygleGAN V2 Models')
    if os.path.isdir(v2_dir):
        try:
            engines.append(CycleGANEngine(v2_dir, version_name='CycleGAN_V2'))
        except Exception as e:
            print(f"[WARN] Could not load CycleGAN V2: {e}")

    # ILHSAF
    engines.append(ILHSAFEngine())

    return engines
