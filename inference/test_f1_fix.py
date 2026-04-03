import numpy as np
from scipy.signal import resample
import wfdb
import os

from db_loaders import load_adfecgdb_record
from metrics import calculate_f1_precision_recall, detect_fetal_r_peaks
from extraction_engines import CycleGANEngine

# Load V1
engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

def overlap_add_process(engine_wrapper, signal_200, window_size=128, step=64):
    """Process with 50% overlap and Hanning blending."""
    out_length = len(signal_200)
    output = np.zeros(out_length)
    weight = np.zeros(out_length)
    window = np.hanning(window_size)
    
    # Process
    for start in range(0, out_length - window_size, step):
        chunk = signal_200[start:start+window_size]
        
        # We need to min-max scale it just like the training logic
        s_min, s_max = chunk.min(), chunk.max()
        if s_max - s_min < 1e-6:
            c_norm = np.zeros_like(chunk)
        else:
            c_norm = 2 * (chunk - s_min) / (s_max - s_min) - 1
            
        # Inference using the internal torch engine
        # We process one at a time for this test
        # (Actually engine.engine.process_signal expects full arrays and breaks them into 128 chunks)
        # We can bypass the wrappers batching for a sec
        import torch
        with torch.no_grad():
            tensor_in = torch.FloatTensor(c_norm).view(1, 1, window_size).to(engine_wrapper.engine.device)
            tensor_out = engine_wrapper.engine.model(tensor_in).cpu().numpy().flatten()
            
        # Optional: un-normalize back to input scale? Or just leave as [-1, 1]?
        # Let's un-normalize!
        if s_max - s_min >= 1e-6:
            tensor_out = (tensor_out + 1) / 2.0 * (s_max - s_min) + s_min
            
        # Add to output buffer
        output[start:start+window_size] += tensor_out * window
        weight[start:start+window_size] += window
        
    # Normalize by weights
    valid = weight > 0
    output[valid] /= weight[valid]
    return output

print("Evaluating using Channel 0 + OLA + Un-normalization")
for i in [1, 4, 7, 8, 10]:
    rec = f"r{i:02d}.edf"
    data = load_adfecgdb_record(rec)
    
    # 1. Force first abdominal channel instead of max energy
    abd = data['abdominal']
    fs = data['fs']
    abd_1d = abd[0]
    
    # 2. Resample
    original_len = len(abd_1d)
    n_target = int(original_len * 200 / fs)
    sig_200 = resample(abd_1d, n_target)
    
    # 3. Process with OLA
    fecg_200 = overlap_add_process(engine, sig_200)
    
    # 4. Resample back
    fecg = resample(fecg_200, original_len)
    
    # 5. Evaluate
    peaks = detect_fetal_r_peaks(fecg, fs)
    f1, p, r = calculate_f1_precision_recall(peaks, data['fqrs'], tolerance_ms=50, fs=fs)
    print(f"{rec}: F1={f1:.3f} (Det={len(peaks)}, GT={len(data['fqrs'])})")
