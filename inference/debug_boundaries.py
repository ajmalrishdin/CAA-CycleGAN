import numpy as np
from db_loaders import load_adfecgdb_record
from extraction_engines import CycleGANEngine

# Load V1 engine
engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

# Load data
data = load_adfecgdb_record('r01.edf')
abd = data['abdominal']
fs = data['fs']
original_len = len(abd[0])

# Downsample to 200 Hz exactly as extraction_engines does
from scipy.signal import resample
n_target = int(original_len * 200 / fs)
sig_200 = resample(abd[0], n_target)

# Process it chunks
out_200 = engine.engine.process_signal(sig_200, window_size=128)

# Compute absolute diffs between adjacent samples
diffs = np.abs(np.diff(out_200))

# Mean diff across all samples
mean_diff = np.mean(diffs)

# Mean diff exactly at 128-sample boundaries
boundary_idx = np.arange(127, len(diffs), 128)
boundary_diffs = diffs[boundary_idx]
mean_boundary_diff = np.mean(boundary_diffs)

print(f"Mean sample-to-sample difference: {mean_diff:.4f}")
print(f"Mean boundary (every 128 samples) difference: {mean_boundary_diff:.4f}")
print(f"Ratio Boundary/Normal: {mean_boundary_diff/mean_diff:.2f}")

# Also check how many windows are EXACTLY [-1, 1] bounds
windows = [out_200[i:i+128] for i in range(0, len(out_200)-128+1, 128)]
min_vals = [w.min() for w in windows]
max_vals = [w.max() for w in windows]

print(f"\nAcross {len(windows)} windows:")
print(f"Mean min value: {np.mean(min_vals):.4f}")
print(f"Mean max value: {np.mean(max_vals):.4f}")
