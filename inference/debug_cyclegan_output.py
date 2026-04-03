import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from db_loaders import load_adfecgdb_record
from extraction_engines import CycleGANEngine

# Load one record
data = load_adfecgdb_record('r01.edf')
abd = data['abdominal']
fs = data['fs']

# Pick one channel
best_ch = np.argmax([np.var(ch) for ch in abd])
abd_1d = abd[best_ch]

# Load V1 engine
engine = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)

# Extract fECG
fecg = engine.extract_fecg(abd_1d, fs=fs)

# Plot 5 seconds (5000 samples)
t = np.arange(5000) / 1000.0
plt.figure(figsize=(12, 4))
plt.plot(t, fecg[:5000])
plt.title("Extracted fECG from CycleGAN V1 (First 5 seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("Output_checkpoints/debug_fecg_plot.png", dpi=150)
print("Saved debug_fecg_plot.png")
