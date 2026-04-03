import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from db_loaders import load_adfecgdb_record, load_nifeadb_record
from extraction_engines import CycleGANEngine

# Setup output directory
output_dir = "Output_comparison"
os.makedirs(output_dir, exist_ok=True)

print("Loading Models...")
# Initialize CycleGAN Engines
engine_v1 = CycleGANEngine('models/sagan_1', version_name='CycleGAN_V1', step=633)
engine_v2 = CycleGANEngine('models/CygleGAN V2 Models', version_name='CycleGAN_V2', step=4386)

# Configuration for plotting
plot_duration_sec = 4.0

def plot_waveforms(data_dict, title_prefix, save_filename, has_gt=False):
    abd_signal = data_dict['abdominal'][0]  # First channel
    fs = data_dict['fs']
    
    # Calculate samples to plot
    n_samples = int(plot_duration_sec * fs)
    t = np.linspace(0, plot_duration_sec, n_samples)
    
    print(f"[{title_prefix}] Extracting with V1...")
    fecg_v1 = engine_v1.extract_fecg(abd_signal, fs=fs)
    print(f"[{title_prefix}] Extracting with V2...")
    fecg_v2 = engine_v2.extract_fecg(abd_signal, fs=fs)
    
    # Plotting
    num_plots = 4 if has_gt else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)
    
    # Plot 1: Input Abdominal
    axes[0].plot(t, abd_signal[:n_samples], color='black', alpha=0.8)
    axes[0].set_title(f"{title_prefix} - Input (Abdomen 1)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylabel('Amplitude')
    
    # Plot 2: V1 fECG
    axes[1].plot(t, fecg_v1[:n_samples], color='blue', alpha=0.9)
    axes[1].set_title("CycleGAN V1 - Extracted fECG")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylabel('Amplitude')
    
    # Plot 3: V2 fECG
    axes[2].plot(t, fecg_v2[:n_samples], color='green', alpha=0.9)
    axes[2].set_title("CycleGAN V2 - Extracted fECG")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_ylabel('Amplitude')
    
    # Plot 4 (Optional): Ground Truth
    if has_gt and data_dict.get('direct_fecg') is not None:
        gt_signal = data_dict['direct_fecg']
        if gt_signal.ndim >= 1:
            axes[3].plot(t, gt_signal[:n_samples], color='red', alpha=0.8)
            axes[3].set_title("Ground Truth - Direct Fetal Scalp ECG")
            axes[3].grid(True, linestyle='--', alpha=0.6)
            axes[3].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[{title_prefix}] Saved plot to {save_path}")

# ==========================================
# 1. ADFECGDB Processing
# ==========================================
print("\n--- Processing ADFECGDB ---")
try:
    data_adf = load_adfecgdb_record('r01.edf')
    plot_waveforms(data_adf, "ADFECGDB (r01.edf)", "cyclegan_waveforms_adfecgdb.png", has_gt=True)
except Exception as e:
    print(f"Error loading ADFECGDB: {e}")

# ==========================================
# 2. NIFEADB (Arrhythmia) Processing
# ==========================================
print("\n--- Processing NIFEADB (Arrhythmia Database) ---")
try:
    data_nif = load_nifeadb_record('ARR_01')
    plot_waveforms(data_nif, "NIFEADB Arrhythmia (ARR_01)", "cyclegan_waveforms_arrhythmia.png", has_gt=False)
except Exception as e:
    print(f"Error loading NIFEADB: {e}")
