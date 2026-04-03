import os
import numpy as np
import matplotlib.pyplot as plt
from db_loaders import load_adfecgdb_record, load_nifeadb_record, list_adfecgdb_records, list_nifeadb_records
from extraction_engines import CycleGANEngine

def plot_extraction(record_name, abdominal, extracted, ground_truth=None, fs=1000, duration=5.0, save_path=None):
    """
    Plots a 5-second window of the extraction comparison.
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # We only show the first n_samples
    abd_window = abdominal[0][:n_samples]
    ext_window = extracted[:n_samples]
    
    rows = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(15, 4 * rows), sharex=True)
    
    # Original Abdomen
    axes[0].plot(t, abd_window, color='gray', alpha=0.7)
    axes[0].set_title(f"Record: {record_name} - Original Abdominal (Channel 1)")
    axes[0].set_ylabel("Amplitude (std_unit)")
    
    # Extracted fECG
    axes[1].plot(t, ext_window, color='blue')
    axes[1].set_title("Extracted fetal ECG (CycleGAN_V2 Step 26)")
    axes[1].set_ylabel("Amplitude")
    
    # Ground Truth if available
    if ground_truth is not None:
        gt_window = ground_truth[:n_samples]
        axes[2].plot(t, gt_window, color='red', alpha=0.8)
        axes[2].set_title("Ground Truth (Direct fECG)")
        axes[2].set_ylabel("Amplitude")
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def main():
    # 1. Load Model
    print("Loading CycleGAN_V2 Step 26...")
    model_dir = "/Users/ajmalrishdin/Documents/Projects/Pre-term Birth Detector/Code/CAA-CycleGAN/models/CygleGAN V2 Models"
    engine = CycleGANEngine(model_dir, version_name='CycleGAN_V2', step=26)
    
    out_dir = "Output_Plots_V2_Step26"
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Process ADFECGDB (5 records)
    print("\nProcessing ADFECGDB...")
    adf_records, adf_folder = list_adfecgdb_records()
    for rec in adf_records:
        if '_ARR_' in rec: continue # Only standard ones
        print(f" -> {rec}")
        try:
            data = load_adfecgdb_record(rec, folder=adf_folder)
            fs = data['fs']
            # Abdomen leading for CycleGAN
            input_abd = data['abdominal'][0]
            fecg = engine.extract_fecg(input_abd, fs=fs)
            
            save_path = os.path.join(out_dir, f"ADFECGDB_{rec.replace('.edf', '')}.png")
            plot_extraction(rec, data['abdominal'], fecg, ground_truth=data['direct_fecg'], fs=fs, save_path=save_path)
        except Exception as e:
            print(f"Failed {rec}: {e}")

    # 3. Process Arrhythmia Database (26 records)
    print("\nProcessing Arrhythmia DB...")
    nifea_records, nifea_folder = list_nifeadb_records()
    for rec in nifea_records:
        print(f" -> {rec}")
        try:
            data = load_nifeadb_record(rec, folder=nifea_folder)
            fs = data['fs']
            input_abd = data['abdominal'][0]
            fecg = engine.extract_fecg(input_abd, fs=fs)
            
            save_path = os.path.join(out_dir, f"Arrhythmia_{rec}.png")
            plot_extraction(rec, data['abdominal'], fecg, ground_truth=None, fs=fs, save_path=save_path)
        except Exception as e:
            print(f"Failed {rec}: {e}")

    print(f"\nDone! Plots saved to '{out_dir}/'")

if __name__ == "__main__":
    main()
