
import os
import glob
import pandas as pd
import numpy as np
import torch
import sys
import wfdb
from scipy import signal
from sklearn.preprocessing import scale
from scipy.signal import find_peaks

# Ensure current directory is in path
sys.path.append(os.getcwd())

from Utils.DataUtils import DataUtils
from inference import InferenceEngine

def load_wfdb_record(record_path, data_utils):
    """
    Load and preprocess a WFDB record (header + dat)
    """
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"Error reading record {record_path}: {e}")
        return None, None

    fs = record.fs
    data = record.p_signal.T # (Channels, Samples)
    sig_names = record.sig_name
    
    # Identify Abdominal channels
    # Look for names containing 'Abdomen'
    abd_indices = [i for i, name in enumerate(sig_names) if 'Abdomen' in name]
    
    if not abd_indices:
        print(f"No Abdominal channels found in {record_path}")
        return None, None
        
    abd_data = data[abd_indices, :]
    
    # Process using DataUtils methods
    # 1. Bandpass 1-100Hz
    abd_data = data_utils.butter_bandpass_filter(abd_data, 1, 100, fs, order=3)
    
    # 2. Signal Filter (Comb/Notch)
    abd_data = data_utils.SignalFilter(abd_data)
    
    # 3. Scale
    abd_data = scale(abd_data, axis=1)
    
    # 4. Resample to 200Hz
    target_fs = 200
    if fs != target_fs:
        num_samples = int(abd_data.shape[1] * target_fs / fs)
        abd_data = signal.resample(abd_data, num_samples, axis=1)
        
    return abd_data, fs

def evaluate_nifeadb(data_dir, model_path, output_dir='nifeadb_results'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model using InferenceEngine
    print(f"Loading model from {model_path}...")
    try:
        engine = InferenceEngine(model_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    du = DataUtils()
    
    # Get Records
    records_file = os.path.join(data_dir, 'RECORDS')
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            record_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback to listing .hea files
        record_names = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(data_dir, "*.hea"))]
    
    print(f"Found {len(record_names)} records.")
    
    results = []
    
    print("\nProcessing records...")
    print(f"{'Record':<10} {'Duration(s)':<12} {'Orig_FS':<10} {'Est_BPM':<10}")
    print("-" * 50)
    
    for rec_name in record_names:
        rec_path = os.path.join(data_dir, rec_name)
        
        try:
            abdECG, orig_fs = load_wfdb_record(rec_path, du)
            
            if abdECG is None:
                continue
            
            # Use InferenceEngine on the first abdominal channel
            fecg_out = engine.process_signal(abdECG[0, :])
            
            if len(fecg_out) == 0:
                print(f"{rec_name:<10} {'Error':<12}")
                continue
            
            # Simple Peak Detection to estimate BPM
            peaks, _ = find_peaks(fecg_out, distance=30, height=0.1) # distance ~150ms at 200Hz
            
            if len(peaks) > 1:
                # Average RR interval in seconds (fs=200)
                rr_intervals = np.diff(peaks) / 200.0
                avg_rr = np.mean(rr_intervals)
                bpm = 60.0 / avg_rr if avg_rr > 0 else 0
            else:
                bpm = 0
            
            duration = len(fecg_out) / 200.0
            
            print(f"{rec_name:<10} {duration:<12.2f} {orig_fs:<10} {bpm:<10.1f}")
            
            results.append({
                'Record': rec_name,
                'Duration_Sec': duration,
                'Original_FS': orig_fs,
                'Estimated_BPM': bpm
            })
            
            # Save extracted signal
            np.save(os.path.join(output_dir, f"{rec_name}_fecg.npy"), fecg_out)
            
        except Exception as e:
            print(f"Error processing {rec_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save statistics
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "nifeadb_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nProcessing complete. Results saved to {output_dir}/")
    print("\nNOTE: No ground truth annotations were found for this database.")
    print("Metrics (Sensitivity, Precision, F1) could not be calculated.")
    print("Estimated BPM is provided for reference.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing NIFEA DB records')
    parser.add_argument('--model_path', type=str, default='models/sagan_1/245_G_AECG2FECG.pth', help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str, default='nifeadb_results', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_nifeadb(args.input_dir, args.model_path, args.output_dir)
