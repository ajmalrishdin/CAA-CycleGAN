
import os
import glob
import pandas as pd
import numpy as np
import torch
import sys
from scipy import signal
from sklearn.preprocessing import scale

# Ensure current directory is in path
sys.path.append(os.getcwd())

from Utils.DataUtils import DataUtils
from inference import InferenceEngine
from evaluate_metrics import calculate_metrics

def load_physionet_record(csv_path, fqrs_path, data_utils):
    """
    Load and preprocess a PhysioNet record (CSV + FQRS txt)
    Mimics DataUtils processing: Filter -> Scale -> Resample
    """
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        # Drop units row (asssuming row 1 is units, row 0 is header)
        df = df.drop(0) 
        
        # Extract data columns (Timestamp, Ch1, Ch2, Ch3, Ch4)
        # We need data columns. Assuming columns 1-4 are signals.
        data = df.iloc[:, 1:5].values.astype(float).T # Shape (4, N)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None, None
    
    # Process using DataUtils methods
    # 1. Bandpass 1-100Hz (fs=1000)
    data = data_utils.butter_bandpass_filter(data, 1, 100, 1000, order=3)
    
    # 2. Signal Filter (Comb/Notch)
    data = data_utils.SignalFilter(data)
    
    # 3. Scale (StandardScaler per channel)
    data = scale(data, axis=1)
    
    # 4. Resample to 200Hz (Downsample by 5)
    # Original is 1000Hz.
    new_len = int(data.shape[1] / 5)
    data = signal.resample(data, new_len, axis=1)
    
    # Load FQRS annotations
    if os.path.exists(fqrs_path):
        try:
            fqrs = np.loadtxt(fqrs_path)
            if fqrs.size > 0:
                fqrs = np.atleast_1d(fqrs)
                # Scale annotations to 200Hz
                fqrs = np.round(fqrs / 5).astype(int)
            else:
                fqrs = np.array([])
        except:
            fqrs = np.array([])
    else:
        fqrs = np.array([])
        
    return data, fqrs

def evaluate_physionet(data_dir, model_path, output_file='physionet_results.csv'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        engine = InferenceEngine(model_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Init DataUtils for match filters
    du = DataUtils()
    
    # Find files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "a*.csv")))
    print(f"Found {len(csv_files)} records.")
    
    results = []
    
    print("\nStarting evaluation...")
    print(f"{'Record':<10} {'Sens':<10} {'Prec':<10} {'F1':<10} {'Corr':<10}")
    print("-" * 60)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        fqrs_file = os.path.join(data_dir, f"{base_name}.fqrs.txt")
        
        try:
            abdECG, fqrs = load_physionet_record(csv_file, fqrs_file, du)
            
            if abdECG is None or len(fqrs) == 0:
                continue
                
            # Create a dummy ground truth signal for correlation calculation.
            # Since PhysioNet only provides peak locations, we cannot calculate signal correlation.
            # Passing zeros will result in 0 correlation from calculate_metrics.
            dummy_gt_fecg = np.zeros_like(abdECG)
            
            # REFACTORED: Use InferenceEngine
            # Pass first channel
            fecg_reconstructed = engine.process_signal(abdECG[0, :])
            
            # REFACTORED: Use shared metric calculation
            tp, fp, fn, corr = calculate_metrics(
                fecg_reconstructed, 
                dummy_gt_fecg, 
                fqrs
            )
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0
            
            print(f"{base_name:<10} {sens:.4f}     {prec:.4f}     {f1:.4f}     {'N/A':<10}")
            
            results.append({
                'Record': base_name,
                'TP': tp, 
                'FP': fp, 
                'FN': fn,
                'Sensitivity': sens,
                'Precision': prec,
                'F1_Score': f1
            })
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

    # Overall metrics
    avg_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    avg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    avg_f1 = 2 * avg_sens * avg_prec / (avg_sens + avg_prec) if (avg_sens + avg_prec) > 0 else 0
    
    print("-" * 60)
    print(f"{'Overall':<10} {avg_sens:.4f}     {avg_prec:.4f}     {avg_f1:.4f}     {'N/A':<10}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing PhysioNet set-a-ext-text csv files')
    parser.add_argument('--model_path', type=str, default='models/sagan_1/245_G_AECG2FECG.pth', help='Path to generator checkpoint')
    parser.add_argument('--output', type=str, default='physionet_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    evaluate_physionet(args.input_dir, args.model_path, args.output)
