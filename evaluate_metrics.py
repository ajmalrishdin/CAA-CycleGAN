
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import glob
import pandas as pd
import time

# Ensure current directory is in path (optional if running from root)
sys.path.append(os.getcwd())

from Utils.DataUtils import DataUtils
from inference import InferenceEngine

def calculate_metrics(fecg_reconstructed, gt_fecg, fqrs_rpeaks):
    """
    Calculate metrics (Sensitivity, Precision, F1, Correlation) 
    given the reconstructed signal and ground truth.
    """
    valid_len = len(fecg_reconstructed)
    
    # Peak detection on reconstructed FECG
    # Using scipy find_peaks
    # distance=30 samples (approx 150ms at 200Hz)
    # height=0.1 (normalized signal [-1, 1], peaks usually distinct)
    detected_peaks, _ = find_peaks(fecg_reconstructed, distance=30, height=0.1) 
    
    # Compare with ground truth
    # Filter GT peaks to be within valid length
    gt_peaks = fqrs_rpeaks[fqrs_rpeaks < valid_len]
    
    tp = 0
    fp = 0
    fn = 0
    
    detected_peaks_set = set(detected_peaks)
    
    # Tolerance window 50ms = 10 samples at 200Hz
    tolerance = 10 
    
    for gt in gt_peaks:
        # Find detected peaks within tolerance
        candidates = [p for p in detected_peaks_set if abs(p - gt) <= tolerance]
        
        if len(candidates) > 0:
            tp += 1
            # Greedy match: take closest
            best_match = min(candidates, key=lambda x: abs(x - gt))
            # Remove from set to avoid double counting (though usually 1-to-1)
            if best_match in detected_peaks_set:
                detected_peaks_set.remove(best_match)
        else:
            fn += 1
            
    # Remaining peaks are False Positives
    fp = len(detected_peaks_set)
    
    # Calculate Pearson Correlation
    # Slice ground truth to match the processed length
    gt_signal = gt_fecg[0, :valid_len]
    
    # Calculate correlation (handle edge cases like constant signal)
    if np.std(fecg_reconstructed) < 1e-6 or np.std(gt_signal) < 1e-6:
        correlation = 0.0
    else:
        correlation = np.corrcoef(fecg_reconstructed, gt_signal)[0, 1]

    return tp, fp, fn, correlation


def evaluate_all_models(model_dir='./models/sagan_1', output_file='evaluation_results.csv'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all checkpoints
    if not os.path.exists(model_dir):
         print(f"Model directory {model_dir} does not exist.")
         return

    model_files = glob.glob(os.path.join(model_dir, '*_G_AECG2FECG.pth'))
    if not model_files:
        print(f"No models found in {model_dir}")
        return

    # Extract steps and sort
    checkpoints = []
    for f in model_files:
        try:
            step = int(os.path.basename(f).split('_')[0])
            checkpoints.append((step, f))
        except ValueError:
            pass
    
    checkpoints.sort(key=lambda x: x[0])
    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")

    # Pre-load data to save time (Read once, evaluate many times)
    print("Pre-loading datasets...")
    data_utils = DataUtils()
    datasets = []
    # Dataset loop (ADFECGDB has 5 records usually r01, r04...)
    for i in range(5):
        try:
            print(f"Loading record {data_utils.fileNames[i]}...")
            abdECG, fetalECG, fqrs_rpeaks = data_utils.readData(i)
            datasets.append({
                'name': data_utils.fileNames[i],
                'abdECG': abdECG, # shape (channels, samples)
                'fetalECG': fetalECG,
                'fqrs_rpeaks': fqrs_rpeaks
            })
        except Exception as e:
            print(f"Error loading record {i}: {e}")
    
    results = []
    
    print("\nStarting evaluation loop...")
    print(f"{'Step':<10} {'Avg Sens':<10} {'Avg Prec':<10} {'Avg F1':<10} {'Avg Corr':<10} {'Time':<10}")
    print("-" * 75)

    # Initialize Engine once (architecture is shared)
    engine = None

    for step, model_path in checkpoints:
        start_time = time.time()
        
        # Load / Update Inference Engine for this checkpoint
        try:
            if engine is None:
                engine = InferenceEngine(model_path, device=device)
            else:
                engine.load_weights(model_path)
        except Exception as e:
            print(f"Failed to load model step {step}: {e}")
            continue

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_corr = 0
        
        # Evaluate on all records
        valid_datasets = 0
        for data in datasets:
            try:
                # Use only the first channel of Abdominal ECG for inference
                # InferenceEngine expects 1D array or (1, N)
                aecg_input = data['abdECG'][0, :] 
                
                # REFACTORED: Use InferenceEngine to get the signal
                fecg_reconstructed = engine.process_signal(aecg_input)
                
                # Calculate metrics
                tp, fp, fn, corr = calculate_metrics(fecg_reconstructed, data['fetalECG'], data['fqrs_rpeaks'])
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_corr += corr
                valid_datasets += 1
            except Exception as e:
                print(f"Error evaluating {data['name']} at step {step}: {e}")
        
        if valid_datasets == 0:
            continue

        # Calculate macro-averages
        avg_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        avg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        avg_f1 = 2 * avg_sens * avg_prec / (avg_sens + avg_prec) if (avg_sens + avg_prec) > 0 else 0
        avg_corr = total_corr / valid_datasets
        
        elapsed = time.time() - start_time
        print(f"{step:<10} {avg_sens:.4f}     {avg_prec:.4f}     {avg_f1:.4f}     {avg_corr:.4f}     {elapsed:.1f}s")
        
        results.append({
            'Step': step,
            'Sensitivity': avg_sens,
            'Precision': avg_prec,
            'F1_Score': avg_f1,
            'Pearson_Corr': avg_corr
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['F1_Score'])
        print(f"\nBest Model: Step {best_model['Step']}")
        print(f"F1 Score: {best_model['F1_Score']:.4f}")
        print(f"Sensitivity: {best_model['Sensitivity']:.4f}")
        print(f"Precision: {best_model['Precision']:.4f}")
        print(f"Correlation: {best_model['Pearson_Corr']:.4f}")

if __name__ == "__main__":
    evaluate_all_models()
