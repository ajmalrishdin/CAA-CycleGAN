#!/usr/bin/env python3
# ============================================================================
# run_all_checkpoints.py
# ============================================================================
# Evaluates ALL checkpoints of CycleGAN V1 and V2 on the ADFECGDB dataset.
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from db_loaders import DATABASE_REGISTRY
from extraction_engines import CycleGANEngine
from run_comparison import evaluate_one

OUTPUT_DIR = 'Output_checkpoints'

def get_checkpoint_engines():
    base_dir = os.path.dirname(__file__)
    engines = []
    
    # Collect V1
    v1_dir = os.path.join(base_dir, 'models', 'sagan_1')
    if os.path.isdir(v1_dir):
        files = [f for f in os.listdir(v1_dir) if f.endswith('_G_AECG2FECG.pth')]
        steps = sorted([int(f.split('_')[0]) for f in files])
        for step in steps:
            engines.append(CycleGANEngine(v1_dir, version_name=f'CycleGAN_V1_step_{step}', step=step))
            
    # Collect V2
    v2_dir = os.path.join(base_dir, 'models', 'CygleGAN V2 Models')
    if os.path.isdir(v2_dir):
        files = [f for f in os.listdir(v2_dir) if f.endswith('_G_AECG2FECG.pth')]
        steps = sorted([int(f.split('_')[0]) for f in files])
        for step in steps:
            engines.append(CycleGANEngine(v2_dir, version_name=f'CycleGAN_V2_step_{step}', step=step))
            
    return engines

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading all CycleGAN checkpoint engines...")
    try:
        engines = get_checkpoint_engines()
    except Exception as e:
        print(f"Error loading engines: {e}")
        return
        
    print(f"Found {len(engines)} checkpoint engines.")
    
    db_name = 'ADFECGDB'
    db_info = DATABASE_REGISTRY[db_name]
    records, folder = db_info['list_fn']()
    
    print(f"\nRunning evaluation on {db_name} ({len(records)} records).")
    
    all_results = []
    for engine in engines:
        print(f"\n--- {engine.name} ---")
        for rec_name in records:
            data = db_info['load_fn'](rec_name, folder)
            try:
                res = evaluate_one(engine, data, db_name)
                # Parse out the actual step and model name
                parts = engine.name.split('_step_')
                res['Model'] = parts[0]
                res['Step'] = int(parts[1])
                all_results.append(res)
                
                print(f"    {rec_name:<16} F1={res['F1']:.3f}  BPM={res['BPM_mean']:.0f}")
            except Exception as e:
                print(f"    {rec_name:<16} [FAILED] {e}")

    df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(OUTPUT_DIR, 'checkpoint_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Plotting F1 over steps
    agg = df.groupby(['Model', 'Step']).agg(
        F1_mean=('F1', 'mean'),
        BPM_mean=('BPM_mean', 'mean')
    ).reset_index()
    
    # Write aggregated results
    agg_path = os.path.join(OUTPUT_DIR, 'checkpoint_agg_results.csv')
    agg.to_csv(agg_path, index=False)
    
    plt.figure(figsize=(12, 7))
    for model in agg['Model'].unique():
        sub = agg[agg['Model'] == model].sort_values('Step')
        plt.plot(sub['Step'], sub['F1_mean'], marker='o', linewidth=2, label=model)
        
    plt.title('CycleGAN F1 Score vs Checkpoint Step on ADFECGDB', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Mean F1 Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt_path = os.path.join(OUTPUT_DIR, 'checkpoint_f1_plot.png')
    plt.savefig(plt_path, dpi=150, bbox_inches='tight')
    print(f"Saved F1 progression plot to {plt_path}")
    plt.close()

if __name__ == '__main__':
    main()
