#!/usr/bin/env python3
# ============================================================================
# run_comparison.py
# ============================================================================
# Main pipeline: run every compatible (technique × database) combination,
# compute metrics, save CSV + comparison plots.
#
# Usage:
#   python run_comparison.py                  # full run
#   python run_comparison.py --quick          # 1 record per DB (smoke test)
#   python run_comparison.py --db ADFECGDB    # single database
#   python run_comparison.py --engine ILHSAF  # single technique
# ============================================================================

import os
import sys
import time
import argparse
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from db_loaders import DATABASE_REGISTRY
from extraction_engines import get_all_engines
from metrics import detect_fetal_r_peaks, calculate_f1_precision_recall, calculate_bpm

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'Output_comparison'


# ============================================================================
# Compatibility rules
# ============================================================================

# Databases where CycleGAN is experimental due to domain mismatch
EXPERIMENTAL_CYCLEGAN = {'NINFEA'}


def is_compatible(engine, db_name, db_info):
    """Check if an engine–database combination should be run."""
    # ILHSAF with real chest — only if DB has chest
    # ILHSAF with synthetic ref — always allowed
    # CycleGAN — always allowed (experimental flag for NINFEA)
    return True


# ============================================================================
# Main evaluation logic
# ============================================================================

def evaluate_one(engine, data, db_name):
    """
    Run one extraction engine on one record and return a metrics dict.
    """
    abd = data['abdominal']
    chest_raw = data['chest']
    fqrs_true = data['fqrs']
    fs = data['fs']
    record = data['record']

    # The CycleGAN models were trained aggressively on Abdomen_1 (channel 0 index).
    # Selecting by highest variance biases towards channels with the strongest maternal ECG,
    # destroying the fECG signal characteristics the model expects. 
    # Use the first channel for consistency across methods unless specifically analyzing.
    if abd.ndim == 2:
        abd_1d = abd[0]
    else:
        abd_1d = abd

    # Prepare chest signal (1‑D)
    chest_1d = None
    if chest_raw is not None and chest_raw.ndim >= 1:
        if chest_raw.ndim == 2:
            chest_1d = chest_raw[0]  # first chest channel
        else:
            chest_1d = chest_raw

    # Determine reference mode for ILHSAF
    ref_mode = 'N/A'
    if engine.name == 'ILHSAF':
        ref_mode = 'real_chest' if chest_1d is not None else 'synthetic'

    # Run extraction
    t0 = time.time()
    fecg = engine.extract_fecg(abd_1d, chest=chest_1d, fs=fs)
    elapsed = time.time() - t0

    # Detect R‑peaks in extracted fECG
    detected_peaks = detect_fetal_r_peaks(fecg, fs)

    # BPM
    avg_bpm, std_bpm, _ = calculate_bpm(fecg, fs)

    # F1 / Precision / Recall (only if ground truth available)
    f1, prec, recall = 0.0, 0.0, 0.0
    has_gt = fqrs_true is not None and len(fqrs_true) > 0
    if has_gt:
        f1, prec, recall = calculate_f1_precision_recall(
            detected_peaks, fqrs_true, tolerance_ms=50, fs=fs)

    experimental = db_name in EXPERIMENTAL_CYCLEGAN and 'CycleGAN' in engine.name

    return {
        'Database': db_name,
        'Record': record,
        'Technique': engine.name,
        'ILHSAF_Ref': ref_mode,
        'F1': f1,
        'Precision': prec,
        'Recall': recall,
        'BPM_mean': avg_bpm,
        'BPM_std': std_bpm,
        'Detected_Peaks': len(detected_peaks),
        'GT_Peaks': len(fqrs_true) if has_gt else -1,
        'Has_GT': has_gt,
        'Experimental': experimental,
        'Time_s': elapsed,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(df, output_dir):
    """Generate comparison bar charts from aggregate results."""
    # Only plot databases that have ground truth
    df_gt = df[df['Has_GT']].copy()

    if df_gt.empty:
        print("  No ground‑truth data to plot.")
        return

    agg = df_gt.groupby(['Database', 'Technique']).agg(
        F1=('F1', 'mean'),
        Precision=('Precision', 'mean'),
        Recall=('Recall', 'mean'),
        BPM=('BPM_mean', 'mean'),
    ).reset_index()

    databases = agg['Database'].unique()
    techniques = agg['Technique'].unique()
    n_db = len(databases)
    n_tech = len(techniques)

    # Color palette
    colors = ['#4361EE', '#F72585', '#4CC9F0', '#7209B7', '#3A0CA3']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ['F1', 'Precision', 'Recall']
    titles = ['F1 Score', 'Precision', 'Recall / Sensitivity']

    bar_width = 0.8 / max(n_tech, 1)

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        x = np.arange(n_db)
        for j, tech in enumerate(techniques):
            vals = []
            for db in databases:
                row = agg[(agg['Database'] == db) & (agg['Technique'] == tech)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (j - (n_tech - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width * 0.9,
                   label=tech, color=colors[j % len(colors)], edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(databases, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('fECG Extraction — Technique Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved comparison_plot.png")
    plt.close()

    # BPM plot (all databases including those without GT)
    agg_all = df.groupby(['Database', 'Technique']).agg(
        BPM=('BPM_mean', 'mean'),
    ).reset_index()

    databases_all = agg_all['Database'].unique()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x2 = np.arange(len(databases_all))
    for j, tech in enumerate(techniques):
        vals = []
        for db in databases_all:
            row = agg_all[(agg_all['Database'] == db) & (agg_all['Technique'] == tech)]
            vals.append(row['BPM'].values[0] if len(row) > 0 else 0)
        offset = (j - (n_tech - 1) / 2) * bar_width
        ax2.bar(x2 + offset, vals, bar_width * 0.9,
                label=tech, color=colors[j % len(colors)], edgecolor='white')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(databases_all, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('BPM', fontsize=11)
    ax2.set_title('Estimated Fetal Heart Rate (BPM)', fontsize=13, fontweight='bold')
    ax2.axhspan(110, 160, alpha=0.1, color='green', label='Normal range')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bpm_comparison_plot.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved bpm_comparison_plot.png")
    plt.close()


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare fECG extraction techniques across databases')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 1 record per database')
    parser.add_argument('--db', type=str, default=None,
                        help='Run only this database (e.g. ADFECGDB)')
    parser.add_argument('--engine', type=str, default=None,
                        help='Run only this engine (e.g. ILHSAF, CycleGAN_V1)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load engines
    print("=" * 70)
    print("fECG Extraction — Comparative Evaluation")
    print("=" * 70)
    print("\nLoading extraction engines...")
    engines = get_all_engines()
    if args.engine:
        engines = [e for e in engines if e.name == args.engine]
    print(f"  Engines: {[e.name for e in engines]}")

    # Select databases
    db_names = list(DATABASE_REGISTRY.keys())
    if args.db:
        db_names = [args.db]
    print(f"  Databases: {db_names}")
    print()

    all_results = []

    for db_name in db_names:
        db_info = DATABASE_REGISTRY[db_name]
        print(f"{'=' * 70}")
        print(f"Database: {db_name} — {db_info['description']}")
        print(f"{'=' * 70}")

        try:
            records, folder = db_info['list_fn']()
        except Exception as e:
            print(f"  [ERROR] Could not list records: {e}")
            continue

        if args.quick:
            records = records[:1]

        print(f"  Records to process: {len(records)}")

        for engine in engines:
            if not is_compatible(engine, db_name, db_info):
                print(f"  [{engine.name}] SKIPPED — incompatible")
                continue

            exp_tag = " [EXPERIMENTAL]" if (db_name in EXPERIMENTAL_CYCLEGAN
                                            and 'CycleGAN' in engine.name) else ""
            print(f"\n  --- {engine.name}{exp_tag} ---")

            for rec_name in records:
                try:
                    data = db_info['load_fn'](rec_name, folder)
                    result = evaluate_one(engine, data, db_name)
                    all_results.append(result)

                    status = f"F1={result['F1']:.3f}" if result['Has_GT'] else "no GT"
                    print(f"    {rec_name:<16} {status}  "
                          f"BPM={result['BPM_mean']:.0f}  "
                          f"det={result['Detected_Peaks']}  "
                          f"({result['Time_s']:.1f}s)")
                except Exception as e:
                    print(f"    {rec_name:<16} [FAILED] {e}")
                    traceback.print_exc()

    # Save results
    if not all_results:
        print("\nNo results to save.")
        return

    df = pd.DataFrame(all_results)

    csv_path = os.path.join(args.output, 'comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nPer‑record results saved to {csv_path}")

    # Aggregate summary
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    df_gt = df[df['Has_GT']]
    if not df_gt.empty:
        agg = df_gt.groupby(['Database', 'Technique']).agg(
            F1_mean=('F1', 'mean'),
            F1_std=('F1', 'std'),
            Prec_mean=('Precision', 'mean'),
            Recall_mean=('Recall', 'mean'),
            BPM_mean=('BPM_mean', 'mean'),
            N=('Record', 'count'),
        ).reset_index()

        print("\nDatabases WITH ground‑truth annotations:")
        print(agg.to_string(index=False, float_format='%.4f'))

        agg.to_csv(os.path.join(args.output, 'aggregate_summary.csv'), index=False)
        print(f"\nAggregate summary saved to {args.output}/aggregate_summary.csv")

    df_nogt = df[~df['Has_GT']]
    if not df_nogt.empty:
        agg2 = df_nogt.groupby(['Database', 'Technique']).agg(
            BPM_mean=('BPM_mean', 'mean'),
            BPM_std=('BPM_mean', 'std'),
            N=('Record', 'count'),
        ).reset_index()

        print("\nDatabases WITHOUT ground‑truth annotations (BPM only):")
        print(agg2.to_string(index=False, float_format='%.1f'))

    # Plots
    print("\nGenerating plots...")
    plot_comparison(df, args.output)

    print(f"\n{'=' * 70}")
    print("Done! All outputs in:", args.output)
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
