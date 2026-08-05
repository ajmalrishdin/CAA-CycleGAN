#!/usr/bin/env python3
"""
Evaluate all checkpoints in selected CycleGAN model folders on ADFECGDB.

Default target folders:
- models/sagan_1_SynDB1_bs128_1n8
- models/sagan_1_SynDB1_bsdef_1n8

Outputs:
- detailed per-record CSV
- aggregated per-checkpoint CSV
- aggregated per-folder CSV
"""

import argparse
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from db_loaders import DATABASE_REGISTRY
from extraction_engines import CycleGANEngine
from run_comparison import evaluate_one


DEFAULT_MODEL_DIRS = [
    'models/sagan_1_SynDB1_bs128_1n8',
    'models/sagan_1_SynDB1_bsdef_1n8',
]


def collect_steps(model_dir):
    files = [f for f in os.listdir(model_dir) if f.endswith('_G_AECG2FECG.pth')]
    steps = sorted({int(f.split('_')[0]) for f in files})
    return steps


def resolve_adfecgdb_folder(project_root, override_folder=None):
    if override_folder:
        folder = os.path.abspath(override_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'ADFECGDB folder not found: {folder}')
        return folder

    candidates = [
        os.path.join(project_root, 'Databases', 'ADFECGDB'),
        os.path.join(project_root, 'inference', 'Databases', 'ADFECGDB'),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return folder

    raise FileNotFoundError(
        'Could not locate ADFECGDB folder. Use --adfecgdb-folder to provide it explicitly.'
    )


def list_adfecgdb_edf_records(adfecgdb_folder, include_arr=True):
    """List EDF records directly from disk, optionally including ARR variants."""
    files = sorted([f for f in os.listdir(adfecgdb_folder) if f.lower().endswith('.edf')])
    if include_arr:
        return files
    return [f for f in files if '_ARR_' not in f]


def main():
    parser = argparse.ArgumentParser(description='Evaluate selected CycleGAN model folders on ADFECGDB')
    parser.add_argument('--model-dirs', nargs='+', default=DEFAULT_MODEL_DIRS,
                        help='Model directories relative to project root or absolute paths')
    parser.add_argument('--adfecgdb-folder', type=str, default=None,
                        help='Path to ADFECGDB folder (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default='inference/Output_selected_models',
                        help='Directory to save result CSVs')
    parser.add_argument('--max-records', type=int, default=0,
                        help='Optional cap for records per checkpoint (0 = all records)')
    parser.add_argument('--max-checkpoints', type=int, default=0,
                        help='Optional cap for checkpoints per folder (0 = all checkpoints)')
    parser.add_argument('--include-arr', action='store_true',
                        help='Include ARR EDF records (e.g. r01_ARR_1.edf) in evaluation')
    parser.add_argument('--all-edf', action='store_true',
                        help='List records from all EDFs in folder (ignores RECORDS file)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    resolved_model_dirs = []
    for rel_or_abs in args.model_dirs:
        model_dir = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(project_root, rel_or_abs)
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f'Model directory not found: {model_dir}')
        resolved_model_dirs.append(model_dir)

    adfecgdb_folder = resolve_adfecgdb_folder(project_root, args.adfecgdb_folder)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    db_name = 'ADFECGDB'
    db_info = DATABASE_REGISTRY[db_name]
    if args.all_edf:
        records = list_adfecgdb_edf_records(adfecgdb_folder, include_arr=args.include_arr)
    else:
        records, _ = db_info['list_fn'](adfecgdb_folder)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]

    all_results = []
    failures = []

    print('=' * 80)
    print('Evaluating selected model folders on ADFECGDB')
    print('=' * 80)
    print(f'ADFECGDB folder: {adfecgdb_folder}')
    print(f'Record count: {len(records)}')
    print(f'Model folders: {len(resolved_model_dirs)}')

    for model_dir in resolved_model_dirs:
        folder_name = os.path.basename(model_dir)
        steps = collect_steps(model_dir)
        if args.max_checkpoints and args.max_checkpoints > 0:
            steps = steps[:args.max_checkpoints]

        print(f'\n[{folder_name}] checkpoints: {len(steps)} -> {steps}')

        for step in steps:
            version_name = f'{folder_name}_step_{step}'
            print(f'\n  Loading {folder_name}: step {step}')
            try:
                engine = CycleGANEngine(model_dir, version_name=version_name, step=step)
            except Exception as ex:
                failures.append({
                    'Folder': folder_name,
                    'Step': step,
                    'Record': None,
                    'Error': f'ENGINE_LOAD_FAILED: {ex}',
                })
                print(f'    [ENGINE_LOAD_FAILED] {ex}')
                continue

            for rec_name in records:
                try:
                    data = db_info['load_fn'](rec_name, adfecgdb_folder)
                    res = evaluate_one(engine, data, db_name)
                    res['Folder'] = folder_name
                    res['Step'] = step
                    all_results.append(res)
                    gt_tag = (
                        f"det={res['Detected_Peaks']} gt={res['GT_Peaks']}"
                        if res.get('Has_GT')
                        else "NO_GT"
                    )
                    print(
                        f"    {rec_name:<16} F1={res['F1']:.3f} "
                        f"Acc={res['Accuracy']:.3f} "
                        f"Sens={res['Sensitivity']:.3f}  ({gt_tag})"
                    )
                except Exception as ex:
                    failures.append({
                        'Folder': folder_name,
                        'Step': step,
                        'Record': rec_name,
                        'Error': str(ex),
                    })
                    print(f'    {rec_name:<16} [FAILED] {ex}')

    if not all_results:
        raise RuntimeError('No evaluation results were produced. Check paths and model checkpoints.')

    df = pd.DataFrame(all_results)

    detailed_csv = os.path.join(output_dir, f'selected_models_detailed_{ts}.csv')
    df.to_csv(detailed_csv, index=False)

    per_checkpoint = df.groupby(['Folder', 'Step']).agg(
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std'),
        Accuracy_mean=('Accuracy', 'mean'),
        Sensitivity_mean=('Sensitivity', 'mean'),
        Precision_mean=('Precision', 'mean'),
        Recall_mean=('Recall', 'mean'),
        BPM_mean=('BPM_mean', 'mean'),
        TP_total=('TP', 'sum'),
        FP_total=('FP', 'sum'),
        FN_total=('FN', 'sum'),
        Num_records=('Record', 'count'),
    ).reset_index().sort_values(['Folder', 'Step'])

    checkpoint_csv = os.path.join(output_dir, f'selected_models_per_checkpoint_{ts}.csv')
    per_checkpoint.to_csv(checkpoint_csv, index=False)

    per_folder = df.groupby(['Folder']).agg(
        F1_mean=('F1', 'mean'),
        Accuracy_mean=('Accuracy', 'mean'),
        Sensitivity_mean=('Sensitivity', 'mean'),
        Precision_mean=('Precision', 'mean'),
        Recall_mean=('Recall', 'mean'),
        BPM_mean=('BPM_mean', 'mean'),
        TP_total=('TP', 'sum'),
        FP_total=('FP', 'sum'),
        FN_total=('FN', 'sum'),
        Num_samples=('Record', 'count'),
    ).reset_index().sort_values(['F1_mean', 'Accuracy_mean'], ascending=False)

    folder_csv = os.path.join(output_dir, f'selected_models_per_folder_{ts}.csv')
    per_folder.to_csv(folder_csv, index=False)

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_csv = os.path.join(output_dir, f'selected_models_failures_{ts}.csv')
        failures_df.to_csv(failures_csv, index=False)
    else:
        failures_csv = None

    print('\n' + '=' * 80)
    print('Evaluation complete')
    print('=' * 80)
    print(f'Detailed results:      {detailed_csv}')
    print(f'Per-checkpoint summary:{checkpoint_csv}')
    print(f'Per-folder summary:    {folder_csv}')
    if failures_csv:
        print(f'Failures:              {failures_csv}')

    print('\nTop checkpoints by F1:')
    print(per_checkpoint.sort_values('F1_mean', ascending=False).head(10).to_string(index=False))

    print('\nFolder summary:')
    print(per_folder.to_string(index=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
