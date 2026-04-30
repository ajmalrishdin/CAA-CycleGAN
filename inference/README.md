# Inference Wiki

This folder now holds the modular implementation for inference and evaluation.
Use the root-level [inference.py](../inference.py) as the single entrypoint.

## 1) Main entrypoint

From the project root:

```bash
source en/bin/activate
python inference.py --help
```

Available workflows:

- `signal`
  - Runs the AECG to MECG/FECG conversion workflow.

- `target-folders`
  - Evaluates one or more CycleGAN model folders on ADFECGDB.
  - This is the default workflow when you run `python inference.py` with no workflow name.
  - Supports `--all-edf`, `--include-arr`, `--max-records`, and `--max-checkpoints`.

- `compare`
  - Runs the generic multi-engine / multi-database comparison pipeline.
  - Useful when you want to compare CycleGAN against other extraction methods.

- `all-checkpoints`
  - Sweeps the legacy default V1/V2 checkpoint folders.
  - Best for broad historical comparisons.

- `nifeadb-step`
  - Evaluates one CycleGAN checkpoint on NIFEADB.
  - Uses a proxy SVD reference because that dataset does not provide fetal QRS annotations.

Examples:

```bash
# Signal extraction from a single AECG file
python inference.py signal --input demo

# Current recommended ADFECGDB sweep
python inference.py target-folders \
  --model-dirs models/sagan_1_SynDB1_bs128_1n8 models/sagan_1_SynDB1_bsdef_1n8 \
  --adfecgdb-folder Databases/ADFECGDB \
  --all-edf --include-arr

# Quick comparison smoke test
python inference.py compare --db ADFECGDB --quick

# Legacy checkpoint sweep
python inference.py all-checkpoints

# NIFEADB proxy-metric evaluation
python inference.py nifeadb-step --model-dir models/sagan_1_SynDB1_bs128_1n8 --step 113
```

## 2) What lives in `inference/`

- `db_loaders.py`
  - Unified dataset loaders.
  - Responsible for listing records and loading waveform + annotations.
  - For ADFECGDB, it loads EDF data and `.qrs` annotations.

- `inference_core.py`
  - Shared MECG/FECG model wrapper used by the signal workflow and other callers.

- `extraction_engines.py`
  - Wrapper engines used by evaluation scripts.
  - `CycleGANEngine`: loads CycleGAN checkpoints and extracts fetal ECG.
  - `ILHSAFEngine`: adaptive filter baseline.

- `metrics.py`
  - Peak detection and metrics logic.
  - Includes F1, precision, recall, event-level accuracy, TP/FP/FN counts, and BPM metrics.

- `run_comparison.py`
  - Shared evaluation core.
  - Runs one engine on one record via `evaluate_one`, then builds CSV summaries and plots.

- `run_target_model_folders.py`
  - Targeted CycleGAN folder sweep for ADFECGDB.
  - Handles all EDFs, optional ARR inclusion, and checkpoint/folder summaries.

- `run_all_checkpoints.py`
  - Legacy fixed-folder checkpoint sweep for the default V1/V2 model folders.

- `run_nifeadb_step_metrics.py`
  - NIFEADB-specific checkpoint evaluation with proxy-reference metrics.

- `run_cyclegan.py`
  - The signal conversion CLI used by the `signal` workflow.

- `export_edfs.py`, `generate_waveforms.py`
  - Utility scripts for waveform export/generation.
  - Not required for the primary inference workflows.

## 3) Metric definitions

- `Sensitivity` = `Recall` = `TP / (TP + FN)`
- `Precision` = `TP / (TP + FP)`
- `F1` = harmonic mean of precision and recall
- `Accuracy` = event-level beat-detection accuracy:
  - `TP / (TP + FP + FN)`
- `Specificity` is not used for event-based peak matching and is kept as `NaN`.

## 4) Record selection

ADFECGDB often has both base records and ARR sub-records.

- Base-only mode uses the legacy `RECORDS` file.
- Full mode scans all `.edf` files in the database folder.

Use full mode to include files like:

- `r01_ARR_1.edf`
- `r08_ARR_12.edf`

## 5) Outputs

Primary output folder for target-folder evaluation:

- `inference/Output_selected_models/`

Signal workflow outputs:

- `outputs/`
- Custom output directories when passed via `--output_dir`

Typical files created by `target-folders`:

- `selected_models_detailed_<timestamp>.csv`
- `selected_models_per_checkpoint_<timestamp>.csv`
- `selected_models_per_folder_<timestamp>.csv`
- `selected_models_failures_<timestamp>.csv` when something fails

Stable latest copies, when you promote them manually for reporting:

- `outputs/inference_eval/selected_models_detailed_latest.csv`
- `outputs/inference_eval/selected_models_per_checkpoint_latest.csv`
- `outputs/inference_eval/selected_models_per_folder_latest.csv`

## 6) Typical workflow

1. Put checkpoint files into a model folder.
2. Run `python inference.py target-folders` with that folder and ADFECGDB path.
3. Inspect the per-checkpoint CSV to find the best step.
4. Inspect the per-folder CSV for the overall comparison.
5. Copy the preferred results into `outputs/inference_eval/` if you want a stable report artifact.

## 7) Cleanup guidance

Safe to remove if you want to reclaim space:

- `inference/__pycache__/`
- `inference/.DS_Store`
- Old CSVs in `inference/Output_selected_models/` if you only keep the latest outputs

Keep these files:

- `db_loaders.py`
- `extraction_engines.py`
- `metrics.py`
- `run_comparison.py`
- `run_target_model_folders.py`

Only remove the utility scripts if you are certain you do not use them.

## 8) ARR-inclusive results

Including ARR files increases difficulty and usually lowers aggregate F1 and accuracy compared with base-only records. That is expected because ARR segments are more challenging and noisy.

Always compare runs with the same record-selection mode so the numbers stay meaningful.
