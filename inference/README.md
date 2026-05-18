# Inference Wiki

This folder now holds the modular implementation for inference and evaluation.
Use the root-level [inference.py](../inference.py) as the single entrypoint.

## 1) Quick Start: Interactive FECG Extraction & Visualization

For an interactive workflow to extract fetal ECG from your own recordings:

```bash
source en/bin/activate
python inference/final_inference.py \
  --input Databases/ADFECGDB/r01.edf \
  --pth_path models/sagan_1/100_G_AECG2FECG.pth \
  --device cpu
```

**Features:**
- Load `.edf` or `.dat` ECG recordings
- Select any available channel from the file
- Extract FECG using a trained CycleGAN generator
- View extracted FECG and original AECG in real-time interactive viewer
- Navigate with arrow keys, save PNG snapshots with 'S' key

**Interactive Controls:**
- **Left/Right arrows**: Scroll ±0.5 seconds
- **Ctrl+Left/Ctrl+Right**: Jump ±4 seconds
- **S key** or **Save PNG button**: Save current view as image
- **Navigation buttons**: Use GUI buttons at bottom for scrolling

**Arguments:**
- `--input`: Path to input ECG file (.edf or .dat)
- `--pth_path`: Path to generator model (.pth file)
- `--device`: Device to use (cpu, cuda, mps) — default: cpu
- `--window-size`: Viewer window duration in seconds — default: 4.0
- `--save-dir`: Output directory for saved PNG images — default: outputs/fecg_extraction

**For .dat files (optional):**
- `--raw`: Treat as raw binary instead of WFDB format
- `--auto-fallback`: Fallback to raw loading if WFDB parsing fails
- `--raw-int16`: Read raw data as int16 (default: float32)
- `--n-channels`: Number of channels for raw files — default: 4
- `--fs`: Sampling rate for raw files (Hz) — default: 1000
- `--gain`: Scaling factor for int16 data — default: 1000

**Example with interactive prompts (no --pth_path needed):**
```bash
python inference/final_inference.py --input Databases/ADFECGDB/r01.edf
# Script will prompt you to select:
# 1. Which channel to extract from
# 2. Path to .pth model file
```

## 2) Main entrypoint

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

## 3) What lives in `inference/`

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

- `final_inference.py`
  - **Interactive FECG extraction and visualization pipeline.**
  - Combines file loading (EDF/DAT), FECG extraction (via trained generator), and live signal viewing.
  - User can select which channel to extract from and choose the generator model interactively or via CLI args.
  - Displays both input (AECG) and extracted (FECG) signals side-by-side with real-time navigation.
  - See Quick Start section above for usage.

- `plot_signal.py`
  - Standalone interactive ECG viewer for .edf and .dat files.
  - Useful for browsing and inspecting raw recordings before/after processing.

## 4) Metric definitions

- `Sensitivity` = `Recall` = `TP / (TP + FN)`
- `Precision` = `TP / (TP + FP)`
- `F1` = harmonic mean of precision and recall
- `Accuracy` = event-level beat-detection accuracy:
  - `TP / (TP + FP + FN)`
- `Specificity` is not used for event-based peak matching and is kept as `NaN`.

## 5) Record selection

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

## 8) Interactive signal viewer

Use the interactive ECG viewer to browse EDF and DAT files with keyboard/button controls:

```bash
python inference/plot_signal.py Databases/ADFECGDB/r01.edf
python inference/plot_signal.py Databases/non-invasive-fetal-ecg-arrhythmia-database-1.0.0/example.dat
```

**Features:**
- **Display:** Up to 5 channels stacked vertically with time-locked view
- **Navigation:**
  - `←` / `→` arrow keys: scroll ±0.5 seconds (configurable with `--step-small`)
  - `Ctrl+←` / `Ctrl+→`: jump one full window (default 4 seconds)
  - Buttons at bottom for mouse-based navigation
- **Save:** Press `S` or click "Save PNG" to capture the current view

**Optional flags:**
- `--window-size 4.0`: Time window width in seconds (default 4.0)
- `--step-small 0.5`: Amount to shift per arrow key press (default 0.5)
- `--max-channels 5`: Limit display to N channels (default 5)
- `--save-dir outputs/matplotlib`: Where to save PNG snapshots
- `--raw`: Load `.dat` as raw binary instead of WFDB
- `--raw-int16`: Load raw `.dat` as int16 (instead of float32)
- `--fs 1000`: Sampling rate for raw DAT (default 1000)
- `--gain 1000`: Gain factor for int16→float scaling (default 1000)

## 9) ARR-inclusive results

Including ARR files increases difficulty and usually lowers aggregate F1 and accuracy compared with base-only records. That is expected because ARR segments are more challenging and noisy.

Always compare runs with the same record-selection mode so the numbers stay meaningful.

It is not just comparison. The unified runner in inference.py:16 dispatches to 5 workflows, and several of them write files.

signal
Script: run_cyclegan.py:74
Saves: 3 NumPy arrays + 1 PNG plot
Files:
result_aecg.npy
result_mecg.npy
result_fecg.npy
result_comparison.png
Default folder: outputs
Save implementation: inference_core.py:139
target-folders
Script: run_target_model_folders.py:170
Saves: CSV files only
Files:
selected_models_detailed_timestamp.csv
selected_models_per_checkpoint_timestamp.csv
selected_models_per_folder_timestamp.csv
selected_models_failures_timestamp.csv (only if failures)
Default folder: inference/Output_selected_models
compare
Script: run_comparison.py:318
Saves: CSV + PNG
Files:
comparison_results.csv
aggregate_summary.csv
comparison_plot.png
bpm_comparison_plot.png
Default folder: Output_comparison
all-checkpoints
Script: run_all_checkpoints.py:80
Saves: CSV + PNG
Files:
checkpoint_results.csv
checkpoint_agg_results.csv
checkpoint_f1_plot.png
Default folder: Output_checkpoints
nifeadb-step
Script: run_nifeadb_step_metrics.py:379
Saves: CSV + per-record PNG waveforms
Files:
nifeadb_step_STEP_detailed_timestamp.csv
nifeadb_step_STEP_summary_timestamp.csv
record_waveforms.png for each processed record
Default folder: inference/Output_selected_models
So yes, it does save PNGs and CSVs (and in signal mode also NPY). It is not only a comparison printout.

