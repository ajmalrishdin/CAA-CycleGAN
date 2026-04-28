# Inference Wiki

This wiki explains how the inference pipeline works, how to run evaluations, where results are saved, and which files are safe to remove.

## 1) What the inference folder contains

- `db_loaders.py`
  - Unified dataset loaders.
  - Responsible for listing records and loading waveform + annotations.
  - For ADFECGDB, it loads EDF data and `.qrs` annotations.

- `extraction_engines.py`
  - Wrapper engines used by evaluation scripts.
  - `CycleGANEngine`: loads CycleGAN checkpoints and extracts fetal ECG.
  - `ILHSAFEngine`: adaptive filter baseline.

- `metrics.py`
  - Peak detection and metrics logic.
  - Includes:
    - F1
    - Precision
    - Sensitivity (Recall)
    - Accuracy (event-level: `TP / (TP + FP + FN)`)
    - TP/FP/FN counts
    - BPM metrics

- `run_comparison.py`
  - Generic comparison runner across engines/databases.
  - Produces CSV and comparison plots.

- `run_all_checkpoints.py`
  - Evaluates checkpoint sweeps for the default V1/V2 model folders.

- `run_target_model_folders.py`
  - Targeted runner for arbitrary model folders.
  - This is the script used to evaluate:
    - `models/sagan_1_SynDB1_bs128_1n8`
    - `models/sagan_1_SynDB1_bsdef_1n8`
  - Supports all ADFECGDB EDFs including ARR variants.

- `export_edfs.py`, `generate_waveforms.py`
  - Utility scripts for waveform export/generation.
  - Not required for basic evaluation pipeline.

- `Output_comparison/`, `Output_selected_models/`
  - Output folders created by evaluation runs.

## 2) Current metric definitions

- `Sensitivity` = `Recall` = `TP / (TP + FN)`
- `Precision` = `TP / (TP + FP)`
- `F1` = harmonic mean of precision and recall
- `Accuracy` = event-level accuracy for beat detection:
  - `TP / (TP + FP + FN)`
- `Specificity` is not used for event-based peak matching and is kept as `NaN`.

## 3) How records are selected (important)

ADFECGDB often has both base records and ARR sub-records.

- Base-only mode (legacy): uses `RECORDS` file.
- Full mode (recommended): scans all `.edf` files in folder.

Use full mode to include files like:
- `r01_ARR_1.edf`
- `r08_ARR_12.edf`

## 4) How to run evaluation for your two model folders

From project root:

```bash
source en/bin/activate
python inference/run_target_model_folders.py \
  --model-dirs models/sagan_1_SynDB1_bs128_1n8 models/sagan_1_SynDB1_bsdef_1n8 \
  --adfecgdb-folder Databases/ADFECGDB \
  --all-edf --include-arr
```

Optional controls:

```bash
# Use only first N records for smoke test
--max-records 5

# Use only first N checkpoints per folder
--max-checkpoints 2
```

## 5) Output files and where they are saved

Primary run output folder:
- `inference/Output_selected_models/`

Each run creates timestamped CSV files:
- `selected_models_detailed_<timestamp>.csv`
- `selected_models_per_checkpoint_<timestamp>.csv`
- `selected_models_per_folder_<timestamp>.csv`

Stable latest copies (manually copied for convenience):
- `outputs/inference_eval/selected_models_detailed_latest.csv`
- `outputs/inference_eval/selected_models_per_checkpoint_latest.csv`
- `outputs/inference_eval/selected_models_per_folder_latest.csv`

## 6) CSV schema quick reference

Detailed CSV includes per-record/per-checkpoint rows with fields such as:
- `Folder`, `Step`, `Database`, `Record`, `Technique`
- `F1`, `Accuracy`, `Sensitivity`, `Precision`, `Recall`
- `TP`, `FP`, `FN`
- `BPM_mean`, `BPM_std`, `Detected_Peaks`, `GT_Peaks`, `Time_s`

Per-checkpoint CSV aggregates by `Folder + Step`.

Per-folder CSV aggregates by `Folder` over all checkpoints and records.

## 7) Safe cleanup guidance

Safe to remove (regenerable):
- `inference/__pycache__/`
- `inference/.DS_Store`
- Old CSVs in `inference/Output_selected_models/` if you only keep latest

Keep these:
- `db_loaders.py`, `extraction_engines.py`, `metrics.py`
- `run_target_model_folders.py` (main script for your model-folder sweeps)

Remove utility scripts only if you are certain they are unused in your workflow.

## 8) Typical workflow to evaluate new checkpoints

1. Put checkpoint files into a model folder.
2. Run `run_target_model_folders.py` with that folder and ADFECGDB path.
3. Inspect:
   - per-checkpoint CSV for best step
   - per-folder CSV for overall comparison
4. Promote/copy preferred CSV outputs into `outputs/inference_eval/` for reporting.

## 9) Notes on interpreting ARR-inclusive results

Including ARR files increases difficulty and usually lowers aggregate F1/Accuracy vs base-only records.
This is expected because ARR segments are more challenging/noisy and include arrhythmia variants.

Always compare runs with the same record-selection mode (base-only vs all-EDF) to avoid misleading conclusions.
