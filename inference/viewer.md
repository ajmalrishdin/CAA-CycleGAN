# viewer.py

`inference/viewer.py` is a standalone Matplotlib viewer for saved fetal ECG extraction sessions. It does not perform extraction itself; it loads a previously saved session and lets you inspect the abdominal ECG and extracted fetal ECG side by side.

## What it does

- Opens a saved `.npz` session file.
- Displays the input AECG and extracted FECG in two stacked plots.
- Lets you pan through the recording with buttons or keyboard shortcuts.
- Saves the current viewport as a PNG snapshot.

## Session format

The viewer expects a compressed NumPy archive created by `save_view_session(...)`.

It also accepts older window-based archives produced by earlier extraction scripts.

Stored fields:

- `aecg_signal`: abdominal ECG samples as `float32`
- `fecg_signal`: extracted fetal ECG samples as `float32`
- `channel_label`: label for the selected channel
- `fs`: sampling rate
- `duration`: total duration in seconds
- `save_dir`: output folder used for PNG exports

Legacy window-based archives may also contain:

- `aecg_windows`: non-overlapping AECG windows
- `fecg_ext_windows`: extracted FECG windows
- `fecg_gt_windows`: optional ground-truth windows
- `record_name`, `channel_idx`, `start_sec`, `duration_sec`, `source_record`

The helper `default_session_path(channel_label, save_dir)` creates a timestamped filename such as:

`<channel_label>_session_YYYYMMDD-HHMMSS.npz`

## CLI usage

```bash
python inference/viewer.py --session path/to/session.npz
```

Only one argument is required:

- `--session`: path to a saved `.npz` session

## Viewer controls

- Left button or `Left Arrow`: move back by 0.5 seconds
- Right button or `Right Arrow`: move forward by 0.5 seconds
- `Ctrl+Left Arrow`: jump back by 4 seconds
- `Ctrl+Right Arrow`: jump forward by 4 seconds
- `S` key or `Save PNG` button: save the current 4-second window as a PNG

The visible window is fixed at 4 seconds. When the requested range goes beyond the end of the recording, the viewer clamps it to the valid duration.

## Display behavior

- The AECG trace is plotted in navy.
- The FECG trace is plotted in dark red.
- By default, each plot uses its own y-axis limits for clearer shape inspection.
- Pass `--shared-y-axis` if you want both plots to use the same y-axis limits for direct amplitude comparison.
- The window title shows the selected channel and current time range.
- The status line shows the current time span or the most recent saved filename.

## Functions

- `sanitize_filename_part(value)`
  - Replaces unsafe filename characters with `_`.
- `default_session_path(channel_label, save_dir)`
  - Builds a unique session path inside `save_dir`.
- `save_view_session(...)`
  - Writes the session archive to disk.
- `load_view_session(session_path)`
  - Loads a saved session and returns a dictionary with arrays and metadata.
- `show_interactive_viewer(...)`
  - Launches the interactive two-panel viewer.

## Typical workflow

1. Run your extraction pipeline.
2. Save the resulting signals with `save_view_session(...)`.
3. Open the saved archive with `python inference/viewer.py --session ...`.
4. Inspect the AECG/FECG overlay and export PNG snapshots if needed.

## Notes

- The script uses Matplotlib’s GUI backend, so it should be launched in an environment that supports interactive windows.
- The viewer assumes `aecg_signal` and `fecg_signal` have the same length.