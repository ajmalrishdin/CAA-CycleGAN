# viewer.py

`inference/viewer.py` is the unified launcher for the interactive viewers in this folder. It does not perform extraction or parsing itself; it routes the input file to the right existing script based on the file extension.

## What it does

- Opens `.npz` session archives with `inference/npz.py`.
- Opens `.edf` and `.dat` recordings with `inference/edfndat.py`.
- Preserves the existing viewer behavior instead of reimplementing it.
- Keeps the command surface in one place so users can remember a single entrypoint.

## CLI usage

```bash
python inference/viewer.py path/to/session.npz
python inference/viewer.py path/to/recording.edf
python inference/viewer.py path/to/recording.dat
```

The launcher inspects the file extension and dispatches automatically:

- `.npz` -> `inference/npz.py --session ...`
- `.edf` -> `inference/edfndat.py ...`
- `.dat` -> `inference/edfndat.py ...`

## NPZ sessions

The `.npz` path is handled by `inference/npz.py`, which opens a saved fetal ECG session and shows the AECG and extracted FECG side by side.

Supported session formats include:

- Current sessions with `aecg_signal` and `fecg_signal`
- Older window-based archives with `aecg_windows` and `fecg_ext_windows`

Useful flag:

- `--shared-y-axis`: force both NPZ plots to use the same y-axis limits

## EDF and DAT recordings

The `.edf` and `.dat` paths are handled by `inference/edfndat.py`, which shows up to 5 channels in a stacked interactive plot.

Useful flags forwarded by the launcher:

- `--window-size`: visible window width in seconds, default `4.0`
- `--step-small`: arrow-key step size in seconds, default `0.5`
- `--max-channels`: maximum number of channels to display, default `5`
- `--raw`: treat `.dat` as raw binary instead of WFDB
- `--auto-fallback`: fall back to raw `.dat` loading if WFDB parsing fails
- `--raw-int16`: read raw `.dat` as `int16`
- `--n-channels`: channel count for raw `.dat` fallback, default `4`
- `--fs`: sampling rate for raw `.dat` fallback, default `1000.0`
- `--gain`: scaling factor for raw `int16` `.dat` values, default `1000.0`
- `--save-dir`: directory for PNG snapshots, default `outputs/matplotlib`

## Viewer controls

The underlying viewers keep their original controls:

- Left button or `Left Arrow`: move back by 0.5 seconds
- Right button or `Right Arrow`: move forward by 0.5 seconds
- `Ctrl+Left Arrow`: jump back by one full window
- `Ctrl+Right Arrow`: jump forward by one full window
- `S` key or `Save PNG` button: save the current window as a PNG

## Notes

- The launcher resolves the input file first and then runs the underlying viewer in the project root so relative paths keep working.
- Launch it from an environment that supports Matplotlib GUI windows.
- For `.dat` files, the existing viewer still decides whether to read WFDB or raw binary based on the flags and presence of a `.hea` file.