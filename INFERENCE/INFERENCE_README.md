# Inference Guide: Converting AECG to MECG and FECG

This guide explains how to use the trained models to extract MECG (Maternal ECG) and FECG (Fetal ECG) from AECG (Abdominal ECG) signals.

## Quick Start

### Option 1: Using the simple script (Recommended)

```bash
# With your own AECG signal file (.npy format)
python run_inference.py --input your_aecg_signal.npy

# With a demo signal (for testing)
python run_inference.py --input demo

# Specify a specific model checkpoint
python run_inference.py --input your_aecg_signal.npy --step 633

# Custom output directory
python run_inference.py --input your_aecg_signal.npy --output_dir my_results
```

### Option 2: Using the inference module programmatically

```python
import numpy as np
from inference import convert_aecg_to_mecg_fecg, save_results, preprocess_aecg

# Load your AECG signal
aecg_signal = np.load('your_aecg_signal.npy')  # Shape: (length,) or (channels, length)

# Convert to MECG and FECG
mecg, fecg = convert_aecg_to_mecg_fecg(
    aecg_signal, 
    model_dir='models/sagan_1',
    step=633  # or None for latest
)

# Save results
# Note: save_results handles AECG trimming to match the reconstructed output length
save_results(aecg_signal, mecg, fecg, output_dir='outputs', filename_prefix='result')
```

## Input Format

Your AECG signal can be a numpy array of **any length**. The system uses a sliding window (128 samples) to process the entire signal and stitch it back together.

Supported shapes:
- `(length,)` - Single channel signal
- `(channels, length)` - Multi-channel signal (first channel will be used)
- `(batch, channels, length)` - Batch of signals

**Note**: If the signal length is not a multiple of 128, the engine will automatically pad the signal to ensure complete reconstruction.

## Output

The script generates:
1. **Numpy arrays** (`.npy` files):
   - `result_aecg.npy` - Preprocessed input AECG
   - `result_mecg.npy` - Extracted MECG signal
   - `result_fecg.npy` - Extracted FECG signal

2. **Visualization** (`.png` file):
   - `result_comparison.png` - Plot showing AECG, MECG, and FECG

## Model Checkpoints

The models are saved in `models/sagan_1/` with format:
- `{step}_G_AECG2MECG.pth` - Generator for AECG → MECG
- `{step}_G_AECG2FECG.pth` - Generator for AECG → FECG

If you don't specify `--step`, the script will automatically use the latest checkpoint.

## Example: Loading from your data loader

If you want to use signals from your existing data loader:

```python
from data_loader import Data_Item, FECGDataset
from inference import convert_aecg_to_mecg_fecg
import numpy as np

# Load your dataset
data_item = Data_Item()
dataset = FECGDataset(data_item, train=False)

# Get an AECG signal from the dataset
aecg_signal, _, _, _ = dataset[0]  # Get first sample
aecg_signal = aecg_signal.numpy()   # Convert to numpy

# Convert to MECG and FECG
mecg, fecg = convert_aecg_to_mecg_fecg(aecg_signal)
```

## Troubleshooting

1. **CUDA out of memory**: The script will automatically use CPU if CUDA is not available. If you have memory issues, process signals one at a time.

2. **Model not found**: Make sure the `--model_dir` path is correct and contains the model files.

3. **Shape errors**: Ensure your input signal can be reshaped to (batch, 1, 128). The preprocessing function handles most common formats automatically.

## Notes

- The models are set to evaluation mode (no gradient computation)
- Input signals are automatically normalized to [-1, 1] range
- Output signals maintain the same normalization as training data