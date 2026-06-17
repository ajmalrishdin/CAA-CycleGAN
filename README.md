
This is a fork of "Correlation-Aware Attention CycleGAN for Accurate Fetal ECG Extraction” 

Added Features:

- minor tweaks to minmaxscaler to work resolve dependency version issues
- Inference scripts to use the output models to extract fECG
- new optional flag to change between Nvidia & Apple Silicon

If you need any help for the code and data, do not hesitate to leave issues in this repository.


## Usage

### Requirements

```
Python < 3.11
```

### Installing Dependencies
using venv is recommended 
```
pip install -r requirements.txt
```

### Device configuration

Copy the example env file and set your local accelerator:

```
cp .env.example .env
```

Edit `.env` to choose the compute backend:

| Variable | Values | Description |
|----------|--------|-------------|
| `DEVICE_BACKEND` | `mps`, `cuda`, `cpu` | Apple Silicon GPU, NVIDIA GPU, or CPU |
| `CUDA_DEVICES` | e.g. `0` or `0,1` | Optional. Sets `CUDA_VISIBLE_DEVICES` when using CUDA |

Example for Apple Silicon (default):

```
DEVICE_BACKEND=mps
CUDA_DEVICES=
```

Example for NVIDIA GPU:

```
DEVICE_BACKEND=cuda
CUDA_DEVICES=0
```

`.env` is gitignored. Commit `.env.example` as the template for other machines.

CLI flags (`--device_backend`, `--cuda_devices`, `--device`) still override `.env` when provided.

### Training
#### To train with default parameters:
```
python main.py
```
#### Optional Flags:

`--total_step` [no. of step the model is trained for] (default:1000)
`—num_workers` [no. of CPU cores utilized] (default:2)
`--device_backend` [mps, cuda, or cpu] (default: `DEVICE_BACKEND` from `.env`)
`--cuda_devices` [CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1] (default: `CUDA_DEVICES` from `.env`)
`--batch_size` [No. ECG windows being trained at the same time] (default:32)

Note: increasing batch size will increase memory usage, decrease time for training and change output accuracy

Check Parameters.py for more optional flags

#### Training Command with Flags:

```
python main.py --batch_size 128 --num_workers 8
```

