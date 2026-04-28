
This is a fork of "Correlation-Aware Attention CycleGAN for Accurate Fetal ECG Extraction” 

Added Features:

minor tweaks to minmaxscaler to work resolve dependency version issues
Inference scripts to use the output models to extract fECG
new optional flag to change between Nvidia & Apple Silicon

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

### Training
#### To train with default parameters:
```
python main.py
```
#### Optional Flags:

--total_step [no. of step the model is trained for] (default:1000)

--num_workers [no. of CPU cores utilized] (default:2)

--device_backend [mps, cuda, or cpu] (default:mps)

--cuda_devices [CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1] (default:none)

--batch_size [No. ECG windows being trained at the same time] (default:32)

Note: increasing batch size will increase memory usage, decrease time for training and change output accuracy

Check Parameters.py for more optional flags

#### Training Command with Flags:

```
python main.py --batch_size 128 --num_workers 8
```

