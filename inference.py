import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sagan_models import Generator
import sys

class InferenceEngine:
    def __init__(self, model_path, batch_size=32, device=None):
        """
        Initialize the Inference Engine with a specific model checkpoint.
        """
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # specific parameters for the architecture
        self.imsize = 64
        self.z_dim = 128
        self.g_conv_dim = 64
        
        # Load Generator
        self.model = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
        self.model.eval()

    def process_signal(self, signal_array, window_size=128):
        """
        Process a 1D signal array: Window -> Normalize -> Infer -> Stitch
        """
        # Ensure flat numpy array
        if isinstance(signal_array, torch.Tensor):
            signal_array = signal_array.cpu().numpy()
        signal_array = np.squeeze(signal_array)
        
        signal_len = len(signal_array)
        
        if signal_len < window_size:
            # Pad with zeros if shorter than one window
            pad_len = window_size - signal_len
            signal_array = np.pad(signal_array, (0, pad_len))
            signal_len = window_size
            
        # 1. Windowing (Non-overlapping)
        windows = []
        for i in range(0, signal_len - window_size + 1, window_size):
            segment = signal_array[i:i+window_size]
            windows.append(segment)
            
        if not windows:
            return np.array([])

        # 2. Batch Processing
        outputs = []
        
        # Convert all windows to normalized tensors first could be memory heavy for huge signals, 
        # but fine for standard ECG recordings. Let's do it per batch to be safe.
        
        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i:i+self.batch_size]
            
            # Prepare batch tensor
            batch_tensor = []
            for w in batch_windows:
                # Robust Min-Max Normalization to [-1, 1]
                s_min, s_max = w.min(), w.max()
                if s_max - s_min < 1e-6:
                    w_norm = np.zeros_like(w)
                else:
                    w_norm = 2 * (w - s_min) / (s_max - s_min) - 1
                batch_tensor.append(w_norm)
            
            # Stack and convert to tensor (Batch, 1, WindowSize)
            batch_input = np.array(batch_tensor)
            batch_input = torch.FloatTensor(batch_input).unsqueeze(1).to(self.device)
            
            # Inference
            with torch.no_grad():
                batch_out = self.model(batch_input)
                
            outputs.append(batch_out.cpu().numpy())
            
        # 3. Stitching
        # Flatten batches and then flatten the window sequence
        if not outputs:
            return np.array([])
        reconstructed = np.concatenate([out.reshape(-1) for out in outputs])
        
        return reconstructed

def convert_aecg_to_mecg_fecg(aecg_signal, model_dir='models/sagan_1', step=None):
    """
    Wrapper function to load both MECG and FECG models and process a signal.
    """
    # Determine Checkpoint Step
    if step is None:
        try:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_G_AECG2MECG.pth')]
            steps = [int(f.split('_')[0]) for f in model_files]
            step = max(steps) if steps else None
        except FileNotFoundError:
            print(f"Model directory not found: {model_dir}")
            return None, None
            
        if step is None:
            raise ValueError(f"No model files found in {model_dir}")
        print(f"Using model checkpoint: {step}")
    
    mecg_path = os.path.join(model_dir, f'{step}_G_AECG2MECG.pth')
    fecg_path = os.path.join(model_dir, f'{step}_G_AECG2FECG.pth')
    
    # Initialize Engines
    try:
        print("Loading MECG Engine...")
        mecg_engine = InferenceEngine(mecg_path)
        print("Loading FECG Engine...")
        fecg_engine = InferenceEngine(fecg_path)
    except Exception as e:
        print(e)
        return None, None

    # Process
    print("Extracting MECG...")
    mecg_signal = mecg_engine.process_signal(aecg_signal)
    
    print("Extracting FECG...")
    fecg_signal = fecg_engine.process_signal(aecg_signal)
    
    return mecg_signal, fecg_signal

def save_results(aecg, mecg, fecg, output_dir='outputs', filename_prefix='result'):
    """Save the results as numpy arrays and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Trim to match lengths if necessary (inference might drop last non-full window)
    min_len = min(len(aecg), len(mecg), len(fecg))
    aecg = aecg[:min_len]
    mecg = mecg[:min_len]
    fecg = fecg[:min_len]
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f'{filename_prefix}_aecg.npy'), aecg)
    np.save(os.path.join(output_dir, f'{filename_prefix}_mecg.npy'), mecg)
    np.save(os.path.join(output_dir, f'{filename_prefix}_fecg.npy'), fecg)
    print(f"Saved numpy arrays to {output_dir}/")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot AECG
    axes[0].plot(aecg)
    axes[0].set_title('AECG (Abdominal ECG - Input)')
    axes[0].grid(True)
    
    # Plot MECG
    axes[1].plot(mecg)
    axes[1].set_title('MECG (Maternal ECG - Extracted)')
    axes[1].grid(True)
    
    # Plot FECG
    axes[2].plot(fecg)
    axes[2].set_title('FECG (Fetal ECG - Extracted)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{filename_prefix}_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()
