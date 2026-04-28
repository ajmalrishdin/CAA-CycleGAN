"""Unified entrypoint for inference and evaluation workflows.

This file stays at the repository root so it can be used as the single command
you run for inference. The implementation for each workflow remains inside
inference/ as modular scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from Utils.device_utils import resolve_device

class InferenceEngine:
    def __init__(self, model_path, batch_size=32, device=None, device_backend="mps"):
        """
        Initialize the Inference Engine with a specific model checkpoint.
        """
        self.device = device if device else resolve_device(device_backend)
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

WORKFLOWS = {
    "signal": "inference/run_cyclegan.py",
    "target-folders": "inference/run_target_model_folders.py",
    "compare": "inference/run_comparison.py",
    "all-checkpoints": "inference/run_all_checkpoints.py",
    "nifeadb-step": "inference/run_nifeadb_step_metrics.py",
}

DEFAULT_WORKFLOW = "target-folders"


def convert_aecg_to_mecg_fecg(aecg_signal, model_dir='models/sagan_1', step=None, device=None, device_backend="mps"):
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
        mecg_engine = InferenceEngine(mecg_path, device=device, device_backend=device_backend)
        print("Loading FECG Engine...")
        fecg_engine = InferenceEngine(fecg_path, device=device, device_backend=device_backend)
    except Exception as e:
        print(e)
        return None, None


def _resolve_workflow(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return DEFAULT_WORKFLOW, []

    first = argv[0]
    if first in {"-h", "--help", "help"}:
        _print_help()
        raise SystemExit(0)

    if first in WORKFLOWS:
        return first, argv[1:]

    if first.startswith("-"):
        return DEFAULT_WORKFLOW, argv

    raise SystemExit(f"Unknown workflow: {first}\n\nRun `python inference.py --help` for available workflows.")


def _run_workflow(script_relative_path: str, workflow_args: list[str]) -> int:
    script_path = PROJECT_ROOT / script_relative_path
    if not script_path.is_file():
        raise FileNotFoundError(f"Workflow script not found: {script_path}")

    command = [sys.executable, str(script_path), *workflow_args]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        workflow = DEFAULT_WORKFLOW
        workflow_args: list[str] = []
    else:
        workflow, workflow_args = _resolve_workflow(args)

    return _run_workflow(WORKFLOWS[workflow], workflow_args)


if __name__ == "__main__":
    raise SystemExit(main())
