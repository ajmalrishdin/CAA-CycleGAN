import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.device_utils import configure_runtime, resolve_device
from inference_core import convert_aecg_to_mecg_fecg, save_results


def main():
    parser = argparse.ArgumentParser(description='Extract MECG and FECG from AECG signal')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input AECG signal (.npy file) or "demo" for demo signal')
    parser.add_argument('--model_dir', type=str, default='models/sagan_1 V1',
                        help='Directory containing trained models')
    parser.add_argument('--step', type=int, default=None,
                        help='Model checkpoint step number (default: uses latest)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--output_prefix', type=str, default='result',
                        help='Output filename prefix')
    parser.add_argument('--device_backend', type=str, default='mps', choices=['mps', 'cuda', 'cpu'],
                        help='Execution backend to use')
    parser.add_argument('--cuda_devices', type=str, default=None,
                        help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1"')

    args = parser.parse_args()

    configure_runtime(args.device_backend, args.cuda_devices)
    device = resolve_device(args.device_backend)

    if args.input.lower() == 'demo':
        print('Creating demo AECG signal...')
        t = np.linspace(0, 1, 1000)
        aecg_signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    else:
        print(f'Loading AECG signal from {args.input}...')
        if args.input.endswith('.npy'):
            aecg_signal = np.load(args.input)
        else:
            raise ValueError('Input file must be a .npy file. Got: {}'.format(args.input))

    print(f'Input signal shape: {aecg_signal.shape}')

    print('\n' + '=' * 50)
    print('Running inference...')
    print('=' * 50)

    try:
        mecg, fecg = convert_aecg_to_mecg_fecg(
            aecg_signal,
            model_dir=args.model_dir,
            step=args.step,
            device=device,
            device_backend=args.device_backend,
        )
    except Exception as e:
        print(f'Inference failed: {e}')
        return

    if mecg is None or fecg is None:
        print('Failed to generate outputs.')
        return

    print('\n' + '=' * 50)
    print('Saving results...')
    print('=' * 50)

    save_results(aecg_signal, mecg, fecg, args.output_dir, args.output_prefix)

    print('\n' + '=' * 50)
    print('✓ Conversion complete!')
    print('=' * 50)
    print(f'Input AECG shape: {aecg_signal.shape}')
    print(f'Output MECG shape: {mecg.shape}')
    print(f'Output FECG shape: {fecg.shape}')
    print(f'\nResults saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()