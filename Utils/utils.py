import glob
import os
import random

import numpy as np
import torch
from torch.autograd import Variable


CHECKPOINT_MARKER = 'G_AECG2FECG'

# CHECKPOINT_MARKER is listed last because find_latest_checkpoint keys on it: a
# step must not become discoverable before the rest of its files are on disk.
GENERATOR_SUFFIXES = (
    'G_AECG2MECG',
    'G_MECG2AECG',
    'G_FECG2AECG',
    'G_AECG2BIAS',
    'G_BIAS2AECG',
    CHECKPOINT_MARKER,
)

DISCRIMINATOR_SUFFIXES = (
    'D_AECG2MECG',
    'D_MECG2AECG',
    'D_AECG2FECG',
    'D_FECG2AECG',
    'D_AECG2BIAS',
    'D_BIAS2AECG',
)

ALL_MODULE_SUFFIXES = GENERATOR_SUFFIXES + DISCRIMINATOR_SUFFIXES

RESUME_STATE_FILENAME = 'resume.pth'


def previous_generation_path(path):
    root, ext = os.path.splitext(path)
    return f'{root}_prev{ext}'


def atomic_torch_save(obj, path, keep_previous=False):
    """torch.save that a killed process cannot leave half-written.

    Bytes go to a temp file that is fsynced and then renamed over the target.
    os.replace is atomic, so a reader sees either the previous complete file or
    the new complete one, never a truncated mix of the two.

    With keep_previous, the current file is hard-linked aside first, so the
    generation before last survives too. The link costs no extra bytes and,
    unlike renaming the old file away, never leaves `path` missing.
    """
    tmp_path = f'{path}.tmp'
    try:
        with open(tmp_path, 'wb') as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    if keep_previous and os.path.isfile(path):
        prev_path = previous_generation_path(path)
        try:
            if os.path.exists(prev_path):
                os.remove(prev_path)
            os.link(path, prev_path)
        except OSError:
            pass

    os.replace(tmp_path, path)


def torch_load(path, map_location=None):
    """torch.load that works either side of the weights_only default change."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_module(module):
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def checkpoint_step_is_complete(model_dir, step, suffixes):
    return all(
        os.path.isfile(os.path.join(model_dir, f'{step}_{suffix}.pth'))
        for suffix in suffixes
    )


def find_latest_checkpoint(model_dir, marker=CHECKPOINT_MARKER, required_suffixes=None):
    """Return the highest training step with saved checkpoints in model_dir.

    When required_suffixes is given, a step only counts if every one of those
    files exists, so an incomplete step is skipped rather than picked up.
    """
    pattern = os.path.join(model_dir, f'*_{marker}.pth')
    steps = []
    for path in glob.glob(pattern):
        try:
            step = int(os.path.basename(path).split('_')[0])
        except ValueError:
            continue
        if required_suffixes and not checkpoint_step_is_complete(model_dir, step, required_suffixes):
            continue
        steps.append(step)
    return max(steps) if steps else None


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore RNG state; never fatal, since a resumed run stays valid without it."""
    if not state:
        return
    try:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu().to(torch.uint8))
        cuda_state = state.get('cuda')
        if cuda_state and torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        print(f'Could not restore RNG state ({exc}); continuing with fresh randomness')


def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm1(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def denorm(v):
    v_min = v.min(axis=2).reshape((v.shape[0],1,1))
    v_max = v.max(axis=2).reshape((v.shape[0],1,1))
    return (v - v_min) / (v_max-v_min)