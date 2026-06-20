import glob
import os
import torch
from torch.autograd import Variable


def find_latest_checkpoint(model_dir, marker='G_AECG2FECG'):
    """Return the highest training step with saved checkpoints in model_dir."""
    pattern = os.path.join(model_dir, f'*_{marker}.pth')
    steps = []
    for path in glob.glob(pattern):
        try:
            steps.append(int(os.path.basename(path).split('_')[0]))
        except ValueError:
            continue
    return max(steps) if steps else None


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