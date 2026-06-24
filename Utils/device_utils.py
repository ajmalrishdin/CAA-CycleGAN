import os
from pathlib import Path


VALID_DEVICE_BACKENDS = {"cpu", "mps", "cuda"}
ENV_DEVICE_BACKEND = "DEVICE_BACKEND"
ENV_CUDA_DEVICES = "CUDA_DEVICES"
DEFAULT_DEVICE_BACKEND = "mps"


def _parse_env_file(env_path):
    """Load KEY=VALUE pairs from .env when python-dotenv is unavailable."""
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _load_env():
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"

    try:
        from dotenv import load_dotenv
    except ImportError:
        _parse_env_file(env_path)
        return

    load_dotenv(env_path)


_load_env()


def get_device_backend():
    backend = os.environ.get(ENV_DEVICE_BACKEND, DEFAULT_DEVICE_BACKEND).lower()
    if backend not in VALID_DEVICE_BACKENDS:
        raise ValueError(
            f"Unsupported {ENV_DEVICE_BACKEND}={backend!r}. "
            f"Choose one of: {', '.join(sorted(VALID_DEVICE_BACKENDS))}"
        )
    return backend


def get_cuda_devices():
    value = os.environ.get(ENV_CUDA_DEVICES)
    return value if value else None


def configure_runtime(device_backend=None, cuda_devices=None):
    backend = (device_backend or get_device_backend()).lower()
    if backend not in VALID_DEVICE_BACKENDS:
        raise ValueError(f"Unsupported device backend: {device_backend}")

    devices = cuda_devices if cuda_devices is not None else get_cuda_devices()
    if backend == "cuda" and devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)

    return backend


def resolve_device(device_backend=None):
    backend = configure_runtime(device_backend)
    import torch

    if backend == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA backend requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    if backend == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS backend requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")
