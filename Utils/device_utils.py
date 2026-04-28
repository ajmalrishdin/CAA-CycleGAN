import os


VALID_DEVICE_BACKENDS = {"cpu", "mps", "cuda"}


def configure_runtime(device_backend="mps", cuda_devices=None):
    backend = (device_backend or "mps").lower()
    if backend not in VALID_DEVICE_BACKENDS:
        raise ValueError(f"Unsupported device backend: {device_backend}")

    if backend == "cuda" and cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    return backend


def resolve_device(device_backend="mps"):
    backend = (device_backend or "mps").lower()

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