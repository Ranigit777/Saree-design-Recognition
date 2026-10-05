"""Shared utilities."""

from __future__ import annotations



import os

import random



import numpy as np

import torch





def set_seed(seed: int = 42, deterministic: bool = True) -> None:

    """

    Set seeds for reproducibility.



    Full GPU determinism may reduce performance; cudnn benchmark is disabled when

    deterministic=True.

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False





def get_env_info(device: torch.device) -> dict:

    import sys



    import torchvision



    info = {

        "python_version": sys.version.split()[0],

        "pytorch_version": torch.__version__,

        "torchvision_version": torchvision.__version__,

        "cuda_available": torch.cuda.is_available(),

        "device": str(device),

    }

    if torch.cuda.is_available():

        info["gpu_name"] = torch.cuda.get_device_name(0)

    return info





def print_env_info(device: torch.device) -> dict:

    info = get_env_info(device)

    print(f"Device: {info['device']}")

    if info["cuda_available"]:

        print(f"GPU: {info['gpu_name']}")

    print(f"PyTorch: {info['pytorch_version']}")

    print(f"torchvision: {info['torchvision_version']}")

    return info


