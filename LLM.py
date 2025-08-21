import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
from sklearn.metrics import classification_report
import time
import pickle
from tqdm import tqdm
import os
import argparse
import numpy as np
from PIL import Image
import pyplot


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument()

    args = parser.parse_args()
    return args


args = get_args()


def install_requirements():
    if args.install_req:
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(
            ["%s==%s" % (i.key, i.version) for i in installed_packages])
        for package in requirements:
            if package not in installed_packages_list:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
                print(f"Installed {package}")
        print("All required packages are installed.")


def select_device(device_choice: str):
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    # elif device_choice == "RocM":
    #     if
    else:
        # device_choice == "cpu"
        return torch.device("cpu")


device = select_device(args.device)
