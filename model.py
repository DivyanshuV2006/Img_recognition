import torch
import torch.nn as nn
from torchvision import models
import time
import torch.onnx
import os

def load_resnet_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    return model

def save_model(model, epoch, model_save_dir="utils"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    onnx_file_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}_{timestamp}.onnx")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=["input"], output_names=["output"], opset_version=12)
    print(f"Model saved at {onnx_file_path}")
