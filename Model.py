import torch
import torch.onnx
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import onnx
import onnxruntime as ort

import time
import pickle
import cv2
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
from PIL import Image
import csv

# Initialize the params
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument('--device', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection')
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of model")
    parser.add_argument("--build", action='store_true', default=False, help="Build data from scratch")
    parser.add_argument("--train", action='store_true', default=False, help="Flag to indicate training mode")

    args = parser.parse_args()
    return args

args = get_args()

data_path = r"C:\Users\Test01\Desktop\CS3-Midterm\Scratch\animals"

# Format Data
class buildData():
    def __init__(self, path):
        self.IMG_SIZE = 128
        self.data_path = path
        self.LABELS = {}
        self.traningData = []
        self.animalCount = {}

    def process_folders(self):
        for idx, folder in enumerate(tqdm(os.listdir(self.data_path), desc="Processing folders", delay=0.1)):
            self.LABELS[folder] = idx
            self.animalCount[folder] = 0

    def trainBuild(self):
        for label in tqdm(self.LABELS, desc="Building Data"):
            for f in os.listdir(self.data_path + "\\" + label):
                try:
                    path = os.path.join(self.data_path + "\\" + label, f)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.traningData.append([np.array(img), np.eye(len(self.LABELS))[self.LABELS[label]]])  # One hot encode

                    if label in self.animalCount:
                        self.animalCount[label] += 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.traningData)
        file_path = r"utils\training_data.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.traningData, f)

if args.build:
    Builder = buildData(data_path)
    Builder.process_folders()
    Builder.trainBuild()

def loadData():
    try:
        with open(r"utils\training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
            return training_data
    except Exception as e:
        print("Please build the data with \"model.py --build\"")

# Create Model
class LoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # Compute final size after convolution and pooling
        self.final_size = self.compute_output_size(128)
        self.fc1 = nn.Linear(256 * (self.final_size ** 2), 256)
        self.fc2 = nn.Linear(256, 90)

    @staticmethod
    def compute_output_size(input_size, kernel_size=5, stride=1, pooling=2):
        output = input_size
        for _ in range(3):  # Three conv layers
            output = (output - (kernel_size - 1) - 1) // stride + 1  # Convolution
            output = (output - (pooling - 1) - 1) // pooling + 1    # Pooling
        return output
    
    def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
def ExportOnnx(model):
    dummy_input = torch.randn(100, 3, 128, 128) 
    onnx_file_path = r"utils\model.onnx"

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        input_names=['input'], 
        output_names=['output'], 
        opset_version=12)
    
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_np = np.array(img) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.tensor(img_np).unsqueeze(0).float()
    return img_tensor

def test_onnx_model(ort_session, image_tensor):
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    outputs = ort_session.run(None, inputs)
    return outputs

if args.img is not None:
    onnx_model_path = r"utils\model.onnx"
    test_image_path = args.img

    ort_session = ort.InferenceSession(onnx_model_path)
    image_tensor = preprocess_image(test_image_path)
    output = test_onnx_model(ort_session, image_tensor)
    print("Model Output:", output)

def save_model(model, epoch):
    model_save_dir = "utils"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    onnx_file_path = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}_{timestamp}.onnx")
    ExportOnnx(model, onnx_file_path)
    print(f"Model saved at {onnx_file_path}")
    manage_saved_models()

def manage_saved_models():
    model_save_dir = "utils" 
    max_models = 3
    all_files = [f for f in os.listdir(model_save_dir) if f.endswith(".onnx")]
    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_save_dir, f)))    
    if len(all_files) > max_models:
        files_to_delete = all_files[:-max_models] 
        for file in files_to_delete:
            file_path = os.path.join(model_save_dir, file)
            os.remove(file_path)
            print(f"Deleted old model: {file_path}")

def train():
    training_data = loadData()
    model = LoadModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    BATCH_SIZE = 100

    try:
        EPOCHS = args.num_epochs
    except Exception:
        print("Please pass a valid num of epochs.")

    X = torch.tensor(np.array([i[0] for i in training_data])).view(-1, 3, 128, 128).float() / 255.0
    y = torch.tensor([np.argmax(i[1]) for i in training_data]).long()

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 3 == 0:
            save_model(model, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")


    model.eval()
    ExportOnnx(model)
    onnx.checker.check_model(r"utils\model.onnx")

if args.train:
    train()
