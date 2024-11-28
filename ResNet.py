import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort

from pathlib import Path
import time
import pickle
import cv2
from tqdm import tqdm
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

# Initialize the params
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument('--device', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection (Work in progress)')
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Starting learning rate of model")
    parser.add_argument("--build", action='store_true', default=False, help="Build data from scratch")
    parser.add_argument("--train", action='store_true', default=False, help="Flag to indicate training mode")

    args = parser.parse_args()
    return args

args = get_args()

data_path = r"C:\Users\Test01\Desktop\CS3-Midterm\Scratch\animals"
def createLabels(data_path):
    LABELS = {}
    for idx, folder in enumerate(tqdm(os.listdir(data_path), desc="Processing folders", delay=0.1)):
            LABELS[folder] = idx
    return LABELS

LABELS = createLabels(data_path)
# Format Data
class buildData():
    def __init__(self, path):
        self.IMG_SIZE = 224
        self.data_path = path
        self.traningData = []
        self.animalCount = {}

    def process_folders(self):
        for _, folder in enumerate(tqdm(os.listdir(self.data_path), desc="Processing folders", delay=0.1)):
            self.animalCount[folder] = 0

    def trainBuild(self):
        for label in tqdm(LABELS, desc="Building Data"):
            for f in os.listdir(self.data_path + "\\" + label):
                try:
                    path = os.path.join(self.data_path + "\\" + label, f)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img_tensor = transforms.ToTensor()(img)  # Convert to Tensor
                    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)  # Normalize image
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
                    self.traningData.append([np.array(img_tensor), np.eye(len(LABELS))[LABELS[label]]])  # One hot encode

                    if label in self.animalCount:
                        self.animalCount[label] += 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.traningData)
        file_path = r"utils\training_data.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.traningData, f)

def loadData():
    try:
        with open(r"utils\training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
            return training_data
    except Exception as e:
        print("Please build the data with \"model.py --build\"")

def load_resnet_model():
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 128), 
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=True), 
        torch.nn.Linear(128, 90)) # 90 output classes
    return resnet

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.tensor(img_np).unsqueeze(0).float()
    img_tensor = img_tensor.repeat(100, 1, 1, 1)
    return img_tensor

# Function to test the ONNX model
def test_onnx_model(ort_session, image_tensor):
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    outputs = ort_session.run(None, inputs)
    return outputs

def load_onnx_model(onnx_model_path):
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers, provider_options=provider_options)
    return ort_session

def ExportOnnx(model, onnx_file_path):
    dummy_input = torch.randn(100, 3, 224, 224)  # Example dummy input
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        input_names=['input'], 
        output_names=['output'], 
        opset_version=12
    )
    print(f"Model exported to {onnx_file_path}")

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
class LossBasedScheduler:
    def __init__(self, optimizer, initial_lr, patience=3, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
    
    def step(self, current_loss):
        if current_loss < self.best_loss - 1e-6:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            self.num_bad_epochs = 0
            
def train():
    training_data = loadData()
    model = load_resnet_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = LossBasedScheduler(optimizer, initial_lr=0.01, patience=5, factor=0.5)

    BATCH_SIZE = 32

    try:
        EPOCHS = args.num_epochs
    except Exception:
        print("Please pass a valid num of epochs.")

    X = torch.tensor(np.array([i[0] for i in training_data])).view(-1, 3, 224, 224).float() / 255.0
    y = torch.tensor([np.argmax(i[1]) for i in training_data]).long()

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train() 
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step(loss)

        if (epoch + 1) % 3 == 0:
            save_model(model, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        for param in model.parameters():
            if param.grad is not None:
                print(param.grad.abs().mean())

    model.eval()
    ExportOnnx(model)
    onnx.checker.check_model(r"utils\model.onnx")


providers = ['CPUExecutionProvider']
provider_options = [{}]

if args.build:
    Builder = buildData(data_path)
    Builder.process_folders()
    Builder.trainBuild()

if args.train:
    model = load_resnet_model()
    train()

if args.img is not None:
        onnx_model_path = r"utils\model.onnx"
        test_image_path = args.img

        ort_session = ort.InferenceSession(onnx_model_path)
        image_tensor = preprocess_image(test_image_path)
        output = test_onnx_model(ort_session, image_tensor)

        predictions = output[0][0]
        predicted_index = predictions.argmax()
        predicted_label = list(LABELS.keys())[list(LABELS.values()).index(predicted_index)]

        print(f"Predicted Animal: {predicted_label}")