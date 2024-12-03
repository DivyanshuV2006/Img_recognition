import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
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
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Starting learning rate of model")
    parser.add_argument("--build", action='store_true', help="Build data from scratch")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training mode")

    args = parser.parse_args()
    return args

args = get_args()

data_path = r"C:\Users\Test01\Desktop\CS3-Midterm\Scratch\animals"
file_path = r"utils\training_data.pkl"

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
        transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        for label in tqdm(LABELS, desc="Building Data"):
            for f in os.listdir(self.data_path + "\\" + label):
                try:
                    path = os.path.join(self.data_path + "\\" + label, f)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    # Convert the image to a PIL Image for augmentation
                    img_pil = Image.fromarray(img)
                    img_tensor = transform_augment(img_pil)  # Apply augmentation and transform to tensor

                    self.traningData.append([img_tensor.numpy(), np.eye(len(LABELS))[LABELS[label]]])  # One hot encode

                    if label in self.animalCount:
                        self.animalCount[label] += 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.traningData)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.traningData, f)

def loadData():
    try:
        with open(file_path, "rb") as f:
            training_data = pickle.load(f)
            return training_data
    except Exception as e:
        print("Please build the data with \"model.py --build\"")

def load_resnet_model():
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)

    # Adding batch normalization and other layers
    resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.bn1 = torch.nn.BatchNorm2d(64)  # Adding batch normalization after conv1
    resnet.relu = torch.nn.ReLU(inplace=True)
    resnet.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Adding Batch Normalization after the fully connected layers
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 128),
        torch.nn.BatchNorm1d(128),  # Batch Normalization after the fully connected layer
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
        torch.nn.Dropout(0.5),  # Existing dropout layer
        torch.nn.Linear(128, len(LABELS))
    )
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
    #print(f"Model exported to {onnx_file_path}")

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
        print(f"Best Loss: {self.best_loss}, Number of bad epochs: {self.num_bad_epochs}")
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
    #Rebalence the Loss fn
    class_counts = [len(os.listdir(os.path.join(data_path, label))) for label in LABELS.keys()]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(LABELS) * count) for count in class_counts]
    class_weights = torch.tensor(class_weights).float()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = LossBasedScheduler(optimizer, initial_lr=args.lr, patience=3, factor=0.5)

    BATCH_SIZE = 32

    try:
        EPOCHS = args.num_epochs
    except Exception:
        print("Please pass a valid number of epochs.")

    # Prepare the dataset
    X = torch.tensor(np.array([i[0] for i in training_data])).view(-1, 3, 224, 224).float() / 255.0
    y = torch.tensor([np.argmax(i[1]) for i in training_data]).long()
    dataset = TensorDataset(X, y)

    # Split the dataset into training and validation sets
    train_size = int(0.7 * len(dataset))  # 80% training
    val_size = len(dataset) - train_size  # 20% validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_pred = []
    for epoch in range(EPOCHS):
        # Training loop
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"):
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        scheduler.step(val_loss)

        if (epoch + 1) % 3 == 0:
            save_model(model, epoch)

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(classification_report(y_true, y_pred, target_names=LABELS))
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n"
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

    model.eval()
    ExportOnnx(model, r"utils\model.onnx")
    onnx.checker.check_model(r"utils\model.onnx")

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
        confidence = predictions[predicted_index]
        print(f"Predicted Animal: {predicted_label}, (Confidence: {confidence:.2f})")
