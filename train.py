import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processing import BuildDataset
from model import load_resnet_model, save_model
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(data_path, labels, num_epochs=100, lr=0.01, batch_size=32):
    transform_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BuildDataset(data_path, labels, transform=transform_augment)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_resnet_model(len(labels))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 3 == 0:
            save_model(model, epoch)

    save_model(model, num_epochs - 1)