import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms
import argparse


import os
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import Counter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Starting learning rate of the model")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training mode")
    parser.add_argument("--data_path", type=str, default="training_data", help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="utils/model.pth", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"Data path {args.data_path} does not exist.")
        exit(1)

    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),  # Additional augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load dataset
    full_dataset = datasets.ImageFolder(args.data_path)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Assign transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Calculate class weights
    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Print class distribution
    class_distribution = Counter(train_labels)
    print("Training class distribution:")
    for idx, count in class_distribution.items():
        print(f"Class {class_names[idx]}: {count} images")

    # Create WeightedRandomSampler
    samples_weights = np.array([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

    # Data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Dataset sizes: {dataset_sizes}")

    # Model
    model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights in loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args.train:
        best_model_wts = model.state_dict()
        best_acc = 0.0
        patience = 5
        counter = 0
        train_losses, val_losses = [], []

        total_start_time = time.time()  # Record total start time

        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()  # Record epoch start time

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                y_true, y_pred = [], []

                for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{args.num_epochs}'):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping")
                            model.load_state_dict(best_model_wts)
                            break

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

            if counter >= patience:
                break  # Breaks out of the 'epoch' loop due to early stopping

            epoch_duration = time.time() - epoch_start_time
            total_time_elapsed = time.time() - total_start_time
            avg_epoch_time = total_time_elapsed / (epoch + 1)
            estimated_total_time = avg_epoch_time * args.num_epochs
            estimated_time_remaining = estimated_total_time - total_time_elapsed

            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")
            print(f"Estimated total training time: {estimated_total_time/60:.2f} minutes.")
            print(f"Estimated time remaining: {estimated_time_remaining/60:.2f} minutes.\n")

        total_training_time = time.time() - total_start_time
        print(f"Total training time: {total_training_time/60:.2f} minutes.")

        model.load_state_dict(best_model_wts)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

        # Plot loss curves
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.show()

    if args.img:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        img = Image.open(args.img).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]
            print(f"Predicted class: {predicted_class}")
        
        #show image in a seperate tap
        z = plt.imread(args.img)
        plt.imshow(z)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
