import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_resnet18():
    # 1. Hardware acceleration setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths & Hyperparameters
    ROOT_DIR = Path.cwd()
    DATA_DIR = ROOT_DIR / "data" / "raw"
    SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning pre-trained weights
    EPOCHS = 10

    # 3. Data Transformations (ResNet expects ImageNet normalization statistics)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load CIFAKE Dataset
    train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=transform_train)
    test_dataset = datasets.ImageFolder(root=DATA_DIR / "test", transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")

    # 5. Initialize Pre-trained ResNet18 & Modify Classification Head
    print("Loading pre-trained ResNet18 backbone...")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace the final fully connected layer for binary classification (REAL vs FAKE)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # 6. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    # 7. Training & Evaluation Loop
    print("\n--- Starting ResNet18 Fine-Tuning ---")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = (correct_train / total_train) * 100.0

        # Evaluation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(test_dataset)
        val_acc = (correct_val / total_val) * 100.0
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{EPOCHS}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = SAVED_MODELS_DIR / "resnet18_cifake.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best ResNet18 model checkpoint with {val_acc:.2f}% accuracy to {save_path}")

    print(f"\n✅ Training complete! Best Test Accuracy achieved: {best_acc:.2f}%")

if __name__ == "__main__":
    train_resnet18()
