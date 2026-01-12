import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter

def main():
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    

    # DATASET PATH (ONLY TRAIN FOLDER)
    DATASET_DIR = r"D:\Machine Learning\Codefest\emotion_dataset\train"
    

    # TRANSFORMS
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    

    # LOAD DATASET
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    print("Classes:", full_dataset.classes)
    print("Total images:", len(full_dataset))
    

    # TRAIN / VAL SPLIT (80 / 20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    

    # DATA LOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    

    # CLASS WEIGHTS (IMBALANCE FIX)
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(targets)
    
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.tensor(class_weights).to(device)
    
    print("Class weights:", class_weights)
    
    # MODEL (ResNet-18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Adjust for 48x48 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Output layer
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)
    
    # FREEZE BACKBONE (INITIAL)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # TRAINING LOOP
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
    
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
        train_acc = 100 * correct / total
    
        
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
    
        val_acc = 100 * val_correct / val_total
    
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")
    
        # UNFREEZE AFTER 5 EPOCHS
        if epoch == 4:
            print("Unfreezing entire model...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    
    # SAVE MODEL
    torch.save(model.state_dict(), "emotion_resnet18.pth")
    print("Model saved as emotion_resnet18.pth")

if __name__ == "__main__":
    main()