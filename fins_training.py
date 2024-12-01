import random

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

DATA_DIR = 'fins'
MODEL_PATH = 'fins/fins.pt'
BATCH_SIZE = 32
NUM_EPOCHS = 10

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def display_result(dataset, device, model, val_transform):
    """Display model predictions on random images from the dataset."""
    dataset.transform = val_transform
    indices = random.sample(range(len(dataset)), 10)
    images = []
    predicted_labels = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]
            input_img = img.unsqueeze(0).to(device)
            output = model(input_img)
            _, pred = torch.max(output, 1)
            images.append(img)
            predicted_labels.append(dataset.classes[pred.item()])
            true_labels.append(dataset.classes[label])

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for img, pred_label, true_label, ax in zip(
            images, predicted_labels, true_labels, axes):
        img = (
            img * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2) +
            torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        )
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('fins/random_ten.png')


def main():
    """Main function to train and evaluate the model."""
    # Create dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomRotation(degrees=30),
            transforms.RandomRotation(degrees=45)
        ]),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR)
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train Size: {train_size}, Validation Size: {val_size}")

    device = torch.device("mps" if torch.has_mps else "cpu")
    print(f"Device: {device}")

    # Create model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size

        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / val_size
        val_accuracy = correct / total

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model, MODEL_PATH)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")

    # Check correctness of the model
    display_result(dataset, device, model, val_transform)


if __name__ == '__main__':
    main()
