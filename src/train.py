import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_model
from config import DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def plot_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train loss")
    ax1.plot(val_losses, label="Val loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train acc")
    ax2.plot(val_accs, label="Val acc")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png")
    print("Training history saved to models/training_history.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes ({len(classes)}): {classes}")

    model = get_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss() # For multi-class classifcation this is the standard approach (we have 15 classes)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE) # Adam converges faster and is thus the best approach for transfer-learning

    best_val_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved (best val acc: {best_val_acc:.4f})")

    plot_history(train_losses, val_losses, train_accs, val_accs)
    print(f"\nTraining Done! Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()