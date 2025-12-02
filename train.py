import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os

from datasets.robo_dataset import RoboMNISTDataset
from models.multimodal_model import MultimodalModel
import config


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0

    for batch in tqdm(loader, desc="Training"):
        # Multi-view inputs
        front = batch["front"].to(device)
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        csi = batch["csi"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Model expects a dict
        input_dict = {
            "front": front,
            "left": left,
            "right": right,
            "csi": csi
        }

        outputs = model(input_dict)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * front.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_correct / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            front = batch["front"].to(device)
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            csi = batch["csi"].to(device)
            labels = batch["label"].to(device)

            input_dict = {
                "front": front,
                "left": left,
                "right": right,
                "csi": csi
            }

            outputs = model(input_dict)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * front.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_correct / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets
    dataset_root = config.data_root
    train_set = RoboMNISTDataset(dataset_root, mode="train")
    val_set = RoboMNISTDataset(dataset_root, mode="val")

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    # Model
    model = MultimodalModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch+1}/{config.epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)

        val_loss, val_acc = validate(
            model, val_loader, criterion, device)

        print(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        print(f"Val   Loss={val_loss:.4f}, Val   Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Saved best model!")

    print("\nTraining finished.")
    print("Train samples:", len(train_set))
    print("Val   samples:", len(val_set))


if __name__ == "__main__":
       main()



print("hello world")