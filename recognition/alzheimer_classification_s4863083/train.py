import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm


from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, RandomBrightnessContrast, Normalize
)
from albumentations.pytorch import ToTensorV2
from dataset import TrainDataset 

def main():
    class CFG:
        batch_size = 16
        epochs = 10
        lr = 1e-3
        img_size = 224
        n_splits = 5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('train_df.csv')

    sgkf = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
    train_idx, val_idx = list(sgkf.split(df, df['label'], df['patient_id']))[0]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)


    train_transform = Compose([
        RandomResizedCrop(size=(CFG.img_size,CFG.img_size), scale=(0.8, 1.0)),
        Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = TrainDataset(train_df, transform=train_transform)
    val_dataset = TrainDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    model_name = "convnextv2_tiny" 
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model = model.to(CFG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = GradScaler()


    for epoch in range(CFG.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CFG.epochs}]")

        for images, labels in loop:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        scheduler.step()

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CFG.device), labels.to(CFG.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

        print(f"\nEpoch {epoch+1}/{CFG.epochs} "
            f"| Train Loss: {running_loss/len(train_loader):.4f} "
            f"| Val Loss: {val_loss/len(val_loader):.4f} "
            f"| Val Acc: {100 * val_correct/val_total:.2f}%\n")

    torch.save(model.state_dict(), "convnextv2_tiny.pth")
    print("Training complete, model saved.")


if __name__ == "__main__":
    main()