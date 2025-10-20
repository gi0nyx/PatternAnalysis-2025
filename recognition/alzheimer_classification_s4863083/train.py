import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from modules import ConvNext, create_base

from albumentations import (
    Compose, RandomResizedCrop, Normalize, Resize
)
from albumentations.pytorch import ToTensorV2
from dataset import TrainDataset

class CFG:
    batch_size = 16
    epochs = 25
    lr = 1e-4 
    img_size = 224
    n_splits = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transform = Compose([
        RandomResizedCrop(size=(CFG.img_size, CFG.img_size), scale=(0.8, 1.0)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

val_transform = Compose([
        Resize(CFG.img_size, CFG.img_size, always_apply=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def calculate_patient_level_accuracy(df_subset, preds):
    temp_df = df_subset.copy()
    temp_df['pred'] = preds
    
    # Group by patient and perform voting
    patient_groups = temp_df.groupby('patient_id')
    patient_preds = []
    true_patient_labels = []

    for _, group in patient_groups:
        # Majority vote on slice predictions
        majority_vote = mode(group['pred'].values, keepdims=False).mode
        patient_preds.append(majority_vote)
        
        # True label is the same for all slices of a given patient
        true_patient_labels.append(group['label'].iloc[0])
        
    return accuracy_score(true_patient_labels, patient_preds)

def main():

    df = pd.read_csv('train_df.csv')

    sgkf = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
    train_idx, val_idx = list(sgkf.split(df, df['label'], df['patient_id']))[0]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_df, transform=train_transform)
    val_dataset = TrainDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model_name = "convnextv2_tiny" 
    
    model = create_base()
    model = model.to(CFG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    scaler = GradScaler()

    for epoch in range(CFG.epochs):
        model.train()
        train_loss = 0.0
        train_epoch_preds = []
        train_epoch_labels = []
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CFG.epochs}] Training")
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
            train_epoch_preds.extend(preds.cpu().numpy())
            train_epoch_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        train_slice_acc = accuracy_score(train_epoch_labels, train_epoch_preds)
        train_patient_acc = calculate_patient_level_accuracy(train_df, train_epoch_preds)

       
        model.eval()
        val_loss = 0.0
        val_epoch_preds = []
        val_epoch_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{CFG.epochs}] Validation"):
                images, labels = images.to(CFG.device), labels.to(CFG.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_epoch_preds.extend(preds.cpu().numpy())
                val_epoch_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()
        
        val_slice_acc = accuracy_score(val_epoch_labels, val_epoch_preds)
        val_patient_acc = calculate_patient_level_accuracy(val_df, val_epoch_preds)

        print(f"\nEpoch {epoch+1}/{CFG.epochs} Summary:")
        print(f"  Train -> Loss: {train_loss/len(train_loader):.4f} | Slice Acc: {train_slice_acc:.4f}")
        print(f"  Valid -> Loss: {val_loss/len(val_loader):.4f} | Slice Acc: {val_slice_acc:.4f} | Patient Acc: {val_patient_acc:.4f}\n")

    torch.save(model.state_dict(), "convnext_base.pth")
    print("Training complete, model saved.")

if __name__ == "__main__":
    main()