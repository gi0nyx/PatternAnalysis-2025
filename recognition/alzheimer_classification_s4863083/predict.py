import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import timm
from dataset import TrainDataset 
from train import CFG, val_transform, calculate_patient_level_accuracy

device = CFG.device
model_path = "convnextv2_tiny.pth"
test_df = pd.read_csv('test_df.csv')

model = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

test_dataset = TrainDataset(test_df, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size * 2, shuffle=False)

test_preds = []
test_labels = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Getting Slice Predictions"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels)

test_slice_acc = accuracy_score(test_labels, test_preds)
test_patient_acc = calculate_patient_level_accuracy(test_df, test_preds)

print(f"Slice-Level Accuracy: {test_slice_acc:.4f}")
print(f"Patient-Level Accuracy (using Majority Vote): {test_patient_acc:.4f}")