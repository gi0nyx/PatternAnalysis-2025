import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.stats import mode

preds_df = pd.read_csv('preds.csv')

def get_patient_id(idx):
    id = idx[22:].split('_')[0]
    return id
preds_df['patient_ids'] = preds_df['file_path'].apply(get_patient_id)

def display_random_patient_grid(df, num_patients=4):
    patient_ids = df['patient_id'].unique()
    
    if len(patient_ids) < num_patients:
        num_patients = len(patient_ids)

    random_patient_ids = random.sample(list(patient_ids), num_patients)

    grid_size = int(np.ceil(np.sqrt(num_patients)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i, patient_id in enumerate(random_patient_ids):
        ax = axes[i]
        patient_df = df[df['patient_id'] == patient_id]

        image = plt.imread(patient_df['file_path'].iloc[0])
        
        true_label = patient_df['label'].iloc[0]
        pred_label = mode(patient_df['pred'].values, keepdims=False).mode
        
        ax.imshow(image)
        ax.set_title(f"Patient ID: {patient_id}\nTrue: {true_label}, Pred: {pred_label}")
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('patient_grid.png')
    print("Patient grid image saved to 'patient_grid.png'")


display_random_patient_grid(preds_df, num_patients=4)