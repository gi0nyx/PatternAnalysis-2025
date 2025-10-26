from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import pandas as pd
import numpy as np
import warnings 
import os
warnings.filterwarnings('ignore')

class TrainDataset(Dataset):
    def __init__(self,df,transform=None):
        super().__init__()
        self.df = df
        self.file_path = df['file_path'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        center_path = self.file_path[idx]
        label = self.labels[idx]
        
        try:
            base_path, slice_info = center_path.rsplit('_', 1)
            slice_num_str, ext = slice_info.split('.')
            slice_num = int(slice_num_str)

            prev_slice_path = f"{base_path}_{slice_num - 1}.{ext}" # find next and previous slice
            next_slice_path = f"{base_path}_{slice_num + 1}.{ext}"

            img_center = cv2.imread(center_path, cv2.IMREAD_GRAYSCALE)
            
            if os.path.exists(prev_slice_path):
                img_prev = cv2.imread(prev_slice_path, cv2.IMREAD_GRAYSCALE)
            else:
                img_prev = img_center 
            
            
            if os.path.exists(next_slice_path):
                img_next = cv2.imread(next_slice_path, cv2.IMREAD_GRAYSCALE)
            else:
                img_next = img_center 

           
            image = np.stack([img_prev, img_center, img_next], axis=-1)

        except Exception as e:
            image = cv2.imread(center_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label