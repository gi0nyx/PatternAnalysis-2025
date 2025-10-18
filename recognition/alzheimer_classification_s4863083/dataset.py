from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2

import warnings 
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
        file_path = self.file_paths[idx]
        label = self.labels[idx]

       
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

       
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            
            image = ToTensorV2()(image=image)['image']

        return image, label

        
 