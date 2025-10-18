from glob import glob
import pandas as pd
import os

train_dataset_path = 'dataset/AD_NC/train'
test_dataset_path = 'dataset/AD_NC/test'

train_AD_path = glob(f'{train_dataset_path}/AD/*.jpeg')
train_NC_path = glob(f'{train_dataset_path}/NC/*.jpeg')

test_AD_path = glob(f'{test_dataset_path}/AD/*.jpeg')
test_NC_path = glob(f'{test_dataset_path}/NC/*.jpeg')

def extract_patient_id(path):
    filename = os.path.basename(path)
    return filename.split('_')[0]  

train_df = pd.DataFrame({
    'file_path': train_AD_path + train_NC_path,
    'label': [1] * len(train_AD_path) + [0] * len(train_NC_path)
})

train_df['patient_id'] = train_df['file_path'].apply(extract_patient_id)


test_df = pd.DataFrame({
    'file_path': test_AD_path + test_NC_path,
    'label': [1] * len(test_AD_path) + [0] * len(test_NC_path)
})

test_df['patient_id'] = test_df['file_path'].apply(extract_patient_id)


train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df.to_csv('train_df.csv',index=False)
test_df.to_csv('test_df.csv',index=False)