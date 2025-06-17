import os
import cv2
import numpy as np
import pandas as pd
import torch
from datasets.utils import resize_img
from torch.utils.data import Dataset
from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
import ast
import pydicom
from PIL import Image

np.random.seed(42)

class RSNASegmentDataset(Dataset):
    def __init__(self, config, split='train',transform=None,data_pct=1.0) -> None:
        super().__init__()
        self.config = config
        if split == 'train':
            csv_path = config['dataset']['train_csv']
        elif split == 'valid':
            csv_path = config['dataset']['valid_csv']
        else:
            csv_path = config['dataset']['test_csv']

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        if data_pct != 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.imsize = self.config["seg"].get("imsize", 256)
        self.split = split
        self.seg_transform = self.get_transforms()
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["path"]
        x = self.read_from_dicom(img_path)
        mask = np.zeros([1024, 1024])

        bbox = np.array(ast.literal_eval(row["bbox"]))
        new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)
        if len(new_bbox) > 0:
            for i in range(len(new_bbox)):
                mask[new_bbox[i, 1]:new_bbox[i, 3],
                new_bbox[i, 0]:new_bbox[i, 2]] += 1
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)
        x = augmented["image"]
        y = augmented["mask"].squeeze()
        return x, y
    
    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms
    

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)
    



# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt

# # Assuming you have `RSNAImageDataset` class imported and defined

# # Define some basic transformations (resize, convert to tensor, normalize)
# transform = transforms.Compose([
#     transforms.RandomCrop(224),  
#     transforms.ToTensor(),  
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
# ])

# # Instantiate the dataset
# csv_path = '/u/home/lj0/Code/VLP-Seminars/annotations/train.csv'  # Path to your CSV file
# dataset = RSNASegmentDataset(csv_path=csv_path, transform=transform, data_pct=0.1, imsize=256)  # Using 10% of data

# # Create DataLoader for batching
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Iterate over a batch of data
# for batch_idx, (images, labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Images shape: {images.shape}")  # Should be [batch_size, channels, height, width]
#     print(f"Masks shape: {labels.shape}")
#     break  # Remove this if you want to loop through all batches

