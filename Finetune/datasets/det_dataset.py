import os

import numpy as np
import pandas as pd
import torch
from datasets.utils import get_imgs, read_from_dicom
from torch.utils.data import Dataset
import ast
import cv2
from PIL import Image
np.random.seed(42)

class RSNADetectionDataset(Dataset):
    def __init__(self, config, split='train', transform=None,data_pct=1.0) -> None:
        super().__init__()
        self.config = config
        if split == 'train':
            csv_path = config['dataset']['train_csv']
        elif split == 'valid':
            csv_path = config['dataset']['valid_csv']
        else:
            csv_path = config['dataset']['test_csv']
        max_objects = self.config["det"].get("max_objects", 10)
        imsize = self.config["cls"].get("imsize", 224)

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        if data_pct != 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.imsize = imsize

        # Get the bounding boxes
        filenames = self.df['path'].to_list()
        bboxs = self.df['bbox'].apply(ast.literal_eval).to_list()

        # transfer the bounding boxes to the format of [class, x_center, y_center, width, height]
        # yolo usually need to adrast like this
        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = bboxs[i]
            bbox = np.array(bbox)
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            n = new_bbox.shape[0]
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])
            pad = np.zeros((max_objects - n, 5))
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
       

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.filenames_list[index]
        x = read_from_dicom(img_path, None, None)

        x = cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB)
        h, w, _ = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        x = Image.fromarray(x, "RGB")

        if self.transform:
            x = self.transform(x)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": x,
            "labels": y
        }

        return sample
    

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
# dataset = RSNADetectionDataset(csv_path=csv_path, transform=transform, data_pct=0.1, imsize=256)  # Using 10% of data

# # Create DataLoader for batching
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Iterate over a batch of data
# for batch_idx, samples in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Images shape: {samples['imgs'].shape}")  # Should be [batch_size, channels, height, width]
#     print(f"Labels shape: {samples['labels'].shape}")  # Should be [batch_size, max_objects, 5]

