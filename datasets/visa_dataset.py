import glob
import os
import pandas as pd

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

class VISADataset(Dataset):
    def __init__(
        self,
        is_train,
        visa_dir,
        category,
        csv_path,
        resize_shape=[224,224],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.visa_dir = visa_dir
        self.csv_path = csv_path
        self.category = category

        self.resize_shape = resize_shape
        self.is_train = is_train
        
        self.df = pd.read_csv(csv_path)

        split = "train" if is_train else "test"
        self.df = self.df[(self.df["object"]==category) & (self.df["split"]==split)]

        if is_train:
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )

        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.visa_dir, self.df.iloc[index]["image"])).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            image = self.final_preprocessing(image)
            return image
        else:
            image = self.final_preprocessing(image)
            label = 0 if self.df.iloc[index]["label"] == "normal" else 1
            path = self.df.iloc[index]["image"]

            mask = None
            if self.df.iloc[index]["label"] == "normal":
                mask = torch.zeros((1, self.resize_shape[1], self.resize_shape[0]))
            else:
                mask_path = os.path.join(self.visa_dir, self.df.iloc[index]["mask"])
                mask = Image.open(mask_path).convert("L")
                mask = self.mask_preprocessing(mask)

                # Tutto ciò che è maggiore di 0 (quindi i tuoi 1 e 6 convertiti) diventa 1 (Bianco puro)
                mask = torch.where(
                    mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask)
                )

            return image, label, mask, path