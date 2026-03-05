from typing import Optional
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms

from utilities.configurations import TaskType, Split, LabelName

"""Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
"""

class MVTecDataset(torch.utils.data.Dataset):

    CATEGORIES = (
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    )

    IMG_EXTENSIONS = (".png", ".PNG")

    IMG_SIZE = (3, 900, 900)


    def __init__(
        self,
        task: TaskType,
        root: str,
        category: str,
        split: Split,
        norm: bool = True,
        img_size=(224,224),
        gt_mask_size:Optional[tuple]=None,
    ) -> None:
        super(MVTecDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples: pd.DataFrame = None

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        else:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(gt_mask_size, antialias=True, interpolation=InterpolationMode.NEAREST),
        ])

        self.load_dataset()

    def load_dataset(self):

        root = Path(self.root_category)

        samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in MVTecDataset.IMG_EXTENSIONS]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

        # Modify image_path column by converting to absolute path
        samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:

            # separate masks from samples
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
            samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

            # assign mask paths to anomalous test images
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

            # assert that the right mask files are associated with the right test images
            abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
            if (
                len(abnormal_samples)
                and not abnormal_samples.apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1).all()
            ):
                msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                he mask files in 'ground_truth' folder follow the same naming convention as the
                anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                raise Exception(msg)

        self.samples = samples[samples.split == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        #open the image and get the tensor
        image = Image.open(self.samples.iloc[index].image_path).convert("RGB")
        image = self.transform_image(image)

        if self.split == Split.TRAIN:
            return image
        else:
            #return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            path = self.samples.iloc[index].image_path
            if label == LabelName.ABNORMAL:
                mask = Image.open(self.samples.iloc[index].mask_path).convert('L')
                mask = self.transform_mask(mask)

            else:
                mask = torch.zeros(1, *self.gt_mask_size)

            return image, label, mask.int(), path
