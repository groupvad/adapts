import os

from torch.utils.data.dataset import Subset
from typing import List
from torch.utils.data import DataLoader
import numpy as np
import torch
from typing import Tuple

from data.mvtec_dataset import MVTecDataset
from data.visa_dataset import VISADataset
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY_MVTEC

class TaskStream:

    def __init__(self,
                 dataset_path: str,
                 dtd_path: str,
                 categories: List[str],
                 batch_size: int
                ):

        """
        This class manage a tasks stream
        """

        self.dataset_path = dataset_path
        self.dtd_path = dtd_path
        self.batch_size = batch_size
        self.categories = categories

        self.no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
        ]
        self.slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        self.rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]

    def __len__(self):
        return len(self.categories)

    def get_task_data(self, task_index:int) -> Tuple[DataLoader, DataLoader]:

        """
        Get the training and test data for the given task.

        Args:
        ----
        - task_index (int)
            task index in the task sequence
        - dataset_path (str)
            where the dataset is stored
        - batch_size (int)
            batch_size

        Returns:
        -------
        (tuple):
            0: train dataloader
            1: test dataloader
        """
        category = self.categories[task_index]

        rotate_90 = category in self.rotation_category
        random_rotate = 5 if category in self.slight_rotation_category + self.rotation_category else 0

        # train_dataset = MVTecDataset(
        #     is_train=True,
        #     mvtec_dir=self.dataset_path + category + "/train/good/",
        #     resize_shape=RESIZE_SHAPE,
        #     normalize_mean=NORMALIZE_MEAN,
        #     normalize_std=NORMALIZE_STD,
        #     dtd_dir=self.dtd_path,
        #     rotate_90=rotate_90,
        #     random_rotate=random_rotate,
        # )

        # test_dataset = MVTecDataset(
        #     is_train=False,
        #     mvtec_dir=self.dataset_path + category + "/test/",
        #     resize_shape=RESIZE_SHAPE,
        #     normalize_mean=NORMALIZE_MEAN,
        #     normalize_std=NORMALIZE_STD,
        # )

        train_dataset = VISADataset(
            is_train=True,
            visa_dir=self.dataset_path,
            category=category,
            csv_path=self.dataset_path + "annotations.csv",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=self.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
        )

        test_dataset = VISADataset(
            is_train=False,
            visa_dir=self.dataset_path,
            category=category,
            csv_path=self.dataset_path + "annotations.csv",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )

        return (
            DataLoader(train_dataset, self.batch_size, shuffle = True),
            DataLoader(test_dataset,  self.batch_size, shuffle = True)
        )

    def get_task_data_evaluation(self, task_index:int) -> torch.utils.data.DataLoader:

        """
        Get the test data for the given task.

        Args:
        ----
        - task_index (int)
            task index in the task sequence

        Returns:
        -------
        - torch.utils.data.DataLoader: test dataloader for the given task
        """

        test_dataset = VISADataset(
            is_train=False,
            visa_dir=self.dataset_path,
            category=self.categories[task_index],
            csv_path=self.dataset_path + "annotations.csv",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )

        return DataLoader(test_dataset,  self.batch_size, shuffle = True)

    def get_all_tasks_data(self) -> torch.utils.data.DataLoader:

        """
        Get the data for all tasks.

        Args:
        -----
        - split (str)
            "train" or "test"
        """
        all_datasets_train = []
        all_datasets_test = []

        for category in self.categories:

            rotate_90 = category in self.rotation_category
            random_rotate = 5 if category in self.slight_rotation_category + self.rotation_category else 0

            train_dataset = VISADataset(
                is_train=True,
                visa_dir=self.dataset_path,
                category=category,
                csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=self.dtd_path,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
            )

            test_dataset = VISADataset(
                is_train=False,
                visa_dir=self.dataset_path,
                category=category,
                csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=self.dtd_path,
            )

            print("Category:", category)
            print("Length of train dataset:", len(train_dataset))
            print("Length of test dataset:", len(test_dataset))

            all_datasets_train.append(train_dataset)
            all_datasets_test.append(test_dataset)

        train_dataset = torch.utils.data.ConcatDataset(all_datasets_train)
        test_dataset = torch.utils.data.ConcatDataset(all_datasets_test)
        return DataLoader(train_dataset, self.batch_size, shuffle = True), DataLoader(test_dataset, self.batch_size, shuffle = True)


    def get_previous_tasks(self, task_index):
        return range(len(self.categories))[:task_index+1]