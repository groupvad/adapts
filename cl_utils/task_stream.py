from torch.utils.data import dataset
from torch.utils.data.dataset import Subset
from typing import List
from torch.utils.data import DataLoader
import numpy as np
import torch
from typing import Tuple
import os

from datasets.mvtec_dataset import MVTecDataset
from datasets.visa_dataset import VISADataset
from utilities.configurations import TaskType, Split

class TaskStream:

    def __init__(self,
                 dataset_path: str,
                 dataset: str,
                 categories: List[str],
                 batch_size: int,
                 dtd_path: str = None
                ):

        """
        This class manage a tasks stream
        """

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.categories = categories
        self.dataset = dataset
        self.dtd_path = dtd_path

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

        if self.dataset == "mvtec":
            train_dataset = MVTecDataset(TaskType.SEGMENTATION, self.dataset_path, self.categories[task_index], Split.TRAIN)
            test_dataset  = MVTecDataset(TaskType.SEGMENTATION, self.dataset_path, self.categories[task_index], Split.TEST)
        else: 

            RESIZE_SHAPE = [256, 256]  # width * height
            NORMALIZE_MEAN = [0.485, 0.456, 0.406]
            NORMALIZE_STD = [0.229, 0.224, 0.225]

            train_dataset = VISADataset(
                is_train=True,
                visa_dir=self.dataset_path,
                category=self.categories[task_index],
                csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                #dtd_dir=self.dtd_path,
            )

            test_dataset = VISADataset(
                is_train=False,
                visa_dir=self.dataset_path,
                category=self.categories[task_index],
                csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
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

        if self.dataset == "mvtec":
            test_dataset  = MVTecDataset(TaskType.SEGMENTATION, self.dataset_path, self.categories[task_index], Split.TEST)
        else: 

            RESIZE_SHAPE = [256, 256]  # width * height
            NORMALIZE_MEAN = [0.485, 0.456, 0.406]
            NORMALIZE_STD = [0.229, 0.224, 0.225]

            test_dataset = VISADataset(
                is_train=False,
                visa_dir=self.dataset_path,
                category=self.categories[task_index],
                csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
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

            if self.dataset == "mvtec":
                train_dataset = MVTecDataset(TaskType.SEGMENTATION, self.dataset_path, category, Split.TRAIN)
                train_dataset.load_dataset()
                test_dataset  = MVTecDataset(TaskType.SEGMENTATION, self.dataset_path, category, Split.TEST)
                test_dataset.load_dataset()
            else: 

                RESIZE_SHAPE = [224,224]  # width * height
                NORMALIZE_MEAN = [0.485, 0.456, 0.406]
                NORMALIZE_STD = [0.229, 0.224, 0.225]

                train_dataset = VISADataset(
                    is_train=True,
                    visa_dir=self.dataset_path,
                    category=category,
                    csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
                    resize_shape=RESIZE_SHAPE,
                    normalize_mean=NORMALIZE_MEAN,
                    normalize_std=NORMALIZE_STD,
                )
                

                test_dataset = VISADataset(
                    is_train=False,
                    visa_dir=self.dataset_path,
                    category=category,
                    csv_path=os.path.join(self.dataset_path, "split_csv", "1cls.csv"),
                    resize_shape=RESIZE_SHAPE,
                    normalize_mean=NORMALIZE_MEAN,
                    normalize_std=NORMALIZE_STD,
                )

            all_datasets_train.append(train_dataset)
            all_datasets_test.append(test_dataset)

        train_dataset = torch.utils.data.ConcatDataset(all_datasets_train)
        test_dataset = torch.utils.data.ConcatDataset(all_datasets_test)
        return DataLoader(train_dataset, self.batch_size, shuffle = True), DataLoader(test_dataset, self.batch_size, shuffle = True)


    def get_previous_tasks(self, task_index):
        return range(len(self.categories))[:task_index+1]