import argparse
import warnings

from datasets.visa_dataset import VISA_CATEGORIES
warnings.filterwarnings("ignore")

import wandb
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

from datasets.mvtec_dataset import MVTecDataset
from cl_utils.task_stream import TaskStream
from utilities.configurations import *
from models.rd4ad import RD4AD
from trainers.rd4ad_trainer import RD4AD_Trainer

SEEDS = [1,3,1024]

wandb.login(key="8fd610417fd4144fee676e5be4e212315f697550")

def set_seed(seed):

    # Basic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

def train_single_models(model_name, epochs, batch_size, device, use_wandb):

    for seed in SEEDS:
        set_seed(seed)

        for category in MVTecDataset.CATEGORIES:

            if use_wandb:
                wandb.init(project="stfpm_adapters", name=f"rd4ad_single_{model_name}_{category}", reinit=True)
                wandb.config.update({
                    "model_name": model_name,
                    "seed": seed,
                })

            print(f"Training model on category: {category}")

            train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train")
            print(f"Length train dataset: {len(train_dataset)}")
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)

            test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test")
            print(f"Length test dataset: {len(test_dataset)}")
            test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

            # define the model
            model = RD4AD(model_name, device, (224,224))

            #train the model
            RD4AD_Trainer.single_model_training(model, epochs, train_dataloader, test_dataloader, device, use_wandb, category)

def joint_train(model_name, dataset, epochs, batch_size, device, use_wandb=False):

    for seed in SEEDS:
        set_seed(seed)

        if args.wandb:
            wandb.init(project="stfpm_adapters", name=f"rd4ad_joint_{dataset}", reinit=True)
            wandb.config.update({
                "model_name": model_name,
                "seed": seed,
                "dataset": dataset,
            })

        if dataset == "mvtec":
            categories = MVTecDataset.CATEGORIES
        elif dataset == "visa":
            categories = VISA_CATEGORIES

        # get the full dataset
        task_stream = TaskStream(dataset_path, dataset, categories,  batch_size)
        train_dataloader, test_dataloader = task_stream.get_all_tasks_data()

        print("Training model on all categories jointly")
        print(f"Length train dataset: {len(train_dataloader.dataset)}")
        print(f"Length test dataset: {len(test_dataloader.dataset)}")

        # define the model
        model = RD4AD(model_name, device, (224,224))

        #train the model
        RD4AD_Trainer.single_model_training(model, epochs, train_dataloader, test_dataloader, device, use_wandb)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["resnet18", "wide_resnet50_2"])
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mode", choices=["joint", "single_model"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    dataset_path = ""
    output_masks_size = (224,224)

    if args.mode == "single_model":
        train_single_models(args.model_name, args.epochs, args.batch_size, args.device, args.wandb)
    elif args.mode == "joint":
        joint_train(args.model_name, args.dataset, args.epochs, args.batch_size, args.device, args.wandb)
