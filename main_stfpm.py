import argparse
import warnings
warnings.filterwarnings("ignore")

import wandb
from torch.utils.data import DataLoader

from datasets.mvtec_dataset import MVTecDataset
from cl_utils.task_stream import TaskStream
from utilities.configurations import *
from models.stfpm import STFPM
from trainers.stfpm_trainer import STFPM_Trainer

import random
import numpy as np
import torch

SEEDS = [1,3,1024]

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
                wandb.init(project="stfpm_adapters", name=f"stfpm_single_{category}", reinit=True)
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
            model = STFPM(model_name, ["layer1","layer2","layer3"], output_masks_size)

            #train the model
            STFPM_Trainer.single_model_training(model, STFPM_Trainer.stfpm_loss, epochs, train_dataloader, test_dataloader, device, use_wandb)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["wide_resnet50_2", "mobilenet_v2", "resnet18"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mode", choices=["single_model", "test"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    dataset_path = ""
    output_masks_size = (224,224)

    if args.mode == "single_model":
        train_single_models(args.model_name, args.epochs, args.batch_size, args.device, args.wandb)