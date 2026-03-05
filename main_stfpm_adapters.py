import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import wandb
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from datasets.mvtec_dataset import MVTecDataset
from datasets.visa_dataset import VISADataset, VISA_CATEGORIES
from cl_utils.task_stream import TaskStream
from utilities.configurations import *
from trainers.stfpm_cladapters_trainer import STFPMAdaptersTrainer
from trainers.stfpm_trainer import STFPM_Trainer
from models.stfpm_adapters import STFPMAdapters, LinearAdapter, LinearAdapterExpansion, BottleneckAdapter
from models.stfpm import STFPM
from utilities.evaluator import Evaluator


LAYERS_BACKBONE = {
    "resnet18": ["layer1", "layer2", "layer3"],
    "mobilenet_v2": ["features.3", "features.8", "features.14"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"]
}

SEEDS = [1,3,1024]

def set_seed(seed):

    # Basic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

ADAPTERS = {
    "linear": LinearAdapter,
    "bottleneck": BottleneckAdapter,
    "expansion": LinearAdapterExpansion,
}

def train_single_models_adapters(model_name, adapter_type, epochs, batch_size, device, use_wandb=False):

    for category in MVTecDataset.CATEGORIES:

        for seed in SEEDS:

            set_seed(seed)

            if args.wandb:
                wandb.init(project="stfpm_adapters", name=f"single_adapter_{model_name}_{category}_{adapter_type}", reinit=True)
                wandb.config.update({
                    "model_name": model_name,
                    "adapter_type": adapter_type,
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
            adapter_class = ADAPTERS[adapter_type]
            model = STFPMAdapters(model_name, LAYERS_BACKBONE[model_name], adapter_class, output_masks_size, device)
            model.adapters_save_path = f"./adapters/{adapter_type}_{model_name}/"
            #if not os.path.exists(model.adapters_save_path):
            #    os.makedirs(model.adapters_save_path)

            #train the model
            STFPMAdaptersTrainer.single_model_training(model, STFPM_Trainer.stfpm_loss, epochs, train_dataloader, test_dataloader, category, device, use_wandb)

def train_cl_adapters(model_name, epochs, batch_size, device, use_wandb=False):

    for seed in SEEDS:
        set_seed(seed)

        if args.wandb:
            wandb.init(project="stfpm_adapters", name="cl_adapters_linear", reinit=True)
            wandb.config.update({
                "model_name": model_name,
                "seed": seed,
            })

        model = STFPMAdapters(model_name, LAYERS_BACKBONE[model_name], LinearAdapter, output_masks_size, device)
        model.adapters_save_path = f"./adapters/saved_adapters_cl/linear_{model_name}/"
        os.makedirs(model.adapters_save_path, exist_ok=True)
        tasks_stream = TaskStream(dataset_path, MVTecDataset.CATEGORIES, batch_size)

        #train the model
        STFPMAdaptersTrainer.continual_training(model, epochs, tasks_stream, device, use_wandb)

def eval_adapters_cl(model_name, batch_size, device, use_wandb=False):

    model = STFPMAdapters(model_name, ["layer1","layer2","layer3"], output_masks_size, device)
    model.to(device)
    model.load_prototypes(f"./adapters/saved_adapters_cl/class_prototypes.pth")
    tasks_stream = TaskStream(dataset_path, MVTecDataset.CATEGORIES, batch_size)

    #train the model
    STFPMAdaptersTrainer.eval_trained_adapters(model, tasks_stream, device, use_wandb)

def joint_train(model_name, dataset, epochs, batch_size, device, use_wandb=False):

    for seed in SEEDS:
        set_seed(seed)

        if args.wandb:
            wandb.init(project="stfpm_adapters", name=f"stfpm_joint_{dataset}", reinit=True)
            wandb.config.update({
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
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

        model = STFPM(model_name, ["layer1","layer2","layer3"], output_masks_size)

        #train the model
        STFPM_Trainer.single_model_training(model, STFPM_Trainer.stfpm_loss, epochs, train_dataloader, test_dataloader, device, use_wandb)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["adapters", "continual_adapters", "eval_adapters", "joint", "joint_random"])
    parser.add_argument("--adapter_type", choices=["linear"], default="linear")
    parser.add_argument("--model_name", choices=["wide_resnet50_2", "mobilenet_v2", "resnet18"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--device", type=str)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    dataset_path = "./datasets/mvtec"

    output_masks_size = (224,224)

    if args.mode == "continual_adapters":
        train_cl_adapters(args.model_name, args.epochs, args.batch_size, args.device, args.wandb)
    elif args.mode == "eval_adapters":
        eval_adapters_cl(args.model_name, args.batch_size, args.device, args.wandb)
    elif args.mode == "adapters":
        train_single_models_adapters(args.model_name, args.adapter_type, args.epochs, args.batch_size, args.device, args.wandb)
    elif args.mode == "joint":
        joint_train(args.model_name, args.dataset, args.epochs, args.batch_size, args.device, args.wandb)
