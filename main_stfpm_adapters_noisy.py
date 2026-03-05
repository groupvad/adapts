import argparse
import os
import warnings

from datasets.visa_dataset import VISA_CATEGORIES

warnings.filterwarnings("ignore")

import wandb
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from datasets.mvtec_dataset import MVTecDataset
from datasets.noisy_mvtec.visa_dataset import VISADataset
from datasets.noisy_mvtec.mvtec_dataset import MVTecDataset as NoisyMVTecDataset
from cl_utils.task_stream_noisy import TaskStream
from cl_utils.task_stream import TaskStream as TaskStreamClean
from utilities.configurations import *
from trainers.stfpm_adapters_trainer_seg import STFPMAdaptersTrainerSeg
from trainers.stfpm_trainer import STFPM_Trainer
from models.stfpm_adapters import AttentionAdapter, STFPMAdapters
from models.stfpm_adapters import LinearAdapter, MultiScaleAdapter, LinearAdapterExpansion, DepthwiseSeparableAdapter, LinearAdapterDropout
from models.stfpm_kmprompt import STFPMPKprompts
from models.stfpm_random import STFPMRandom
from models.stfpm import STFPM
from utilities.evaluator import Evaluator


LAYERS_BACKBONE = {
    "resnet18": ["layer1", "layer2", "layer3"],
    "mobilenet_v2": ["features.3", "features.8", "features.14"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"]
}

SEEDS = [1, 3, 1024]

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
}

def train_single_models_adapters(model_name, dataset, adapter_type, layers_idx, epochs, batch_size, device, use_wandb=False):

    CATEGORIES = MVTecDataset.CATEGORIES if dataset == "mvtec" else VISA_CATEGORIES

    for category in CATEGORIES:

        for seed in SEEDS:
            set_seed(seed)

            if args.wandb:
                wandb.init(project="stfpm_adapters", name=f"single_adapter_{model_name}_{category}_{adapter_type}_seg_layers_{str(layers_idx)}_{dataset}", reinit=True)
                wandb.config.update({
                    "model_name": model_name,
                    "adapter_type": adapter_type,
                    "seed": seed,
                    "layers_idx": layers_idx
                })

            print(f"Training model on category: {category}")

            RESIZE_SHAPE = [224, 224]  # width * height
            NORMALIZE_MEAN = [0.485, 0.456, 0.406]
            NORMALIZE_STD = [0.229, 0.224, 0.225]

            if dataset == "mvtec":

                train_dataset = NoisyMVTecDataset(
                    is_train=True,
                    mvtec_dir=dataset_path + category + "/train/good/",
                    resize_shape=RESIZE_SHAPE,
                    normalize_mean=NORMALIZE_MEAN,
                    normalize_std=NORMALIZE_STD,
                    dtd_dir=dtd_path,
                    rotate_90=False,
                    random_rotate=False,
                )
                print(f"Length train dataset: {len(train_dataset)}")
                train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)

                test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test")
                print(f"Length test dataset: {len(test_dataset)}")
                test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

            else:

                train_dataset = VISADataset(
                    is_train=True,
                    visa_dir="/home/u0052/disk/datasets/visa",
                    category=category,
                    csv_path = "/home/u0052/disk/datasets/visa/split_csv/1cls.csv",
                    dtd_dir="/home/u0052/disk/datasets/dtd/images/",
                    resize_shape=RESIZE_SHAPE,
                    normalize_mean=NORMALIZE_MEAN,
                    normalize_std=NORMALIZE_STD,
                    rotate_90=False,
                    random_rotate=False,
                )
                print(f"Length train dataset: {len(train_dataset)}")
                train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)

                test_dataset = VISADataset(
                    is_train=False,
                    visa_dir="/home/u0052/disk/datasets/visa",
                    dtd_dir="/home/u0052/disk/datasets/dtd/images/",
                    category=category,
                    csv_path = "/home/u0052/disk/datasets/visa/split_csv/1cls.csv",
                    resize_shape=RESIZE_SHAPE,
                    normalize_mean=NORMALIZE_MEAN,
                    normalize_std=NORMALIZE_STD,
                )
                print(f"Length test dataset: {len(test_dataset)}")
                test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

            # define the model
            adapter_class = ADAPTERS[adapter_type]
            model = STFPMAdapters(model_name, layers_idx, adapter_class, output_masks_size, device, use_cosine_loss=False)
            model.adapters_save_path = f"./saved_adapters/linear_seq_visa/{str(layers_idx)}"
            os.makedirs(model.adapters_save_path, exist_ok=True)

            #train the model
            STFPMAdaptersTrainerSeg.single_model_training_noisy(model, epochs, train_dataloader, test_dataloader, category, device, use_wandb)

def train_cl_adapters(model_name, layers_idx, epochs, batch_size, device, use_wandb=False):

    for seed in SEEDS:
        set_seed(seed)

        if args.wandb:
            wandb.init(project="stfpm_adapters", name="cl_adapters_linear_seg", reinit=True)
            wandb.config.update({
                "model_name": model_name,
                "seed": seed,
            })

        model = STFPMAdapters(model_name, layers_idx, LinearAdapter, output_masks_size, device)
        model.adapters_save_path = f"./adapters/linear_seg_{model_name}/{str(layers_idx)}"
        os.makedirs(model.adapters_save_path, exist_ok=True)
        tasks_stream = TaskStream(dataset_path, dtd_path, MVTecDataset.CATEGORIES, batch_size)

        #train the model
        STFPMAdaptersTrainerSeg.continual_training(model, epochs, tasks_stream, device, use_wandb)

def eval_cl_quantized_adapters(model_name, device, use_wandb=False):

    model = STFPMAdapters(model_name, LAYERS_BACKBONE[model_name], LinearAdapter,(224,224), device, True)
    model.eval()
    model.is_eval_during_training = False
    model.adapters_save_path = "./adapters"
    model.class_prototypes = torch.load("./adapters/class_prototypes.pth", map_location=device)

    tasks_stream = TaskStreamClean(dataset_path, MVTecDataset.CATEGORIES, batch_size=16)

    wandb.init(project="stfpm_adapters", name="cl_adapters_linear_seg_quant", reinit=True)
    for task_index in range(len(tasks_stream)):


        category = tasks_stream.categories[task_index]

        print(f"Training for task: {category}")

        train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

        print("Len train dataset:", len(train_dataloader.dataset))
        print("Len test dataset:", len(test_dataloader.dataset))

        #evaluate the model on all seen tasks
        model.is_eval_during_training = False

        summary_metrics = {
            "img_roc" : [],
            "pxl_roc" : [],
            "img_pr" : [],
            "pxl_pr" : [],
            "f1_img" : [],
            "f1_pxl" : [],
            "pxl_pro" : [],
            "task_classification_accuracy" : []
        }

        for previous_task_index in tasks_stream.get_previous_tasks(task_index):
            _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, task_classification_accuracy = Evaluator.evaluate_task(model, test_dataloader, device, all_metrics=True, task_index=previous_task_index)

            print(f"Performances on task: {previous_task_index} after training on task {task_index}:")
            print(f"Image-level AUROC: {img_roc}")
            print(f"Pixel-level AUROC: {pxl_roc}")
            print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            print(f"Pixel-level Pro: {pxl_pro}")
            print(f"Task Classification Accuracy: {task_classification_accuracy}")

            # udate summary metrics
            summary_metrics["img_roc"].append(img_roc)
            summary_metrics["pxl_roc"].append(pxl_roc)
            summary_metrics["img_pr"].append(img_pr)
            summary_metrics["pxl_pr"].append(pxl_pr)
            summary_metrics["f1_img"].append(f1_img)
            summary_metrics["f1_pxl"].append(f1_pxl)
            summary_metrics["pxl_pro"].append(pxl_pro)
            summary_metrics["task_classification_accuracy"].append(task_classification_accuracy)

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/eval/img_roc": img_roc,
                        f"Task_T{task_index}/eval/pxl_roc": pxl_roc,
                        f"Task_T{task_index}/eval/img_pr": img_pr,
                        f"Task_T{task_index}/eval/pxl_pr": pxl_pr,
                        f"Task_T{task_index}/eval/f1_img": f1_img,
                        f"Task_T{task_index}/eval/f1_pxl": f1_pxl,
                        f"Task_T{task_index}/eval/pxl_pro": pxl_pro,
                        f"Task_T{task_index}/eval/task_classification_accuracy": task_classification_accuracy,
                    }
                )

        print(f"Summary metrics after training on task {category}:")
        for metric_name, values in summary_metrics.items():
            avg_value = sum(values) / len(values)
            summary_metrics[metric_name] = avg_value
            print(f"Average {metric_name}: {avg_value}")

        if use_wandb:
            wandb.log(
                {
                    "Summary/img_roc": summary_metrics["img_roc"],
                    "Summary/pxl_roc": summary_metrics["pxl_roc"],
                    "Summary/img_pr": summary_metrics["img_pr"],
                    "Summary/pxl_pr": summary_metrics["pxl_pr"],
                    "Summary/f1_img": summary_metrics["f1_img"],
                    "Summary/f1_pxl": summary_metrics["f1_pxl"],
                    "Summary/pxl_pro": summary_metrics["pxl_pro"],
                    "Summary/task_classification_accuracy": summary_metrics["task_classification_accuracy"],
                }
            )

def eval_cl_adapters(model_name, device, use_wandb=False):

    model = STFPMAdapters(model_name, ['layer2', 'layer3'], LinearAdapter,(224,224), device, True)
    model.eval()
    model.is_eval_during_training = False
    model.adapters_save_path = "./adapters/linear_seq/['layer2', 'layer3']"
    model.class_prototypes = torch.load("./adapters/class_prototypes.pth", map_location=device)

    tasks_stream = TaskStreamClean(dataset_path, MVTecDataset.CATEGORIES, batch_size=16)

    wandb.init(project="stfpm_adapters", name="cl_adapters_linear_seg_['layer2', 'layer3']", reinit=True)
    for task_index in range(len(tasks_stream)):


        category = tasks_stream.categories[task_index]

        print(f"Training for task: {category}")

        train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

        print("Len train dataset:", len(train_dataloader.dataset))
        print("Len test dataset:", len(test_dataloader.dataset))

        #evaluate the model on all seen tasks
        model.is_eval_during_training = False

        summary_metrics = {
            "img_roc" : [],
            "pxl_roc" : [],
            "img_pr" : [],
            "pxl_pr" : [],
            "f1_img" : [],
            "f1_pxl" : [],
            "pxl_pro" : [],
            "task_classification_accuracy" : []
        }

        for previous_task_index in tasks_stream.get_previous_tasks(task_index):
            _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, task_classification_accuracy = Evaluator.evaluate_task(model, test_dataloader, device, all_metrics=True, task_index=previous_task_index)

            print(f"Performances on task: {previous_task_index} after training on task {task_index}:")
            print(f"Image-level AUROC: {img_roc}")
            print(f"Pixel-level AUROC: {pxl_roc}")
            print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            print(f"Pixel-level Pro: {pxl_pro}")
            print(f"Task Classification Accuracy: {task_classification_accuracy}")

            # udate summary metrics
            summary_metrics["img_roc"].append(img_roc)
            summary_metrics["pxl_roc"].append(pxl_roc)
            summary_metrics["img_pr"].append(img_pr)
            summary_metrics["pxl_pr"].append(pxl_pr)
            summary_metrics["f1_img"].append(f1_img)
            summary_metrics["f1_pxl"].append(f1_pxl)
            summary_metrics["pxl_pro"].append(pxl_pro)
            summary_metrics["task_classification_accuracy"].append(task_classification_accuracy)

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/eval/img_roc": img_roc,
                        f"Task_T{task_index}/eval/pxl_roc": pxl_roc,
                        f"Task_T{task_index}/eval/img_pr": img_pr,
                        f"Task_T{task_index}/eval/pxl_pr": pxl_pr,
                        f"Task_T{task_index}/eval/f1_img": f1_img,
                        f"Task_T{task_index}/eval/f1_pxl": f1_pxl,
                        f"Task_T{task_index}/eval/pxl_pro": pxl_pro,
                        f"Task_T{task_index}/eval/task_classification_accuracy": task_classification_accuracy,
                    }
                )

        print(f"Summary metrics after training on task {category}:")
        for metric_name, values in summary_metrics.items():
            avg_value = sum(values) / len(values)
            summary_metrics[metric_name] = avg_value
            print(f"Average {metric_name}: {avg_value}")

        if use_wandb:
            wandb.log(
                {
                    "Summary/img_roc": summary_metrics["img_roc"],
                    "Summary/pxl_roc": summary_metrics["pxl_roc"],
                    "Summary/img_pr": summary_metrics["img_pr"],
                    "Summary/pxl_pr": summary_metrics["pxl_pr"],
                    "Summary/f1_img": summary_metrics["f1_img"],
                    "Summary/f1_pxl": summary_metrics["f1_pxl"],
                    "Summary/pxl_pro": summary_metrics["pxl_pro"],
                    "Summary/task_classification_accuracy": summary_metrics["task_classification_accuracy"],
                }
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["adapters", "continual_adapters", "eval_adapters_cl_quant", "eval_adapters_cl"], required=True)
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--adapter_type", choices=["linear"], default="linear")
    parser.add_argument("--model_name", choices=["wide_resnet50_2", "mobilenet_v2", "resnet18"])
    parser.add_argument("--layers_idx", type=str, nargs="+", default=["layer1", "layer2", "layer3"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    dataset_path = "datasets/mvtec/"
    dtd_path = "datasets/dtd/images/"

    output_masks_size = (224,224)

    if args.mode == "continual_adapters":
        train_cl_adapters(args.model_name, args.layers_idx, args.epochs, args.batch_size, args.device, args.wandb)
    if args.mode == "eval_adapters_cl_quant":
        eval_cl_quantized_adapters(args.model_name, args.device, args.wandb)
    if args.mode == "eval_adapters_cl":
        eval_cl_adapters(args.model_name, args.device, args.wandb)
    elif args.mode == "adapters":
        train_single_models_adapters(args.model_name, args.dataset, args.adapter_type, args.layers_idx, args.epochs, args.batch_size, args.device, args.wandb)