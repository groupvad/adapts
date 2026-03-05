from typing import Union
from typing import Union
from tqdm import *
import copy

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.stfpm_adapters import STFPMAdapters
from datasets.mvtec_dataset import MVTecDataset
from utilities.evaluator import Evaluator
from utilities.configurations import WANDB_CONF
from cl_utils.task_stream import TaskStream
from cl_utils.strategies.ewc import *

class STFPMAdaptersTrainer:

    def single_model_training(
        model: STFPMAdapters,
        loss_fn,
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        category: str,
        device:torch.device,
        use_wandb: bool,
    ):

        task_index = MVTecDataset.CATEGORIES.index(category)
        model.reset_adapters()
        model.to(device)

        lr = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for n,p in model.named_parameters():
            if p.requires_grad:
                print(f"Parameter name: {n} - requires_grad: {p.requires_grad}")

        if use_wandb:
            wandb.config.update({
                "epochs": epochs,
                "batch_size": train_dataloader.batch_size,
                "learning_rate": lr,
                "optimizer": "SGD"
            })

        class_prototype = None
        class_prototypes = []

        # best_pxl_roc = 0.0
        best_pxl_f1 = 0.0

        mse_loss = torch.nn.MSELoss(reduction="sum")

        for epoch in trange(epochs):

            model.train()

            #train the model
            for batch in tqdm(train_dataloader):

                batch = batch.to(device)
                class_vectors, teacher_features, student_features = model(batch)

                loss = 0
                for i in range(len(student_features)):
                    # da valutare la normalizzazione
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)

                    # best loss
                    height, width = teacher_features[i].shape[2:]
                    loss += (0.5 / (width * height)) * mse_loss(teacher_features[i], student_features[i])

                    # loss += loss_fn(teacher_features[i], student_features[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == 0:
                    class_prototypes.append(class_vectors)

            # update the class_prototype
            if epoch == 0:
                class_prototype = torch.cat(class_prototypes, dim=0).mean(dim=0, keepdim=True)

                if model.class_prototypes is None:
                    model.class_prototypes = class_prototype
                    print("Initialized class prototypes shape:", model.class_prototypes.shape)
                else:
                    model.class_prototypes = torch.cat([model.class_prototypes, class_prototype], dim=0)
                    print("Updated class prototypes shape:", model.class_prototypes.shape)
                    # save the class prototypes
                    model.save_prototypes()

            print(f"Epoch: {epoch} Final Training loss: {loss}")

            #evaluate the model, we are during the training so no need to load again the adapters
            model.is_eval_during_training = True
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=True)

            print(f"Epoch: {epoch}")
            print(f"Image-level AUROC: {img_roc}")
            print(f"Pixel-level AUROC: {pxl_roc}")
            print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            print(f"Pixel-level Pro: {pxl_pro}")

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/train/epoch": epoch,
                        f"Task_T{task_index}/train/train_loss": loss,
                        f"Task_T{task_index}/train/img_roc": img_roc,
                        f"Task_T{task_index}/train/pxl_roc": pxl_roc,
                        f"Task_T{task_index}/train/img_pr": img_pr,
                        f"Task_T{task_index}/train/pxl_pr": pxl_pr,
                        f"Task_T{task_index}/train/f1_img": f1_img,
                        f"Task_T{task_index}/train/f1_pxl": f1_pxl,
                        f"Task_T{task_index}/train/pxl_pro": pxl_pro,
                    }
                )

            #save the adapters if the pixel f1 is the best until now
            if f1_pxl > best_pxl_f1:
                model.save_adapters(category)
                best_pxl_f1 = f1_pxl

    def continual_training(model:STFPMAdapters, epochs, tasks_stream: TaskStream, device, use_wandb: bool):

        for task_index in range(len(tasks_stream)):

            category = tasks_stream.categories[task_index]

            print(f"Training for task: {category}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            print("Len train dataset:", len(train_dataloader.dataset))
            print("Len test dataset:", len(test_dataloader.dataset))

            # train the adapters for the new task
            STFPMAdaptersTrainer.single_model_training(model, STFPMAdaptersTrainer.stfpm_loss, epochs, train_dataloader, test_dataloader, category ,device, use_wandb)

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

    def eval_trained_adapters(model:STFPMAdapters, tasks_stream: TaskStream, device, use_wandb: bool):

        print(f"Eval for all tasks:")

        #evaluate the model on all seen tasks
        for previous_task_index in tasks_stream.get_previous_tasks(14):
            _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = Evaluator.evaluate_task(model, test_dataloader, device)
            print(f"F1 pixel level on task {tasks_stream.categories[previous_task_index]}: {f1_pxl}")
