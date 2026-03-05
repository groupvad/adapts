from typing import Union, List
from tqdm import *
import copy

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.rd4ad import RD4AD
from utilities.evaluator import Evaluator
from utilities.configurations import WANDB_CONF
from cl_utils.task_stream import TaskStream
from cl_utils.strategies.ewc import *

class RD4AD_Trainer:

    def loss_function(teacher_features: List[torch.Tensor], student_features: List[torch.Tensor]):

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0

        #iterate over the feature extraction layers batches
        #every feature maps shape is (B C H W)
        for i in range(len(teacher_features)):
            loss += torch.mean(
                1 - cos_loss(
                    teacher_features[i].view(teacher_features[i].shape[0],-1),
                    student_features[i].view(student_features[i].shape[0],-1)
                )
            )
        return loss

    def single_model_training(
        model: RD4AD,
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        device:torch.device,
        use_wandb: bool,
        category: str = None,
    ):

        model.train()
        model.to(device)

        learning_rate = 0.005

        optimizer = torch.optim.Adam(
            list(model.decoder.parameters())+list(model.bn.parameters()),
            lr=learning_rate,
            betas=(0.5,0.999)
        )

        for epoch in trange(epochs):

            model.train()

            loss_list = []
            #train the model
            for batch in tqdm(train_dataloader):

                batch = batch.to(device)
                teacher_features, bn_features, student_features,_, _  = model(batch)

                loss = RD4AD_Trainer.loss_function(teacher_features, student_features)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": np.mean(loss_list),
                    }
                )

        #evaluate the model
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=True)

        if use_wandb:
            wandb.log(
                {
                    "loss": np.mean(loss_list),
                    "img_roc": img_roc,
                    "pxl_roc": pxl_roc,
                    "f1_img": f1_img,
                    "f1_pxl": f1_pxl,
                    "img_pr": img_pr,
                    "pxl_pr": pxl_pr,
                    "pxl_pro": pxl_pro,
                }
            )

    def lwf_podnet_training(model:RD4AD, epochs, lamb, reset_student:bool, tasks_stream: TaskStream, device, use_wandb: bool):

        def pod_width_loss(teacher_features, student_features):
            # shape: B, C, H, W

            teacher_features_width_pooled = teacher_features.sum(3)
            student_features_width_pooled = student_features.sum(3)

            # shape: B, C, H, 1

            cos_loss = torch.nn.CosineSimilarity()
            return torch.mean(
                1 - cos_loss(
                    teacher_features_width_pooled.view(teacher_features_width_pooled.shape[0], -1),
                    student_features_width_pooled.view(student_features_width_pooled.shape[0], -1)
                )
            )

        def pod_height_loss(teacher_features, student_features):
            teacher_features_width_pooled = teacher_features.sum(2)
            student_features_width_pooled = student_features.sum(2)

            # shape: B, C, 1, W

            cos_loss = torch.nn.CosineSimilarity()
            return torch.mean(
                1 - cos_loss(
                    teacher_features_width_pooled.view(teacher_features_width_pooled.shape[0], -1),
                    student_features_width_pooled.view(student_features_width_pooled.shape[0], -1)
                )
            )

        def pod_spatial_loss(teacher_features, student_features):
            return pod_width_loss(teacher_features, student_features) + pod_height_loss(teacher_features, student_features)

        def pod_embedding_loss(bn_teacher_features, bn_student_features):
            bn_teacher_features = bn_teacher_features.unsqueeze(0)
            bn_student_features = bn_student_features.unsqueeze(0)
            # shape C, H, W tp 1, C, H, W
            spatial_loss = pod_spatial_loss(bn_teacher_features, bn_student_features)

            return 2 - spatial_loss

        def pod_loss(teacher_features, bn_features, student_features, old_teacher_features, bn_old_features):

            n_layers = len(teacher_features)
            spatial_loss = 0
            for i in range(n_layers):
                spatial_loss += pod_spatial_loss(old_teacher_features[i], student_features[i])
            spatial_loss /= (n_layers - 1)

            return spatial_loss + pod_embedding_loss(bn_old_features, bn_features)

        for task_index in range(len(tasks_stream)):

            print(f"Training for task: {tasks_stream.categories[task_index]}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            if task_index == 0:
                RD4AD_Trainer.single_model_training(model, epochs, train_dataloader, test_dataloader, device, use_wandb)
            else:
                model.set_old_tasks_teacher()
                model.to(device)

                if reset_student:
                    print("Resetting student")
                    model.reset_student(device)

                model.train()

                optimizer = torch.optim.Adam(
                    list(model.decoder.parameters())+list(model.bn.parameters()),
                    lr=0.0005,
                    betas=(0.5,0.999)
                )

                previous_tasks_losses = list()
                actual_tasks_losses = list()

                for epoch in trange(epochs):

                    for batch in tqdm(train_dataloader):

                        batch = batch.to(device)
                        enc_batch, bn_batch, dec_batch, old_dec_batch, old_bn_batch = model(batch)

                        actual_task_loss = 0
                        previous_tasks_loss = 0

                        # compute the loss for the actual task and for the previous loss
                        actual_task_loss = RD4AD_Trainer.loss_function(enc_batch, dec_batch)
                        previous_tasks_loss = pod_loss(enc_batch, bn_batch, dec_batch, old_dec_batch, old_bn_batch)

                        total_loss = actual_task_loss + lamb * previous_tasks_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    if use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch,
                                f"train_loss_{task_index}": total_loss,
                                f"actual_task_loss_{task_index}": actual_task_loss,
                                f"previous_tasks_loss_{task_index}": previous_tasks_loss,
                                f"train_loss": total_loss,
                                f"actual_task_loss": actual_task_loss,
                                f"previous_tasks_loss": previous_tasks_loss,
                            }
                        )

                #evaluate the model on all seen tasks
                for previous_task_index in tasks_stream.get_previous_tasks(task_index):
                    _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
                    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = Evaluator.evaluate_task(model, test_dataloader, device)
                    print(f"F1 pixel level on task {tasks_stream.categories[previous_task_index]}: {f1_pxl}")

                    if use_wandb:
                        wandb.log(
                            {
                                f"f1_pixel_{task_index}": f1_pxl,
                            }
                        )