from typing import Union
from typing import Union
from tqdm import tqdm, trange
import copy

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.stfpm import STFPM
from utilities.evaluator import Evaluator
from utilities.configurations import WANDB_CONF
from cl_utils.task_stream import TaskStream
from cl_utils.strategies.ewc import *

class STFPM_Trainer:

    def stfpm_loss(teacher_features, student_features):
        return torch.sum((teacher_features - student_features) ** 2, 1).mean()

    def stfpm_cosine_loss(teacher_features, student_features):
        # shape: B, C, H, W
        cosine_loss = torch.nn.CosineSimilarity()
        return torch.mean(
            1 - cosine_loss(
                student_features.reshape(student_features.shape[0], -1),
                teacher_features.reshape(teacher_features.shape[0], -1)
            )
        )


    def single_model_training(
        model: STFPM,
        loss_fn: Union[stfpm_cosine_loss, stfpm_loss],
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        device:torch.device,
        use_wandb: bool,
        category: str = None,
    ):

        model.train()
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.4) # 0.4

        # for n,p in model.named_parameters():
        #     if p.requires_grad:
        #         print(f"Parameter name: {n} - requires_grad: {p.requires_grad}")

        mse_loss = torch.nn.MSELoss(reduction="sum")

        for epoch in trange(epochs):

            model.train()

            #train the model
            for batch in tqdm(train_dataloader):

                batch = batch.to(device)
                teacher_features, student_features = model(batch)

                loss = 0
                for i in range(len(student_features)):
                    # da valutare la normalizzazione
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)
                    loss += loss_fn(teacher_features[i], student_features[i])

                    # height, width = teacher_features[i].shape[2:]
                    # loss += (0.5 / (width * height)) * mse_loss(teacher_features[i], student_features[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": loss,
                    }
                )

        #evaluate the model
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=True)

        if use_wandb:
            wandb.log(
                {
                    "loss": loss,
                    "img_roc": img_roc,
                    "pxl_roc": pxl_roc,
                    "f1_img": f1_img,
                    "f1_pxl": f1_pxl,
                    "img_pr": img_pr,
                    "pxl_pr": pxl_pr,
                    "pxl_pro": pxl_pro,
                }
            )



    def lwf_training(model:STFPM, epochs, lamb, reset_student:bool, tasks_stream: TaskStream, device, use_wandb: bool):

        for task_index in range(len(tasks_stream)):

            print(f"Training for task: {tasks_stream.categories[task_index]}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            if task_index == 0:
                STFPM_Trainer.single_model_training(model, epochs, train_dataloader, test_dataloader, device, use_wandb)
            else:
                model.set_old_tasks_teacher()
                model.to(device)

                if reset_student:
                    print("Resetting student")
                    model.reset_student(device)

                model.train()

                optimizer = torch.optim.SGD(model.student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)

                previous_tasks_losses = list()
                actual_tasks_losses = list()
                for epoch in trange(epochs):

                    for batch in tqdm(train_dataloader):

                        batch = batch.to(device)
                        old_tasks_teacher_features, teacher_features, student_features = model(batch)

                        actual_task_loss = 0
                        previous_tasks_loss = 0

                        for i in range(len(student_features)):
                            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                            student_features[i] = F.normalize(student_features[i], dim=1)
                            old_tasks_teacher_features[i] = F.normalize(old_tasks_teacher_features[i], dim=1)
                            actual_task_loss += STFPM_Trainer.stfpm_loss(teacher_features[i], student_features[i])
                            previous_tasks_loss += STFPM_Trainer.stfpm_loss(old_tasks_teacher_features[i], student_features[i])

                        # total loss minimization
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

                    previous_tasks_losses.append(previous_tasks_loss.item())
                    actual_tasks_losses.append(actual_task_loss.item())

                print(f"Epoch: {epoch} Previous tasks losses progression: {previous_tasks_losses}\n")
                print(f"Epoch: {epoch} Actual tasks losses progression: {actual_tasks_losses}\n")

                #evaluate the model on all seen tasks
                for previous_task_index in tasks_stream.get_previous_tasks(task_index):
                    _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
                    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = Evaluator.evaluate_task(model, test_dataloader, device)
                    print(f"F1 pixel level on task {tasks_stream.categories[previous_task_index]}: {f1_pxl}")

    def ewc_training(model:STFPM, epochs, tasks_stream: TaskStream, lamb, device, use_wandb: bool):

        # train the model on the first task
        for task_index in range(len(tasks_stream)):

            print(f"Training for task: {tasks_stream.categories[task_index]}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            if task_index == 0:
                STFPM_Trainer.single_model_training(model, epochs, train_dataloader, test_dataloader, device, use_wandb)

            else:
                model.train()

                # train with ewc for the next tasks #0.4
                #optimizer = torch.optim.SGD(model.student.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
                optimizer = torch.optim.Adam(model.student.parameters(), 0.01, weight_decay = 1e-4)

                for epoch in trange(epochs):

                    for batch in tqdm(train_dataloader):

                        batch = batch.to(device)

                        _, teacher_features, student_features = model(batch)

                        stfpm_loss = 0
                        for i in range(len(teacher_features)):
                            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                            student_features[i] = F.normalize(student_features[i], dim=1)
                            stfpm_loss += STFPM_Trainer.stfpm_loss(teacher_features[i], student_features[i])

                        ewc_loss_value = ewc_loss(model, lamb, estimated_fishers, estimated_means)

                        loss = ewc_loss_value #+ stfpm_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    print(f"Epoch: {epoch}, STFPM Loss: {stfpm_loss}")
                    print(f"Epoch: {epoch}, EWC Loss: {ewc_loss_value}")
                    print(f"Epoch: {epoch} Loss: {loss.item()}\n")

                #evaluate the model on all seen tasks
                for previous_task_index in tasks_stream.get_previous_tasks(task_index):
                    _, test_dataloader = tasks_stream.get_task_data(previous_task_index)
                    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = Evaluator.evaluate_task(model, test_dataloader, device)
                    print(f"F1 pixel level on task {tasks_stream.categories[previous_task_index]}: {f1_pxl}")


            # compute fisher and mean parameters for EWC loss
            estimated_means, estimated_fishers = estimate_ewc_params(model, STFPM_Trainer.stfpm_loss, train_dataloader, device)

    def lwf_podnet_training(model:STFPM, loss, epochs, lamb, reset_student:bool, tasks_stream: TaskStream, device, use_wandb: bool):

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

        def pod_loss(teacher_features, student_features):
            return pod_width_loss(teacher_features, student_features) + pod_height_loss(teacher_features, student_features)

        loss = STFPM_Trainer.stfpm_loss if loss == "mse" else STFPM_Trainer.stfpm_cosine_loss

        for task_index in range(len(tasks_stream)):

            print(f"Training for task: {tasks_stream.categories[task_index]}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            if task_index == 0:
                STFPM_Trainer.single_model_training(model, loss, epochs, train_dataloader, test_dataloader, device, use_wandb)
            else:
                model.set_old_tasks_teacher()
                model.to(device)

                if reset_student:
                    print("Resetting student")
                    model.reset_student(device)

                model.train()

                optimizer = torch.optim.SGD(model.student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)

                previous_tasks_losses = list()
                actual_tasks_losses = list()

                for epoch in trange(epochs):

                    for batch in tqdm(train_dataloader):

                        batch = batch.to(device)
                        old_tasks_teacher_features, teacher_features, student_features = model(batch)

                        actual_task_loss = 0
                        previous_tasks_loss = 0

                        for i in range(len(student_features)):
                            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                            student_features[i] = F.normalize(student_features[i], dim=1)
                            old_tasks_teacher_features[i] = F.normalize(old_tasks_teacher_features[i], dim=1)
                            actual_task_loss += STFPM_Trainer.stfpm_loss(teacher_features[i], student_features[i])
                            #previous_tasks_loss += STFPM_Trainer.stfpm_loss(old_tasks_teacher_features[i], student_features[i]) prev version
                            previous_tasks_loss += pod_loss(old_tasks_teacher_features[i], student_features[i])

                        # total loss minimization
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
