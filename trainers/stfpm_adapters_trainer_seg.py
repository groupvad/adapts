from typing import Union
from typing import Union
from tqdm import *
import copy

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.stfpm_adapters import STFPMAdapters
from datasets.mvtec_dataset import MVTecDataset
from datasets.visa_dataset import VISADataset, VISA_CATEGORIES
from trainers.stfpm_cladapters_trainer import STFPMAdaptersTrainer
from utilities.evaluator import Evaluator
from utilities.configurations import WANDB_CONF
from cl_utils.task_stream import TaskStream
from cl_utils.strategies.ewc import *
from models.seg_utils import ASPP, BasicBlock, make_layer

class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        inplanes = 1792
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

class STFPMAdaptersTrainerSeg:

    def calcola_statistiche_pesi_pytorch(model):
        somma_totale = 0.0
        somma_assoluta_totale = 0.0
        numero_totale_elementi = 0

        # Itera attraverso tutti i tensori dei pesi (e bias) del modello
        for param in model.parameters():
            if param.requires_grad:
                # Aggiunge la somma dei pesi
                somma_totale += param.sum().item()
                # Aggiunge la somma dei valori assoluti
                somma_assoluta_totale += param.abs().sum().item()
                # Conta quanti singoli valori (scalari) ci sono nel tensore
                numero_totale_elementi += param.numel()

        if numero_totale_elementi == 0:
            return 0.0, 0.0

        media_semplice = somma_totale / numero_totale_elementi
        media_assoluta = somma_assoluta_totale / numero_totale_elementi

        return media_semplice, media_assoluta

    def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
        inputs = inputs.float()
        targets = targets.float()
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


    def l1_loss(inputs, targets, reduction="mean"):
        return F.l1_loss(inputs, targets, reduction=reduction)

    def single_model_training_noisy(
        model: STFPMAdapters,
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        category: str,
        device:torch.device,
        use_wandb: bool,
    ):

        # versione che allena congiuntamente adapters e segmentation network

        task_index = MVTecDataset.CATEGORIES.index(category) if category in MVTecDataset.CATEGORIES else VISA_CATEGORIES.index(category)
        model.reset_adapters()
        model.to(device)

        segmentation_net = nn.Sequential(
            nn.Conv2d(model.n_features, 1, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        lr = 0.1

        optimizer = torch.optim.SGD(list(model.parameters()) + list(segmentation_net.parameters()), lr=lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,     
            eta_min=0.001     
        )

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

        # best_pxl_roc = 0.0
        best_pxl_f1 = 0.0

        class_prototype = None
        class_prototypes = []

        mse_loss = torch.nn.MSELoss(reduction="sum")

        for epoch in trange(epochs):

            model.train()

            avg_loss = 0
            avg_adapters_loss = 0
            avg_segm_loss = 0

            for imgs_aug , imgs, masks in tqdm(train_dataloader):

                imgs = imgs.to(device)
                masks = masks.to(device)
                imgs_aug = imgs_aug.to(device)

                ## FORWARD WITH ONLY NORMAL IMAGES
                class_vectors, teacher_features, student_features = model(imgs)
                adapters_loss = 0
                for i in range(len(student_features)):
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)

                    height, width = teacher_features[i].shape[2:]
                    adapters_loss += (0.5 / (width * height)) * mse_loss(teacher_features[i], student_features[i])

                # FORWARD WITH AUGMENTED IMAGES
                _ , teacher_features, student_features = model(imgs_aug)

                upscale_dim = teacher_features[0].shape[2:]
                upscaled_diff = []

                for i in range(len(student_features)):
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)

                    diff = (teacher_features[i] - student_features[i]) ** 2

                    upscaled_diff.append(F.interpolate(diff, size=upscale_dim, mode="bilinear", align_corners=False))

                upscaled_diff = torch.cat(upscaled_diff, dim=1)

                output_segmentation = segmentation_net(upscaled_diff)

                masks = F.interpolate(
                    masks,
                    size=output_segmentation.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                masks = torch.where(
                    masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks)
                )

                focal_loss_val = STFPMAdaptersTrainerSeg.focal_loss(output_segmentation, masks, gamma=4)
                l1_loss_val = STFPMAdaptersTrainerSeg.l1_loss(output_segmentation, masks)
                loss = focal_loss_val + l1_loss_val + adapters_loss
                avg_loss += loss.item()
                avg_adapters_loss += adapters_loss.item()
                avg_segm_loss += (focal_loss_val.item() + l1_loss_val.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == 0:
                    class_prototypes.append(class_vectors)

            avg_loss /= len(train_dataloader)
            print(f"Epoch: {epoch}")
            print(f"AVG loss: {avg_loss}")
            print(f"AVG adapters loss: {avg_adapters_loss / len(train_dataloader)}")
            print(f"AVG segm loss: {avg_segm_loss / len(train_dataloader)}")
            
            scheduler.step()
    
            # Print the learning rate for the current epoch to see it decay
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:2d} | Learning Rate: {current_lr:.6f}")
            
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

            #evaluate the model, we are during the training so no need to load again the adapters
            model.is_eval_during_training = True
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=True)

            print(f"Epoch: {epoch}")
            print(f"Image-level AUROC: {img_roc}")
            print(f"Pixel-level AUROC: {pxl_roc}")
            print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            print(f"Pixel-level Pro: {pxl_pro}")

            media_semplice, media_assoluta = STFPMAdaptersTrainer.calcola_statistiche_pesi_pytorch(model)

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/train/epoch": epoch,
                        f"Task_T{task_index}/train/media_semplice_pesi": media_semplice,
                        f"Task_T{task_index}/train/media_assoluta_pesi": media_assoluta,
                        f"Task_T{task_index}/train/loss": avg_loss,
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


    def single_model_training_noisy_v1(
        model: STFPMAdapters,
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        category: str,
        device:torch.device,
        use_wandb: bool,
    ):

        # versione che allena per tot epochs gli adapters, poi continua ad allenarli usando il segnale
        # prodotto dalla rete di segmentazione, che però è molto semplice e dovrebbe quindi obbligare lo student
        # ad essere preciso nelle diff tra features

        task_index = MVTecDataset.CATEGORIES.index(category)
        model.reset_adapters()
        model.to(device)

        segmentation_net = nn.Sequential(
            nn.Conv2d(1792, 1, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        lr_adapters = 0.1
        lr_res = 0.1
        lr_seghead = 0.01

        adpaters_optimizer = torch.optim.SGD(list(model.parameters()), lr=lr_adapters)
        seg_optimizer = torch.optim.SGD(
            [
                {"params": segmentation_net.parameters(), "lr": lr_seghead},
                {"params": model.parameters()}
            ],
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )

        for n,p in model.named_parameters():
            if p.requires_grad:
                print(f"Parameter name: {n} - requires_grad: {p.requires_grad}")

        if use_wandb:
            wandb.config.update({
                "epochs": epochs,
                "batch_size": train_dataloader.batch_size,
                "learning_rate_adapters": lr_adapters,
                "learning_rate_res": lr_res,
                "learning_rate_seghead": lr_seghead,
                "optimizer": "SGD"
            })

        # best_pxl_roc = 0.0
        best_pxl_f1 = 0.0

        mse_loss = torch.nn.MSELoss(reduction="sum")

        for epoch in trange(epochs):

            model.train()

            segmentation_avg_loss = 0
            adapters_avg_loss = 0

            if epoch < 1:
                # train the adapters
                for _ , batch, _ in tqdm(train_dataloader):

                    batch = batch.to(device)
                    class_vectors, teacher_features, student_features = model(batch)

                    adapters_loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)

                        height, width = teacher_features[i].shape[2:]
                        adapters_loss += (0.5 / (width * height)) * mse_loss(teacher_features[i], student_features[i])

                    adapters_avg_loss += adapters_loss.item()
                    adpaters_optimizer.zero_grad()
                    adapters_loss.backward()
                    adpaters_optimizer.step()
            else:
                #train the segmentation head
                for aug_images, _, aug_masks in tqdm(train_dataloader):

                    batch = aug_images.to(device)
                    masks = aug_masks.to(device)
                    class_vectors, teacher_features, student_features = model(batch)

                    upscale_dim = teacher_features[0].shape[2:]

                    upscaled_diff = []

                    adapters_loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)

                        diff = (teacher_features[i] - student_features[i]) ** 2

                        upscaled_diff.append(F.interpolate(diff, size=upscale_dim, mode="bilinear", align_corners=False))

                    upscaled_diff = torch.cat(upscaled_diff, dim=1)

                    output_segmentation = segmentation_net(upscaled_diff)

                    masks = F.interpolate(
                        masks,
                        size=output_segmentation.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    masks = torch.where(
                        masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks)
                    )

                    focal_loss_val = STFPMAdaptersTrainerSeg.focal_loss(output_segmentation, masks, gamma=4)
                    l1_loss_val = STFPMAdaptersTrainerSeg.l1_loss(output_segmentation, masks)
                    segmentation_loss = focal_loss_val + l1_loss_val
                    segmentation_avg_loss += segmentation_loss.item()

                    seg_optimizer.zero_grad()
                    segmentation_loss.backward()
                    seg_optimizer.step()

            segmentation_avg_loss /= len(train_dataloader)
            adapters_avg_loss /= len(train_dataloader)
            print(f"Epoch: {epoch}")
            print(f"AVG adapters loss: {adapters_avg_loss}")
            print(f"AVG segmentation loss: {segmentation_avg_loss}")

            #evaluate the model, we are during the training so no need to load again the adapters
            model.is_eval_during_training = True
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=False)

            print(f"Epoch: {epoch}")
            # print(f"Image-level AUROC: {img_roc}")
            # print(f"Pixel-level AUROC: {pxl_roc}")
            # print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            # print(f"Pixel-level Pro: {pxl_pro}")

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/train/epoch": epoch,
                        f"Task_T{task_index}/train/segmentation_loss": segmentation_avg_loss,
                        f"Task_T{task_index}/train/adapters_loss": adapters_avg_loss,
                        f"Task_T{task_index}/train/img_roc": img_roc,
                        f"Task_T{task_index}/train/pxl_roc": pxl_roc,
                        f"Task_T{task_index}/train/img_pr": img_pr,
                        f"Task_T{task_index}/train/pxl_pr": pxl_pr,
                        f"Task_T{task_index}/train/f1_img": f1_img,
                        f"Task_T{task_index}/train/f1_pxl": f1_pxl,
                        f"Task_T{task_index}/train/pxl_pro": pxl_pro,
                    }
                )

    def single_model_training_noisy_v0(
        model: STFPMAdapters,
        epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        category: str,
        device:torch.device,
        use_wandb: bool,
    ):
        # versione che allena per tot epochs gli adapters, poi continua ad allenarli usando il segnale
        # prodotto dalla rete di segmentazione

        task_index = MVTecDataset.CATEGORIES.index(category)
        model.reset_adapters()
        model.to(device)

        segmentation_net = SegmentationNet()
        segmentation_net.train()
        segmentation_net.to(device)

        lr_adapters = 0.1
        lr_res = 0.1
        lr_seghead = 0.01

        adpaters_optimizer = torch.optim.SGD(list(model.parameters()), lr=lr_adapters)
        seg_optimizer = torch.optim.SGD(
            [
                {"params": segmentation_net.res.parameters(), "lr": lr_res},
                {"params": segmentation_net.head.parameters(), "lr": lr_seghead},
                {"params": model.parameters()}
            ],
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )

        for n,p in model.named_parameters():
            if p.requires_grad:
                print(f"Parameter name: {n} - requires_grad: {p.requires_grad}")

        if use_wandb:
            wandb.config.update({
                "epochs": epochs,
                "batch_size": train_dataloader.batch_size,
                "learning_rate_adapters": lr_adapters,
                "learning_rate_res": lr_res,
                "learning_rate_seghead": lr_seghead,
                "optimizer": "SGD"
            })

        # best_pxl_roc = 0.0
        best_pxl_f1 = 0.0

        mse_loss = torch.nn.MSELoss(reduction="sum")

        for epoch in trange(epochs):

            model.train()

            segmentation_avg_loss = 0
            adapters_avg_loss = 0

            if epoch < 1:
                # train the adapters
                for _ , batch, _ in tqdm(train_dataloader):

                    batch = batch.to(device)
                    class_vectors, teacher_features, student_features = model(batch)

                    adapters_loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)

                        height, width = teacher_features[i].shape[2:]
                        adapters_loss += (0.5 / (width * height)) * mse_loss(teacher_features[i], student_features[i])

                    adapters_avg_loss += adapters_loss.item()
                    adpaters_optimizer.zero_grad()
                    adapters_loss.backward()
                    adpaters_optimizer.step()
            else:
                #train the segmentation head
                for aug_images, _, aug_masks in tqdm(train_dataloader):

                    batch = aug_images.to(device)
                    masks = aug_masks.to(device)
                    class_vectors, teacher_features, student_features = model(batch)

                    upscale_dim = teacher_features[0].shape[2:]

                    upscaled_diff = []

                    adapters_loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)

                        diff = (teacher_features[i] - student_features[i]) ** 2

                        upscaled_diff.append(F.interpolate(diff, size=upscale_dim, mode="bilinear", align_corners=False))

                    upscaled_diff = torch.cat(upscaled_diff, dim=1)

                    output_segmentation = segmentation_net(upscaled_diff)

                    masks = F.interpolate(
                        masks,
                        size=output_segmentation.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    masks = torch.where(
                        masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks)
                    )

                    focal_loss_val = STFPMAdaptersTrainerSeg.focal_loss(output_segmentation, masks, gamma=4)
                    l1_loss_val = STFPMAdaptersTrainerSeg.l1_loss(output_segmentation, masks)
                    segmentation_loss = focal_loss_val + l1_loss_val
                    segmentation_avg_loss += segmentation_loss.item()

                    seg_optimizer.zero_grad()
                    segmentation_loss.backward()
                    seg_optimizer.step()

            segmentation_avg_loss /= len(train_dataloader)
            adapters_avg_loss /= len(train_dataloader)
            print(f"Epoch: {epoch}")
            print(f"AVG adapters loss: {adapters_avg_loss}")
            print(f"AVG segmentation loss: {segmentation_avg_loss}")

            #evaluate the model, we are during the training so no need to load again the adapters
            model.is_eval_during_training = True
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, device, category, all_metrics=False)

            print(f"Epoch: {epoch}")
            # print(f"Image-level AUROC: {img_roc}")
            # print(f"Pixel-level AUROC: {pxl_roc}")
            # print(f"F1 Score Image-level: {f1_img}")
            print(f"F1 Score Pixel-level: {f1_pxl}")
            # print(f"Pixel-level Pro: {pxl_pro}")

            if use_wandb:
                wandb.log(
                    {
                        f"Task_T{task_index}/train/epoch": epoch,
                        f"Task_T{task_index}/train/segmentation_loss": segmentation_avg_loss,
                        f"Task_T{task_index}/train/adapters_loss": adapters_avg_loss,
                        f"Task_T{task_index}/train/img_roc": img_roc,
                        f"Task_T{task_index}/train/pxl_roc": pxl_roc,
                        f"Task_T{task_index}/train/img_pr": img_pr,
                        f"Task_T{task_index}/train/pxl_pr": pxl_pr,
                        f"Task_T{task_index}/train/f1_img": f1_img,
                        f"Task_T{task_index}/train/f1_pxl": f1_pxl,
                        f"Task_T{task_index}/train/pxl_pro": pxl_pro,
                    }
                )

    def continual_training(model:STFPMAdapters, epochs, tasks_stream: TaskStream, device, use_wandb: bool):

        for task_index in range(len(tasks_stream)):

            category = tasks_stream.categories[task_index]

            print(f"Training for task: {category}")

            train_dataloader, test_dataloader = tasks_stream.get_task_data(task_index)

            print("Len train dataset:", len(train_dataloader.dataset))
            print("Len test dataset:", len(test_dataloader.dataset))

            # train the adapters for the new task
            STFPMAdaptersTrainerSeg.single_model_training_noisy(model, epochs, train_dataloader, test_dataloader, category ,device, use_wandb)

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