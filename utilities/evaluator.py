from __future__ import annotations
import os
from typing import Optional
import pandas as pd
import logging
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utilities.metrics import *
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    @staticmethod
    def evaluate_task(model, test_dataloader, device, category = None, all_metrics=False, task_index: int = None):
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            output_path (str): path where to store the detection results
        """

        model.eval()

        task_ids = []

        # Initialize results.
        gt_masks_list, true_img_scores = (list(), list())
        pred_masks, pred_img_scores = (list(), list())

        for images, labels, masks, _ in tqdm(test_dataloader):
            # get anomaly map and score
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model(images.to(device), category=category)

                if task_index is not None:
                    task_ids.extend(model.loadeded_adapters_ids)

            if len(anomaly_maps.shape) == 3:
                anomaly_maps = anomaly_maps.squeeze(0)

            if anomaly_maps.shape[2:] != masks.shape[2:]:
                raise Exception(
                    "The output anomaly maps should have the same resolution as the target masks."
                    + f"Expected shape: {masks.shape[2:]}, got: {anomaly_maps.shape[2:]}"
                )

            # add true masks and img anomaly scores
            gt_masks_list.extend(masks.cpu().numpy().astype(int))
            true_img_scores.extend(labels.cpu().numpy())

            # add predicted masks and img anomaly scores (check for numpy arrays or tensors)
            if isinstance(anomaly_maps, torch.Tensor):
                pred_masks.extend(anomaly_maps.cpu().numpy())
                pred_img_scores.extend(anomaly_scores.cpu().numpy())
            else:
                pred_masks.extend(anomaly_maps)
                pred_img_scores.extend(anomaly_scores)

        gt_masks_list = np.asarray(gt_masks_list)
        true_img_scores = np.asarray(true_img_scores)
        pred_masks = np.asarray(pred_masks)
        pred_img_scores = np.asarray(pred_img_scores)
        pred_masks = min_max_norm(pred_masks)

        if task_index is not None:
            task_ids = np.array(task_ids)

        if all_metrics:


            # """Image-level AUROC"""
            fpr, tpr, img_roc_auc = cal_img_roc(pred_img_scores, true_img_scores)

            # """Pixel-level AUROC"""
            fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_masks_list, pred_masks)

            # """F1 Score Image-level"""
            f1_img = cal_f1_img(pred_img_scores, true_img_scores)

            """F1 Score Pixel-level"""
            f1_pxl = cal_f1_pxl(pred_masks, gt_masks_list)

            # """Image-level PR-AUC"""
            pr_auc_img = cal_pr_auc_img(pred_img_scores, true_img_scores)

            # # """Pixel-level PR-AUC"""
            pr_auc_pxl = cal_pr_auc_pxl(pred_masks, gt_masks_list)

            # """Pixel-level AU-PRO"""
            au_pro_pxl = cal_pro_auc_pxl(np.squeeze(pred_masks, axis=1), gt_masks_list)

            if task_index is not None:
                """Task Classification Accuracy"""
                task_classification_accuracy = (task_ids == task_index).sum() / len(task_ids)
            else:
                task_classification_accuracy = None

            return (
                img_roc_auc,
                per_pixel_rocauc,
                f1_img,
                f1_pxl,
                pr_auc_img,
                pr_auc_pxl,
                au_pro_pxl,
                task_classification_accuracy
            )

        else:
            """F1 Score Pixel-level"""
            f1_pxl = cal_f1_pxl(pred_masks, gt_masks_list)

            return (
                0,
                0,
                0,
                f1_pxl,
                0,
                0,
                0,
                0,
            )
