import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision, F1Score
from torchmetrics.classification import BinaryPrecisionRecallCurve
from tqdm import tqdm
import wandb

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.destseg import DeSTSeg
from model.metrics import AUPRO, IAPS

warnings.filterwarnings("ignore")

def compute_f1_max(prc_metric):
    precision, recall, thresholds = prc_metric.compute()

    # Calculate F1 score for all threshold points
    # Added 1e-10 to the denominator to prevent division by zero
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)

    # Find the index of the maximum F1 score
    max_f1_idx = torch.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]

    # Note: torchmetrics returns precision/recall of length N+1 and thresholds of length N.
    # We cap the index to avoid an IndexError if the max F1 occurs at the very end.
    best_threshold_idx = min(max_f1_idx, len(thresholds) - 1)
    best_threshold = thresholds[best_threshold_idx]

    return max_f1, best_threshold


def evaluate(args, task_index, task_stream, model, visualizer, global_step=0):
    model.eval()

    metrics = [
        "DeSTSeg_IAP",
        "DeSTSeg_AUPRO",
        "DeSTSeg_AP",
        "DeSTSeg_AUC",
        "DeSTSeg_detect_AUC",
        "DeSTSeg_detect_F1",
        "DeSTSeg_F1"
    ]

    # Evaluate on all previous tasks and get summary metrics
    summary_metrics = { metric: [] for metric in metrics }

    # eval on all previous tasks
    for task_index in task_stream.get_previous_tasks(task_index):

        print("Evaluating on Task", task_index)
        test_dataloader = task_stream.get_task_data(task_index)[1]  # Get the test dataloader for the task

        with torch.no_grad():
            seg_IAPS = IAPS().cuda()
            seg_AUPRO = AUPRO().cuda()
            seg_AUROC = AUROC().cuda()
            seg_AP = AveragePrecision().cuda()
            seg_detect_AUROC = AUROC().cuda()
            seg_detect_PRC = BinaryPrecisionRecallCurve().cuda()
            seg_PRC = BinaryPrecisionRecallCurve().cuda()

            for sample_batched in tqdm(test_dataloader):
                img = sample_batched["img"].cuda()
                mask = sample_batched["mask"].to(torch.int64).cuda()

                output_segmentation, output_de_st, output_de_st_list = model(img)

                output_segmentation = F.interpolate(
                    output_segmentation,
                    size=mask.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                output_de_st = F.interpolate(
                    output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
                )

                mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
                output_segmentation_sample, _ = torch.sort(
                    output_segmentation.view(output_segmentation.size(0), -1),
                    dim=1,
                    descending=True,
                )
                output_segmentation_sample = torch.mean(
                    output_segmentation_sample[:, : args.T], dim=1
                )
                output_de_st_sample, _ = torch.sort(
                    output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
                )
                output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)

                seg_IAPS.update(output_segmentation, mask)
                seg_AUPRO.update(output_segmentation, mask)
                seg_AP.update(output_segmentation.flatten(), mask.flatten())
                seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
                seg_detect_AUROC.update(output_segmentation_sample, mask_sample)
                seg_detect_PRC.update(output_segmentation_sample, mask_sample)
                seg_PRC.update(output_segmentation.flatten(), mask.flatten())

            iap_seg, _ = seg_IAPS.compute()
            aupro_seg, ap_seg, auc_seg, auc_detect_seg = (
                seg_AUPRO.compute(),
                seg_AP.compute(),
                seg_AUROC.compute(),
                seg_detect_AUROC.compute(),
            )

            # Replace the old f1_detect and f1_seg compute lines with:
            f1_detect_max, best_detect_thresh = compute_f1_max(seg_detect_PRC)
            f1_seg_max, best_seg_thresh = compute_f1_max(seg_PRC)

            visualizer.add_scalar("DeSTSeg_IAP", iap_seg, global_step)
            visualizer.add_scalar("DeSTSeg_AUPRO", aupro_seg, global_step)
            visualizer.add_scalar("DeSTSeg_AP", ap_seg, global_step)
            visualizer.add_scalar("DeSTSeg_AUC", auc_seg, global_step)
            visualizer.add_scalar("DeSTSeg_detect_AUC", auc_detect_seg, global_step)
            visualizer.add_scalar("DeSTSeg_detect_F1", f1_detect_max, global_step)
            visualizer.add_scalar("DeSTSeg_F1", f1_seg_max, global_step)

            wandb.log(
                {
                    f"Task_T{task_index}/eval/DeSTSeg_IAP": iap_seg,
                    f"Task_T{task_index}/eval/DeSTSeg_AUPRO": aupro_seg,
                    f"Task_T{task_index}/eval/DeSTSeg_AP": ap_seg,
                    f"Task_T{task_index}/eval/DeSTSeg_AUC": auc_seg,
                    f"Task_T{task_index}/eval/DeSTSeg_detect_AUC": auc_detect_seg,
                    f"Task_T{task_index}/eval/DeSTSeg_detect_F1": f1_detect_max,
                    f"Task_T{task_index}/eval/DeSTSeg_F1": f1_seg_max,

                },
            )

            print("Evaluation results for Task", task_index)
            print("================================")
            print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
            print("pixel_AUC:", round(float(auc_seg), 4))
            print("pixel_AP:", round(float(ap_seg), 4))
            print("PRO:", round(float(aupro_seg), 4))
            print("image_AUC:", round(float(auc_detect_seg), 4))
            print("IAP:", round(float(iap_seg), 4))
            print("F1:", round(float(f1_seg_max), 4))
            print("Detect F1:", round(float(f1_detect_max), 4))
            print()

            summary_metrics["DeSTSeg_IAP"].append(iap_seg)
            summary_metrics["DeSTSeg_AUPRO"].append(aupro_seg)
            summary_metrics["DeSTSeg_AP"].append(ap_seg)
            summary_metrics["DeSTSeg_AUC"].append(auc_seg)
            summary_metrics["DeSTSeg_detect_AUC"].append(auc_detect_seg)
            summary_metrics["DeSTSeg_detect_F1"].append(f1_detect_max)
            summary_metrics["DeSTSeg_F1"].append(f1_seg_max)


            seg_IAPS.reset()
            seg_AUPRO.reset()
            seg_AUROC.reset()
            seg_AP.reset()
            seg_detect_AUROC.reset()
            seg_detect_PRC.reset()
            seg_PRC.reset()

    print(f"Summary metrics after training on task {task_index}:")
    for metric_name, values in summary_metrics.items():
        avg_value = sum(values) / len(values)
        summary_metrics[metric_name] = avg_value
        print(f"Average {metric_name}: {avg_value}")

    wandb.log(
        {
            f"Summary/{metric}": summary_metrics[metric] for metric in summary_metrics.keys()
        }
    )