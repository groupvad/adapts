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

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY_MVTEC
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


def evaluate(args, test_dataloader, model, visualizer, global_step=0):
    model.eval()
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
            # seg_detect_F1.update(output_segmentation_sample, mask_sample)
            # seg_F1.update(output_segmentation.flatten(), mask.flatten())
            # Replace the old seg_detect_F1 and seg_F1 updates with:
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
                "DeSTSeg_IAP": iap_seg,
                "DeSTSeg_AUPRO": aupro_seg,
                "DeSTSeg_AP": ap_seg,
                "DeSTSeg_AUC": auc_seg,
                "DeSTSeg_detect_AUC": auc_detect_seg,
                "DeSTSeg_detect_F1": f1_detect_max,
                "DeSTSeg_F1": f1_seg_max,

            },
        )

        print("Eval at step", global_step)
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

        seg_IAPS.reset()
        seg_AUPRO.reset()
        seg_AUROC.reset()
        seg_AP.reset()
        seg_detect_AUROC.reset()
        seg_detect_PRC.reset()
        seg_PRC.reset()



def test(args, category):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"DeSTSeg_MVTec_test_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    assert os.path.exists(
        os.path.join(args.checkpoint_path, args.base_model_name + category + ".pckl")
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.checkpoint_path, args.base_model_name + category + ".pckl"
            )
        )
    )

    evaluate(args, category, model, visualizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="./datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_5000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    args = parser.parse_args()

    obj_list = args.category
    for obj in obj_list:
        assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            test(args, obj)
