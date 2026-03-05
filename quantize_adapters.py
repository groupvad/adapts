import pandas as pd
import os
import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from models.stfpm_adapters import LinearAdapter, STFPMAdapters,BottleneckAdapter
from datasets.mvtec_dataset import MVTecDataset, TaskType
from utilities.evaluator import Evaluator

def main():

    model_name = "wide_resnet50_2"
    layers_idx = ["layer1","layer2","layer3"]
    device = torch.device("cpu")
    dataset_path = ""

    pre_perf = {
        "category": [],
        "img_roc": [],
        "pxl_roc": [],
        "f1_img": [],
        "f1_pxl": [],
        "img_pr": [],
        "pxl_pr": [],
        "pxl_pro": []
    }

    post_perf = {
        "category": [],
        "img_roc": [],
        "pxl_roc": [],
        "f1_img": [],
        "f1_pxl": [],
        "img_pr": [],
        "pxl_pr": [],
        "pxl_pro": []
    }

    for category in MVTecDataset.CATEGORIES:

        # load the STFPM with adapters
        model = STFPMAdapters(model_name, layers_idx, LinearAdapter,(224,224), device, True)
        model.is_eval_during_training = True
        model.adapters_save_path = ""
        model.load_adapters_from_path(category)
        model.to(device)
        model.eval()

        test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, torch.device("cpu"), category, True)

        print(f"Model Evaluation on category: {category}q")
        print(f"Image-level AUROC: {img_roc}")
        print(f"Pixel-level AUROC: {pxl_roc}")
        print(f"F1 Score Image-level: {f1_img}")
        print(f"F1 Score Pixel-level: {f1_pxl}")
        print(f"Image-level Precision: {img_pr}")
        print(f"Pixel-level Precision: {pxl_pr}")
        print(f"Pixel-level Pro: {pxl_pro}")

        pre_perf["category"].append(category)
        pre_perf["img_roc"].append(img_roc)
        pre_perf["pxl_roc"].append(pxl_roc)
        pre_perf["f1_img"].append(f1_img)
        pre_perf["f1_pxl"].append(f1_pxl)
        pre_perf["img_pr"].append(img_pr)
        pre_perf["pxl_pr"].append(pxl_pr)
        pre_perf["pxl_pro"].append(pxl_pro)

        # calibration data
        dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train")
        calibration_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # quantize the adapters
        model.quantize_adapters(calibration_loader)

        # save the quantized adapters
        os.makedirs("./quantized_adapters/linear_quantized", exist_ok=True)
        model.adapters_save_path = "./quantized_adapters/linear_quantized"
        model.save_adapters(category)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro, _ = Evaluator.evaluate_task(model, test_dataloader, torch.device("cpu"), category, True)

        print(f"Quantized Model Evaluation on category: {category}")
        print(f"Image-level AUROC: {img_roc}")
        print(f"Pixel-level AUROC: {pxl_roc}")
        print(f"F1 Score Image-level: {f1_img}")
        print(f"F1 Score Pixel-level: {f1_pxl}")
        print(f"Image-level Precision: {img_pr}")
        print(f"Pixel-level Precision: {pxl_pr}")
        print(f"Pixel-level Pro: {pxl_pro}")

        post_perf["category"].append(category)
        post_perf["img_roc"].append(img_roc)
        post_perf["pxl_roc"].append(pxl_roc)
        post_perf["f1_img"].append(f1_img)
        post_perf["f1_pxl"].append(f1_pxl)
        post_perf["img_pr"].append(img_pr)
        post_perf["pxl_pr"].append(pxl_pr)
        post_perf["pxl_pro"].append(pxl_pro)

if __name__ == "__main__":
    main()