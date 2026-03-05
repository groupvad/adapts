import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY_MVTEC
from data.visa_dataset import VISA_CATEGORIES
from data.task_stream import TaskStream
from data.mvtec_dataset import MVTecDataset
from eval_joint import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss

warnings.filterwarnings("ignore")


def train(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_joint"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    task_stream = TaskStream(
        dataset_path=args.mvtec_path,
        dtd_path=args.dtd_path,
        categories=VISA_CATEGORIES,
        batch_size=args.bs
    )

    train_dataloader, test_dataloader = task_stream.get_all_tasks_data()

    print("Length of train dataset:", len(train_dataloader.dataset))
    print("Length of test dataset:", len(test_dataloader.dataset))

    global_step = 0

    flag = True

    while flag:
        for sample_batched in tqdm(train_dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation, output_de_st, output_de_st_list = model(
                img_aug, img_origin
            )

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()
                de_st_optimizer.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            wandb.log(
                {
                    "cosine_loss": cosine_loss_val,
                    "focal_loss": focal_loss_val,
                    "l1_loss": l1_loss_val,
                    "total_loss": total_loss_val,
                }
            )

            if global_step % args.eval_per_steps == 0:
                evaluate(args, test_dataloader, model, visualizer, global_step)

            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                    )

            if global_step >= args.steps:
                flag = False
                break

    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="/home/u0052/disk/datasets/visa/")
    parser.add_argument("--dtd_path", type=str, default="/home/u0052/disk/datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3000)#5000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=50)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        wandb.init(project="stfpm_adapters", name=f"destseg_joint_visa", reinit=True)
        print("Training DestSeg on multi task setting")
        train(args)
