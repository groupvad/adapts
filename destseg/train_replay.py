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

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.task_stream import TaskStream
from data.mvtec_dataset import MVTecDataset
from data.replay_memory import Memory
from eval_replay import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss

warnings.filterwarnings("ignore")


def train(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_replay_{args.replay_size}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()
    memory = Memory(memory_size=args.replay_size)
    replay_ratio = 0.5

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
        categories=ALL_CATEGORY,
        batch_size=args.bs
    )

    for task_index in range(len(task_stream)):

        print(f"Training on Task {task_index}")

        train_dataloader, _ = task_stream.get_task_data(task_index)
        n_replay_samples = int(args.bs * replay_ratio)

        global_step = 0
        flag = True 

        while flag:
            for sample_batched in tqdm(train_dataloader):
                # 1. Extract the current batch tensors (usually on CPU from DataLoader)
                img_origin = sample_batched["img_origin"]
                img_aug = sample_batched["img_aug"]
                mask = sample_batched["mask"]

                if task_index > 0:
                    # 2. Get replay samples (unpacking the 3 returned tensors)
                    num_to_sample = min(n_replay_samples, img_aug.size(0))
                    mem_img_origin, mem_img_aug, mem_mask = memory.get_samples(num_to_sample)

                    num_retrieved = mem_img_origin.size(0)

                    if num_retrieved > 0:
                        # 3. Ensure memory samples are on the same device as the current batch
                        mem_img_origin = mem_img_origin.to(img_origin.device)
                        mem_img_aug = mem_img_aug.to(img_aug.device)
                        mem_mask = mem_mask.to(mask.device)

                        # 4. Select random indices to replace in the current batch
                        replace_idx = torch.randperm(img_aug.size(0))[:num_retrieved]

                        # 5. Overwrite the selected indices for all three modalities synchronously
                        img_origin[replace_idx] = mem_img_origin
                        img_aug[replace_idx] = mem_img_aug
                        mask[replace_idx] = mem_mask

                # Zero the optimizer gradients
                seg_optimizer.zero_grad()
                de_st_optimizer.zero_grad()

                # 6. Move the final mixed batches to the GPU for the forward pass
                img_origin = img_origin.cuda()
                img_aug = img_aug.cuda()
                mask = mask.cuda()

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

                if global_step >= args.steps:
                    flag = False
                    break

                visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
                visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
                visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
                visualizer.add_scalar("total_loss", total_loss_val, global_step)

                wandb.log(
                    {
                        f"Task_T{task_index}/train/cosine_loss": cosine_loss_val,
                        f"Task_T{task_index}/train/focal_loss": focal_loss_val,
                        f"Task_T{task_index}/train/l1_loss": l1_loss_val,
                        f"Task_T{task_index}/train/total_loss": total_loss_val,
                    }
                )

        print("End of Task Training, Evaluating on all seen tasks...")

        evaluate(args, task_index,task_stream, model, visualizer, global_step)

        for batch in train_dataloader:
            img_origin = batch["img_origin"]
            img_aug = batch["img_aug"]
            mask = batch["mask"]

            memory.add_samples(
                task_id=task_index,
                images=img_origin.detach().cpu(),
                aug_images=img_aug.detach().cpu(),
                masks=mask.detach().cpu()
            )

    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="/home/u0052/disk/datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="/home/u0052/disk/datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2000)#5000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--replay_size", default=100, type=int)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        wandb.init(project="stfpm_adapters", name=f"destseg_replay_{args.replay_size}", reinit=True)
        print("Training DestSeg with replay")
        train(args)
