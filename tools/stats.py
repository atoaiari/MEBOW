# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Chenyan Wu (czw390@psu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from enum import auto
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from lib.config import cfg
from lib.config import update_config

from lib.core.loss import JointsMSELoss
from lib.core.loss import DepthLoss
from lib.core.loss import hoe_diff_loss
from lib.core.loss import Bone_loss

from lib.core.function import train
from lib.core.function import validate

from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

import dataset
import models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics")

    # general
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        # required=True,
        type=str,
        default="experiments/coco/segm-4_lr1e-3.yaml",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # philly
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    parser.add_argument(
        "--subset", help="Use full dataset or not: 1 || 0", type=int, default=1
    )
    parser.add_argument(
        "--plot_hists", help="Plot histograms: 1 || 0", type=int, default=1
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    subset = True if args.subset == 1 else False
    plot_hists = True if args.plot_hists == 1 else False

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg=cfg,
        coco_root=cfg.DATASET.TRAIN_ROOT,
        annot_root=cfg.DATASET.ANNOT_ROOT,
        is_train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg=cfg,
        coco_root=cfg.DATASET.TRAIN_ROOT,
        annot_root=cfg.DATASET.ANNOT_ROOT,
        is_train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if subset:
        torch.manual_seed(21)
        np.random.seed(21)
        sub_idx_path = os.path.join(cfg.DATASET.ANNOT_ROOT, "subset_idx.pt")
        if os.path.isfile(
            sub_idx_path
        ):  # if the idx has been saved before then load it
            sub_idx = torch.load(sub_idx_path)
        else:
            sub_pct = 0.35
            sub_idx = torch.randint(
                0, len(train_dataset), (int(len(train_dataset) * sub_pct),)
            )
            torch.save(sub_idx, sub_idx_path)  # save indices to have a fixed idx

        train_dataset = torch.utils.data.Subset(train_dataset, sub_idx)  # Load subset

    # Basic stats
    print("*" * 100)
    print("Dataset statistics\n")
    total = len(train_dataset) + len(valid_dataset)
    pct_train = round(len(train_dataset) / total, 3)
    pct_test = round(len(valid_dataset) / total, 3)
    print(f"Train-test ratio: {pct_train} - {pct_test}")

    
    dataset_dict = {"train": train_dataset, "test": valid_dataset}
    # dataset_dict = { "test": valid_dataset}
    for dset in dataset_dict:
        print(f"\nCalculating {dset} statistics")
        data = dataset_dict[dset]
        box_x0, box_y0, box_w, box_h, box_size, bins = [], [], [], [], [], []
        for item in tqdm(data, total=len(data)):
            _, _, _, degree, _, bbox = item
            x, y, w, h = bbox
            sz = (w * h) ** 0.5

            # Append res
            box_x0.append(x)
            box_y0.append(y)
            box_w.append(w)
            box_h.append(h)
            box_size.append(sz)
            bins.append(degree.argmax())

        # Plot and present res
        stat_dict = {
            "box_x0": box_x0,
            "box_y0": box_y0,
            "box_w": box_w,
            "box_h": box_h,
            "box_size": box_size,
            "bins": bins,
        }

        for stat in stat_dict:
            stat_data = stat_dict[stat]  # Select list containing values

            # Optional histogram to visualize distribution
            if plot_hists:
                plt.hist(stat_data, bins="auto")
                savename = None
                title = None
                if dset == "train":
                    savename = f"{dset}_{stat}_hist_sub{subset}.png"
                    title = f"Distribution of {stat} over {dset}_sub{subset}.png"
                else:
                    savename = f"{dset}_{stat}_hist_sub.png"
                    title = f"Distribution of {stat} over {dset}.png"

                plt.title(title)
                plt.savefig(f"tools/stat_plots/{savename}")
                plt.close()

            # Descriptive stats
            print(f"Descriptive statistcs for {stat} over {dset}")
            mu, med, sig, rng = (
                round(np.mean(stat_data), 2),
                round(np.median(stat_data), 2),
                round(np.std(stat_data), 2),
                (round(np.min(stat_data), 2), round(np.max(stat_data), 2)),
            )
            print(f"Mean {mu}, Median {med}, Std {sig}, Range {rng}")


if __name__ == "__main__":
    main()
