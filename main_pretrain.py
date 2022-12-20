# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import datetime
import os
import time
from pathlib import Path

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import models_mae
import util.misc as misc
from engine_pretrain import train_one_epoch
from util.argparsers import PreTrainArgumentParser
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from wandb_manager import wandb_init, wandb_login


def main(args: PreTrainArgumentParser):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    base_lr = args.lr * 256 / eff_batch_size
    print(f"base lr: {base_lr:.2f}")
    print(f"actual lr: {args.lr:.2f}")

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler,
    )
    
    wandb_login()
    wandb_config = args.__dict__()
    wandb_config['base_lr'] = base_lr
    wandb_init('test', wandb_config)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(
            model, data_loader_train, optimizer, 
            device, epoch, loss_scaler, args=args,
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, 
                model_without_ddp=model_without_ddp, 
                optimizer=optimizer,
                loss_scaler=loss_scaler, 
                epoch=epoch,
            )

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    # args = get_args_parser()
    # args = args.parse_args()
    args = PreTrainArgumentParser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
