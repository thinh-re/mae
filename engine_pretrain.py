import math
import sys
from typing import Iterable, Optional

import torch
import wandb
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import util.lr_sched as lr_sched
import util.misc as misc
from util.argparsers import PreTrainArgumentParser
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def train_one_epoch(
    global_rank: int,
    model: torch.nn.Module,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler: NativeScaler,
    log_writer: Optional[SummaryWriter] = None,
    args: Optional[PreTrainArgumentParser] = None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if global_rank == 0 and log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples: Tensor = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            rs = model(samples, mask_ratio=args.mask_ratio)
            loss: Tensor = rs[0]

        loss_value: float = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss, optimizer, parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if global_rank == 0 and log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            wandb.log({
                'train_loss': loss_value_reduce,
                'lr': lr,
            }, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if global_rank == 0:
        print("Averaged stats:", metric_logger)
