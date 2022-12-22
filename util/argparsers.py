import argparse
from typing import Optional

from tap import Tap

class PreTrainArgumentParser(Tap):
    batch_size: Optional[int] = 64 # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
    epochs: Optional[int] = 400 
    accum_iter: Optional[int] = 1 # Accumulate gradient iterations, (for increasing the effective batch size under memory constraints)
    
    # Model parameters
    model: Optional[str] = 'mae_vit_base_patch16' # Name of model to train
    input_size: Optional[int] = 224 # Images input size
    mask_ratio: Optional[float] = 0.75 # Masking ratio (percentage of removed patches)
    norm_pix_loss: Optional[bool] = False # Use (per-patch) normalized pixels as targets for computing loss
    
    # Optimizer parameters
    weight_decay: Optional[float] = 0.05 # weight decay
    lr: Optional[float] = None # learning rate (absolute lr)
    blr: Optional[float] = 1e-3 # base learning rate: absolute_lr = base_lr * total_batch_size / 256
    min_lr: Optional[float] = 0. # lower lr bound for cyclic schedulers that hit 0
    warmup_epochs: Optional[int] = 40 # epochs to warmup LR
    
    # Dataset parameters
    data_path: Optional[str] = '/kaggle/input/rgbdsod-set4/' # dataset path
    output_dir: Optional[str] = './output_dir' # path where to save, empty for no saving
    log_dir: Optional[str] = './output_dir' # path where to tensorboard log
    device: Optional[str] = 'cuda' # device to use for training / testing
    seed: Optional[int] = 0
    resume: Optional[str] = '' # resume from checkpoint
    
    start_epoch: Optional[int] = 0 
    num_workers: Optional[int] = 10
    pin_mem: Optional[bool] = True # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    no_pin_mem: Optional[bool] = False
    
    # Distributed training parameters
    world_size: Optional[int] = 1 # Number of distributed processes
    local_rank: Optional[int] = -1
    rank: Optional[int] = -1
    dist_on_itp: Optional[bool] = False
    dist_url: Optional[str] = 'env://' # Url used to set up distributed training
    distributed: Optional[bool] = False

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/', type=str,
    #                     help='dataset path')
    parser.add_argument('--data_path', default='/kaggle/input/rgbdsod-set4/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser
