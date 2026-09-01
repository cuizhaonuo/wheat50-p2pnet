import argparse
import datetime
import math
import os
import random
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
from util.repository import get_shared_repository_root, resolve_shared_path

warnings.filterwarnings('ignore', category=FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=[3500], type=int, nargs='+',
                        help='epoch(s) at which to drop LR by 0.1. Pass multiple values for multi-step.')
    parser.add_argument('--lr_schedule', default='multistep', choices=('multistep', 'cosine'),
                        help='decay applied after the warmup, either the --lr_drop milestones '
                             'or a cosine decay to zero over the remaining epochs')
    parser.add_argument('--lr_warmup_epochs', default=0, type=int,
                        help='epochs of linear warmup before the decay starts')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use, either a VGG "
                             "variant or one of dinov3_convnext_{tiny,small,base,large}")
    parser.add_argument('--backbone_weights', default='', type=str,
                        help="checkpoint for the DINOv3 ConvNeXt backbone, accepting "
                             "either the Hugging Face or the upstream naming")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="train the detection head on top of a frozen backbone")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA',
                        help='dataset adapter: SHHA or COCO')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    parser.add_argument('--coco_train_annotations',
                        default='annotations/instances_train2017.json')
    parser.add_argument('--coco_val_annotations',
                        default='annotations/instances_val2017.json')
    parser.add_argument('--coco_train_images', default='train2017')
    parser.add_argument('--coco_val_images', default='val2017')
    parser.add_argument('--coco_category_name', default='wheat_2')
    parser.add_argument('--coco_crop_size', default=512, type=int)
    parser.add_argument('--coco_crops_per_image', default=1, type=int)
    parser.add_argument('--coco_scale_min', default=0.7, type=float)
    parser.add_argument('--coco_scale_max', default=1.3, type=float)
    parser.add_argument('--coco_hflip_probability', default=0.5, type=float)
    parser.add_argument(
        '--coco_copy_paste_probability', default=0.0, type=float,
        help='probability of applying instance Copy-Paste to a training crop')
    parser.add_argument(
        '--coco_copy_paste_max_instances', default=3, type=int,
        help='maximum number of instances pasted into each selected crop')
    parser.add_argument(
        '--coco_copy_paste_placement_attempts', default=20, type=int,
        help='maximum zero-overlap placement attempts per pasted instance')
    
    parser.add_argument('--experiment_name', default='',
                        help='optional subdirectory name shared by all artifact paths')
    parser.add_argument('--output_dir', default='logs',
                        help='log directory relative to the main worktree')
    parser.add_argument('--checkpoints_dir', default='checkpoints',
                        help='checkpoint directory relative to the main worktree')
    parser.add_argument('--tensorboard_dir', default='runs',
                        help='TensorBoard directory relative to the main worktree')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_threshold', default=0.5, type=float,
                        help='score above which an anchor point is counted; its best value '
                             'moves with the training configuration, see '
                             'tools/sweep_eval_threshold.py')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser


def build_param_dicts(model, args):
    """Split the parameters by backbone membership and by decay eligibility.

    Weight decay is skipped for biases and for the one-dimensional weights of
    the normalization layers, where decaying towards zero shrinks activations
    rather than regularizing the fit.
    """
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = 'backbone' in name
        decay = parameter.ndim > 1
        groups.setdefault((backbone, decay), []).append(parameter)
    param_dicts = []
    for (backbone, decay), parameters in sorted(groups.items()):
        param_dicts.append({
            'params': parameters,
            'lr': args.lr_backbone if backbone else args.lr,
            'weight_decay': args.weight_decay if decay else 0.0,
        })
    return param_dicts


def build_lr_scheduler(optimizer, args):
    """Linear warmup followed by the configured decay.

    With no warmup and the multistep schedule this reproduces the MultiStepLR
    with gamma 0.1 that the earlier experiments used.
    """
    warmup = max(0, args.lr_warmup_epochs)

    def factor(epoch):
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if args.lr_schedule == 'cosine':
            progress = (epoch - warmup) / max(1, args.epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 0.1 ** sum(epoch >= milestone for milestone in args.lr_drop)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    shared_root = get_shared_repository_root(__file__)
    for attribute in (
        'output_dir',
        'checkpoints_dir',
        'tensorboard_dir',
    ):
        setattr(
            args,
            attribute,
            resolve_shared_path(
                getattr(args, attribute), shared_root, args.experiment_name
            ),
        )
    for directory in (
        args.output_dir,
        args.checkpoints_dir,
        args.tensorboard_dir,
    ):
        if directory:
            os.makedirs(directory, exist_ok=True)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = build_param_dicts(model_without_ddp, args)
    # AdamW with weight_decay=0 is Adam, which is what the earlier runs used
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    lr_scheduler = build_lr_scheduler(optimizer, args)
    # create the dataset
    train_set, val_set = build_dataset(args=args)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    try:
        from tensorboardX import SummaryWriter
    except ImportError as error:
        raise ImportError(
            'tensorboardX is required for training; install requirements.txt'
        ) from error
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(
                    epoch, stat['loss_ce']))
                log_file.write("loss/loss_point@{}: {}".format(
                    epoch, stat['loss_point']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)
            writer.add_scalar('loss/loss_point', stat['loss_point'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device,
                                               threshold=args.eval_threshold)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0], 
                                result[1], t2 - t1, np.min(mae)))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
