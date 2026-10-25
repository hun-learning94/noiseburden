import argparse
import datetime
import json
import numpy as np
import os
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--patch_embed', default='bottleneck', choices=['bottleneck', 'linear'],
                        help='bottleneck: released low-rank embedding; linear: single patch projection')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--pred_space', default='x', type=str, choices=['x', 'eps', 'v'],
                        help='Space the network output is interpreted in (Table 2)')
    parser.add_argument('--loss_space', default='v', type=str, choices=['x', 'eps', 'v'],
                        help='Space the l2 loss is computed in (Table 2)')
    parser.add_argument('--loss_impl', default='direct', choices=['direct', 'weighted_v'],
                        help='direct: existing x-hat conversions; weighted_v: pl229 loss and sampling conversions')
    parser.add_argument('--output_shortcut', default='none', choices=['none', 'learned_a'],
                        help='Optional output-only learned A(t)*z bypass for eps prediction')
    parser.add_argument('--max_steps', default=0, type=int,
                        help='Benchmark mode: stop after this many steps (0 = full training)')
    parser.add_argument('--bench_warmup', default=20, type=int,
                        help='Steps excluded from benchmark timing')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--eval_completed_epochs', type=int, nargs='*', default=None,
                        help='Explicit completed-epoch evaluation milestones; overrides eval_freq')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--exclude_image_manifest', default='', type=str,
                        help='JSON with image paths relative to train/ reserved for diagnostics')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency (in epochs) to keep numbered checkpoints')
    parser.add_argument('--ema_snapshot_epochs', type=int, nargs='*', default=[],
                        help='Completed epochs at which to save compact EMA1-only snapshots')
    parser.add_argument('--save_rng_state', action='store_true',
                        help='Save/restore per-rank RNG state in rolling checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    if args.exclude_image_manifest:
        manifest = json.loads(Path(args.exclude_image_manifest).read_text())
        excluded = set(manifest['images'])
        before = len(dataset_train.samples)
        train_root = Path(args.data_path) / 'train'
        excluded_paths = {str(train_root / name) for name in excluded}
        dataset_train.samples = [
            (name, label) for name, label in dataset_train.samples
            if name not in excluded_paths
        ]
        if before - len(dataset_train.samples) != len(excluded):
            raise ValueError("Not all reserved diagnostic images were found in the training dataset")
        dataset_train.imgs = dataset_train.samples
        dataset_train.targets = [label for _, label in dataset_train.samples]
        print("Reserved diagnostic images:", len(excluded))
    assert len(dataset_train) == 1281167, len(dataset_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size() * args.grad_accum_steps
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        saved_args = checkpoint.get('args')
        if saved_args is None:
            raise ValueError("Checkpoint has no args; cannot verify experiment configuration.")
        for key, default in [('patch_embed', 'bottleneck'), ('pred_space', 'x'),
                             ('loss_space', 'v'), ('loss_impl', 'direct'),
                             ('output_shortcut', 'none')]:
            saved_value = getattr(saved_args, key, default)
            if saved_value != getattr(args, key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"checkpoint={saved_value!r}, requested={getattr(args, key)!r}"
                )
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        if args.save_rng_state:
            states = checkpoint.get('rng_states')
            if states is not None and len(states) == num_tasks:
                state = states[global_rank]
                torch.set_rng_state(state['torch'])
                torch.cuda.set_rng_state(state['cuda'], device)
                np.random.set_state(state['numpy'])
                random.setstate(state['python'])
                print("RESTORED_RANK_RNG", global_rank, force=True)
            else:
                assert checkpoint['epoch'] == 39, 'Unexpected RNG migration after initial resume'
                continuation_seed = 20260929 + global_rank
                torch.manual_seed(continuation_seed)
                torch.cuda.manual_seed(continuation_seed)
                np.random.seed(continuation_seed)
                random.seed(continuation_seed)
                print("INITIALIZED_CONTINUATION_RNG", global_rank, continuation_seed, force=True)
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    if (args.online_eval and args.start_epoch > 0
            and args.eval_completed_epochs is not None
            and args.start_epoch in args.eval_completed_epochs):
        # A preemption may happen after saving the epoch but before its FID.
        # Experiment evaluators can skip an already verified metrics file.
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(seed)
            evaluate(model_without_ddp, args, args.start_epoch - 1,
                     batch_size=args.gen_bsz, log_writer=log_writer)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        if args.max_steps:
            break

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % args.save_freq == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        if epoch + 1 in args.ema_snapshot_epochs:
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch,
                epoch_name=f"ema1-ep{epoch + 1:03d}", ema_only=True,
            )

        # Perform online evaluation at specified intervals
        evaluate_now = (
            epoch + 1 in args.eval_completed_epochs
            if args.eval_completed_epochs is not None
            else epoch % args.eval_freq == 0 or epoch + 1 == args.epochs
        )
        if args.online_eval and evaluate_now:
            torch.cuda.empty_cache()
            with torch.random.fork_rng(), torch.no_grad():
                torch.manual_seed(seed)
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
