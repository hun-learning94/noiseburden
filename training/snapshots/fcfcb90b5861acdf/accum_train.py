"""JiT training with an optimizer/EMA update once per effective batch."""
from contextlib import nullcontext
import json
import math
import time

import torch

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device,
                    epoch, log_writer=None, args=None):
    model.train(True)
    accum = int(args.grad_accum_steps)
    if accum < 1 or len(data_loader) % accum:
        raise ValueError("Microbatch count must be divisible by accumulation; refuse a smaller final batch.")
    updates = len(data_loader) // accum
    max_steps = int(getattr(args, "max_steps", 0))
    if max_steps and max_steps % accum:
        raise ValueError("Benchmark microsteps must end on a complete optimizer update.")
    logger = misc.MetricLogger(delimiter="  ")
    logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    completed = 0
    for step, (x, labels) in enumerate(logger.log_every(data_loader, 20, f"Epoch: [{epoch}]")):
        if step % accum == 0:
            lr_sched.adjust_learning_rate(optimizer, (step // accum) / updates + epoch, args)
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255).mul_(2).sub_(1)
        labels = labels.to(device, non_blocking=True)
        update = (step + 1) % accum == 0
        sync = nullcontext() if update or not hasattr(model, "no_sync") else model.no_sync()
        with sync:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, labels)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, microstep {step}: {value}")
            (loss / accum).backward()
        if update:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_without_ddp.update_ema()
        torch.cuda.synchronize()
        logger.update(loss=value, lr=optimizer.param_groups[0]["lr"])
        mean_loss = misc.all_reduce_mean(value)
        if log_writer is not None and step % args.log_freq == 0:
            axis = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", mean_loss, axis)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], axis)
        completed += 1
        if max_steps and completed >= max_steps:
            break
    elapsed = time.monotonic() - started
    if misc.is_main_process():
        print("ACCUM_EPOCH_RESULT", json.dumps({
            "epoch": epoch, "microsteps": completed,
            "optimizer_updates": completed // accum,
            "effective_batch": args.batch_size * accum * misc.get_world_size(),
            "mean_training_loss": logger.meters["loss"].global_avg,
            "seconds": elapsed,
            "images_per_second": completed * args.batch_size * misc.get_world_size() / elapsed,
        }), flush=True)
