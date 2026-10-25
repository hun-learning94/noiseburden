"""Training and evaluation entry points for the frozen EDM Table 1 runs."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import main_jit
import util.misc as misc
import util.lr_sched as lr_sched
from edm import EDMDenoiser, EPOCHS, VARIANTS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def arguments(root, variant):
    a = main_jit.get_args_parser().parse_args([])
    a.edm_variant, a.sigma_data = variant, .5
    a.model, a.img_size, a.class_num = 'JiT-B/16', 256, 1000
    a.pred_type, a.loss_type, a.shortcut, a.schedule = 'x', 'x', 'none', 'linear'
    a.batch_size, a.grad_accum_steps, a.num_workers = 64, 8, 8
    a.lr, a.blr, a.lr_schedule, a.warmup_epochs = .0002, .00005, 'constant', 5
    a.epochs, a.save_last_freq, a.save_freq = 200, 1, 1000000
    a.cfg, a.interval_min, a.interval_max = 2.9, .1, 1.
    a.num_images, a.gen_bsz = 10000, 64
    a.sampling_method, a.num_sampling_steps = 'heun', 50
    a.online_eval = False
    a.data_path = os.environ['EDM_DATA_PATH']
    a.output_dir = a.resume = str(root/'train')
    return a


def verify_sources():
    bundle = Path(__file__).resolve().parent.parent
    manifest = json.loads((bundle/'sources.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if digest(bundle/relative) != expected:
            raise RuntimeError('Frozen source changed: '+relative)


def train(root, variant):
    a = arguments(root, variant)
    resume_rng = None
    cp = root/'train/checkpoint-last.pth'
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', mmap=True, weights_only=False)
        assert saved['args'].edm_variant == variant
        assert saved['args'].sigma_data == .5 and saved['args'].grad_accum_steps == 8
        assert len(saved['rng_states']) == 2
        resume_rng = saved['rng_states'][int(os.environ['RANK'])]
        completed = saved['epoch']+1
        if int(os.environ['RANK']) == 0:
            retained = cp.parent/f'checkpoint-ep{completed:03d}.pth'
            if completed in EPOCHS and not retained.exists():
                os.link(cp, retained)
            publish(root/'progress.json', dict(variant=variant, completed_epochs=completed,
                    optimizer_updates=completed*1251, saved_at=time.time()))
        del saved
    main_jit.Denoiser = EDMDenoiser
    resume_applied = False

    def train_epoch(model, bare, loader, optimizer, device, epoch, log_writer=None, args=None):
        nonlocal resume_applied
        if not resume_applied and resume_rng is not None:
            torch.set_rng_state(resume_rng['torch'])
            torch.cuda.set_rng_state(resume_rng['cuda'])
            np.random.set_state(resume_rng['numpy'])
            random.setstate(resume_rng['python'])
        resume_applied = True
        model.train(True)
        accum = args.grad_accum_steps
        updates = len(loader)//accum
        assert updates == 1251, (len(loader), updates)
        before = {int(state['step']) for state in optimizer.state.values()}
        assert not before if epoch == 0 else before == {epoch*updates}
        optimizer.zero_grad(set_to_none=True)
        loss_sum, start = 0., time.monotonic()
        for step, (images, labels) in enumerate(loader):
            if step >= updates*accum:
                break
            if step % accum == 0:
                lr_sched.adjust_learning_rate(optimizer, epoch+(step//accum)/updates, args)
            images = images.to(device, non_blocking=True).float().div_(255).mul_(2).sub_(1)
            labels = labels.to(device, non_blocking=True)
            boundary = (step+1) % accum == 0
            with nullcontext() if boundary else model.no_sync():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(images, labels)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, microstep {step}')
                (loss/accum).backward()
            if boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                bare.update_ema()
            loss_sum += value
            if step % 100 == 0:
                reduced = misc.all_reduce_mean(value)
                print(f'epoch={epoch+1} microstep={step}/{updates*accum} loss={reduced:.6f}', flush=True)
                if log_writer:
                    log_writer.add_scalar('train_loss', reduced, epoch*updates+step//accum)
        after = {int(state['step']) for state in optimizer.state.values()}
        assert after == {(epoch+1)*updates}
        print('EPOCH_COMPLETE', epoch+1, 'seconds', time.monotonic()-start,
              'loss', loss_sum/(updates*accum), flush=True)

    def save(args, model_without_ddp, optimizer, epoch, epoch_name=None):
        rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                   numpy=np.random.get_state(), python=random.getstate())
        rng_states = [None]*dist.get_world_size()
        dist.all_gather_object(rng_states, rng)
        if misc.is_main_process():
            payload = dict(model=model_without_ddp.state_dict(), optimizer=optimizer.state_dict(),
                           epoch=epoch, args=args, rng_states=rng_states)
            for k in (1, 2):
                state = copy.copy(model_without_ddp.state_dict())
                for (name, _), value in zip(model_without_ddp.named_parameters(), getattr(model_without_ddp, f'ema_params{k}')):
                    state[name] = value
                payload[f'model_ema{k}'] = state
            if shutil.disk_usage(root).free < 10*1024**3:
                raise OSError('At least 10 GiB free space required to save a checkpoint')
            temp = cp.with_suffix('.tmp')
            torch.save(payload, temp)
            with temp.open('rb') as f:
                os.fsync(f.fileno())
            temp.replace(cp)
            completed = epoch+1
            if completed in EPOCHS:
                retained = cp.parent/f'checkpoint-ep{completed:03d}.pth'
                if not retained.exists():
                    # Hard link retains the old inode when checkpoint-last is atomically replaced.
                    os.link(cp, retained)
            publish(root/'progress.json', dict(variant=variant, completed_epochs=completed,
                    optimizer_updates=completed*1251, saved_at=time.time()))
        dist.barrier()
        if epoch+1 < 200 and (root/'stop-requested').exists():
            raise ResumeLater()

    main_jit.train_one_epoch, misc.save_model = train_epoch, save
    try:
        main_jit.main(a)
        if misc.is_main_process():
            assert json.loads((root/'progress.json').read_text())['completed_epochs'] == 200
            assert all((cp.parent/f'checkpoint-ep{epoch:03d}.pth').exists() for epoch in EPOCHS)
            publish(root/'train-complete.json', dict(variant=variant, completed_epochs=200))
    except ResumeLater:
        return 75
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


class ResumeLater(Exception):
    pass


def evaluate(root, variant, epoch):
    import cv2
    import torch_fidelity
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('FID protocol requires a GPU with native BF16 support')
    a = arguments(root, variant)
    cp = root/'train'/f'checkpoint-ep{epoch:03d}.pth'
    checkpoint_sha = digest(cp)
    out = root/'eval'/f'ep{epoch:03d}'
    result_file = out/'metrics.json'
    if result_file.exists():
        record = json.loads(result_file.read_text())
        assert record['checkpoint_sha256'] == checkpoint_sha and record['verified_image_count'] == 10000
        return 0
    saved = torch.load(cp, map_location='cpu', mmap=True, weights_only=False)
    assert saved['epoch']+1 == epoch and saved['args'].edm_variant == variant
    model = EDMDenoiser(a).cuda().eval()
    model.load_state_dict(saved['model_ema1'], strict=True)
    del saved
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    folder = out/'images'
    folder.mkdir(parents=True, exist_ok=True)
    # A smaller one-GPU generation batch is recorded explicitly. Keep the
    # historical class ordering, quantization, BF16, guidance, and EMA1.
    batch = 64
    for first in range(0, 10000, batch):
        labels = torch.arange(first, min(first+batch, 10000), device='cuda')//10
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            images = model.generate(labels)
        if not torch.isfinite(images).all():
            raise FloatingPointError(f'Nonfinite generated image at index {first}')
        images = ((images+1)/2).float().cpu().numpy()
        for offset, img in enumerate(images):
            img = np.round(np.clip(img.transpose(1, 2, 0)*255, 0, 255)).astype(np.uint8)[:, :, ::-1]
            if not cv2.imwrite(str(folder/f'{first+offset:05d}.png'), img):
                raise OSError('PNG write failed')
        print('generated', first+len(images), '/10000', flush=True)
    assert {p.name for p in folder.glob('*.png')} == {f'{i:05d}.png' for i in range(10000)}
    stats = Path(__file__).parent/'fid_stats/jit_in256_stats.npz'
    metrics = torch_fidelity.calculate_metrics(input1=str(folder), input2=None,
        fid_statistics_file=str(stats), cuda=True, isc=True, fid=True,
        kid=False, prc=False, verbose=False)
    fid = float(metrics['frechet_inception_distance'])
    if not math.isfinite(fid):
        raise FloatingPointError('Nonfinite FID')
    record = dict(variant=variant, completed_epochs=epoch, fid=fid,
        inception_score=float(metrics['inception_score_mean']), verified_image_count=10000,
        num_images=10000, cfg=2.9, interval=[.1, 1.], sampling_method='heun', sampling_steps=50,
        ema='EMA1', ema_decay=.9999, precision='bf16', generation_batch=batch, seed=0,
        checkpoint_sha256=checkpoint_sha, reference_sha256=digest(stats),
        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        sources_sha256=digest(Path(__file__).parent.parent/'sources.json'))
    publish(result_file, record)
    shutil.rmtree(folder)
    print('VERIFIED_FID', json.dumps(record), flush=True)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('train', 'eval'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--epoch', type=int, choices=EPOCHS)
    opts = p.parse_args()
    verify_sources()
    if opts.mode == 'eval' and opts.epoch is None:
        p.error('--epoch is required for evaluation')
    sys.exit(train(opts.root, opts.variant) if opts.mode == 'train' else evaluate(opts.root, opts.variant, opts.epoch))
