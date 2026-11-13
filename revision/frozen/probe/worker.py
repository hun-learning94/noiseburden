"""Frozen DiT/ADM final-head input recovery; one H200, adaptive large batches."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from probe_math import Moments, fit, metrics, neighborhoods, patchify, sample_pixels

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
DEVICE = 'cuda'
DATA = Path(os.environ['PUBLIC_PROBE_CACHE'])
ASSETS = json.loads((HERE / 'assets.json').read_text())['assets']
TIMES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
ALPHAS = np.cumprod(1 - np.linspace(.0001, .02, 1000, dtype=np.float64))
SIGMAS = np.sqrt((1 - ALPHAS) / ALPHAS)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def log(event, **kw):
    print(json.dumps(dict(event=event, **kw)), flush=True)


def summarize(errors, means, seconds):
    """One shared image bootstrap for all readouts, without huge JSON arrays."""
    mean = torch.as_tensor(np.asarray(means), device=DEVICE, dtype=torch.float64)
    second = torch.as_tensor(np.asarray(seconds), device=DEVICE, dtype=torch.float64)
    n = len(mean)
    rng = np.random.default_rng(240924)
    ids = rng.integers(n, size=(1000, n))
    counts = np.zeros((1000, n), dtype=np.float64)
    np.add.at(counts, (np.arange(1000)[:, None], ids), 1/n)
    weights = torch.from_numpy(counts).to(DEVICE)
    variance = (second.mean(0)-mean.mean(0).square()).mean()
    bootvar = (weights@second-(weights@mean).square()).mean(1)
    quantiles = torch.tensor([.025,.975], device=DEVICE, dtype=torch.float64)
    result = {}
    for key, values in errors.items():
        mse = torch.tensor(values, dtype=torch.float64, device=DEVICE)
        boot = weights@mse
        result[key] = dict(mse=float(mse.mean()), variance=float(variance),
            ev=float(1-mse.mean()/variance), r2_percent=float(100*(1-mse.mean()/variance)),
            mse_ci95=torch.quantile(boot, quantiles).cpu().tolist(),
            ev_ci95=torch.quantile(1-boot/bootvar, quantiles).cpu().tolist(),
            mse_by_image=values)
    return result


def mapping(t):
    sigma = (1-t)/t
    timestep = int(np.argmin(np.abs(np.log(SIGMAS) - np.log(sigma))))
    return dict(jit_reference_t=t, requested_sigma=sigma, timestep=timestep,
                alpha_bar=float(ALPHAS[timestep]), sigma=float(SIGMAS[timestep]))


@torch.inference_mode()
def encode_latents():
    path = HERE / 'results/clean-latents.npy'
    done = HERE / 'results/clean-latents.json'
    if done.exists():
        record = json.loads(done.read_text())
        assert record['vae_sha256'] == ASSETS['vae_weights']['sha256']
        assert digest(path) == record['sha256']
        return path
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(Path(ASSETS['vae_config']['path']).parent), local_files_only=True,
                                       use_safetensors=True).float().to(DEVICE).eval().requires_grad_(False)
    images = np.load(DATA / 'images256.npy', mmap_mode='r')
    ids = np.load(DATA / 'original_indices.npy')
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(images), 4, 32, 32))
    batch, lo = 64, 0
    while lo < len(images):
        hi = min(lo+batch, len(images))
        try:
            x = torch.from_numpy(np.array(images[lo:hi])).to(DEVICE).float().div_(127.5).sub_(1)
            distribution = vae.encode(x).latent_dist
            noise = torch.stack([torch.randn((4, 32, 32), generator=torch.Generator().manual_seed(7200000+int(i)))
                                 for i in ids[lo:hi]]).to(DEVICE)
            value = (distribution.mean + distribution.std * noise) * .18215
            assert torch.isfinite(value).all()
            out[lo:hi] = value.cpu().numpy()
            del x, distribution, noise, value
            lo = hi
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch //= 2
            gc.collect()
            torch.cuda.empty_cache()
            log('VAE_BATCH_REDUCED', batch=batch)
    out.flush()
    del out, vae
    gc.collect()
    torch.cuda.empty_cache()
    save(done, dict(sha256=digest(path), vae_sha256=ASSETS['vae_weights']['sha256'], scale=.18215,
                    posterior='one fixed posterior sample per image', batch=batch))
    log('LATENTS_READY', batch=batch)
    return path


class Extractor:
    def __init__(self, name):
        self.name, self.feature, self.replay_error = name, None, 0.
        self.labels = np.load(DATA / 'labels.npy')
        self.ids = np.load(DATA / 'original_indices.npy')
        self.positions = torch.from_numpy(np.stack([
            np.random.default_rng(9300000+int(i)).choice(256*256, size=512, replace=False)
            for i in self.ids])).to(DEVICE)
        if name == 'dit':
            self.clean = np.load(encode_latents(), mmap_mode='r')
            from dit_models import DiT_XL_2
            self.model = DiT_XL_2(input_size=32)
            self.head = self.model.final_layer.linear
            self.keys = {'token': (1152, 16), 'epsilon_output': (16, 16)}
        else:
            self.clean = np.load(DATA / 'images256.npy', mmap_mode='r')
            from guided_diffusion.script_util import create_model
            self.model = create_model(image_size=256, num_channels=256, num_res_blocks=2,
                                      learn_sigma=True, class_cond=True, attention_resolutions='32,16,8',
                                      num_heads=4, num_head_channels=64, resblock_updown=True,
                                      use_fp16=False, use_scale_shift_norm=True)
            self.head = self.model.out[-1]
            self.keys = {'conv3x3': (2304, 3), 'conv1x1': (256, 3), 'epsilon_output': (3, 3)}
        path = ASSETS[name]['path']
        assert digest(path) == ASSETS[name]['sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.float().to(DEVICE).eval().requires_grad_(False)
        self.handle = self.head.register_forward_pre_hook(lambda mod, args: setattr(self, 'feature', args[0].detach()))
        log('MODEL_READY', model=name, parameters=sum(p.numel() for p in self.model.parameters()))

    @torch.inference_mode()
    def extract(self, lo, hi, noise_mapping, replay=False):
        x = torch.from_numpy(np.array(self.clean[lo:hi])).to(DEVICE).float()
        if self.name == 'adm':
            x = x.div_(127.5).sub_(1)
        noise = torch.stack([torch.randn(tuple(x.shape[1:]), generator=torch.Generator().manual_seed(8100000+int(i)))
                             for i in self.ids[lo:hi]]).to(DEVICE)
        alpha = noise_mapping['alpha_bar'] ** .5
        beta = (1-noise_mapping['alpha_bar']) ** .5
        z = alpha*x + beta*noise
        ts = torch.full((len(x),), noise_mapping['timestep'], device=DEVICE, dtype=torch.long)
        labels = torch.from_numpy(self.labels[lo:hi]).to(DEVICE)
        raw = self.model(z, ts, labels)
        features = self.feature
        if replay:
            reproduced = self.head(features)
            if self.name == 'dit':
                reproduced = self.model.unpatchify(reproduced)
            difference = float((reproduced-raw).abs().max())
            assert difference < 1e-5, difference
            self.replay_error = max(self.replay_error, difference)
            del reproduced
        eps = raw[:, :x.shape[1]]
        clean_mse = (((z-beta*eps)/alpha).double()-x.double()).square().mean((1, 2, 3))
        if self.name == 'dit':
            assert features.shape[1:] == (256, 1152)
            rows = {'token': features, 'epsilon_output': patchify(eps)}
            target = patchify(z)
        else:
            assert features.shape[1:] == (256, 256, 256)
            positions = self.positions[lo:hi]
            rows = dict(conv3x3=neighborhoods(features, positions), conv1x1=sample_pixels(features, positions),
                        epsilon_output=sample_pixels(eps, positions))
            target = sample_pixels(z, positions)
        assert torch.isfinite(raw).all() and torch.isfinite(target).all()
        self.feature = None
        return rows, target, clean_mse

    def tune(self):
        measurements = []
        limit = torch.cuda.get_device_properties(0).total_memory * .82
        for b in ([16, 64] if self.name == 'dit' else [8, 24]):
            torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            values = self.extract(0, b, mapping(.5), replay=True)
            torch.cuda.synchronize()
            measurements.append(dict(batch=b, peak_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-tic))
            del values
            gc.collect()
            torch.cuda.empty_cache()
        slope = max(1, (measurements[-1]['peak_bytes']-measurements[0]['peak_bytes']) /
                    (measurements[-1]['batch']-measurements[0]['batch']))
        intercept = measurements[0]['peak_bytes']-slope*measurements[0]['batch']
        batch = int((limit-intercept)/slope)//8*8
        batch = max(8, min(batch, 512 if self.name == 'dit' else (256 if self.name.startswith('jit') else 64)))
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                values = self.extract(0, batch, mapping(.5), replay=True)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                del values
                gc.collect()
                torch.cuda.empty_cache()
                if peak > limit and batch > 8:
                    batch = max(8, batch-8)
                    continue
                break
            except torch.cuda.OutOfMemoryError:
                self.feature = None
                batch = max(1, batch//2)
                gc.collect()
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
        save(HERE / f'results/{self.name}-batch.json', dict(batch=batch, measurements=measurements,
             selected_peak_gib=peak/2**30, target_fraction=.82, gpu=torch.cuda.get_device_name(),
             precision='FP32 backbone; FP64 probe statistics and fitting'))
        log('BATCH_SELECTED', model=self.name, batch=batch, peak_gib=peak/2**30)
        return batch


@torch.inference_mode()
def run_time(extractor, tv, batch):
    destination = HERE / f'results/{extractor.name}-t{tv:g}.json'
    if destination.exists():
        assert json.loads(destination.read_text())['complete']
        return batch
    start = time.time()
    mp = mapping(tv)
    if extractor.name.startswith('jit'):
        mp = dict(jit_reference_t=tv, requested_sigma=(1-tv)/tv, sigma=(1-tv)/tv,
                  native_time=tv, a=tv, b=1-tv, schedule='JiT linear pixel corruption')
    train = {k: Moments(d, q, DEVICE) for k, (d, q) in extractor.keys.items()}
    dev = {k: Moments(d, q, DEVICE) for k, (d, q) in extractor.keys.items()}
    fitted, fits = {}, {}
    errors = {k: [] for k in extractor.keys}
    means, seconds, native = [], [], []
    for phase, first, end in [('fit', 0, 384), ('tune', 384, 512), ('test', 512, 2512)]:
        lo = first
        while lo < end:
            hi = min(lo+batch, end)
            try:
                rows, target, clean_mse = extractor.extract(lo, hi, mp, replay=lo == first)
            except torch.cuda.OutOfMemoryError:
                extractor.feature = None
                gc.collect()
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch//2)
                log('BATCH_REDUCED', model=extractor.name, t=tv, batch=batch)
                continue
            if phase in ('fit', 'tune'):
                for key, value in rows.items():
                    (train if phase == 'fit' else dev)[key].add(value, target)
            else:
                truth = target.double()
                for key, value in rows.items():
                    weight, bias = fitted[key]
                    prediction = value.double() @ weight + bias
                    errors[key].extend((prediction-truth).square().mean((1, 2)).cpu().tolist())
                means.extend(truth.mean(1).cpu().tolist())
                seconds.extend(truth.square().mean(1).cpu().tolist())
                native.extend(clean_mse.cpu().tolist())
                del truth, prediction
            del rows, target, clean_mse
            lo = hi
            log('PROGRESS', model=extractor.name, t=tv, phase=phase, images_done=lo-first,
                images_total=end-first, batch=batch, seconds=time.time()-start)
        if phase == 'tune':
            for key in extractor.keys:
                weight, bias, info = fit(train[key], dev[key])
                fitted[key] = (weight, bias)
                fits[key] = info
            del train, dev
            state = {key: dict(weight=w.cpu(), bias=b.cpu()) for key, (w, b) in fitted.items()}
            tmp = destination.with_suffix('.pt.tmp')
            torch.save(state, tmp)
            tmp.replace(destination.with_suffix('.pt'))
            del state
    scores = summarize(errors, means, seconds)
    np.savez_compressed(destination.with_suffix('.npz'), target_mean=np.asarray(means),
                        target_second_moment=np.asarray(seconds), native_mse=np.asarray(native),
                        **{k+'_mse':np.asarray(v) for k,v in errors.items()})
    save(destination, dict(complete=True, model=extractor.name, mapping=mp, probes=scores, fits=fits,
         fit_images=384, dev_images=128, test_images=2000,
         observations_per_image=512 if extractor.name == 'adm' else 256,
         native_clean_mse=float(np.mean(native)), native_clean_mse_by_image=native,
         head_reconstruction_max_abs=extractor.replay_error,
         seconds=time.time()-start, batch=batch, gpu=torch.cuda.get_device_name(),
         peak_gib=torch.cuda.max_memory_allocated()/2**30, job=os.environ.get('SLURM_JOB_ID'),
         worker_sha256=digest(__file__), protocol_sha256=digest(HERE / 'PROTOCOL.md'),
         precision='FP32 model and inputs; FP64 affine statistics, solve, predictions and metrics; TF32 disabled',
         checkpoint_sha256=ASSETS[extractor.name]['sha256']))
    log('TIME_COMPLETE', model=extractor.name, t=tv, r2={k:v['r2_percent'] for k,v in scores.items()})
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dit', 'adm'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.94)
    torch.manual_seed(0)
    extractor = Extractor(args.model)
    batch = extractor.tune()
    for tv in TIMES:
        batch = run_time(extractor, tv, batch)
    log('MODEL_COMPLETE', model=args.model)


if __name__ == '__main__':
    main()
