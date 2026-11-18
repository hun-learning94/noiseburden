"""Portable preparation, fitting, denoising and generation for scalar calibration."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import gc
import json

import numpy as np
import torch
from PIL import Image

from .core import (ROOT, CONFIG, METRICS, read, digest, save, key, predictions,
    project_discarded, load_model, load_coefficients, multiplier, sample, preflight)


@contextmanager
def writer(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def record_identity(folder, identity):
    path = Path(folder)/'identity.json'
    if path.exists() and read(path) != identity:
        raise ValueError(f'Existing outputs use a different protocol: {folder}')
    save(path, identity)


def identity(args):
    return {'checkpoint_sha256': read(CONFIG)['checkpoint']['sha256'],
            'protocol_sha256': digest(CONFIG),
            'release_manifest_sha256': digest(ROOT/'CALIBRATION_MANIFEST.json'),
            'batch_size': args.batch_size, 'precision': 'FP32, TF32 disabled'}


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def prepare(args):
    from .frozen.crop import center_crop_arr
    protocol = read(CONFIG)
    records = read(ROOT/'configs/calibration/samples.json')
    assets = args.output
    torch.set_num_threads(4)
    with writer(assets):
        record_identity(assets, {'samples_sha256': digest(ROOT/'configs/calibration/samples.json'),
                                'protocol_sha256': digest(CONFIG)})
        for bank in ['fit0', 'fit1', 'test']:
            names = [f'{bank}-{suffix}.npy' for suffix in ['images', 'noise', 'labels', 'indices']]
            expected = protocol['data_bank_sha256']
            if all((assets/n).exists() and digest(assets/n) == expected[n] for n in names):
                continue
            cohort = [r for r in records if r['bank'] == bank]
            shape = (len(cohort), 3, 256, 256)
            images = np.lib.format.open_memmap(assets/names[0], mode='w+', dtype=np.uint8, shape=shape)
            noise = np.lib.format.open_memmap(assets/names[1], mode='w+', dtype=np.float32, shape=shape)
            for i, row in enumerate(cohort):
                directory = args.validation if row['split'] == 'validation' else args.train
                with Image.open(directory/row['filename']) as im:
                    images[i] = np.asarray(center_crop_arr(im.convert('RGB'), 256)).transpose(2, 0, 1)
                noise[i] = torch.randn((3, 256, 256),
                    generator=torch.Generator().manual_seed(row['noise_seed'])).numpy()
            images.flush(); noise.flush()
            np.save(assets/names[2], np.array([r['label'] for r in cohort], dtype=np.int64 if bank=='test' else np.int32))
            np.save(assets/names[3], np.array([r['source_index'] for r in cohort], dtype=np.int64))
            for name in names:
                if digest(assets/name) != expected[name]:
                    raise ValueError(f'{name} differs from the recorded bank. Check JPEG version and preprocessing environment.')
            print(f'Prepared and verified {bank}: {len(cohort)} images', flush=True)
        save(assets/'complete.json', {'sha256': protocol['data_bank_sha256']})


def verify_banks(assets, banks):
    expected = read(CONFIG)['data_bank_sha256']
    for bank in banks:
        for suffix in ['images', 'noise', 'labels', 'indices']:
            name = f'{bank}-{suffix}.npy'
            if digest(assets/name) != expected[name]:
                raise ValueError(f'Unmatched calibration data: {name}')


def batches(assets, bank, size):
    images = np.load(assets/f'{bank}-images.npy', mmap_mode='r')
    noise = np.load(assets/f'{bank}-noise.npy', mmap_mode='r')
    labels = np.load(assets/f'{bank}-labels.npy')
    for lo in range(0, len(images), size):
        hi = min(lo+size, len(images))
        x = torch.from_numpy(np.array(images[lo:hi])).cuda().float().div_(127.5).sub_(1)
        e = torch.from_numpy(np.array(noise[lo:hi])).cuda()
        y = torch.from_numpy(labels[lo:hi]).cuda()
        yield x, e, y


@torch.inference_mode()
def fit(args):
    verify_banks(args.assets, ['fit0', 'fit1'])
    with writer(args.output):
        record_identity(args.output, identity(args))
        model, rows = load_model(args.checkpoint)
        save(args.output/'preflight.json', preflight(model, rows))
        for tv in read(CONFIG)['fit_times']:
            path = args.output/f'{key(tv)}.json'
            if path.exists():
                continue
            moments = {'unguided': [], 'guided': []}
            for bank in ['fit0', 'fit1']:
                for x, e, y in batches(args.assets, bank, args.batch_size):
                    z = tv*x+(1-tv)*e
                    for mode, (f, _) in predictions(model, z, tv, y).items():
                        squared = project_discarded(f.double()-x.double(), rows).square()
                        moments[mode].extend(squared.flatten(1).mean(1).cpu().tolist())
            coefficients = {}
            for mode, values in moments.items():
                assert len(values) == 10000
                v = float(np.mean(values))/(640/768)
                sigma = (1-tv)/tv
                coefficients[mode] = {'vbar': v, 'c_t': v/(v+sigma*sigma),
                                      'discarded_mse': float(np.mean(values))}
            atomic_npz(path.with_suffix('.npz'), **{m: np.asarray(v) for m,v in moments.items()})
            save(path, {'complete': True, 't': tv, 'images': 10000, 'coefficients': coefficients})
            print('FIT', tv, coefficients, flush=True)


@torch.inference_mode()
def denoise(args):
    verify_banks(args.assets, ['test'])
    coefficients = load_coefficients(args.coefficients)
    with writer(args.output):
        info = identity(args)
        info['coefficients'] = coefficients
        record_identity(args.output, info)
        model, rows = load_model(args.checkpoint)
        save(args.output/'preflight.json', preflight(model, rows))
        for tv in read(CONFIG)['test_times']:
            path = args.output/f'{key(tv)}.json'
            if path.exists():
                continue
            values = {m: {method: [] for method in ['baseline','full','late_half']}
                      for m in ['unguided','guided']}
            for x, e, y in batches(args.assets, 'test', args.batch_size):
                z = tv*x+(1-tv)*e
                for mode, (f, _) in predictions(model, z, tv, y).items():
                    error = f.double()-x.double()
                    delta = project_discarded(z.double()/tv-f.double(), rows)
                    for method in values[mode]:
                        c = multiplier(tv, method)*coefficients[key(tv)][mode]['c_t']
                        values[mode][method].extend((error+c*delta).square().flatten(1).mean(1).cpu().tolist())
            records = {}
            for mode, methods in values.items():
                records[mode] = {}
                baseline = np.asarray(methods['baseline'])
                for method, v in methods.items():
                    vv = np.asarray(v); gain = baseline-vv
                    records[mode][method] = {'mse': float(vv.mean()), 'improvement': float(gain.mean()),
                        'reduction_percent': float(100*gain.mean()/baseline.mean()),
                        'paired_se': float(gain.std(ddof=1)/np.sqrt(len(gain))),
                        'c_t': multiplier(tv,method)*coefficients[key(tv)][mode]['c_t']}
            atomic_npz(path.with_suffix('.npz'), **{m+'__'+method: np.asarray(v)
                for m, methods in values.items() for method, v in methods.items()})
            save(path, {'complete': True, 't': tv, 'images': 2000, 'records': records})
            print('DENOISE', tv, records['guided'], flush=True)


@torch.inference_mode()
def generate(args):
    coefficients = load_coefficients(args.coefficients)
    with writer(args.output):
        info = identity(args)
        info.update(method=args.method, endpoint_clamp=args.floor, coefficients=coefficients,
                    images=10000, images_per_class=10, seed_offset=0, steps=50, cfg=3.)
        record_identity(args.output, info)
        model, rows = load_model(args.checkpoint, args.floor)
        save(args.output/'preflight.json', preflight(model, rows))
        rows = rows.float()
        output = args.output/'images'; output.mkdir(exist_ok=True)
        for temp in output.glob('*.tmp.png'):
            temp.unlink()
        expected = {f'{i:05d}.png' for i in range(10000)}
        done = {p.name for p in output.glob('*.png') if not p.name.endswith('.tmp.png')}
        if not done <= expected:
            raise ValueError('Unexpected image files in output directory')
        for name in done:
            with Image.open(output/name) as im:
                if im.size != (256,256):
                    raise ValueError(name)
                im.verify()
        batch = args.batch_size
        while len(done) < 10000:
            ids = [i for i in range(10000) if f'{i:05d}.png' not in done][:batch]
            try:
                labels = torch.tensor([i//10 for i in ids], device='cuda')
                z = torch.stack([torch.randn((3,256,256), device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(i)) for i in ids])
                result = sample(model, rows, z, labels, args.method, coefficients)
                pixels = ((result+1)*127.5).clamp(0,255).round().byte().permute(0,2,3,1).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                gc.collect(); torch.cuda.empty_cache(); batch //= 2
                print('Reduced batch size after OOM:', batch, flush=True)
                continue
            for i, pixel in zip(ids, pixels):
                name = f'{i:05d}.png'
                temp = output/f'{i:05d}.tmp.png'
                Image.fromarray(pixel).save(temp); temp.replace(output/name)
                done.add(name)
            save(args.output/'progress.json', {'images': len(done), 'total': 10000,
                'batch_size': batch, 'complete': len(done)==10000})
            print('GENERATED', len(done), '/ 10000', flush=True)


def score(args):
    protocol = read(CONFIG)
    if digest(args.fid_stats) != protocol['fid_reference_sha256']:
        raise ValueError('Reference statistics differ from the matched FID-10K protocol.')
    output = args.input/'images'
    names = {p.name for p in output.glob('*.png') if not p.name.endswith('.tmp.png')}
    if names != {f'{i:05d}.png' for i in range(10000)}:
        raise ValueError('FID requires all 10,000 generated images.')
    if any(output.glob('*.tmp.png')):
        raise ValueError('Incomplete temporary PNGs remain; resume sampling first.')
    info = read(args.input/'identity.json')
    if read(args.input/'progress.json')['images'] != 10000:
        raise ValueError('Sampling is incomplete.')
    # The campaign used this identical bundled torch-fidelity implementation.
    import sys
    sys.path.insert(0, str(ROOT/'evaluation/third_party'))
    import torch_fidelity
    values = torch_fidelity.calculate_metrics(input1=str(output), input2=None,
        fid_statistics_file=str(args.fid_stats), cuda=True, isc=True, fid=True,
        kid=False, prc=False, verbose=False, batch_size=64)
    result = {'complete': True, 'fid': float(values['frechet_inception_distance']),
        'inception_score': float(values['inception_score_mean']), 'images': 10000,
        'endpoint_clamp': info['endpoint_clamp'], 'method': info['method'],
        'reference_sha256': digest(args.fid_stats)}
    if not np.isfinite(result['fid']):
        raise FloatingPointError('Nonfinite FID')
    save(args.input/'fid.json', result)
    print(json.dumps(result, indent=2))
