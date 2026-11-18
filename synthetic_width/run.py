"""Run the reported constrained synthetic sweeps, or print their commands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def commands(family, output, device):
    jobs = []
    if family == 'gmm':
        common = ['--dist', 'gmm', '--gmm_path', str(HERE/'data/patch_gmm_K128.pt')]
        for t in [.1, .2, .3, .5, .7, .9]:
            spec = output/'spectra'/f'gmm128_t{t}.pt'
            jobs.append([sys.executable, str(HERE/'spectrum.py'), *common,
                '--t', str(t), '--out', str(spec), '--device', device])
            if t not in [.3, .5, .7, .9]:
                continue
            for k in [0, 8, 16, 32, 64, 96, 128, 192]:
                jobs.append([sys.executable, str(HERE/'train_resume.py'), *common,
                    '--t', str(t), '--spectrum', str(spec), '--arch', 'constrained',
                    '--extra', str(k), '--target', 'dstar', '--affine_init',
                    '--g_width', '512', '--adam_eps', '1e-12',
                    '--out', str(output/'results'/f'gmm128_t{t}_constrained_k{k}.json'),
                    '--device', device])
    else:
        for rank in [4, 8, 16, 32]:
            common = ['--dist', 'binary', '--d', '64', '--scales', f'flat:{rank}:1.0',
                '--sigma0', '.1', '--dist_seed', '0', '--t', '.5']
            spec = output/'spectra'/f'flat{rank}_t0.5.pt'
            jobs.append([sys.executable, str(HERE/'spectrum.py'), *common,
                '--out', str(spec), '--device', device])
            for k in [0, 4, 8, 16, 32]:
                jobs.append([sys.executable, str(HERE/'train.py'), *common,
                    '--spectrum', str(spec), '--arch', 'constrained', '--extra', str(k),
                    '--target', 'dstar', '--affine_init', '--adam_eps', '1e-12',
                    '--out', str(output/'results'/f'flat{rank}_t0.5_constrained_k{k}.json'),
                    '--device', device])
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['gmm', 'binary'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    jobs = commands(args.family, args.output, args.device)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    (args.output/'results').mkdir(parents=True, exist_ok=True)
    for command in jobs:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
