"""CPU verification/reporting and portable GPU reproduction commands."""
import argparse
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Check hashes, cohorts, moments, MSEs and source parity on CPU.')
    r = sub.add_parser('report', help='Rebuild tables from included measurements on CPU.')
    r.add_argument('--output', type=Path, default=Path('artifacts/pyramid-calibration'))
    for name in ['prepare', 'fit', 'use-supplied-fit', 'reference', 'sample', 'score']:
        q = sub.add_parser(name)
        q.add_argument('--workspace', type=Path, required=True)
        if name in ['sample', 'score']:
            q.add_argument('--variant', choices=['baseline', 'calibrated', 'late_half'], required=True)
            q.add_argument('--count', type=int, choices=[2000, 10000, 20000, 30000, 50000], required=True)
        if name == 'score':
            q.add_argument('--start', type=int, default=0)
    args = p.parse_args()
    if hasattr(args, 'workspace'):
        os.environ['PYRAMID_CALIBRATION_WORKSPACE'] = str(args.workspace.resolve())
    from .report import verify, render
    checks = verify()
    if args.command == 'verify':
        import json
        print(json.dumps(checks, indent=2))
        return
    if args.command == 'report':
        print(render(args.output))
        return
    from .workspace import setup, use_supplied_fit, score_block
    setup()
    if args.command == 'prepare':
        from . import prepare, projector
        prepare.main()
        projector.main()
    elif args.command == 'use-supplied-fit':
        use_supplied_fit()
    elif args.command == 'score':
        if args.start < 0 or args.start + args.count > 50000:
            p.error('Score interval must lie within [0, 50000).')
        print(score_block(args.variant, args.start, args.count))
    elif args.command == 'reference':
        import fcntl
        from .common import ROOT, STORE, read
        from . import worker
        assert read(ROOT/'PREPARED.json')['complete']
        lock = (STORE/'reference.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker.torch.set_num_threads(4)
        worker.torch.backends.cuda.matmul.allow_tf32 = False
        worker.torch.backends.cudnn.allow_tf32 = False
        worker.torch.backends.cudnn.benchmark = False
        worker.reference_features()
    else:
        from . import worker
        sys.argv = [sys.argv[0], '--stage', 'fit' if args.command == 'fit' else 'generate']
        if args.command == 'sample':
            from .common import STORE
            if not (STORE/'fid/reference-statistics.npz').exists():
                raise RuntimeError('Run reference (or fit) before sample.')
            sys.argv += ['--variant', args.variant, '--count', str(args.count)]
        worker.main()
        if args.command == 'fit':
            from . import assess
            assess.main()


if __name__ == '__main__':
    main()
