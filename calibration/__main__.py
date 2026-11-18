"""Run with python -m calibration --help."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description='Reproduce scalar calibration of released ImageNet256 JiT-B/16.')
    sub=p.add_subparsers(dest='command',required=True)
    report=sub.add_parser('report',help='Rebuild tables and plots from included measurements, on CPU.')
    report.add_argument('--output',type=Path,default=Path('artifacts/calibration'))
    sub.add_parser('verify',help='Verify source/metric hashes and recompute numerical consistency checks, on CPU.')
    prepare=sub.add_parser('prepare',help='Prepare exact training and held-out validation image/noise banks on CPU.')
    prepare.add_argument('--train',type=Path,required=True,help='ImageNet training directory containing wnid subdirectories.')
    prepare.add_argument('--validation',type=Path,required=True,help='Directory of official validation JPEGs.')
    prepare.add_argument('--output',type=Path,required=True)
    for name in ['fit','denoise','sample']:
        parser=sub.add_parser(name)
        parser.add_argument('--checkpoint',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--batch-size',type=int,default=64)
        if name in ['fit','denoise']:
            parser.add_argument('--assets',type=Path,required=True)
        if name in ['denoise','sample']:
            parser.add_argument('--coefficients',type=Path,default=Path(__file__).resolve().parents[1]/'metrics/calibration/fit')
        if name=='sample':
            parser.add_argument('--method',choices=['baseline','full','late_half'],required=True)
            parser.add_argument('--floor',type=float,choices=[.05,.02,0.],required=True)
    score=sub.add_parser('score',help='Score a completed 10K generation directory with the frozen FID implementation.')
    score.add_argument('--input',type=Path,required=True)
    score.add_argument('--fid-stats',type=Path,required=True)
    args=p.parse_args()
    if hasattr(args,'batch_size') and args.batch_size<1:
        p.error('--batch-size must be positive')
    from . import report as reporting
    if args.command=='report':
        print(json.dumps(reporting.render(args.output),indent=2));return
    if args.command=='verify':
        print(json.dumps(reporting.verify(),indent=2));return
    # Require an intact release before using its cohort, coefficients or inference code.
    reporting.verify()
    from . import run
    if args.command=='sample':
        run.generate(args)
    else:
        getattr(run,args.command)(args)


if __name__=='__main__':
    main()
