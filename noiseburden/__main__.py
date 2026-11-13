"""Run with: python -m noiseburden --output artifacts --strict"""
from pathlib import Path
import argparse,json
from . import tables,verify

def main():
    parser=argparse.ArgumentParser(description='Reproduce the revision tables and Figures 2–10 from included metrics.')
    parser.add_argument('--output',type=Path,default=Path('artifacts'))
    parser.add_argument('--metrics',type=Path,help='Use a complete numerical rerun exported by evaluation.collect.')
    parser.add_argument('--strict',action='store_true',help='Require pixel-identical PNGs in the pinned environment.')
    parser.add_argument('--check-only',action='store_true',help='Verify input integrity and recompute metric consistency checks.')
    parser.add_argument('--arxiv',action='store_true',help='Reproduce the archived arXiv outputs instead of the revision.')
    args=parser.parse_args()
    if args.metrics and args.strict:parser.error('--strict compares the original distributed measurements only')
    result=dict(input_validation=verify.inputs(args.metrics))
    if not args.arxiv and not args.metrics:
        from revision import plotting as revised_plots,tables as revised_tables,verify as revised_verify
        result['revision_validation']=revised_verify.inputs()
        if not args.check_only:
            revised_plots.render(args.output);revised_tables.render(args.output)
            result.update(revised_verify.outputs(args.output,args.strict))
            (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    if not args.check_only:
        from . import plotting
        plotting.render(args.output,args.metrics);tables.render(args.output,args.metrics)
        if args.metrics:result.update(reference_comparison=False,note='Outputs reflect the supplied numerical rerun.')
        else:result.update(verify.outputs(args.output,args.strict))
        (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
