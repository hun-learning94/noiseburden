"""Configure once, then run one checkpoint/teacher per process."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from .runtime import ROOT,configure

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['setup','native','input','noise','spectra','analyze'])
    p.add_argument('--workspace',type=Path,default=Path('work/evaluation'))
    p.add_argument('--cache',type=Path);p.add_argument('--checkpoints',type=Path)
    p.add_argument('--accept-new-checkpoints',action='store_true')
    p.add_argument('--case');p.add_argument('--teacher');p.add_argument('--resolution',type=int,choices=[128,256])
    p.add_argument('--smoke',action='store_true');p.add_argument('--cpu',action='store_true');a=p.parse_args()
    if a.mode=='setup':
        if not a.cache or not a.checkpoints:p.error('setup requires --cache and --checkpoints')
        print(configure(a.workspace,a.cache,a.checkpoints,a.accept_new_checkpoints));return
    info=json.loads((a.workspace/'runtime.json').read_text());run=Path(info['run'])
    env=dict(os.environ,VALIDATION_CACHE=info['cache'],EVAL_CACHE=info['cache'],PROBE_CACHE=info['cache'],
             SPECTRAL_CACHE=info['cache'],PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(ROOT/'evaluation/third_party')+os.pathsep+env.get('PYTHONPATH','')
    if a.mode in ['native','input']:
        if not a.case:p.error('this mode requires --case')
        cmd=[sys.executable,str(run/('native.py' if a.mode=='native' else 'input_probe.py')),a.case]
        if a.smoke:cmd.append('--smoke')
    elif a.mode=='noise':
        if a.smoke:p.error('the frozen noise probe evaluates its complete six-time cohort')
        cmd=[sys.executable,str(run/'noise_probe.py')]
    else:
        if not a.resolution:p.error('spectra/analyze requires --resolution')
        cmd=[sys.executable,str(run/f'spectra{a.resolution}'/('extract.py' if a.mode=='spectra' else 'analyze.py'))]
        if a.mode=='spectra':
            if not a.teacher:p.error('spectra requires --teacher')
            cmd+=['--teacher',a.teacher]
            if a.smoke:cmd.append('--smoke')
        elif a.cpu:cmd.append('--cpu')
    subprocess.run(cmd,env=env,check=True)

if __name__=='__main__':main()
