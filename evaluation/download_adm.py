"""Download the two official ADM checkpoints and verify their published run hashes."""
from pathlib import Path
import argparse,urllib.request
from .runtime import SPEC,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resolution',type=int,choices=[128,256],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=next(c for c in SPEC['cases'] if c['id']==f'adm{a.resolution}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not a.output.exists():
        tmp=a.output.with_suffix(a.output.suffix+'.download');urllib.request.urlretrieve(c['checkpoint_url'],tmp)
        assert digest(tmp)==c['checkpoint_sha256'],'Downloaded checkpoint checksum mismatch';tmp.replace(a.output)
    assert digest(a.output)==c['checkpoint_sha256'];print(a.output)

if __name__=='__main__':main()
