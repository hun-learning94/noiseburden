"""Verify the frozen model and loader code without loading weights or images."""
from pathlib import Path
import json
from .runtime import ROOT,SPEC,digest

def main():
    checked=0
    for c in SPEC['cases']:
        for name,expected in c['source_hashes'].items():
            assert digest(ROOT/c['source']/name)==expected,(c['id'],name);checked+=1
        if 'loader' in c:assert digest(ROOT/c['loader'])==c['loader_sha256']
        if c['kind']=='width':assert c['width']==c['heads']*c['head_dim']
    print(json.dumps(dict(checkpoint_configurations=len(SPEC['cases']),source_file_checks=checked,
                         training_base=SPEC['training_base']),indent=2))

if __name__=='__main__':main()
