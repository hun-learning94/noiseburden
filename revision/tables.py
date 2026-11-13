"""Regenerate seven revision tables, including the matched epoch120 controls."""
from pathlib import Path
import csv,json,tempfile
ROOT=Path(__file__).resolve().parents[1]

def render(output):
    from noiseburden import tables as legacy
    specs=json.loads((ROOT/'configs/revision_tables.json').read_text())
    data=json.loads((ROOT/'metrics/revision/width_controls_i256_ep120_provenance.json').read_text())
    target=Path(output)/'tables';target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        legacy.render(Path(temp))
        oldspecs=json.loads((ROOT/'configs/tables.json').read_text())
        for spec in specs:
            number=spec['number']
            if number==4:
                rows=[]
                for r in data['rows']:
                    p=ROOT/'metrics/revision/controls256'/f'{r["case"]}-t0.5.json'
                    d=json.loads(p.read_text()) if p.exists() else None
                    name={'image':'Image','noise':'Noise','restricted':'Restricted','deep':'Restricted + depth'}[r['family']]
                    fields=[name,str(r['width']),str(r['depth']),f"{r['parameters']/1e6:.2f}",
                            f"{r['fid']:.2f}" if r['fid'] is not None else '',
                            f"{d['native_clean_mse']:.5f}" if d else '',
                            f"{d['probes']['token']['r2_percent']:.2f}" if d and r['family']!='image' else '']
                    rows.append(fields)
                body='\n'.join(' & '.join(r)+r'\\' for r in rows)+'\n'
            else:
                oldnumber=number if number<4 else number-1
                old=(Path(temp)/f'tables/table_{oldnumber:02d}.tex').read_text()
                oldspec=oldspecs[oldnumber-1]
                body=old[len(oldspec['prefix']):len(old)-len(oldspec['suffix'])-1]
                rows=list(csv.reader((Path(temp)/f'tables/table_{oldnumber:02d}.csv').open()))
            text=spec['prefix']+body+spec['suffix']+'\n'
            (target/f'table_{number:02d}.tex').write_text(text)
            with (target/f'table_{number:02d}.csv').open('w',newline='') as stream:csv.writer(stream).writerows(rows)
