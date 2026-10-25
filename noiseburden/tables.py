"""Recompute all six manuscript tables; the first two specify definitions."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'metrics'
TIMES=[.05,.1,.2,.5,.8,.9,.95]

def render(output,metrics=None):
    data=Path(metrics).resolve() if metrics else DATA
    output=Path(output)/'tables';output.mkdir(parents=True,exist_ok=True)
    specs=json.loads((ROOT/'configs/tables.json').read_text())
    cases=json.loads(((data/'plot_cases.json') if metrics else ROOT/'configs/plot_cases.json').read_text())['cases']
    native={c['id']:{r['t']:r['clean_mse']['mean'] for r in
            json.loads((data/f'native/native-{c["id"]}.json').read_text())['records']} for c in cases}
    noise=list(csv.DictReader((data/'noise_layerwise_200_ema1.csv').open()))
    for spec in specs:
        number=spec['number'];body=spec['body'];csv_rows=[]
        if number==3:
            times=[.1,.5,.7,.8,.9,.95];lines=[]
            for layer in range(13):
                values=[f'{float(next(r["noise_recovery_score"] for r in noise if int(r["layer"])==layer and float(r["t"])==t)):.5f}' for t in times]
                fields=[str(layer),*values];lines.append(' & '.join(fields)+r'\\');csv_rows.append(fields)
            body='\n'.join(lines)+'\n'
        elif number in [4,5]:
            res=256 if number==4 else 128
            selected=sorted([c for c in cases if c['kind']=='width' and c['image_size']==res],key=lambda c:(c['pred']=='x',c['width']))
            lines=[]
            for c in selected:
                fields=['Noise' if c['pred']=='eps' else 'Image',str(c['width']),*[f'{native[c["id"]][t]:.5g}' for t in TIMES]]
                lines.append(' & '.join(fields)+r'\\');csv_rows.append(fields)
            body='\n'.join(lines)+'\n'
        elif number==6:
            lines=[]
            for res in [256,128]:
                if res==128:lines.append(r'\midrule')
                lines.append(r'\multicolumn{8}{l}{ImageNet'+str(res)+r'}\\')
                records=json.loads((data/f'spectra{res}/summary.json').read_text())['records']
                keys=[f'i{res}_eps{1536 if res==256 else 384}',f'i{res}_x{768 if res==256 else 192}',f'adm{res}']
                names=[rf'JiT noise ($d_h={1536 if res==256 else 384}$)',rf'JiT image ($d_h={768 if res==256 else 192}$)','ADM U-Net']
                values={(r['teacher'],r['t']):r['teacher_full_image_mse'] for r in records if r['patch']==0}
                for key,name in zip(keys,names):
                    cells=[];plain=[]
                    for t in TIMES:
                        value=f'{values[key,t]:.6f}';plain.append(value)
                        if values[key,t]==min(values[k,t] for k in keys):value=r'\textbf{'+value+'}'
                        cells.append(value)
                    lines.append(' & '.join([name,*cells])+r'\\');csv_rows.append([str(res),name,*plain])
            body='\n'.join(lines)+'\n'
        else:
            csv_rows=[line.removesuffix('\\\\').split(' & ') for line in body.splitlines() if ' & ' in line]
        result=spec['prefix']+body+spec['suffix']+'\n'
        (output/f'table_{number:02d}.tex').write_text(result)
        with (output/f'table_{number:02d}.csv').open('w',newline='') as f:csv.writer(f).writerows(csv_rows)
    return [f'table_{s["number"]:02d}.tex' for s in specs]
