"""Regenerate Figures 2–9 from the included, held-out validation measurements."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator,NullFormatter
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'metrics'
NATIVE=DATA/'native'
OUT=ROOT/'artifacts/figures'
CSV_OUT=ROOT/'artifacts/metrics'
M=json.loads((ROOT/'configs/plot_cases.json').read_text())
VALIDATION=json.loads((DATA/'protocol.json').read_text())
BLUE='#0072B2';ORANGE='#C94C00';GREEN='#00856A';PURPLE='#8B52A1';GRAY='#656565';GOLD='#BD8A00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.35,'lines.markersize':3.6,'pdf.fonttype':42,'savefig.pad_inches':.015})
SIZES={}

def source_path(value):
    path=Path(value)
    return path if path.is_absolute() else (ROOT/path if (ROOT/path).is_file() else DATA/path)

def canvas(rows=1,cols=1,height=2.4,**kwargs):
    fig,ax=plt.subplots(rows,cols,figsize=(5.5,height),layout='constrained',**kwargs)
    fig.get_layout_engine().set(w_pad=.025,h_pad=.025,wspace=.035,hspace=.07)
    return fig,ax

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    SIZES[name]=dict(canvas_inches=list(fig.get_size_inches()),body_font_pt=8.5,tick_font_pt=8)
    plt.close(fig)

def style(ax):
    ax.grid(axis='y',color='.9',lw=.55);ax.set_axisbelow(True);ax.tick_params(length=2.5,pad=2)

def noiseaxis(ax):
    ax.set_xscale('log');ax.set_xticks([1/9,1,9],[r'$1/9$',r'$1$',r'$9$']);ax.set_xlabel(r'Effective noise $\sigma_t$');style(ax)

def native(key):
    c=next(c for c in M['cases'] if c['id']==key)
    j=json.loads((NATIVE/f'native-{key}.json').read_text())
    assert j['provenance']['case']['checkpoint_sha256']==c['checkpoint_sha256']
    assert j['provenance']['protocol_sha256']==VALIDATION['manifest_sha256']
    return sorted(j['records'],key=lambda r:r['sigma'])

def curve(ax,key,color,label,marker='o',ls='-',band=True,x_key='sigma'):
    rr=sorted(native(key),key=lambda r:r[x_key]);x=np.array([r[x_key] for r in rr]);y=np.array([r['clean_mse']['mean'] for r in rr]);ci=np.array([r['clean_mse']['ci95'] for r in rr])
    ax.plot(x,y,marker=marker,ls=ls,color=color,label=label,mfc='white' if ls=='--' else color)
    if band:ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.10,lw=0)
    return x,y

def bottleneck_performance():
    """Compact generation and denoising panels for the main text."""
    rows=list(csv.DictReader((DATA/'bottleneck_fid_200.csv').open()))
    fid={r['case_id']:float(r['fid']) for r in rows}
    assert len(rows)==len(fid)==10
    order=[('bottleneck_image','Image',BLUE),('bottleneck_exact','Exact',GREEN),
           ('bottleneck_noise','Noise',ORANGE),('edm_jit_input','EDM: JiT',PURPLE),
           ('edm_reflow_input','EDM: ReFlow',PURPLE),('bottleneck_additive','Additive',GOLD),
           ('bottleneck_learned_exact','Learned 1',GRAY),
           ('bottleneck_learned_noise','Learned 2',GRAY),
           ('bottleneck_learned_additive','Learned 3',GRAY),
           ('bottleneck_velocity','Velocity',GRAY)]
    assert set(fid)=={item[0] for item in order}
    order=sorted(order,key=lambda item:fid[item[0]])
    fig,axs=canvas(cols=2,height=2.45,gridspec_kw={'width_ratios':[.88,1.3]})
    fig.set_size_inches(5.2,2.45)
    a,b=axs
    for i,(idx,label,color) in enumerate(order):
        v=fid[idx]
        a.plot([7,v],[i,i],color=color,alpha=.25,lw=1)
        a.scatter(v,i,color=color,s=22,zorder=3)
        a.annotate(f'{v:.2f}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
    a.set_yticks(range(10),[label for _,label,_ in order]);a.invert_yaxis()
    a.set_xscale('log');a.set_xlim(7,1250);a.set_xticks([10,100,1000],['10','100','1000'])
    a.set_xlabel('Guided FID-10K');a.tick_params(axis='y',length=0,pad=3)
    a.grid(axis='x',color='.9',lw=.55)
    a.set_title('(a) Generation quality',loc='left')
    curve(b,'bottleneck_image',BLUE,'Image',marker='o',band=False,x_key='t')
    curve(b,'bottleneck_exact',GREEN,'Exact',marker='s',ls='--',band=False,x_key='t')
    x,_=curve(b,'bottleneck_noise',ORANGE,'Noise',marker='o',band=False,x_key='t')
    curve(b,'edm_jit_input',PURPLE,'EDM',marker='^',band=False,x_key='t')
    curve(b,'edm_reflow_input',PURPLE,'_nolegend_',marker='v',ls='--',band=False,x_key='t')
    bare=sorted(native('bottleneck_noise'),key=lambda r:r['t']);edm=sorted(native('edm_jit_input'),key=lambda r:r['t'])
    b.plot(x,[r['clean_mse_floor'] for r in bare],color='.2',ls='--',lw=1.15,label='Noise bound',zorder=5)
    b.plot(x,[r['clean_mse_floor'] for r in edm],color=PURPLE,ls=':',lw=1.6,label='EDM bound',zorder=5)
    b.set_yscale('log');b.set_ylim(8e-5,900)
    b.set_yticks([1e-4,1e-3,1e-2,.1,1,10,100],['0.0001','0.001','0.01','0.1','1','10','100'])
    b.set_ylabel('Per-coordinate clean-image MSE')
    b.set_xlim(.025,.975);b.set_xticks([.1,.5,.9],['0.1','0.5','0.9'])
    b.set_xticks([.05,.2,.8,.95],minor=True);b.set_xlabel(r'Time $t$');style(b)
    b.text(0,-.105,'High noise',transform=b.transAxes,ha='left',va='top',fontsize=7.8)
    b.text(1,-.105,'Low noise',transform=b.transAxes,ha='right',va='top',fontsize=7.8)
    b.set_title('(b) Denoising loss',loc='left')
    b.legend(frameon=False,loc='upper right',bbox_to_anchor=(1,1.04),handlelength=1.6,labelspacing=.15,fontsize=7.8)
    save(fig,'paper_bottleneck_performance')

def bottleneck_loss_grid():
    """All ten validation loss curves as an annotated, time-normalized heatmap."""
    cases=[('bottleneck_image','Image'),('bottleneck_noise','Noise'),
           ('bottleneck_velocity','Velocity'),('bottleneck_additive','Additive'),
           ('bottleneck_exact','Exact'),('bottleneck_learned_exact','Learned 1'),
           ('bottleneck_learned_noise','Learned 2'),('bottleneck_learned_additive','Learned 3'),
           ('edm_jit_input','EDM: JiT'),('edm_reflow_input','EDM: ReFlow')]
    times=sorted(M['times'])
    loss=np.array([[r['clean_mse']['mean'] for r in sorted(native(key),key=lambda r:r['t'])]
                   for key,_ in cases])
    ratio=loss/loss[0]
    norm=LogNorm(vmin=.5,vmax=3000)
    fig,ax=canvas(height=2.65)
    im=ax.imshow(ratio,aspect='auto',norm=norm,cmap='YlOrRd',interpolation='nearest')
    ax.set_yticks(range(len(cases)),[label for _,label in cases])
    ax.set_xticks(range(len(times)),[f'{t:g}' for t in times])
    ax.set_xlabel(r'Time $t$',labelpad=13)
    ax.tick_params(length=0,pad=3)
    ax.set_xticks(np.arange(len(times)-1)+.5,minor=True)
    ax.set_yticks(np.arange(len(cases)-1)+.5,minor=True)
    ax.grid(which='minor',color='white',lw=.65)
    ax.tick_params(which='minor',length=0)
    for i in range(len(cases)):
        for j in range(len(times)):
            rgba=im.cmap(norm(ratio[i,j]));luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{loss[i,j]:.3g}',ha='center',va='center',fontsize=7.8,
                    color='white' if luminance<.48 else '#222222')
    ax.text(0,-.11,'High noise',transform=ax.transAxes,ha='left',va='top',fontsize=7.8)
    ax.text(1,-.11,'Low noise',transform=ax.transAxes,ha='right',va='top',fontsize=7.8)
    for spine in ax.spines.values():spine.set_visible(False)
    bar=fig.colorbar(im,ax=ax,pad=.025,fraction=.045)
    bar.set_ticks([1,10,100,1000],labels=['1','10','100','1000'])
    bar.ax.yaxis.set_minor_locator(FixedLocator([]))
    bar.set_label('MSE / image-prediction MSE',fontsize=8.5)
    bar.ax.tick_params(labelsize=8,length=2,pad=2)
    save(fig,'paper_bottleneck_loss_grid')
    rows=[dict(case=key,method=label,t=t,clean_mse=float(loss[i,j]),
               image_mse=float(loss[0,j]),relative_mse=float(ratio[i,j]),
               dataset_split='validation',test_images=2000)
          for i,(key,label) in enumerate(cases) for j,t in enumerate(times)]
    with (CSV_OUT/'bottleneck_validation_loss_grid.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def learned_coefficients():
    source=DATA/'learned'
    provenance=json.loads((source/'coefficient-provenance.json').read_text())
    assert provenance['grid_matches_saved_native_outputs']
    rows=list(csv.DictReader((source/'learned-coefficients.csv').open()))
    fig,axs=canvas(cols=2,height=1.8)
    fig.set_size_inches(4.7,1.8)
    styles=[('bottleneck_learned_exact',GREEN,'Learned 1','-.'),
            ('bottleneck_learned_noise',GRAY,'Learned 2','-'),
            ('bottleneck_learned_additive',GOLD,'Learned 3','--')]
    for key,color,label,ls in styles:
        rr=sorted([r for r in rows if r['case']==key],key=lambda r:float(r['sigma']))
        x=[float(r['sigma']) for r in rr]
        skip=np.array([float(r['t'])*float(r['u']) for r in rr])
        output=np.array([float(r['w']) for r in rr])
        ratios=[skip*np.array(x),-output*np.array(x)]
        assert np.allclose(ratios[0],[float(r['relative_u']) for r in rr],rtol=1e-12,atol=1e-12)
        assert np.allclose(ratios[1],[float(r['relative_w']) for r in rr],rtol=1e-12,atol=1e-12)
        for ax,ratio in zip(axs,ratios):
            ax.plot(x,ratio,color=color,label=label,ls=ls,lw=1.35)
    for ax in axs:
        ax.axhline(1,color='.15',ls=':',lw=1.2,label='Exact coefficients')
        noiseaxis(ax)
    axs[0].set_title('(a) Input-skip coefficient',loc='left',fontsize=8.5)
    axs[0].set_ylabel(r'$c_{\mathrm{skip}}\,/\,(1/\sigma_t)$');axs[0].set_ylim(0,1.13);axs[0].set_yticks([0,.5,1])
    axs[1].set_title('(b) Network-output coefficient',loc='left',fontsize=8.5)
    axs[1].set_ylabel(r'$c_{\mathrm{out}}\,/\,(-1/\sigma_t)$');axs[1].set_ylim(-4.6,4);axs[1].set_yticks([-4,-2,0,1,2,4])
    axs[0].legend(frameon=False,loc='center right',fontsize=7.8,labelspacing=.25,handlelength=1.8)
    save(fig,'paper_learned_coefficients')

def probes():
    """Final-block recovery versus time from one paired six-time cohort."""
    rows=list(csv.DictReader((DATA/'noise_layerwise_200_ema1.csv').open()))
    assert len(rows)==78 and all(float(r['noise_recovery_score'])<1/6 for r in rows)
    assert all(r['dataset_split']=='validation' and int(r['test_images'])==2000
               and hashlib.sha256(source_path(r['source']).read_bytes()).hexdigest()==r['source_sha256'] for r in rows)
    fig,ax=canvas(height=1.5);fig.set_size_inches(2.4,1.5)
    for layer,color,marker,ls,label in [(12,ORANGE,'o','-','Block 12')]:
        rr=sorted([r for r in rows if int(r['layer'])==layer],key=lambda r:float(r['t']))
        assert [float(r['t']) for r in rr]==[.1,.5,.7,.8,.9,.95]
        ax.plot([float(r['t']) for r in rr],[float(r['noise_recovery_score']) for r in rr],color=color,marker=marker,mfc='white' if layer==12 else color,ms=2.8,lw=1.1,ls=ls,label=label,zorder=3)
    ax.axhline(1/6,color='.25',ls=':',lw=1.1,label='Ceiling')
    ax.set_xlim(.06,.99);ax.set_ylim(.08,.17)
    ax.set_xticks([.1,.5,.9],['0.1','0.5','0.9']);ax.set_xticks([.7,.8,.95],minor=True);ax.set_xlabel(r'Time $t$')
    ax.set_yticks([.08,.12,1/6],['0.08','0.12',r'$1/6$'])
    ax.set_ylabel('Noise recovery');style(ax)
    ax.text(0,-.16,'High noise',transform=ax.transAxes,ha='left',va='top',fontsize=7.2)
    ax.text(1,-.16,'Low noise',transform=ax.transAxes,ha='right',va='top',fontsize=7.2)
    ax.legend(frameon=False,loc='lower left',fontsize=7.8,labelspacing=.15,handlelength=1.3,handletextpad=.4)
    save(fig,'paper_noise_ceiling')

def width_relative_gains(image_size=256):
    """Ratio of mean loss reduction to mean reference loss, with paired intervals."""
    cases=sorted([c for c in M['cases'] if c['kind']=='width' and c['image_size']==image_size and c['pred']=='eps'],key=lambda c:c['width'])
    reference=cases[0];reference_width=3*(image_size//16)**2;assert reference['width']==reference_width
    indices=np.random.default_rng(231023).integers(2000,size=(1000,2000))
    records=[]
    with np.load(NATIVE/f"native-{reference['id']}-per-image.npz") as arrays:
        base={tv:np.asarray(arrays[f'x_{tv}'],dtype=float) for tv in M['times']}
    denominator={tv:a[indices].mean(axis=1) for tv,a in base.items()}
    for c in cases[1:]:
        native(c['id'])
        with np.load(NATIVE/f"native-{c['id']}-per-image.npz") as arrays:
            for tv in M['times']:
                a=base[tv];b=np.asarray(arrays[f'x_{tv}'],dtype=float)
                assert a.shape==b.shape==(2000,) and np.isfinite(a).all() and np.isfinite(b).all() and a.mean()>0
                difference=a-b
                value=100*difference.mean()/a.mean()
                samples=100*difference[indices].mean(axis=1)/denominator[tv]
                records.append(dict(width=c['width'],t=tv,reference_width=reference_width,mean=float(value),ci95=np.quantile(samples,[.025,.975]).tolist(),reference_mse=float(a.mean()),model_mse=float(b.mean()),n_images=2000))
    return records

def adm_residuals(image_size=256):
    """Validated ADM-only residual spectra, preserving the fixed-patch analysis."""
    run=DATA/f'spectra{image_size}'
    assert len([r for r in VALIDATION['independent_cpu_adm_spectra'] if r['resolution']==image_size])==7
    summary=json.loads((run/'summary.json').read_text())
    assert summary['complete'] and summary['manifest_sha256']==VALIDATION['historical_spectral_manifest_sha256'][str(image_size)]
    rows=sorted([r for r in summary['records'] if r['teacher']==f'adm{image_size}' and r['patch']==0],key=lambda r:r['t'])
    assert [r['t'] for r in rows]==[.05,.1,.2,.5,.8,.9,.95]
    exported=[]
    for row in rows:
        positions=[r for r in summary['records'] if r['teacher']==f'adm{image_size}' and r['t']==row['t']]
        assert len(positions)==5
        exported.append(dict(t=row['t'],sigma=row['sigma'],directions95=row['k95'],
                             directions95_min=min(r['k95'] for r in positions),directions95_max=max(r['k95'] for r in positions),
                             variance_per_coordinate=row['energy_per_coordinate'],
                             variance_min=min(r['energy_per_coordinate'] for r in positions),variance_max=max(r['energy_per_coordinate'] for r in positions),
                             dataset_split='validation',fit_images=6000,test_images=4000))
    export_name='adm_residual_spectrum.csv' if image_size==256 else 'adm_residual_spectrum_i128.csv'
    with (CSV_OUT/export_name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    return exported

def residual_timeaxis(ax):
    ax.set_xlim(.025,.975);ax.set_xticks([.1,.5,.9],['0.1','0.5','0.9'])
    ax.set_xticks([.05,.2,.8,.95],minor=True);ax.set_xlabel(r'Time $t$',labelpad=14);style(ax)
    ax.text(0,-.115,'High noise',transform=ax.transAxes,ha='left',va='top',fontsize=7)
    ax.text(1,-.115,'Low noise',transform=ax.transAxes,ha='right',va='top',fontsize=7)

def width_performance(image_size=256):
    """Generation, approximation gains, and residual dispersion/magnitude."""
    dp=3*(image_size//16)**2
    fig,axs=canvas(cols=3,height=2.0)
    a,b,c=axs;fid=fid_width();gains=width_relative_gains(image_size)
    exported=[dict(resolution=image_size,width=r['width'],t=r['t'],
                   reference_width=r['reference_width'],relative_mse_reduction_percent=r['mean'],
                   ci95_low=r['ci95'][0],ci95_high=r['ci95'][1],reference_mse=r['reference_mse'],
                   model_mse=r['model_mse'],test_images=r['n_images'],dataset_split='validation') for r in gains]
    with (CSV_OUT/f'width_relative_gains_i{image_size}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    for pred,color,marker in [('Image',BLUE,'s'),('Noise',ORANGE,'o')]:
        values=sorted((w,y) for res,p,w,y in fid if res==image_size and p==pred)
        a.plot(*np.array(values).T,color=color,marker=marker,label=pred,ms=3.5,lw=1.2)
    a.set_yscale('log');a.yaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(42,520);a.set_yticks([50,100,400],['50','100','400'])
    a.set_xlim(90,1600) if image_size==256 else a.set_xlim(20,400)
    a.set_xticks([dp//3,dp,2*dp])
    a.set_xticks(sorted({w for res,p,w,y in fid if res==image_size}-{dp//3,dp,2*dp}),minor=True)
    a.set_title('(a) FID-10K',loc='left',fontsize=8.5)
    a.legend(frameon=False,loc='upper right' if image_size==256 else 'lower left',
             fontsize=8 if image_size==256 else 7.4,
             **({} if image_size==256 else dict(handlelength=1.2,labelspacing=.15)))
    first_wider=min(w for res,p,w,y in fid if res==image_size and p=='Noise' and w>dp)
    a.annotate(str(first_wider),(first_wider,next(y for res,p,w,y in fid if res==image_size and p=='Noise' and w==first_wider)),xytext=(8,12),textcoords='offset points',fontsize=8,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.6))
    wider=sorted({r['width'] for r in gains})
    for width,marker in zip(wider,['o','s','^','D']):
        color=plt.get_cmap('viridis')(.1+.75*(width-wider[0])/(wider[-1]-wider[0]))
        rr=sorted([r for r in gains if r['width']==width],key=lambda r:r['t'])
        y=np.array([r['mean'] for r in rr]);ci=np.array([r['ci95'] for r in rr])
        b.errorbar([r['t'] for r in rr],y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),color=color,marker=marker,label=str(width),capsize=1.5,ms=3.5,lw=1.2)
    b.set_xlim(.025,.975);b.set_xticks([.1,.5,.9],['0.1','0.5','0.9']);b.set_xticks([.05,.2,.8,.95],minor=True)
    b.set_ylim(0,90 if image_size==256 else 100);b.set_yticks([0,20,40,60,80] if image_size==256 else [0,25,50,75,100])
    b.set_ylabel('Relative MSE reduction (%)');b.set_title('(b) Denoising gain',loc='left',fontsize=8.5)
    b.legend(title=r'Width $d_h$',frameon=False,loc='upper right',fontsize=7.5,title_fontsize=8,labelspacing=.2,borderpad=.2,handlelength=1.3,handletextpad=.4)
    a.axvline(dp,color='.55',ls=':',lw=.85)
    a.annotate(r'$d_p$',(dp,.98),xycoords=('data','axes fraction'),xytext=(-4,0),textcoords='offset points',ha='right',va='top',fontsize=8,color='.35')
    for ax in axs:style(ax)
    a.set_xlabel(r'Model width $d_h$',labelpad=14);b.set_xlabel(r'Time $t$',labelpad=14)
    b.text(0,-.115,'High noise',transform=b.transAxes,ha='left',va='top',fontsize=7)
    b.text(1,-.115,'Low noise',transform=b.transAxes,ha='right',va='top',fontsize=7)
    rows=adm_residuals(image_size)
    c.plot([r['t'] for r in rows],[r['directions95'] for r in rows],color=PURPLE,marker='o',ms=3.5,lw=1.2)
    c.fill_between([r['t'] for r in rows],[r['directions95_min'] for r in rows],[r['directions95_max'] for r in rows],color=PURPLE,alpha=.12,lw=0)
    c.set_ylim(0,600 if image_size==256 else 192);c.set_yticks([0,200,400,600] if image_size==256 else [0,64,128,192]);c.set_ylabel('Directions for 95%',color=PURPLE)
    c.tick_params(axis='y',colors=PURPLE);c.spines['left'].set_color(PURPLE)
    c.set_title('(c) Residual structure',loc='left',fontsize=8.5)
    residual_timeaxis(c)
    magnitude=c.twinx()
    magnitude.plot([r['t'] for r in rows],[r['variance_per_coordinate'] for r in rows],color=GREEN,marker='s',ls='--',mfc='white',ms=3.3,lw=1.2)
    magnitude.fill_between([r['t'] for r in rows],[r['variance_min'] for r in rows],[r['variance_max'] for r in rows],color=GREEN,alpha=.09,lw=0)
    magnitude.set_yscale('log');magnitude.set_ylim(5e-4,.2);magnitude.set_yticks([.001,.01,.1],[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])
    magnitude.yaxis.set_minor_locator(FixedLocator([]));magnitude.set_ylabel('Residual magnitude',color=GREEN)
    magnitude.tick_params(axis='y',colors=GREEN,length=2.5,pad=2);magnitude.spines['right'].set_visible(True);magnitude.spines['right'].set_color(GREEN)
    magnitude.spines['left'].set_visible(False);magnitude.grid(False)
    save(fig,'paper_width_performance' if image_size==256 else 'paper_width_performance_i128')

def readouts(resolutions=(256,128)):
    """Directly compare native denoising and affine noisy-input recovery."""
    rows=list(csv.DictReader((DATA/'affine_input_recovery_200_ema1.csv').open()))
    assert len(rows)==16
    assert all(r['dataset_split']=='validation' and int(r['test_images'])==2000
               and hashlib.sha256(source_path(r['source']).read_bytes()).hexdigest()==r['source_sha256'] for r in rows)
    for res,expected in [(256,5),(128,4)]:
        if res not in resolutions:continue
        rr=sorted([r for r in rows if int(r['resolution'])==res and r['prediction']=='Noise'],key=lambda r:int(r['width']))
        assert len(rr)==expected
        x=np.array([float(r['native_clean_mse']) for r in rr])
        y=np.array([float(r['input_r2_percent']) for r in rr])
        size=(2.2,1.8) if res==256 else (2.65,1.85)
        fig,ax=canvas(height=size[1]);fig.set_size_inches(*size)
        if res==256:ax.set_box_aspect(.85)
        width=np.array([int(r['width']) for r in rr])
        colors=plt.get_cmap('viridis')(.1+.75*(width-width.min())/(width.max()-width.min()))
        ax.scatter(x,y,s=28,c=colors,edgecolors='white',linewidths=.4,zorder=3)
        ax.axhline(100,color='.45',ls=':',lw=.9,zorder=1)
        offsets=([(0,8),(13,-11),(12,-2),(16,0),(0,12)] if res==256
                 else [(0,8),(12,-10),(14,-2),(0,12)])
        for r,xv,yv,color,(dx,dy) in zip(rr,x,y,colors,offsets):
            ax.annotate(r['width'],(xv,yv),xytext=(dx,dy),textcoords='offset points',
                        ha='right' if dx<0 else ('center' if dx==0 else 'left'),
                        va='center',fontsize=8,color='.2',
                        arrowprops=dict(arrowstyle='-',color=color,lw=.45,alpha=.7) if dx else None)
        ax.set_ylim(72,107);ax.set_yticks([75,85,95,100]);style(ax)
        if res==256:
            ax.set_xlim(.01475,.02015)
            ax.set_xticks([.015,.017,.019],['0.015','0.017','0.019'])
        else:
            ax.set_xlim(.0178,.0278)
            ax.set_xticks([.019,.023,.027],['0.019','0.023','0.027'])
        ax.set_xlabel(r'Clean-image MSE $\downarrow$',fontsize=8.5)
        ax.set_ylabel('Noisy input recovery' if res==256 else 'Noisy-input recovery (%)',fontsize=8.5)
        if res==256:ax.tick_params(labelsize=8)
        save(fig,'paper_input_recovery' if res==256 else 'paper_input_recovery_i128')
        if res==256:
            SIZES['paper_input_recovery'].update(body_font_pt=8.5,tick_font_pt=8,annotation_font_pt=8,axes_box_aspect=.85,wrap_textwidth_fraction=.40)

def width_probe_panels():
    """Paired denoising and input recovery, with both targets at both resolutions."""
    rows=list(csv.DictReader((DATA/'affine_input_recovery_200_ema1.csv').open()))
    assert len(rows)==16
    assert all(r['dataset_split']=='validation' and int(r['test_images'])==2000
               and hashlib.sha256(source_path(r['source']).read_bytes()).hexdigest()==r['source_sha256'] for r in rows)
    fig,axs=canvas(cols=4,height=1.85)
    axs[1].sharey(axs[0]);axs[3].sharey(axs[2])
    axs[1].tick_params(labelleft=False);axs[3].tick_params(labelleft=False)
    for col,(res,dp) in enumerate([(256,768),(128,192)]):
        for pred,color,marker in [('Image',BLUE,'s'),('Noise',ORANGE,'o')]:
            rr=sorted([r for r in rows if int(r['resolution'])==res and r['prediction']==pred],key=lambda r:int(r['width']))
            x=[int(r['width']) for r in rr]
            for ax,key in zip([axs[col],axs[col+2]],['native_clean_mse','input_r2_percent']):
                ax.plot(x,[float(r[key]) for r in rr],color=color,marker=marker,label=pred,ms=3.5,lw=1.2)
        for row,ax in enumerate([axs[col],axs[col+2]]):
            ax.set_xlim(90,1600) if res==256 else ax.set_xlim(40,400)
            ax.set_xticks([128,768,1536] if res==256 else [64,192,384])
            ax.set_xticks(sorted({int(r['width']) for r in rows if int(r['resolution'])==res}),minor=True)
            ax.axvline(dp,color='.55',ls=':',lw=.85)
            if row==0:
                ax.set_ylim(.013,.029);ax.set_yticks([.015,.020,.025],['0.015','0.020','0.025'])
            else:
                ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100])
                ax.axhline(100,color='.55',ls=':',lw=.85)
            letter=chr(ord('a')+2*row+col)
            ax.set_title(f'({letter}) ImageNet{res}',loc='left',fontsize=8.5)
            style(ax)
    axs[0].set_ylabel('Clean-image MSE')
    axs[2].set_ylabel('Input recovery (%)')
    axs[0].legend(frameon=False,loc='upper right',fontsize=7.5,handlelength=1.1,labelspacing=.15,handletextpad=.3)
    fig.supxlabel(r'Model width $d_h$',fontsize=8.5)
    save(fig,'paper_width_probe_panels')

def export_width_measurements():
    """Retain full numerical grids alongside the visual manuscript summaries."""
    rows=[dict(resolution=r,prediction=p,width=w,fid=y,epoch=200,ema='EMA1')
          for r,p,w,y in sorted(fid_width())]
    losses=[]
    for c in M['cases']:
        if c['kind']!='width':continue
        for r in native(c['id']):
            losses.append(dict(case=c['id'],resolution=c['image_size'],prediction=c['pred'],
                               width=c['width'],t=r['t'],clean_mse=r['clean_mse']['mean'],
                               dataset_split='validation',test_images=2000))
    for name,data in [('width_generation_200_ema1.csv',rows),('width_native_loss_grid.csv',losses)]:
        with (CSV_OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def fid_width():
    rows=list(csv.DictReader((DATA/'width_generation_200_ema1.csv').open()))
    assert len(rows)==16
    return [(int(r['resolution']),r['prediction'],int(r['width']),float(r['fid'])) for r in rows]

def render(output,metrics=None):
    global OUT,CSV_OUT,DATA,NATIVE,M,VALIDATION
    if metrics is not None:
        DATA=Path(metrics).resolve();NATIVE=DATA/'native'
        M=json.loads((DATA/'plot_cases.json').read_text())
        VALIDATION=json.loads((DATA/'protocol.json').read_text())
    output=Path(output).resolve();OUT=output/'figures';CSV_OUT=output/'metrics'
    OUT.mkdir(parents=True,exist_ok=True);CSV_OUT.mkdir(parents=True,exist_ok=True)
    bottleneck_performance();learned_coefficients();bottleneck_loss_grid();probes()
    width_performance(256);readouts((256,));width_performance(128);width_probe_panels()
    export_width_measurements()
    (OUT/'figure_sizes.json').write_text(json.dumps(SIZES,indent=2)+'\n')
    return SIZES
