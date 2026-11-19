"""Paper-sized revision plots from portable numerical records."""
from pathlib import Path
import json,csv,tempfile,shutil
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter,FixedLocator,FuncFormatter
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'metrics/revision'
OUT=ROOT/'artifacts/figures'
controls=SimpleNamespace(FAMILIES={
 'image':('Image','#0072B2','s','-.'),
 'noise':('Noise','#C94C00','o','-'),
 'restricted':('Fixed','#8B52A1','^','--'),
 'deep':('Fixed + depth','#00856A','D',':')})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
def style(ax):
    ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True);ax.tick_params(length=2.5,pad=2)

def width_summary(rows,gains,spectrum,*,image_size=128):
    """One row: width, parameter allocation, denoising, residual structure."""
    assert image_size in (128,256)
    # ImageNet256 depth controls are not part of the current manuscript.
    if image_size==256:
        rows=[r for r in rows if r["family"]!="deep"]
        gains=[r for r in gains if r["family"]!="deep"]
    dp=192 if image_size==128 else 768
    widths=[256,320,384] if image_size==128 else [896,1024,1152,1536]
    color_ticks=[64,128,192,256,320,384] if image_size==128 else [128,384,768,1024,1536]
    fid_limits=(76,440) if image_size==128 else (50,440)
    fid_ticks=[80,100,150,250,400] if image_size==128 else [50,75,100,200,400]
    available_families={r['family'] for r in rows if r.get('fid') is not None}
    available_families.update(r['family'] for r in gains)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8.8,
                         'axes.labelsize':8.8,'axes.titlesize':9,'axes.linewidth':.65,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.7}):
        fig=plt.figure(figsize=(6.6,2.75),layout='constrained')
        grid=fig.add_gridspec(1,5,width_ratios=[1,1,1.12,.055,1.05])
        a,b,c,cax,d=[fig.add_subplot(grid[0,i]) for i in range(5)]
        axes=[a,b,c,d]
        fig.get_layout_engine().set(w_pad=.015,h_pad=.015,wspace=.015,hspace=0)
        norm=Normalize(color_ticks[0],color_ticks[-1])
        for family,(label,color,marker,ls) in controls.FAMILIES.items():
            rr=sorted([r for r in rows if r['family']==family and r.get('fid') is not None],key=lambda r:r['width'])
            if not rr:continue
            if family in ['restricted','deep']:
                rr=[next(r for r in rows if r['family']=='noise' and r['width']==dp)]+rr
            if family!='deep':
                a.plot([r['aligned_width'] for r in rr],[r['fid'] for r in rr],
                       color=color,marker=marker,ls=ls,ms=3,lw=1,label=label)
            b.plot([r['parameters']/1e6 for r in rr],[r['fid'] for r in rr],
                   color='.7',ls=ls,lw=.65)
            for r in rr:
                if family in ['restricted','deep'] and r['width']==dp:continue
                b.scatter(r['parameters']/1e6,r['fid'],c=[r['width']],norm=norm,
                          cmap='viridis',marker=marker,s=24,edgecolors='.2',lw=.4,zorder=4)
        for ax in [a,b]:
            ax.set(yscale='log',ylim=fid_limits)
            ax.set_yticks(fid_ticks,[str(v) for v in fid_ticks])
            ax.yaxis.set_minor_formatter(NullFormatter())
        a.set(xlim=(182,397) if image_size==128 else (728,1588),ylabel='FID-10K',xlabel='$d_h$ (noise)\n$d_h+d_p$ (image)',
              title='(a) Width')
        a.set_xticks([192,256,320,384] if image_size==128 else [768,1024,1536]);a.tick_params(axis='x',labelsize=7)
        a.axvline(dp,color='.55',ls=':',lw=.7);a.text(dp+9*dp/192,410,r'$d_p$',fontsize=7,color='.4')
        a.legend(frameon=False,loc='upper right',handlelength=1.25,labelspacing=.25,borderaxespad=.3)
        b.set(xlim=(0,35) if image_size==128 else (0,550),xlabel='Trainable parameters\n(millions)',title='(b) Parameters')
        b.set_xticks([0,10,20,30] if image_size==128 else [0,250,500]);b.tick_params(labelleft=False)
        handles=[Line2D([],[],color='.4',marker=marker,ls=ls,label=label.replace('Fixed + depth','Fixed\n+ depth'),ms=3,lw=.7)
                 for family,(label,_,marker,ls) in controls.FAMILIES.items() if family in available_families]
        b.legend(handles=handles,frameon=False,loc='upper right',fontsize=7.2,
                 labelspacing=.25,handlelength=1.25,borderaxespad=.2)
        colors={w:plt.get_cmap('viridis')(norm(w)) for w in widths}
        for family in ['noise','restricted','deep']:
            _,_,marker,ls=controls.FAMILIES[family]
            for width,color in colors.items():
                rr=sorted([r for r in gains if r['family']==family and r['width']==width],key=lambda r:r['t'])
                if rr:c.plot([r['t'] for r in rr],[r['mean'] for r in rr],color=color,ls=ls,
                             marker=marker,ms=2.4,lw=.9,mfc='white' if family!='noise' else color)
        gain_min=-2 if image_size==128 else min(-2,5*np.floor(min(r['mean'] for r in gains)/5))
        c.set(ylim=(gain_min,100),ylabel='Relative MSE reduction (%)',title='(c) Denoising gain')
        c.set_yticks([0,25,50,75,100])
        t=[r['t'] for r in spectrum]
        d.plot(t,[r['directions95'] for r in spectrum],color='#8B52A1',marker='o',ms=2.7,lw=1)
        d.set(ylim=(0,dp),title='(d) Residual structure')
        d.set_yticks([0,dp//3,2*dp//3,dp])
        d.text(.06,.67,'Directions for 95%',transform=d.transAxes,
               color='#8B52A1',rotation=90,ha='left',va='top',fontsize=8.3)
        right=d.twinx()
        right.plot(t,[r['variance_per_coordinate'] for r in spectrum],color='#00856A',ls='--',marker='s',mfc='white',ms=2.7,lw=1)
        right.set(yscale='log',ylim=(5e-4,.2),ylabel='Residual magnitude')
        right.set_yticks([.001,.01,.1],[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])
        right.yaxis.set_minor_locator(FixedLocator([]));right.yaxis.label.set_color('#00856A')
        right.yaxis.labelpad=1;right.tick_params(axis='y',pad=1,length=2.5)
        for ax in [c,d]:
            ax.set(xlim=(.025,.975),xlabel=r'Time $t$')
            ax.set_xticks([.1,.5,.9]);ax.xaxis.labelpad=13
            ax.text(0,-.085,'High',transform=ax.transAxes,ha='left',va='top',fontsize=7.5)
            ax.text(.5,-.085,'noise',transform=ax.transAxes,ha='center',va='top',fontsize=7.5)
            ax.text(1,-.085,'Low',transform=ax.transAxes,ha='right',va='top',fontsize=7.5)
        for ax in axes:style(ax)
        # Put the shared width key beside the denoising panel, freeing the footer.
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,
                        orientation='vertical')
        cb.set_ticks(color_ticks);cax.set_title(r'$d_h$',pad=5)
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.tick_params(labelsize=7.5,length=2,pad=1)
        save(fig,f'paper_width_summary_i{image_size}')

def scatter(ax,rows,key,title,resolution,compact=False):
    publiccolors={'ADM':'#0072B2','DiT':'#8B52A1','U-ViT':'#009E73','LDM':'#303030'}
    mixed_epochs=len({r.get('epoch') for r in rows if r['model'].startswith('JiT')})>1
    offsets={768:(0,10),896:(9,-11),1024:(14,-1),1152:(-6,-14),1536:(-2,-2),192:(0,10),256:(12,-11),320:(12,-8),384:(10,17)}
    for r in rows:
        model=r['model'];x,y=r[key],r['recovery_percent'];width=int(r['width']) if r['width']!='' else None
        if model.startswith('JiT'):
            family='deep' if model.endswith('+ depth') else ('restricted' if model!='JiT' else 'noise')
            widths=sorted({int(a['width']) for a in rows if a['model']==model})
            cmap={'noise':'Oranges','restricted':'Greens','deep':'PuRd'}[family]
            color=plt.get_cmap(cmap)(np.linspace(.48,.88,len(widths))[widths.index(width)])
            marker={'noise':'o','restricted':'^','deep':'D'}[family]
            ax.scatter(x,y,color=color,marker=marker,s=25,edgecolor='white',lw=.4,zorder=4)
            if compact:continue
            label=str(width)+('/'+str(int(r['depth'])) if family=='deep' else '')+('*' if mixed_epochs and r.get('epoch')==120 else '')
            dx,dy=offsets[width]
            if resolution==256 and not mixed_epochs and family=='noise':
                dx,dy={768:(0,10),896:(12,-24),1024:(-28,-18),1152:(-28,0),1536:(-5,12)}[width]
            if resolution==256 and not mixed_epochs and family=='noise' and key=='rgb_mse' and width==1152:dx,dy=-15,-8
            if family=='restricted':dx,dy={896:(17,-10),1024:(27,-4),1152:(28,8),1536:(-23,-37)}[width]
            ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else ('center' if dx==0 else 'left'),fontsize=7,color=color if family!='noise' else 'black',arrowprops={'arrowstyle':'-','color':color,'lw':.4} if dx else None)
        else:
            color=publiccolors[model];ax.scatter(x,y,color=color,marker='s' if model=='ADM' else '^',s=31,edgecolor='white',lw=.4,zorder=5)
            if compact:continue
            dx,dy,ha={'ADM':(8,-15,'left'),'DiT':(4,16,'left'),'U-ViT':(-7,17,'right'),'LDM':(-3,17,'right')}[model]
            if key=='rgb_mse':dx,dy,ha=-8,13,'right'
            ax.annotate(model+('\n(latents)' if model!='ADM' else ''),(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,fontsize=8,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.4})
    ax.set_xscale('log');xx=[r[key] for r in rows];ax.set_xlim(min(xx)*.85,max(xx)*1.18);ax.set_ylim(72,107)
    ax.set_yticks([75,85,95,100]);ax.axhline(100,color='.55',ls=':',lw=.7)
    ticks=([20,100,400] if compact else [20,50,100,200,400]) if key=='fid' else ([.015,.025] if compact else [.015,.02,.025])
    ticks=[x for x in ticks if ax.get_xlim()[0]<x<ax.get_xlim()[1]];ax.set_xticks(ticks);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set(xlabel='FID-10K' if key=='fid' else 'Clean-image MSE',ylabel='Noisy-input recovery (%)',title=title)
    hs=[Line2D([],[],color='#c94c00',marker='o',ls='',ms=4,label='Unrestricted'),Line2D([],[],color='#318a47',marker='^',ls='',ms=4,label='Fixed')]
    if any(r['model'].endswith('+ depth') for r in rows):hs.append(Line2D([],[],color='#ba1e60',marker='D',ls='',ms=4,label='Fixed + depth'))
    if compact:hs.append(Line2D([],[],color='#0072B2',marker='s',ls='',ms=4,label='ADM'))
    ax.legend(handles=hs,loc='lower left',frameon=False,fontsize=6.6 if compact else 7.3,handlelength=1,labelspacing=.2)
    if mixed_epochs and any(r.get('epoch')==120 for r in rows):ax.text(.98,.035,'* epoch120',transform=ax.transAxes,ha='right',fontsize=6.8,color='.3')
    style(ax)

def times(ax,rows,title,compact=False):
    colors={'DiT':'#8B52A1','ADM':'#0072B2','U-ViT':'#009E73','LDM':'#303030'}
    for name,color in colors.items():
        rr=[r for r in rows if r['model']==name]
        if rr:ax.plot([r['t'] for r in rr],[r['hidden_recovery_percent'] for r in rr],color=color,lw=1.1,label=name+(' (latents)' if name!='ADM' else ''))
    widths=sorted({int(r['width']) for r in rows if r['model']=='JiT'})
    for w,shade in zip(widths,np.linspace(.48,.9,len(widths))):
        rr=[r for r in rows if r['model']=='JiT' and r['width']==w]
        ax.plot([r['t'] for r in rr],[r['hidden_recovery_percent'] for r in rr],color=plt.get_cmap('Oranges')(shade),marker='o',ms=2,lw=1.,label=f'JiT {w}')
    ax.set(xlim=(.075,.925),ylim=(-2,103),xlabel=r'Time $t$',ylabel='Noisy-input recovery (%)',title=title)
    ax.set_xticks([.1,.5,.9]);ax.set_yticks([0,25,50,75,100]);ax.axhline(100,color='.55',ls=':',lw=.7)
    ax.legend(loc='lower left',frameon=False,fontsize=6.1 if compact else 7,ncol=1 if compact else 2,labelspacing=.2,columnspacing=.6,handlelength=1.2)
    style(ax)

def render(output):
    global OUT
    OUT=Path(output)/'figures';OUT.mkdir(parents=True,exist_ok=True)
    from noiseburden import plotting as legacy
    # Reuse the archived measurements, adding the velocity loss/floor to Figure 2.
    with tempfile.TemporaryDirectory() as temp:
        with plt.rc_context():
            legacy.render(Path(temp),include_velocity=True)
        for name in ['paper_bottleneck_performance','paper_noise_ceiling','paper_learned_coefficients','paper_bottleneck_loss_grid']:
            for ext in ['pdf','png']:shutil.copy2(Path(temp)/f'figures/{name}.{ext}',OUT/f'{name}.{ext}')
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42})
    data=json.loads((DATA/'section42_figures_provenance.json').read_text())
    width_summary(data['rows'],data['gains'],data['spectrum'])
    grid=json.loads((DATA/'width_controls_i256_ep120_provenance.json').read_text())
    width_summary(grid['rows'],grid['gains'],grid['spectrum'],image_size=256)
    public=data['public_panels']
    fig,ax=plt.subplots(figsize=(2.9,2.25),layout='constrained')
    scatter(ax,public['a'],'fid','',256);save(fig,'paper_input_recovery_public256')
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.55),layout='constrained')
    scatter(axes[0],public['b'],'rgb_mse','(a) Denoising and recovery',256)
    times(axes[1],public['c'],'(b) Recovery across noise levels');save(fig,'paper_public_recovery_extra256')
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.5),layout='constrained')
    scatter(axes[0],public['d'],'fid','(a) Generation',128,compact=True)
    scatter(axes[1],public['e'],'rgb_mse','(b) Denoising',128,compact=True)
    times(axes[2],public['f'],'(c) Input recovery',compact=True);save(fig,'paper_public_recovery_extra128')
    target=Path(output)/'metrics';target.mkdir(exist_ok=True)
    rows=[dict(case=case,t=r['t'],sigma=r['sigma'],clean_mse=r['clean_mse']['mean'],
               lower_bound=r['clean_mse_floor'])
          for case in ['bottleneck_image','bottleneck_exact','bottleneck_noise',
                       'bottleneck_velocity','edm_jit_input','edm_reflow_input']
          for r in legacy.native(case)]
    with (target/'figure2_measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for res,records in [(128,data),(256,grid)]:
        for kind in ['rows','gains','spectrum']:
            rows=records[kind]
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (target/f'width{res}_{kind}.csv').open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    for name,rows in public.items():
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (target/f'input_recovery_panel_{name}.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
