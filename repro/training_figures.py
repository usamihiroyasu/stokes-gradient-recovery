"""Main-paper plots from saved AD and shared-network records."""
from pathlib import Path
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = HERE.parent/'latex/images'
BLUE, ORANGE, GREEN, GRAY = '#0072B2','#D55E00','#009E73','#666666'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=300)
    plt.close(fig)


def observed_errors():
    report = json.loads((HERE/'results/ad_route_benchmark.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(7.1,2.55))
    fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.84,wspace=.38)
    delta = np.geomspace(1e-7,1e-1,200)
    for sign,style in [(-1,'-'),(1,':')]:
        r = 1+sign*delta
        radial = abs((1-r*r)**2/(r*(3-r*r)*(1+r*r)))
        tangent = abs((1-r*r)/(r*(1+r*r)))
        axes[0].loglog(delta,radial,style,color=BLUE,label='radial: vector / mixed' if sign==-1 else None)
        axes[0].loglog(delta,tangent,style,color=ORANGE,label='tangent: vector' if sign==-1 else None)
    axes[0].set(xlabel=r'Radius offset $|r-1|$',ylabel='Curvature-input gain / I',title='(a) Analytic input gains')
    axes[0].legend(fontsize=6.7)
    for method,color,label in [('first',ORANGE,'First inverse'),('vector',BLUE,'Vector HVP'),('mixed',GREEN,'Mixed scalar'),('direct',GRAY,'Direct physical')]:
        values=[]
        for power in [7,5,3,1]:
            rows=[r for r in report['accuracy_records'] if r['dtype']=='float32' and r['I_requested']==1.2 and r['method']==method and abs(abs(r['r_requested']-1)-10.**-power)<1e-10]
            values.append(max(r['error'] for r in rows))
        axes[1].loglog([1e-7,1e-5,1e-3,1e-1],values,'o-',color=color,label=label,ms=3,lw=1.1)
    axes[1].set(xlabel='Requested radius offset',ylabel='Euclidean gradient error',title='(b) Observed float32 recovery')
    axes[1].legend(fontsize=6.7)
    save(fig,'near_fold_information')


def training():
    report=json.loads((HERE/'results/trained_fold_diagnostic.json').read_text())
    data=np.load(HERE/'results/trained_fold/high_dolp_standard_0.npz',allow_pickle=False)
    target=data['target'];state=data['s']
    fig,axes=plt.subplots(1,4,figsize=(7.2,2.1),gridspec_kw={'width_ratios':[.9,.9,1.15,1.15]})
    fig.subplots_adjust(left=.02,right=.985,bottom=.25,top=.81,wspace=.65)
    for ax,s,title in [(axes[0],target,'(a) Target DoLP'),(axes[1],state,'(b) Fitted DoLP')]:
        image=ax.imshow((np.linalg.norm(s[:,1:],axis=1)/s[:,0]).reshape(32,32),vmin=.85,vmax=1,cmap='viridis')
        ax.set_title(title,fontsize=8);ax.axis('off')
    cb=fig.colorbar(image,ax=axes[:2],location='bottom',fraction=.06,pad=.10,shrink=.85)
    cb.set_ticks([.85,.9,.95,1]);cb.ax.tick_params(labelsize=6)
    for case,color in [('renderer',BLUE),('high_dolp',ORANGE)]:
        distances=[];curves=[]
        for row in report['runs']:
            if row['dataset']!=case:continue
            stem=f"{case}_{row['initialization']}_{row['seed']}"
            run=np.load(HERE/'results/trained_fold'/(stem+'.npz'),allow_pickle=False)
            distances.extend(abs(np.linalg.norm(run['w'].astype(float),axis=1)-1))
            curves.append(run['loss_curve'])
        axes[2].hist(distances,bins=np.geomspace(1e-3,3,35),histtype='step',lw=1.2,color=color,label={'renderer':'Renderer','high_dolp':'High DoLP'}[case])
        values=np.array(curves);steps=np.arange(values.shape[1])*25
        axes[3].semilogy(steps,np.median(values,axis=0),color=color,label={'renderer':'Renderer','high_dolp':'High DoLP'}[case])
        axes[3].fill_between(steps,values.min(0),values.max(0),color=color,alpha=.15)
    axes[2].set_xscale('log');axes[2].set(xlabel=r'Final $|r-1|$',ylabel='Pixels across ten runs',title='(c) Distance from fold')
    axes[2].axvline(1e-3,color=GRAY,lw=.8,ls='--')
    axes[2].legend(fontsize=6.1)
    axes[3].set(xlabel='Adam updates',ylabel='Mean analyzer loss',title='(d) Learning curves')
    axes[3].legend(fontsize=6.1)
    save(fig,'trained_fold_incidence')


def curvature_histograms():
    report=json.loads((HERE/'results/trained_fold_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5))
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.84,wspace=.35)
    for ax,case,color in zip(axes,['renderer','high_dolp'],[BLUE,ORANGE]):
        values=[]
        for run in report['runs']:
            if run['dataset']!=case:continue
            stem=f"{case}_{run['initialization']}_{run['seed']}"
            data=np.load(HERE/'results/trained_fold'/(stem+'_diagnostic.npz'),allow_pickle=False)
            values.extend(data['normalized_curvature'])
        ax.hist(values,bins=40,histtype='step',color=color,lw=1.2)
        ax.axvline(0,color=GRAY,ls='--',lw=.8)
        ax.set(xlabel=r'Final radial curvature $c_r/I^2$',ylabel='Pixels across ten runs',
               title='Renderer' if case=='renderer' else 'High DoLP')
    save(fig,'trained_curvature_histograms')


if __name__=='__main__':
    observed_errors()
    training()
    curvature_histograms()
