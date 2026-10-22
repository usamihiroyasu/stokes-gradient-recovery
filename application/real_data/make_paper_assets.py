"""Generate figures and transparent aggregate tables from frozen records."""
from pathlib import Path
import collections,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;L=(R/'latex/images') if (R/'latex/images').is_dir() else R/'revision/stokes_gradient_recovery/latex/images'
plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
real=json.loads((P/'RESULTS.json').read_text())['records'];raw=[r for r in real if r['arm']=='warm_start']
colors={'DeepSfP':'#557ca5','PolarNS_means':'#c5742d','PolarBurstSR':'#39947c'}
labels={'DeepSfP':'DeepSfP','PolarNS_means':'PolarNS means','PolarBurstSR':'PolarBurstSR'}
fig,axs=plt.subplots(1,2,figsize=(11.1,3.7),layout='constrained')
for dataset in colors:
    rs=[r for r in raw if r['dataset']==dataset];bad=[r for r in rs if r['false_success']]
    axs[0].scatter([r['radius'] for r in rs],np.maximum([r['gap'] for r in rs],1e-14),s=5,alpha=.28,color=colors[dataset],rasterized=True)
    axs[0].scatter([r['radius'] for r in bad],[r['gap'] for r in bad],s=23,alpha=.8,marker='x',color=colors[dataset],label=labels[dataset],rasterized=True)
    axs[1].scatter(np.maximum([r['gap'] for r in rs],1e-14),np.maximum([r['physical_bound'] for r in rs],1e-14),s=6,alpha=.45,color=colors[dataset],rasterized=True)
axs[0].set(xscale='log',yscale='log',xlabel='Raw radius r',ylabel='Objective gap to cone minimum',title='(a) Real-data false successes miss the fixed fold gate')
axs[0].axvline(1,color='black',ls=':',lw=1);axs[0].axhline(1e-6,color='black',ls='--',lw=1);axs[0].legend(fontsize=9,loc='lower right')
axs[1].plot([1e-14,1],[1e-14,1],color='black',lw=1,ls=':')
axs[1].axvline(1e-6,color='black',lw=1,ls='--');axs[1].axhline(1e-6,color='black',lw=1,ls='--')
axs[1].set(xscale='log',yscale='log',xlabel='Objective gap to cone minimum',ylabel='Physical supporting-half-space bound',title='(b) Physical auditing rejects all inaccurate endpoints')
for ax in axs:ax.grid(alpha=.16)
for dest in [P,L]:
    fig.savefig(dest/'real_stopping_regimes.png',dpi=200,facecolor='white',bbox_inches='tight')
    fig.savefig(dest/'real_stopping_regimes.pdf',facecolor='white',bbox_inches='tight')
plt.close(fig)
fresnel=R/'application/fresnel_sequence';d=json.loads((fresnel/'results.json').read_text())['records'];inp=json.loads((fresnel/'inputs.json').read_text());seq=max([r for r in d if r['method']=='BFGS' and r['arm']=='baseline'],key=lambda r:r['final']['gap'])['sequence']
fig,axs=plt.subplots(1,2,figsize=(10.4,3.6),layout='constrained')
for arm,label,color,ls in [('baseline','Raw warm start','#c04c2a','-'),('curvature_gate','Curvature audit + restart','#176fa3','--'),('physical_audit','Physical audit + restart','#3b8f63',':')]:
    rs=sorted([r for r in d if r['method']=='BFGS' and r['sequence']==seq and r['arm']==arm],key=lambda r:r['frame']);s=np.array([r['final']['physical'] for r in rs]);f=np.arange(len(rs))
    if arm!='physical_audit':axs[0].plot(f,np.linalg.norm(s[:,1:],axis=1)/s[:,0],label=label,color=color,ls=ls,lw=1.8)
    axs[1].semilogy(f,np.maximum([r['final']['gap'] for r in rs],1e-15),label=label,color=color,ls=ls,lw=1.8)
truth=np.array(inp[seq]['truth']);axs[0].plot(f,np.linalg.norm(truth[:,1:],axis=1)/truth[:,0],color='black',ls=':',label='Fresnel truth',lw=1.8)
axs[0].set(xlabel='Frame',ylabel='Degree of linear polarization',title='(a) A rocking dielectric reflection')
axs[1].axhline(1e-6,color='black',lw=1,ls='--');axs[1].set(xlabel='Frame',ylabel='Objective gap to cone minimum',title='(b) Audits remove the false successes')
for ax in axs:ax.grid(alpha=.16);ax.legend(fontsize=9)
for dest in [P,L]:
    fig.savefig(dest/'fresnel_sequence.png',dpi=200,facecolor='white',bbox_inches='tight')
    fig.savefig(dest/'fresnel_sequence.pdf',facecolor='white',bbox_inches='tight')
plt.close(fig)
agg=[]
for dataset in colors:
    methods={}
    for method in ['BFGS','L-BFGS-B']:
        z={arm:[r for r in real if r['dataset']==dataset and r['method']==method and r['arm']==arm] for arm in ['cold_start','warm_start','curvature_gate','physical_audit']}
        methods[method]={'n':len(z['warm_start']),'cold_false':sum(r['false_success'] for r in z['cold_start']),'warm_false':sum(r['false_success'] for r in z['warm_start']),'curvature_false':sum(r['false_success'] for r in z['curvature_gate']),'curvature_detected_on_raw':sum(r['false_success'] and r['trigger'] for r in z['warm_start']),'physical_rejected_on_raw':sum(r['false_success'] and not r['physical_accept'] for r in z['warm_start']),'max_warm_gap':max(r['gap'] for r in z['warm_start']),'max_warm_radius':max(r['radius'] for r in z['warm_start']),'physical_restart_native_false':sum(r['false_success'] for r in z['physical_audit']),'physical_restart_interventions':sum(r['intervene'] for r in z['physical_audit']),'native_failures':sum(not r['success'] for vals in z.values() for r in vals)}
    agg.append({'dataset':dataset,'methods':methods})
(P/'PAPER_AGGREGATES.json').write_text(json.dumps({'datasets':agg,'figure_selection':{'fresnel_sequence':int(seq),'rule':'maximum BFGS raw objective gap across all fixed sequences'},'note':'Counts include separate strata and repeated inputs across histories; no independent-scene or prevalence inference.'},indent=2)+'\n')
print(json.dumps(agg))
