"""Render the complete-grid pilot and an explicitly selected worst-case sequence."""
from pathlib import Path
import json,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'results.json').read_text());I=json.loads((HERE/'inputs.json').read_text())
base=[r for r in D['records'] if r['method']=='BFGS' and r['arm']=='baseline']
seq=max(base,key=lambda r:r['final']['gap'])['sequence']
chosen=[r for r in D['records'] if r['method']=='BFGS' and r['sequence']==seq]
by={a:sorted([r for r in chosen if r['arm']==a],key=lambda r:r['frame']) for a in ['baseline','curvature_gate','physical_audit']}
f=np.arange(41);truth=np.array(I[seq]['truth']);theta=np.rad2deg(I[seq]['incidence_radians'])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(13,8.6),layout='constrained')
ax=axs[0,0]
for name,label,color,style in [('baseline','Raw warm start','#c44e2b','-'),('curvature_gate','Curvature audit + restart','#1472a6','--')]:
    s=np.array([r['final']['physical'] for r in by[name]])
    ax.plot(f,np.linalg.norm(s[:,1:],axis=1)/s[:,0],label=label,color=color,ls=style,lw=1.8)
ax.plot(f,np.linalg.norm(truth[:,1:],axis=1)/truth[:,0],color='black',label='Fresnel truth',ls=':',lw=1.8)
ax.set(xlabel='Frame',ylabel='Degree of linear polarization',title=f'(a) Sequence {seq}: selected by maximum baseline gap')
ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=.15)
ax=axs[0,1]
for name,label,color,style in [('baseline','Raw warm start','#c44e2b','-'),('curvature_gate','Curvature audit + restart','#1472a6','--'),('physical_audit','Physical audit + restart','#368d5e',':')]:
    ax.semilogy(f,np.maximum([r['final']['gap'] for r in by[name]],1e-15),label=label,color=color,ls=style,lw=1.8)
ax.axhline(1e-6,color='black',lw=1,ls='--',label='Gap threshold')
ax.set(xlabel='Frame',ylabel='Objective gap to direct cone solution',title='(b) Both audits remove the false acceptances')
ax.legend(fontsize=9);ax.grid(alpha=.15)
ax=axs[1,0]
raw=by['baseline'];rad=np.array([r['initial_diagnostic']['radius'] for r in raw]);bad=np.array([r['final']['false_success'] for r in raw])
ax.plot(f,rad,color='#7360a4',lw=1.8);ax.axhline(1,color='black',ls=':',lw=1)
ax.scatter(f[bad],rad[bad],color='#c44e2b',marker='x',s=60,label='Native success, gap > 1e-6',zorder=3)
ax.set(xlabel='Frame',ylabel='Raw radius',title='(c) Naturally inherited estimates approach the fold')
ax.legend(fontsize=9);ax.grid(alpha=.15)
ax=axs[1,1];ax.axis('off')
labels={'baseline':'Raw warm start','curvature_gate':'Curvature + restart','physical_audit':'Physical + restart','monotone':'Monotone: matched start'}
rows=[]
for s in D['summary']:
    if s['arm']=='monotone':
        s=dict(s,**json.loads((HERE/'matched_monotone_control.json').read_text())['summary'])
    rows.append([s['method'],labels[s['arm']],str(s['false_success']),str(s['solver_nonacceptances'])])
tab=ax.table(cellText=rows,colLabels=['Solver','Arm','False success','Failures'],cellLoc='left',colLoc='left',bbox=[0,.30,1,.64],colWidths=[.17,.43,.21,.19])
tab.auto_set_font_size(False);tab.set_fontsize(9.4)
for (r,c),cell in tab.get_celld().items():
    cell.set_edgecolor('#d8d8d8');cell.set_linewidth(.5)
    if r==0:cell.set_facecolor('#e8eef3');cell.set_text_props(weight='bold')
ax.set_title('(d) Complete grid: 24 sequences, 984 frames per arm',loc='left')
ax.text(0,.23,'All arms see the same observations; each carries its own state.\nFalse success: native acceptance and objective gap > 1e-6.\nMonotone control matches the initial physical state and tolerance.\nCounts are frame counts in this fixed simulation, not population rates.',va='top',fontsize=9.4,linespacing=1.55)
fig.suptitle('Rocking dielectric reflection with Poisson and read noise',fontsize=17,weight='bold')
fig.savefig(HERE/'fresnel_pilot.png',dpi=200,facecolor='white',bbox_inches='tight')
fig.savefig(HERE/'fresnel_pilot.svg',facecolor='white',bbox_inches='tight')
print(json.dumps({'selected_sequence':int(seq),'selection_rule':'largest BFGS baseline objective gap across the fixed complete grid','PNG':'fresnel_pilot.png','SVG':'fresnel_pilot.svg'}))
