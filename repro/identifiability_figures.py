"""Editable scientific figures from checked numerical arrays."""
from pathlib import Path
import os,json
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=HERE.parent/'latex/images'
BLUE,ORANGE,GREEN,GRAY='#0072B2','#D55E00','#009E73','#666666'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=300);plt.close(fig)
def principle(d):
    fig,aa=plt.subplots(1,3,figsize=(7.2,3.05),gridspec_kw={'width_ratios':[1.05,1.15,1.]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.85,wspace=.72)
    colors=[GREEN,GRAY,ORANGE];ax=aa[0];p=np.linspace(.25,1.75,200)
    ax.fill_between(p,p,1.75,color=BLUE,alpha=.12);ax.plot(p,p,color=BLUE,lw=1.1)
    ax.plot(1,1,'ko',ms=4)
    for case,color in zip(d['cases'],colors):
        g=np.array(case['g']);alpha=case['alpha']
        if alpha:
            end=np.array([1.,1.])+g[[1,0]]
            ax.annotate('',xy=end,xytext=(1.,1.),arrowprops={'arrowstyle':'-|>','color':color,'lw':1.7})
            ax.text(end[0]-.12,end[1]+(.08 if alpha<0 else -.12),r'$\alpha='+str(alpha)+'$',color=color,fontsize=7)
    ax.text(.45,.61,r'$\alpha=0:\ g=0$',fontsize=7,color=GRAY)
    ax.text(.37,1.51,r'$I\geq|p|$',fontsize=8);ax.text(1.05,1.05,r'$s$',fontsize=8)
    ax.set(xlim=(.3,1.7),ylim=(.3,1.7),xlabel='Polarization component p',ylabel='Intensity I')
    ax.set_title('(a) Physical gradients at one output',fontsize=8,loc='left',pad=12)
    ax=aa[1]
    for case,color in zip(d['cases'],colors):
        ax.plot(d['t'],case['radial_loss'],color=color,lw=1.5,label=r'$\alpha='+str(case['alpha'])+'$')
    ax.axhline(0,color='.25',lw=.7,ls=':');ax.plot(0,0,'ko',ms=3)
    ax.set(xlim=(-.35,.35),xlabel=r'$t$, with $w=(1+t)e$',ylabel='Offset radial objective')
    ax.set_title('(b) Indistinguishable at first order',fontsize=8,loc='left',pad=12)
    ax.legend(fontsize=6.4,loc='lower left',framealpha=.9)
    ax.text(.5,.96,'same value\nsame first derivatives',ha='center',va='top',transform=ax.transAxes,fontsize=6.8)
    ax=aa[2];ys=np.array([2,1,0])
    ax.barh(ys,[c['curvature'] for c in d['cases']],color=colors,height=.35)
    ax.axvline(0,lw=.8,color='.3')
    ax.set_yticks(ys,[r'$\alpha=-0.3$'+'\noptimum',r'$\alpha=0$'+'\noptimum',r'$\alpha=0.3$'+'\nstrict saddle'],fontsize=6.7)
    ax.set(xlim=(-.43,.43),ylim=(-.55,2.55),xlabel=r'Radial curvature $\gamma_r$')
    ax.set_title('(c) Curvature',fontsize=8,loc='left',pad=12)
    ax.text(.5,-.33,'Constructed stationary family',ha='center',transform=ax.transAxes,fontsize=6.8)
    save(fig,'identifiability_principle')
def geometry(d):
    fig,aa=plt.subplots(1,3,figsize=(7.2,3.05),gridspec_kw={'width_ratios':[1.15,.9,1.15]})
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.86,wspace=.57)
    ax=aa[0];p=np.linspace(-1.9,1.2,250);h=np.array(d['pI_h']);s=np.array(d['pI_s']);u=np.array(d['pI_opt'])
    ax.fill_between(p,p,1.85,color=GRAY,alpha=.13);ax.fill_between(p,abs(p),1.85,color=BLUE,alpha=.14)
    ax.plot(p,p,'--',color=GRAY,lw=.8);ax.plot(p,abs(p),color=BLUE,lw=1.1)
    P,I=np.meshgrid(p,np.linspace(-.65,1.85,220));vals=.5*(.5*(P-h[0])**2+(I-h[1])**2)
    ax.contour(P,I,vals,levels=[.24,.7,1.39],colors='.6',linewidths=.65)
    for point,marker,color,label,offset in [(s,'o','black',r'$s$',(.05,.04)),(h,'D','#8061A8',r'$h_s=z$',(-.22,-.24)),(u,'*',GREEN,r'$u^*_{\mathcal{C}}$',(-.52,.21))]:
        ax.plot(*point,marker=marker,color=color,ms=7 if marker=='*' else 4);ax.text(point[0]+offset[0],point[1]+offset[1],label,fontsize=8)
    ax.text(-.14,1.58,r'$\mathcal{C}\subset H_e$',fontsize=8)
    ax.set(xlim=(-1.9,1.2),ylim=(-.65,1.85),xlabel='Polarization p',ylabel='Intensity I')
    ax.set_title('(a) Relaxed feasible set',loc='left',fontsize=8,pad=12)
    ax=aa[1];values=d['values'];lo=values['halfspace_minimum'];mid=values['cone_minimum'];hi=values['current']
    for y,label,color in [(lo,r'$L(h_s)$',GRAY),(mid,r'$L(u^*_{\mathcal{C}})$',GREEN),(hi,r'$L(s)$','black')]:
        ax.plot([-.05,.65],[y,y],color=color,lw=1.2);ax.text(.02,y+.045,label,fontsize=8)
    for x,low,color,label in [(1.2,mid,GREEN,'gap = 1.15'),(2.05,lo,BLUE,'bound = 1.39')]:
        ax.annotate('',xy=(x,hi),xytext=(x,low),arrowprops={'arrowstyle':'<->','color':color,'lw':1.2})
        ax.text(x+.11,.5*(hi+low),label,rotation=90,va='center',fontsize=7.2,color=color)
    ax.set(xlim=(-.2,2.7),ylim=(-.1,1.65),ylabel='Objective value');ax.set_xticks([]);ax.set_yticks([0,.24,1.39])
    ax.set_title('(b) Gap and upper bound',loc='left',fontsize=8,pad=12)
    ax=aa[2];colors=[GRAY,BLUE,ORANGE,GREEN]
    for row in d['scatter']:
        col=colors[row['matrix_index']];ax.scatter(row['bound'],row['computed_gap'],s=19,facecolor=col if row['z_feasible'] else 'white',edgecolor=col,lw=.8,zorder=3)
    upper=max(x['bound'] for x in d['scatter'])*1.04;ax.plot([0,upper],[0,upper],':',color='.3',lw=.8)
    handles=[Line2D([],[],marker='o',mfc='.4',mec='.4',ls='none',label=r'$z\in\mathcal{C}$'),Line2D([],[],marker='o',mfc='white',mec='.4',ls='none',label=r'$z\notin\mathcal{C}$')]
    ax.legend(handles=handles,fontsize=6.5,loc='upper left');ax.grid(alpha=.15)
    ax.set(xlim=(-.1,upper),ylim=(-.1,upper),xlabel='Half-space bound',ylabel='Independently computed gap')
    ax.set_title('(c) Stationary tightness',loc='left',fontsize=8,pad=12)
    save(fig,'objective_gap_geometry')
def near(d, anisotropy):
    fig,aa=plt.subplots(2,2,figsize=(7.2,5.0));fig.subplots_adjust(left=.09,right=.985,bottom=.11,top=.93,wspace=.32,hspace=.46)
    ax=aa[0,0]
    for direction,col,label in [('radial',BLUE,'radial: vector / mixed'),('tangential',ORANGE,'tangent: vector')]:
        for side,ls in [('inner','-'),('outer',':')]:
            rows=sorted([x for x in anisotropy['records'] if x['side']==side and x['direction']==direction],key=lambda x:x['distance'])
            x=np.array([row['distance'] for row in rows]);y=[row['normalized_gain'] for row in rows]
            ax.loglog(x,y,ls,color=col,lw=1.5,label=label if side=='inner' else None)
            ax.loglog(x,[row['measured_gain'] for row in rows],'o',color=col,mfc='white',ms=2.2,mew=.6)
    guide=np.array([1e-8,1e-1])
    ax.loglog(guide,.25*guide**2,'--',color=GRAY,lw=.7)
    ax.loglog(guide,.25*guide,'--',color=GRAY,lw=.7)
    ax.text(3e-5,4e-11,'order 2',color=BLUE,fontsize=7)
    ax.text(2e-6,2e-5,'order 1',color=ORANGE,fontsize=7)
    ax.set(xlim=(7e-9,.15),ylim=(1e-17,.3),xlabel=r'Radial distance $|r-1|$',ylabel='Curvature-input gain / I')
    ax.set_title('(a) Curvature-input gains by route',loc='left',fontsize=8.2)
    ax.legend(fontsize=6.4,loc='lower right');ax.grid(alpha=.15)
    ax=aa[0,1];supp=d['support'];sigma=supp['sigma'];phi=supp['phi'];p=np.linspace(.93,1.05,150)
    ax.fill_between(p,p,1.075,color=BLUE,alpha=.12);ax.plot(p,p,color=BLUE,label='true support',lw=1.2)
    ax.plot(p,p+sigma,'--',color=ORANGE,label='plane through interior state',lw=1.2)
    ax.plot(phi,1.,'ko',ms=4);ax.text(phi+.004,1.002,r'$s_r$',fontsize=8)
    ax.annotate('',xy=(phi,1),xytext=(phi,phi),arrowprops={'arrowstyle':'<->','lw':1,'color':GRAY})
    ax.text(phi-.013,.5*(1+phi),r'$\sigma_r$',fontsize=8,ha='right')
    ax.set(xlim=(.93,1.05),ylim=(.93,1.075),xlabel='Polarization p',ylabel='Intensity I')
    ax.legend(fontsize=6.4,loc='lower right',framealpha=.9);ax.set_title('(b) Supporting-plane displacement',loc='left',fontsize=8.2)
    ax=aa[1,0];rr=[x for x in d['comparison'] if x['sheet']=='inner'];x=[a['delta'] for a in rr]
    for key,col,label in [('first',ORANGE,'first-order inverse'),('corrected',GREEN,'model corrected'),('second',BLUE,'naive fold reuse')]:
        ax.loglog(x,[a[key+'_error'] for a in rr],color=col,lw=1.4,label=label+' error')
        ax.loglog(x,[a[key+'_envelope'] for a in rr],'--',color=col,lw=1.1,label=label+' envelope')
    handles,labels=ax.get_legend_handles_labels();ax.legend([handles[0],handles[2],handles[4]],['first inverse','vector corrected','naive reuse'],fontsize=6.4);ax.grid(alpha=.15);ax.set(xlabel=r'Radial distance $|r-1|$',ylabel=r'Gradient error in $W^{-1}$ norm');ax.set_title('(c) Three recovery routes',loc='left',fontsize=8.2)
    ax=aa[1,1];c=d['counterexample'];values=1000*np.array([c['uncorrected_bound'],c['exact_gap'],c['corrected_bound']])
    ax.bar(range(3),values,color=[ORANGE,GREEN,BLUE],width=.58);ax.plot(0,0,'x',color=ORANGE,ms=7)
    for i,y in enumerate(values):ax.text(i,y+.16,f'{y:.3f}' if y>1e-10 else '0',ha='center',fontsize=8)
    ax.set_xticks(range(3),['Uncorrected\nbound','Exact\ngap','Slack-corrected\nbound'],fontsize=7)
    ax.set(ylim=(-.2,6.2),ylabel=r'Objective difference ($\times10^{-3}$)')
    ax.text(.5,.94,'exact physical gradient supplied',transform=ax.transAxes,ha='center',va='top',fontsize=7)
    ax.set_title('(d) Correct gradient, invalid zero bound',loc='left',fontsize=8.2)
    save(fig,'near_fold_information')
if __name__=='__main__':
    d=json.loads((HERE/'results/identifiability_checks.json').read_text());assert d['status']=='PASS';OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((HERE/'results/corrected_recovery_checks.json').read_text());assert c['status']=='PASS'
    d['figure3']['comparison']=c['radius_records']
    anisotropy=json.loads((HERE/'results/anisotropic_curvature_checks.json').read_text());assert anisotropy['status']=='PASS'
    principle(d['figure1']);geometry(d['figure2']);near(d['figure3'],anisotropy);print('generated three checked-data figure pairs, including directional curvature sensitivity')
