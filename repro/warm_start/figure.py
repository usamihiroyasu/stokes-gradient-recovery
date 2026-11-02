"""Principal figure for the registered, dependent two-frame conditions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads((args.input/'main_results.json').read_text());audit=json.loads((args.input/'audited_summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained');colors=['#2b6cb0','#b85824','#4f7c54'];methods=['BFGS','L-BFGS-B','trust-exact']
    x=np.arange(3)
    ax=axes[0,0]
    for j,method in enumerate(methods):
        qs=[q for q in audit['summary'] if q['method']==method and q['gtol']==1e-6]
        vals=[q['baseline_false_success'] for q in qs]
        ax.bar(x+(j-1)*.23,vals,width=.22,color=colors[j],label=method)
    ax.set(xticks=x,xticklabels=['0','0.05','0.20'],xlabel='Change in polarization angle (rad)',ylabel='False successes / 60 conditions',ylim=(0,19),title='(a) Same-solver warm starts, gtol = 10⁻⁶')
    ax.legend(frameon=False,fontsize=7)
    ax=axes[0,1]
    for method,color in zip(methods[:2],colors):
        rs=[r for r in data['records'] if r['method']==method and r['gtol']==1e-6 and r['angle_change']==0]
        xx=[max(abs(r['diagnostic']['radius']-1),1e-10) for r in rs];yy=[max(r['baseline']['gap'],1e-12) for r in rs]
        ax.scatter(xx,yy,s=12,color=color,alpha=.65,label=method)
        fail=[r for r in rs if not r['baseline']['success']]
        for r in fail:ax.scatter(max(abs(r['diagnostic']['radius']-1),1e-10),max(r['baseline']['gap'],1e-12),marker='x',s=50,color='black',linewidth=1.2)
    ax.axvline(1e-3,color='.5',lw=.8,ls='--');ax.axhline(1e-6,color='.5',lw=.8,ls='--')
    ax.set(xscale='log',yscale='log',xlabel='Final |r − 1|',ylabel='Cone objective gap',title='(b) Radius test and the retained solver failure')
    ax.text(.04,.06,'× native solver failure',transform=ax.transAxes,fontsize=7)
    ax=axes[1,0]
    qs=[q for q in audit['summary'] if q['gtol']==1e-6 and q['angle_change']==0]
    for j,(key,label,color) in enumerate([('always_extra_f','Always restart','#718096'),('trigger_extra_f','Triggered restart','#2b6cb0')]):
        vals=[q[key] for q in qs];bars=ax.bar(x+(j-.5)*.3,vals,width=.29,label=label,color=color)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+7,str(val),ha='center',fontsize=7)
    ax.set(xticks=x,xticklabels=methods,ylabel='Additional function/gradient calls',ylim=(0,580),title='(c) Restart work at zero angle change')
    ax.legend(frameon=False,fontsize=7)
    ax.set_xlabel('Gate Hessians: 17 / 21 / 5 (separate)',fontsize=7)
    ax=axes[1,1]
    for method,color in zip(methods[:2],colors):
        qs=[q for q in audit['summary'] if q['method']==method and q['angle_change']==0]
        vals=[q['baseline_false_success'] for q in qs];remaining=[q['triggered_restart_false_success'] for q in qs]
        ax.plot(x,vals,'o-',color=color,lw=1,label=method+' baseline');ax.plot(x,remaining,'s--',color=color,lw=1,label=method+' audited')
    ax.set(xticks=x,xticklabels=['10⁻⁴','10⁻⁶','10⁻⁸'],xlabel='Gradient tolerance',ylabel='False successes / 60 conditions',ylim=(-.7,18.5),title='(d) A local trigger is not a general stopping test')
    ax.legend(frameon=False,fontsize=6.5,ncol=1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(args.output.with_suffix('.png'),dpi=200,bbox_inches='tight')

if __name__=='__main__':main()
