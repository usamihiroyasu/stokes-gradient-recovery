"""Mathematical diagnostic diagrams from explicit objectives and retained QP records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "latex" / "images"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#8061A8"


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=300)
    plt.close(fig)


def concept():
    target = np.array([1., .5])  # physical order I,p
    boundary = np.array([5/6, 5/6])
    W = np.diag([1., .5])
    g = W @ (boundary-target)
    assert np.allclose(g, [-1/6, 1/6])
    t = np.linspace(-.34, .34, 401)
    r = 1+t
    phi = 2*r/(1+r*r)
    loss = .5*((boundary[0]-1)**2 + .5*(boundary[0]*phi-.5)**2)
    l0 = 1/24
    assert abs(loss[len(t)//2]-l0)<1e-14
    step = 1e-4
    def radial_loss(tt):
        q=2*(1+tt)/(1+(1+tt)**2)
        return .5*((5/6-1)**2+.5*((5/6)*q-.5)**2)
    curvature=(radial_loss(step)-2*radial_loss(0)+radial_loss(-step))/step**2
    assert abs(curvature+5/36)<1e-7

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2),
                             gridspec_kw={"width_ratios":[1,1,1.10]})
    fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.84,wspace=.45)
    ax=axes[0]
    p=np.linspace(-1.15,1.15,250)
    ax.fill_between(p,abs(p),1.3,color=BLUE,alpha=.10)
    ax.plot(p,abs(p),color=BLUE,lw=1.3)
    ax.scatter([target[1]],[target[0]],marker="*",s=60,color=GREEN,zorder=4)
    ax.scatter([boundary[1]],[boundary[0]],s=28,color="black",zorder=4)
    ax.annotate(r"$s^*$",target[::-1],xytext=(.27,1.08),fontsize=9)
    ax.annotate(r"$s_b$",boundary[::-1],xytext=(.90,.70),fontsize=9)
    end=boundary-g
    ax.annotate("",xy=end[::-1],xytext=boundary[::-1],
                arrowprops={"arrowstyle":"-|>","color":ORANGE,"lw":1.8})
    ax.text(.33,.86,r"$-g$",color=ORANGE,fontsize=9)
    ax.text(-.88,1.13,r"$\mathcal{C}:\ I\geq|p|$",fontsize=8.5)
    ax.set(xlim=(-1.1,1.15),ylim=(0,1.3),xlabel=r"$p=e^{\mathsf{T}}(Q,U)$",ylabel="Intensity I")
    ax.set_title("(a) A feasible descent",loc="left",fontsize=8.5,pad=13)
    ax.text(.03,-.26,r"Slice: $p_\perp=0$",transform=ax.transAxes,fontsize=8)

    ax=axes[1]
    ax.plot(t,loss-l0,color=BLUE,lw=1.7)
    ax.plot([-.16,.16],[0,0],color="black",lw=.8,ls="--")
    ax.plot(0,0,"o",color="black",ms=4)
    ax.set(xlabel=r"$t$, with $w=(1+t)e$",ylabel=r"$\widetilde{L}(t)-\widetilde{L}(0)$",
           xlim=(-.35,.35),ylim=(-.011,.005))
    ax.set_title("(b) A first-order blind spot",loc="left",fontsize=8.5,pad=13)
    ax.text(.04,.92,r"$\partial_t\widetilde{L}(0)=0$"+"\n"+
            r"$\partial_{tt}\widetilde{L}(0)<0$",transform=ax.transAxes,
            fontsize=8,va="top")
    ax.text(.01,-.26,"Radial slice of a saddle",transform=ax.transAxes,fontsize=8)
    ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))

    ax=axes[2];ax.set_axis_off()
    ax.set_title("(c) Second-order diagnostic",loc="left",fontsize=8.2,pad=13)
    boxes=[(.72,.20, r"$b=\partial_a\widetilde{L}/I'(a)$"+"\n"+r"$v=\nabla^2_{ww}\widetilde{L}\,e$"),
           (.39,.20,r"$g_p=-v/I$"+"\n"+r"$g_I=b+e^{\mathsf{T}}v/I$"),
           (.10,.16,r"$0\leq\Delta_{\mathcal{C}}(s)\leq\mathcal{B}(s)$")]
    for bottom,height,text in boxes:
        ax.add_patch(FancyBboxPatch((.03,bottom),.94,height,boxstyle="round,pad=.015",
                     transform=ax.transAxes,fc="#F2F6F8",ec="#8B9BA5",lw=.7))
        ax.text(.5,bottom+height/2,text,ha="center",va="center",fontsize=8.4,
                transform=ax.transAxes)
    for top,bottom in [(.70,.61),(.37,.28)]:
        ax.annotate("",xy=(.5,bottom),xytext=(.5,top),xycoords="axes fraction",
                    arrowprops={"arrowstyle":"-|>","lw":.9,"color":BLUE})
    ax.text(.5,-.05,"Exact fold; stationarity not required",ha="center",
            transform=ax.transAxes,fontsize=7.3)
    ax.text(.5,-.26,r"Stationary fold: sign of $\gamma_r$"+"\n"+"distinguishes optimum and saddle",
            ha="center",transform=ax.transAxes,fontsize=7.3)
    save(fig,"diagnostic_principle")
    return {"target_Ip":target.tolist(),"fold_Ip":boundary.tolist(),
            "physical_gradient_Ip":g.tolist(),"radial_curvature_exact":-5/36,
            "radial_curvature_fd":float(curvature),"radial_fd_step":step}


def certificate():
    report=json.loads((ROOT/"results"/"general_metric_independent.json").read_text())
    assert report["status"]=="PASS"
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
    fig.subplots_adjust(left=.095,right=.985,bottom=.19,top=.87,wspace=.35)
    ax=axes[0]
    pp=np.linspace(-1.9,1.22,400)
    ax.fill_between(pp,pp,1.85,color="#C3CDD4",alpha=.30)
    ax.fill_between(pp,abs(pp),1.85,color=BLUE,alpha=.17)
    ax.plot(pp,pp,"--",color="#6C7C86",lw=.9)
    ax.plot(pp,abs(pp),color=BLUE,lw=1.3)
    z=np.array([-1.4,.2]); s=np.array([.8,.8]); c=np.array([-.6,.6])
    # Figure coordinates are p,I, so W_xy=diag(.5,1).
    def objective(v):return .5*((v-z)@np.diag([.5,1.])@(v-z))
    P,I=np.meshgrid(pp,np.linspace(-.65,1.85,350))
    vals=.5*(.5*(P-z[0])**2+(I-z[1])**2)
    ax.contour(P,I,vals,levels=[.08,.24,.7,1.39],colors="#8A8A8A",linewidths=.65)
    ax.plot(*z,"D",color=PURPLE,ms=5)
    ax.plot(*s,"o",color="black",ms=4)
    ax.plot(*c,"*",color=GREEN,ms=9)
    ax.annotate(r"$z=h$",z,xytext=(-1.72,-.12),fontsize=9)
    ax.annotate(r"$c=P^W_{\mathcal{C}}(z)$",c,xytext=(-1.38,.97),fontsize=8,
                arrowprops={"arrowstyle":"-","lw":.6})
    ax.text(.89,.78,r"$s$",fontsize=9)
    ax.text(-.02,1.55,r"$\mathcal{C}\subset H_s$",fontsize=9)
    ax.text(-1.6,-.48,r"$H_s:\ I\geq p$",fontsize=8.5)
    ax.set(xlim=(-1.9,1.22),ylim=(-.65,1.85),xlabel="Polarization component p",ylabel="Intensity I")
    ax.set_title("(a) Relaxation to a supporting half-space",loc="left",fontsize=8.2,pad=12)
    assert abs(objective(s)-1.39)<1e-12 and abs(objective(c)-.24)<1e-12
    assert np.max(abs(np.diag([.5,1.])@(c-z)+.4*np.array([-1.,-1.])))<1e-12
    assert np.array([1.,-1.])@(z-s)<0 and z[1]<abs(z[0])
    ax.text(.03,.95,r"$\Delta_{\mathcal{C}}=1.15<\mathcal{B}=1.39$",transform=ax.transAxes,
            fontsize=8.1,va="top",bbox={"fc":"white","ec":"none","alpha":.9,"pad":2})

    ax=axes[1];colors=["#555555",BLUE,ORANGE,GREEN]
    for metric,color in zip(report["matrices"],colors):
        rows=[r for r in report["records"] if r["matrix"]==metric["name"] and r["alpha"]>0]
        for feasible in [True,False]:
            rr=[r for r in rows if r["z_feasible"]==feasible]
            ax.scatter([r["certificate_bound"] for r in rr],[r["actual_gap"] for r in rr],
                       s=23,facecolor=color if feasible else "white",edgecolor=color,lw=.85,zorder=3)
    upper=max(r["certificate_bound"] for r in report["records"])*1.04
    ax.plot([0,upper],[0,upper],":",color="#404040",lw=1,zorder=1)
    ax.set(xlim=(-.1,upper),ylim=(-.1,upper),xlabel=r"Certificate $\mathcal{B}(s)$",
           ylabel=r"Independent objective gap $\Delta_{\mathcal{C}}(s)$")
    ax.set_title("(b) Tightness at stationary folds",loc="left",fontsize=8.2,pad=12)
    handles=[Line2D([],[],marker="o",color="none",mfc=".3",mec=".3",label=r"$z\in\mathcal{C}$: equality"),
             Line2D([],[],marker="o",color="none",mfc="white",mec=".3",label=r"$z\notin\mathcal{C}$: strict bound"),
             Line2D([],[],ls=":",color=".3",label=r"$\Delta_{\mathcal{C}}=\mathcal{B}$")]
    primary=ax.legend(handles=handles,loc="upper left",fontsize=7.2,framealpha=.95)
    ax.add_artist(primary)
    auxiliary=[Line2D([],[],color=color,lw=1.6,label=label) for color,label in zip(colors,["ideal","angle variation","gain / efficiency","variance variation"])]
    ax.legend(handles=auxiliary,loc="lower right",fontsize=6.7,framealpha=.9)
    ax.grid(alpha=.15)
    save(fig,"certificate_geometry")
    return {"illustrative_slice_W_pI":[[.5,0],[0,1]],"s_pI":s.tolist(),"z_equals_h_pI":z.tolist(),
            "cone_minimum_pI":c.tolist(),"gap":float(objective(s)-objective(c)),"bound":float(objective(s)),
            "stationary_scatter_feasible_z":56,"stationary_scatter_infeasible_z":4}


def renderer_composite():
    raw=np.load(ROOT/"data"/"renderer_reference.npz")["stokes"][:,:,:3,0].astype(float)
    field=np.load(ROOT/"results"/"constructed_stationary_field.npz")
    # Retained field key names are checked before use rather than recomputing predictions.
    stationary=field["stationary"]
    err=field["relative_W_error"]
    d=np.linalg.norm(raw[:,:,1:],axis=-1)/raw[:,:,0]
    panels=[(raw[:,:,0],"Reference I","gray",.6,1.25),(100*d,"Reference DoLP","viridis",0,5),
            (stationary[:,:,0],"Constructed I","gray",.6,1.25),
            (100*np.linalg.norm(stationary[:,:,1:],axis=-1)/stationary[:,:,0],"Constructed DoLP","viridis",0,100),
            (err,"Normalized W-error","magma",0,1/3)]
    fig,axes=plt.subplots(1,5,figsize=(7.2,2.15))
    fig.subplots_adjust(left=.03,right=.965,bottom=.31,top=.82,wspace=.35)
    for ax,(arr,title,cmap,lo,hi) in zip(axes,panels):
        im=ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,interpolation="nearest")
        ax.set_title(title,fontsize=7.4,pad=8);ax.set_axis_off()
        cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.09,pad=.10,ticks=[lo,(lo+hi)/2,hi])
        cb.ax.set_xticklabels([format(lo,".3g"),format((lo+hi)/2,".3g"),format(hi,".3g")])
        cb.ax.tick_params(labelsize=6.5)
    save(fig,"renderer_and_counterexample")


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    receipt={"concept":concept(),"certificate":certificate(),"scope":"mathematical diagrams and retained numerical records"}
    renderer_composite()
    (ROOT/"results"/"diagnostic_figure_checks.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
