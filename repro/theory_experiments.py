"""Deterministic Stokes-head diagnostics and figures. No neural network is trained."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "latex" / "images"
RESULTS = ROOT / "results"
FIG.mkdir(exist_ok=True, parents=True)
RESULTS.mkdir(exist_ok=True)
BETA = .5
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
    "pdf.fonttype":42, "ps.fonttype":42, "axes.spines.top":False,
    "axes.spines.right":False, "axes.linewidth":.6})
BLUE, ORANGE, PURPLE, GREY = "#0075A8", "#CB721B", "#8061A8", "#6B7379"


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), metadata={"Title":name.replace("_", " ")})
    fig.savefig(FIG / (name + ".png"), dpi=300)
    plt.close(fig)


def head(w, mode):
    r2 = np.sum(w*w, axis=-1)
    if mode == "F":
        a = 2/(1+r2)
        b = -4/(1+r2)**2
    elif mode == "G":
        a = 1/np.sqrt(1+r2)
        b = -1/(1+r2)**1.5
    elif mode == "P":
        r = np.sqrt(r2)
        a = 1/np.maximum(1,r)
        b = np.where(r>1, -1/np.maximum(r,1)**3, 0.)
    else:
        raise ValueError(mode)
    value = a[...,None]*w
    J = a[...,None,None]*np.eye(2) + b[...,None,None]*w[...,None]*w[...,None,:]
    return value,J


# Independent finite-difference checks of maps, matched-output spectra and full-head stationarity.
max_fd = max_spec = max_stationary = max_hessian = 0.
h = 1e-5
for d in [.05,.3,.8,.99]:
    s = np.sqrt(1-d*d)
    for angle in [0.,.71,2.1]:
        e = np.array([np.cos(angle), np.sin(angle)])
        for r,mode,expected in [(d/(1+s),"F",[s*(1+s),1+s]),
                                 ((1+s)/d,"F",[-s*(1-s),1-s]),
                                 (d/s,"G",[s**3,s])]:
            w=r*e; q,J=head(w,mode)
            fd=np.column_stack([(head(w+h*np.eye(2)[k],mode)[0]-head(w-h*np.eye(2)[k],mode)[0])/(2*h) for k in range(2)])
            max_fd=max(max_fd,float(abs(fd-J).max()))
            max_spec=max(max_spec,float(abs(np.sort(np.linalg.svd(J,compute_uv=False))-np.sort(abs(np.array(expected)))).max()))
            assert np.max(abs(q-d*e))<1e-12


def full_loss_grad(x, d, mode="F", Istar=1.):
    # Coordinates are (I, wx, wy); a regular intensity activation preserves stationary Hessian inertia.
    I=x[0]; q,J=head(x[1:],mode)
    errI=I-Istar; errP=I*q-np.array([Istar*d,0.])
    value=.5*(errI*errI+BETA*(errP@errP))
    grad=np.r_[errI+BETA*(q@errP), BETA*I*(J.T@errP)]
    return value,grad


for d in [.05,.3,.8,.99]:
    for sign in [1.,-1.]:
        I=(1+BETA*sign*d)/(1+BETA)
        x=np.array([I,sign,0.]); value,grad=full_loss_grad(x,d)
        H=np.column_stack([(full_loss_grad(x+h*np.eye(3)[k],d)[1]-full_loss_grad(x-h*np.eye(3)[k],d)[1])/(2*h) for k in range(3)])
        # The tangent basis is the same for these axis-aligned fold points.
        expected=np.diag([1+BETA,-BETA*I*(I-sign*d),BETA*I*sign*d])
        max_stationary=max(max_stationary,float(abs(grad).max()))
        max_hessian=max(max_hessian,float(abs(H-expected).max()))
        if sign>0:
            assert abs(value-(1-d)**2/6)<1e-13
        assert np.linalg.eigvalsh(H).min()<0
assert max_fd<1e-8 and max_spec<1e-12 and max_hessian<1e-7 and max_stationary<1e-12

# Matched physical initialization, fixed I=1, target q=(0.3,0), direct raw-coordinate GD.
d0, target, eta, steps = .8, np.array([.3,0.]), .1, 4000
s0=np.sqrt(1-d0*d0)
starts={"F inner":("F",d0/(1+s0)),"F outer":("F",(1+s0)/d0),
        "G monotone":("G",d0/s0),"Disk interior":("P",d0)}
curves={}; final={}
for name,(mode,r) in starts.items():
    w=np.array([r,0.]); losses=[]
    for k in range(steps+1):
        q,J=head(w,mode); err=q-target
        losses.append(.5*BETA*float(err@err))
        if k<steps: w-=eta*BETA*(J.T@err)
    curves[name]=np.array(losses)
    final[name]={"initial_raw_radius":float(r),"initial_output_degree":d0,
                 "final_loss":losses[-1],"final_raw_radius":float(np.linalg.norm(w))}
assert max(abs(v[0]-curves["F inner"][0]) for v in curves.values())<1e-14
np.savez_compressed(RESULTS/"matched_output_gd.npz",**{k.replace(" ","_"):v for k,v in curves.items()})

# Exact conditioning and pullback mobility; not measured training performance.
d=np.linspace(.005,.999,800); s=np.sqrt(1-d*d)
fig,axes=plt.subplots(1,3,figsize=(7,2.65))
fig.subplots_adjust(left=.07,right=.99,top=.84,bottom=.22,wspace=.48)
axes[0].semilogy(d,1/s,label="F (both sheets)",color=BLUE)
axes[0].semilogy(d,1/s**2,label="G monotone",color=PURPLE,ls="--")
axes[0].set(xlabel="Output DoLP d",ylabel="Jacobian condition number",ylim=(.8,1e3))
axes[0].legend(frameon=False,fontsize=6.7,loc="upper left")
axes[1].semilogy(d,s*s*(1+s)**2,label="F inner",color=BLUE)
axes[1].semilogy(d,s*s*(1-s)**2,label="F outer",color=ORANGE)
axes[1].semilogy(d,s**6,label="G",color=PURPLE,ls="--")
axes[1].set(xlabel="Output DoLP d",ylabel=r"Radial mobility $\lambda_r^2$",ylim=(1e-10,10))
axes[1].legend(frameon=False,fontsize=6.7,loc="lower center")
for (name,loss),color,ls in zip(curves.items(),[BLUE,ORANGE,PURPLE,GREY],["-","-","--",":"]):
    axes[2].semilogy(np.arange(steps+1),np.maximum(loss,1e-20),label=name,color=color,ls=ls,lw=1.1)
axes[2].set(xlabel="Gradient steps",ylabel="Fixed-I quadratic loss",ylim=(1e-20,.2))
axes[2].legend(frameon=False,fontsize=6.4,loc="lower right")
for ax,title in zip(axes,["(a) Conditioning","(b) Radial mobility","(c) Direct-output descent"]):
    ax.set_title(title,loc="left",fontsize=8.1,pad=12)
    ax.tick_params(labelsize=7,width=.6); ax.grid(color="#DFE5E8",lw=.4)
save(fig,"conditioning_and_dynamics")

# Aligned (I,r) slices of the full Stokes W-loss, target I*=1, d*=0.5.
rr=np.linspace(.001,4.1,500); II=np.linspace(.3,1.3,280)
R,V=np.meshgrid(rr,II); db=.5; Ib=(2+db)/3
fig,axes=plt.subplots(1,3,figsize=(7,2.4))
fig.subplots_adjust(left=.07,right=.91,top=.84,bottom=.20,wspace=.22)
for ax,mode,title in zip(axes,["F","G","P"],["(a) Folded map F","(b) Monotone map G","(c) Disk projection P"]):
    radi={"F":2*R/(1+R*R),"G":R/np.sqrt(1+R*R),"P":np.minimum(R,1)}[mode]
    loss=.5*((V-1)**2+BETA*(V*radi-db)**2)
    im=ax.pcolormesh(rr,II,np.maximum(loss,1e-6),norm=LogNorm(1e-6,1),cmap="viridis",shading="nearest",rasterized=True)
    ax.set(xlabel="Raw radius r",ylim=(.3,1.3),xticks=[0,1,2,3,4])
    ax.set_title(title,loc="left",fontsize=8.2,pad=12)
    if mode=="F":
        sd=np.sqrt(1-db*db); ax.plot([db/(1+sd),(1+sd)/db],[1,1],"*",mfc="white",mec="black",ms=7,mew=.5)
        ax.plot(1,Ib,"x",color="#F5B942",ms=6,mew=1.2)
        ax.annotate("strict saddle",(1,Ib),(1.6,.64),fontsize=6.8,color="#23333E",arrowprops={"arrowstyle":"-","color":"#23333E","lw":.6})
    elif mode=="G": ax.plot(db/np.sqrt(1-db*db),1,"*",mfc="white",mec="black",ms=7,mew=.5)
    else:
        ax.plot(db,1,"*",mfc="white",mec="black",ms=7,mew=.5)
        ax.plot([1.01,4.1],[Ib,Ib],color="#F5B942",ls="--",lw=1.3)
        ax.text(1.35,.67,"flat suboptimal minima",color="#23333E",fontsize=6.3)
    ax.tick_params(labelsize=7,width=.6)
axes[0].set_ylabel("Intensity I"); axes[1].set_yticklabels([]);axes[2].set_yticklabels([])
cb=fig.colorbar(im,cax=fig.add_axes([.933,.20,.014,.64])); cb.ax.tick_params(labelsize=7);cb.ax.set_title(r"$\ell$",fontsize=8,pad=7)
save(fig,"stationary_landscapes")

# Authentic saved renderer reference: no interpolation and no new rendering.
reference=ROOT/"data"/"renderer_reference.npz"
raw=np.load(reference); S=raw["stokes"][:,:,:3,0].astype(np.float64)
Istar=S[:,:,0]; p=S[:,:,1:]; dstar=np.linalg.norm(p,axis=-1)/Istar
assert S.shape==(32,32,3) and np.max(dstar)<1 and np.min(Istar)>0
fig,axes=plt.subplots(1,4,figsize=(7,2.35))
fig.subplots_adjust(left=.03,right=.99,bottom=.28,top=.84,wspace=.32)
for ax,arr,title,cmap,lo,hi,label in zip(axes,[Istar,S[:,:,1],S[:,:,2],100*dstar],
    ["(a) Reference I","(b) Reference Q","(c) Reference U","(d) Reference DoLP"],
    ["gray","RdBu_r","RdBu_r","viridis"],[.95,-.05,-.05,0],[1.25,.05,.05,5],
    ["relative radiance","relative radiance","relative radiance","percent"]):
    im=ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,interpolation="nearest")
    ax.set_title(title,fontsize=8,pad=10);ax.set_xticks([]);ax.set_yticks([])
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.08,pad=.08)
    cb.set_ticks([lo,(lo+hi)/2,hi])
    cb.ax.tick_params(labelsize=6.8);cb.set_label(label,fontsize=7)
save(fig,"renderer_reference")

# Constructed full-head non-KKT stationary field, using the reference direction (oracle initialization).
e=np.zeros_like(p); norms=np.linalg.norm(p,axis=-1)
np.divide(p,norms[...,None],out=e,where=norms[...,None]>0);e[norms==0]=[1.,0.]
Ib=Istar*(2+dstar)/3; a=np.log(np.expm1(Ib)); w=e.copy()
a_initial=a.copy(); w_initial=w.copy()
losses=[]; max_grad=0.
for k in range(1001):
    I=np.logaddexp(0,a); f,J=head(w,"F"); errI=I-Istar; errP=I[...,None]*f-p
    L=.5*(errI**2+BETA*np.sum(errP**2,axis=-1));losses.append(float(np.mean(L/Istar**2)))
    ga=(1/(1+np.exp(-a)))*(errI+BETA*np.sum(f*errP,axis=-1))
    gw=BETA*I[...,None]*np.einsum("...ji,...j->...i",J,errP)
    max_grad=max(max_grad,float(np.max(abs(ga))),float(np.max(abs(gw))))
    if k<1000: a-=.1*ga;w-=.1*gw
relative_error=2*L/Istar**2
assert np.max(abs(relative_error-(1-dstar)**2/3))<1e-11
assert max_grad<1e-11
Sstationary=np.concatenate([I[...,None],I[...,None]*f],axis=-1)
np.savez_compressed(RESULTS/"constructed_stationary_field.npz",reference=S,stationary=Sstationary,
                    relative_W_error=relative_error,normalized_half_loss=np.array(losses),
                    initial_a=a_initial,initial_w=w_initial,final_a=a,final_w=w)
fig,axes=plt.subplots(1,3,figsize=(7,2.55))
fig.subplots_adjust(left=.06,right=.985,bottom=.25,top=.84,wspace=.30)
for ax,arr,title,cmap,lo,hi,label in zip(axes,[I,100*np.linalg.norm(f,axis=-1),relative_error],
    ["(a) Stationary intensity","(b) Stationary DoLP","(c) Normalized W-error"],
    ["gray","viridis","magma"],[.6,0,0],[1.25,100,1/3],
    ["relative radiance","percent",r"$\|s-s^*\|_W^2/(I^*)^2$"]):
    im=ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,interpolation="nearest")
    ax.set_title(title,fontsize=8.2,pad=10);ax.set_xticks([]);ax.set_yticks([])
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.08,pad=.08)
    cb.ax.tick_params(labelsize=7);cb.set_label(label,fontsize=7.6)
save(fig,"constructed_stationary_field")

summary={"scope":"direct-output deterministic numerical diagnostics, not shared-network training",
    "maps":["F folded","G monotone","P disk"],"beta":BETA,"seed":"none; deterministic",
    "matched_GD":{"target":[1.,.3,0.],"initial_normalized_output":[.8,0.],
                  "step_size":eta,"steps":steps,"results":final},
    "independent_checks":{"jacobian_cases":36,"full_head_fold_cases":8,
        "finite_difference_step":h,"max_fd_jacobian_error":max_fd,
        "max_singular_value_error":max_spec,"max_stationary_gradient":max_stationary,
        "max_fd_full_hessian_error":max_hessian},
    "renderer":{"source_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),
        "shape":list(S.shape),"first_RGB_output_channel":True,"max_DoLP":float(dstar.max()),
        "mean_DoLP":float(dstar.mean()),"min_I":float(Istar.min()),"max_I":float(Istar.max())},
    "constructed_stationary_field":{"pixels":1024,"step_size":.1,"steps":1000,
        "oracle_reference_direction_initialization":True,"initial_mean_half_normalized_W_loss":losses[0],
        "final_mean_half_normalized_W_loss":losses[-1],"max_raw_gradient_over_steps":max_grad,
        "max_parameter_drift":float(max(abs(a-a_initial).max(),abs(w-w_initial).max())),
        "mean_normalized_W_error":float(relative_error.mean()),
        "min_normalized_W_error":float(relative_error.min()),"max_normalized_W_error":float(relative_error.max())},
    "new_neural_network_training_runs":0,"new_renderer_runs":0}
(RESULTS/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
