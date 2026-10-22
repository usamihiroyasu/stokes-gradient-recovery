import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def panel_a(ax):
    r = np.linspace(0.0, 3.2, 640)
    F = 2*r/(1+r**2)
    G = r/np.sqrt(1+r**2)
    P = np.minimum(r, 1.0)
    ax.plot(r, F, color="#0072B2", lw=1.8, label="F: r- = 0.5, r+ = 2")
    ax.plot(r, G, color="#E69F00", lw=1.8, ls="--", label="G: r = 4/3")
    ax.plot(r, P, color="#009E73", lw=1.8, ls="-.", label="P: r = 0.8")
    ax.axhline(0.8, color="k", lw=1.0, alpha=.55)
    ax.axvline(1.0, color="crimson", lw=1.0, alpha=.75)
    ax.axhline(1.0, color="gray", lw=.8, ls=":")
    ax.text(2.12,1.018,"physical boundary",fontsize=7)
    ax.text(1.025,.08,"fold",rotation=90,fontsize=7,color="crimson")
    pts = [(.5,.8,"#0072B2","o"), (2.0,.8,"#0072B2","s"), (4/3,.8,"#E69F00","^"), (.8,.8,"#009E73","D")]
    for x,y,c,m in pts: ax.plot([x],[y], marker=m, ms=5.5, mfc="white", mec=c, mew=1.4)
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Raw radius r"); ax.set_ylabel("Output DoLP d")
    ax.set_title("(a) Radius maps and matched output",loc="left",fontsize=8.4,pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=.9)

def panel_b(ax):
    # inner: radial .96 tangential 1.6 ; outer singular scales .25 of inner -> .24,.4 (abs values for ellipse radii)
    a_in_r, a_in_t = .96, 1.6
    a_out_r, a_out_t = .24, .40
    th = np.linspace(0, 2*np.pi, 720)
    ax.plot(a_in_r*np.cos(th), a_in_t*np.sin(th), color="#0072B2", lw=1.8, label="inner")
    ax.plot(a_out_r*np.cos(th), a_out_t*np.sin(th), color="#E69F00", lw=1.8, ls="--", label="outer")
    # signed radial arrows from origin: inner +x, outer -x (opposite signs)
    ax.annotate("", xy=(a_in_r,0), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=1.6))
    ax.annotate("", xy=(-a_out_r,0), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color="#E69F00", lw=1.6))
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-2.2, 1.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\delta q_\parallel$ (radial)"); ax.set_ylabel(r"$\delta q_\perp$ (tangential)")
    ax.set_title("(b) Jacobian images at d = 0.8",loc="left",fontsize=8.4,pad=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=.9)
    ax.text(.04,.035,"same condition number,\ndifferent sensitivity scale",fontsize=7,va="bottom",transform=ax.transAxes,bbox=dict(fc="white",ec="none",alpha=.9,pad=1))

def checks():
    # F(r)=2r/(1+r^2). Solve F=d for d=.8 -> 2r = .8(1+r^2) => .8 r^2 -2r +.8=0 => r^2-2.5r+1=0
    d=.8
    roots_F = sorted([ (2.5-math.sqrt(2.5**2-4))/2, (2.5+math.sqrt(2.5**2-4))/2 ])
    # G(r)=r/sqrt(1+r^2)=d => r^2=d^2(1+r^2) => r=sqrt(d^2/(1-d^2))
    root_G = math.sqrt(d*d/(1-d*d))
    # P=min(r,1)=d -> if d<1 then r=d
    root_P = d
    spectra=[];max_fd=0.0
    def fmap(w):return 2*w/(1+w@w)
    for radius in roots_F:
        w=np.array([radius,0.]);den=1+w@w
        J=2*np.eye(2)/den-4*np.outer(w,w)/den**2
        h=1e-6
        Jfd=np.column_stack([(fmap(w+h*np.eye(2)[k])-fmap(w-h*np.eye(2)[k]))/(2*h) for k in range(2)])
        max_fd=max(max_fd,float(np.max(abs(J-Jfd))))
        spectra.append([float(J[0,0]),float(J[1,1])])
    assert max_fd<1e-9
    assert np.allclose(spectra,[[.96,1.6],[-.24,.4]],atol=1e-12)
    lam_in_r,lam_in_t=spectra[0]
    lam_out_r,lam_out_t=spectra[1]
    cond_in = abs(lam_in_t)/abs(lam_in_r)
    cond_out = abs(lam_out_t)/abs(lam_out_r)
    scale_r = abs(lam_out_r)/lam_in_r
    scale_t = abs(lam_out_t)/lam_in_t
    # independent check: outer singular values are .25 of inner magnitudes
    err_r = abs(scale_r - .25)
    err_t = abs(scale_t - .25)
    assert max(err_r,err_t)<1e-12 and abs(cond_in-cond_out)<1e-12
    return {
        "d": d,
        "max_independent_fd_J_error":max_fd,
        "F_roots_at_d": roots_F,
        "G_root_at_d": root_G,
        "P_root_at_d": root_P,
        "inner_eigs_radial_tangential": [lam_in_r, lam_in_t],
        "outer_eigs_radial_tangential": [lam_out_r, lam_out_t],
        "condition_number_inner": cond_in,
        "condition_number_outer": cond_out,
        "outer_over_inner_radial": scale_r,
        "outer_over_inner_tangential": scale_t,
        "abs_err_vs_.25_radial": err_r,
        "abs_err_vs_.25_tangential": err_t,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    panel_a(ax1); panel_b(ax2)
    fig.tight_layout(pad=1.1,w_pad=2.0)
    stem = os.path.join(a.output_dir, "head_geometry")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    with open(os.path.join(a.output_dir, "head_geometry_checks.json"), "w") as f:
        json.dump(checks(), f, indent=2)

if __name__ == "__main__":
    main()
