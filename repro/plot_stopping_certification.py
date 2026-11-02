"""Visualize frozen stopping records; no optimizer or numerical replay is run.

Usage: python plot_stopping_certification.py --root PATH_TO_PACKAGE --output NEW_DIR
Requires ReportLab. Scientific inputs are resolved relative to --root.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import importlib.metadata
import json
import math

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

FILES = {
    "audit": "theory/stopping_guarantee/results/FROZEN_ENDPOINT_AUDIT.json",
    "interval": "theory/validated_stopping/results/ENDPOINTS.json",
    "reference": "theory/validated_stopping/results/REFERENCE_INTERVALS.json",
    "protocol": "theory/validated_stopping/results/PROTOCOL.json",
    "summary": "theory/validated_stopping/results/SUMMARY.json",
    "sharpness": "theory/stopping_guarantee/results/SHARPNESS_AND_COUNTEREXAMPLES.json",
    "observations": "application/real_data/RESULTS.json",
}
TEAL = HexColor("#087E8B")
AMBER = HexColor("#B88618")
RED = HexColor("#C03C32")
INK = HexColor("#232B33")
GRID = HexColor("#D8DEE3")


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(c, x, y, s, size=7, bold=False, align="left", color=INK):
    c.setFillColor(color)
    c.setFont("Times-Bold" if bold else "Times-Roman", size)
    getattr(c, {"left": "drawString", "right": "drawRightString", "center": "drawCentredString"}[align])(x, y, s)


def power(c, x, y, exponent, size=7, align="center"):
    main = "10"
    exp = str(exponent)
    w = c.stringWidth(main, "Times-Roman", size) + c.stringWidth(exp, "Times-Roman", size * .72)
    start = x - (w / 2 if align == "center" else w if align == "right" else 0)
    text(c, start, y, main, size)
    text(c, start + c.stringWidth(main, "Times-Roman", size), y + size * .38, exp, size * .72)


def paragraph(c, x, y, width, markup, size=8.5):
    p = Paragraph(markup, ParagraphStyle("figure", fontName="Times-Roman", fontSize=size, leading=size * 1.15, textColor=INK))
    _, h = p.wrap(width, 80)
    p.drawOn(c, x, y - h)


def scatter_panel(c, origin, label, title, values, gap, statuses, accepted):
    x0, y0 = origin
    w, h = 174, 115
    xmin, xmax, ymin, ymax = -18., 0., -18., 2.
    px = lambda v: x0 + (v - xmin) * w / (xmax - xmin)
    py = lambda v: y0 + (v - ymin) * h / (ymax - ymin)
    text(c, x0, y0 + h + 23, f"({label}) {title}", 9, True)
    n_accept = sum(accepted)
    n_reject = sum(s == "accurate" and not a for s, a in zip(statuses, accepted))
    text(c, x0, y0 + h + 11, f"{n_accept:,} accepted; {n_reject:,} accurate rejected", 7)
    c.setStrokeColor(GRID); c.setLineWidth(.35)
    for z in [-18, -12, -6, 0]:
        c.line(px(z), y0, px(z), y0 + h)
        power(c, px(z), y0 - 11, z)
    for z in [-18, -12, -6, 0, 2]:
        c.line(x0, py(z), x0 + w, py(z))
        power(c, x0 - 5, py(z) - 2.4, z, align="right")
    c.setStrokeColor(HexColor("#707B84")); c.setDash(2, 2); c.setLineWidth(.55)
    c.line(px(-6), y0, px(-6), y0 + h)
    c.line(x0, py(-6), x0 + w, py(-6))
    c.setDash(1, 2)
    c.line(px(-18), py(-18), px(0), py(0))
    c.setDash()
    # Accepted accurate points are drawn first; rejected accurate and all
    # inaccurate endpoints remain visible on top. No points are sampled out.
    categories = [
        ([i for i in range(len(gap)) if accepted[i]], TEAL, .34, .65),
        ([i for i in range(len(gap)) if not accepted[i] and statuses[i] == "accurate"], AMBER, .62, .82),
        ([i for i in range(len(gap)) if statuses[i] == "inaccurate"], RED, .9, 1.05),
    ]
    for indices, color, opacity, radius in categories:
        c.setFillColor(color); c.setFillAlpha(opacity)
        for i in indices:
            xx = px(min(xmax, max(xmin, math.log10(max(gap[i], 1e-300)))))
            yy = py(ymax if values[i] is None else min(ymax, max(ymin, math.log10(max(values[i], 1e-300)))))
            if values[i] is None:
                c.setStrokeColor(color); c.setLineWidth(.65)
                p = c.beginPath(); p.moveTo(xx, yy); p.lineTo(xx-2.2, yy-4);p.lineTo(xx+2.2, yy-4);p.close()
                c.drawPath(p, fill=1, stroke=0)
            else:
                c.circle(xx, yy, radius, fill=1, stroke=0)
    c.setFillAlpha(1)
    c.setStrokeColor(INK); c.setLineWidth(.6)
    c.rect(x0, y0, w, h, fill=0, stroke=1)
    text(c, x0 + w/2, y0 - 26, "Reference nominal gap", 8, align="center")
    c.saveState();c.translate(x0-33,y0+h/2);c.rotate(90);text(c,0,0,"Upper bound",8,align="center");c.restoreState()


def sharpness_panel(c, origin, records):
    x0,y0=origin;w,h=174,115
    px=lambda z:x0+(math.log10(z)+12)*w/11
    py=lambda z:y0+(z-.34)*h/.065
    text(c,x0,y0+h+23,"(d) Sharpness along the stored family",9,True)
    paragraph(c,x0,y0+h+16,w,"Scaled gap tends to 2<super>-4/3</super>",7.5)
    c.setStrokeColor(GRID);c.setLineWidth(.35)
    for z in [-12,-9,-6,-3,-1]:
        xx=px(10.**z);c.line(xx,y0,xx,y0+h);power(c,xx,y0-11,z)
    for z in [.34,.36,.38,.40]:
        yy=py(z);c.line(x0,yy,x0+w,yy);text(c,x0-5,yy-2.4,f"{z:.2f}",7,align="right")
    limit=2**(-4/3)
    c.setStrokeColor(HexColor("#707B84"));c.setDash(3,2);c.setLineWidth(.7);c.line(x0,py(limit),x0+w,py(limit));c.setDash()
    c.setStrokeColor(TEAL);c.setFillColor(TEAL);c.setLineWidth(1)
    pts=[(px(float(r['delta'])),py(float(r['ratio']))) for r in records]
    p=c.beginPath();p.moveTo(*pts[0])
    for xx,yy in pts[1:]:p.lineTo(xx,yy)
    c.drawPath(p,stroke=1,fill=0)
    for xx,yy in pts:c.circle(xx,yy,1.6,stroke=0,fill=1)
    c.setStrokeColor(INK);c.setLineWidth(.6);c.rect(x0,y0,w,h,fill=0,stroke=1)
    text(c,x0+w/2,y0-26,"Radial distance to the fold",8,align="center")
    c.saveState();c.translate(x0-33,y0+h/2);c.rotate(90);text(c,0,0,"Sharpness ratio",8,align="center");c.restoreState()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={k:root/v for k,v in FILES.items()}
    hashes={FILES[k]:digest(p) for k,p in inputs.items()}
    data={k:json.loads(p.read_text()) for k,p in inputs.items()}
    audit=data['audit']; interval=[r for r in data['interval'] if r['bits']==128];ref=data['reference']
    warm=[r for r in data['observations']['records'] if r['arm']=='warm_start']
    assert len(audit)==len(interval)==len(ref)==len(warm)==9284
    for rel,h in data['protocol']['source_sha256'].items():
        assert digest(root/rel)==h,rel
    for i,(a,v,r,w) in enumerate(zip(audit,interval,ref,warm)):
        assert v['case']==i
        assert (a['stream'],a['method'],a['index'])==(w['stream'],w['method'],w['index'])
        assert (a['dataset'],a['stratum'],a['method'])==(v['dataset'],v['stratum'],v['method'])
        assert v['reference_class']==r['status']
        assert a['original_false_success']==v['original_false_success']==w['false_success']
    gaps=[float(Fraction(int(r['lower_integer'])+int(r['upper_integer']),2*(1<<r['bits']))) for r in ref]
    assert all(g>=0 for g in gaps)
    statuses=[r['status'] for r in ref];assert set(statuses)=={'accurate','inaccurate'}
    uniform=[r['bounds']['uniform'] for r in audit]
    pointwise=[r['bounds']['pointwise'] for r in audit]
    validated=[float(Fraction(int(r['bound_upper_integer']),1<<r['bits'])) for r in interval]
    accepted=[[(v is not None and v<=1e-6) for v in uniform],[(v is not None and v<=1e-6) for v in pointwise],[r['certified'] for r in interval]]
    sharp=[r for r in data['sharpness'] if r['family']=='sharp_first_tolerance']
    assert len(sharp)==12
    # The recorded ratios are checked against their recorded Decimal operands.
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=70
        for r in sharp:
            ratio=Decimal(r['gap'])/(Decimal(r['epsilon'])**(Decimal(4)/3))
            assert abs(ratio-Decimal(r['ratio']))<Decimal('1e-60')
    counts={}
    for name,values,acc,n in zip(['uniform','pointwise','validated128'],[uniform,pointwise,validated],accepted,[980,8861,8861]):
        assert sum(acc)==n
        counts[name]={'accepted':sum(acc),'inaccurate_accepted':sum(a and s=='inaccurate' for a,s in zip(acc,statuses)),'accurate_rejected':sum(not a and s=='accurate' for a,s in zip(acc,statuses)),'y_below_display_floor':sum(v is not None and v<1e-18 for v in values),'undefined_bound_top_triangles':sum(v is None for v in values),'y_above_display_ceiling':sum(v is not None and v>100 for v in values)}
        assert counts[name]['inaccurate_accepted']==0
    summary128=next(r for r in data['summary']['precisions'] if r['bits']==128)
    assert summary128['scalar_curvature_evaluations']==52 and summary128['certified']==8861
    c=canvas.Canvas(str(out/'stopping_certification.pdf'),pagesize=(454,390),pageCompression=1)
    c.setTitle('Conditional stopping bounds on frozen polarization endpoints')
    c.setAuthor('Hiroyasu Usami, Ayato Tsuboi, Keisuke Hara')
    scatter_panel(c,(42,233),'a','Uniform',uniform,gaps,statuses,accepted[0])
    scatter_panel(c,(267,233),'b','Pointwise (float64)',pointwise,gaps,statuses,accepted[1])
    scatter_panel(c,(42,53),'c','Validated (128 bit)',validated,gaps,statuses,accepted[2])
    sharpness_panel(c,(267,53),sharp)
    legend=[(45,'Accepted',TEAL),(134,'Accurate, rejected',AMBER),(255,'Inaccurate',RED)]
    for xx,label,color in legend:
        c.setFillColor(color);c.circle(xx,9,2,fill=1,stroke=0);text(c,xx+6,6.8,label,7)
    text(c,446,6.8,'9,284 endpoints; none omitted',7,align='right')
    c.showPage();c.save()
    receipt={'input_sha256':hashes,'endpoint_join':'case index, dataset, stratum, method, stream/index; all 9284 aligned to frozen warm records','reference_gap':'midpoint of independent 256-bit enclosing interval; used only as plotting coordinate','x_below_display_floor':sum(g<1e-18 for g in gaps),'display_floor':1e-18,'display_y_ceiling':100,'counts':counts,'sharpness_records':12,'sharpness_ratio_operands_checked_decimal_precision':70,'scalar_curvature_evaluations':52,'full_head_hessians':summary128['full_head_hessians'],'recorded_128bit_seconds':summary128['elapsed_seconds'],'reportlab_version':importlib.metadata.version('reportlab'),'new_optimizer_calls':0,'scientific_replay':False,'all_points_drawn':True}
    (out/'FIGURE_REPRODUCTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'counts':counts,'x_below_display_floor':receipt['x_below_display_floor'],'output':str(out/'stopping_certification.pdf')},indent=2))


if __name__=='__main__':main()
