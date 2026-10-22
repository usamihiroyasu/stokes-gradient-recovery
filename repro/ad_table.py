"""Typeset the saved common-graph AD measurements without rerunning timings."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'results/ad_route_benchmark.json').read_text());s=report['summary']
labels={'direct':'Direct physical','first':'First inverse','vector':'Vector HVP','mixed':'Mixed scalar'}
def sci(x):
    a,e=f'{x:.2e}'.split('e')
    return rf'${float(a):.2f}\!\times\!10^{{{int(e)}}}$'
rows=[f"{labels[m]} & {s['median_us_float64_r0999'][m]:.1f} & {sci(s['max_error_by_dtype']['float32'][m])} & {sci(s['max_error_by_dtype']['float64'][m])}\\\\" for m in labels]
table=r'''\begin{table}[!t]
\caption{Common-graph AD: median microseconds at $I=1.2$, $r=0.999$, float64; maximum Euclidean gradient error over the intensity/radius grid for each precision. Four exact-fold first-inverse cases are singular and omitted from error maxima.}
\label{tab:adcost}
\centering\small
\begin{tabular}{@{}lrrr@{}}\toprule
Route & $\mu$s & Error32 & Error64\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
(HERE.parent/'latex/ad_results_table.tex').write_text(table)
supp=HERE.parent/'latex/supplement_body.tex'
text=supp.read_text()
timing={x['method']:x for x in report['timing_records'] if x['dtype']=='float64' and x['r']==.999}
parts=[]
for method,label in [('direct','direct'),('first','first inverse'),('vector','vector HVP'),('mixed','mixed')]:
    row=timing[method]
    parts.append(f"{label} ${row['median_us']:.1f}$ [${row['q25_us']:.1f},{row['q75_us']:.1f}$]")
paragraph=r'At float64 and $r=0.999$, median [interquartile] microseconds are: '+', '.join(parts[:-1])+', and '+parts[-1]+'. '
text,n=re.subn(r'At float64 and \$r=0\.999\$, median \[interquartile\] microseconds are: .*?These are observations',lambda m:paragraph+'These are observations',text,flags=re.S)
assert n==1
supp.write_text(text)
print('generated AD table and timing paragraph from retained measurements')
