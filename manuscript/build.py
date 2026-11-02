from pathlib import Path
import re, shutil, subprocess
root=Path(__file__).resolve().parent
out=root/'build'; out.mkdir(exist_ok=True)
tex=shutil.which('pdflatex')
if not tex: raise SystemExit('Install a current TeX Live distribution with the packages listed in README.txt.')
for cycle in range(4):
    warnings=[]
    for stem in ['main','supplement']:
        job='stokes-scientific-reports-'+stem
        subprocess.run([tex,'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-jobname='+job,'-output-directory='+str(out),stem+'.tex'],cwd=root/'source',check=True)
        log=(out/(job+'.log')).read_text(errors='replace')
        warnings += re.findall(r'undefined|Rerun to get|Label.s. may have changed',log)
    if cycle>=1 and not warnings: break
if warnings: raise SystemExit('Check unresolved cross-references in build/*.log.')
print('Main manuscript and supplementary PDFs are in build/.')
