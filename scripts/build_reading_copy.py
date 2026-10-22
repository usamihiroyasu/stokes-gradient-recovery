"""Compile portable reading copies; preserve the authoritative IEEE sources/class.

The bundled XeTeX compiler does not support the official class's pdfTeX spot-color
primitives. Only this wrapper changes the layout, not the scientific body sources.
"""
from pathlib import Path
import re,subprocess,os,shutil
P=Path(__file__).resolve().parent.parent;L=P/'latex'
ENGINE=os.environ.get('TECTONIC') or shutil.which('tectonic')
if not ENGINE: raise SystemExit('Set TECTONIC to the Tectonic executable path')
PREAMBLE=r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{cite,amsmath,amssymb,amsfonts,amsthm}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\usepackage{graphicx,textcomp,mathtools,booktabs,array,tabularx,calc,microtype,placeins,etoolbox,xurl}
\usepackage[hidelinks]{hyperref}
\urlstyle{same}
\newcommand{\PARstart}[2]{#1#2}
\newcommand{\EOD}{}
\setlength{\emergencystretch}{1em}
\setlength{\columnsep}{0.24in}
\renewcommand{\dbltopfraction}{0.9}
\setlength{\parskip}{0pt}
\begin{document}
'''

def main():
    maintex=(L/'main.tex').read_text();abstract=maintex.split('\\begin{abstract}',1)[1].split('\\end{abstract}',1)[0]
    title_match=re.search(r'\\title\{([^{}]+)\}',maintex)
    if title_match is None:raise SystemExit('Cannot read title from latex/main.tex')
    title=title_match.group(1)
    for stem in ['main','supplement']:
        is_supp=stem=='supplement';head='Supplementary Material: '+title if is_supp else title
        pre=PREAMBLE
        if is_supp:
            pre+=r'''\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thetheorem}{S\arabic{theorem}}
'''
        pre+=r'\twocolumn[{\centering\Large\bfseries '+head+r'\par}\vspace{0.6em}{\centering\small Hiroyasu Usami, Ayato Tsuboi, and Keisuke Hara\par Research draft, 22 October 2026\par}\vspace{1em}'+'\n'
        if not is_supp:pre+=r'\begin{minipage}{\textwidth}\small\textbf{Abstract.} '+abstract+r'\end{minipage}\vspace{1.5em}'+'\n'
        pre+=']\n'
        pre+='\\input{'+('supplement_body' if is_supp else 'body')+'.tex}\n\\FloatBarrier\n\\input{'+('supplement_bibliography' if is_supp else 'bibliography')+'.tex}\n\\end{document}\n'
        path=L/('reading_'+stem+'.tex');path.write_text(pre)
        out=P/'build_reading';out.mkdir(exist_ok=True)
        cmd=[ENGINE,'--keep-logs','--keep-intermediates','--outdir',str(out),str(path)]
        result=subprocess.run(cmd,cwd=L,capture_output=True,text=True)
        (out/(stem+'.stdout')).write_text(result.stdout+result.stderr)
        print(stem,result.returncode,result.stdout[-1500:],result.stderr[-3000:],flush=True)
        if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
