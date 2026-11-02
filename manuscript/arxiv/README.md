# Source of the arXiv-generated reader copy

This is the exact 16-file source bundle uploaded for the current arXiv draft. The draft has not been submitted or announced.

- [arXiv-generated PDF](../../paper/arxiv.pdf): the primary reader copy, with the complete Supplementary Information.
- [Download the exact uploaded source ZIP](stokes-arxiv-source.zip).
- [Source file hashes](SOURCE_FILES.sha256).
- [Journal-format sources](../README.txt): maintained separately for Scientific Reports preparation.

## Build locally

From an extracted copy of this directory, with TeX Live and `pdflatex` available:

```sh
pdflatex -interaction=nonstopmode -halt-on-error arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error arxiv.tex
```

The root document is `arxiv.tex`; it includes `supplement-body.tex` and the 14 figure PDFs in `images/`. No `.aux` file or external manuscript directory is required. The source compiled successfully with arXiv's TeX Live 2025. Local builds do not reproduce the arXiv-added draft watermark, and PDF metadata or font extraction may differ.

The journal version and this combined version contain the same scientific text and figures. The combined source uses a standard article preamble and internal supplementary labels in place of cross-file references.

## File identities

- Uploaded source ZIP SHA-256: `b5063648d8e7b4547f886329e09935960509cd441534e25b80d8b931a2b97dee`
- arXiv-generated PDF SHA-256: `18401099efa0d9762a160025264bf8fe24969a279ba2ddaf0083aae02c603d45`
