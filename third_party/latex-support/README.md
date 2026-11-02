# Unmodified third-party source accompaniment

These complete upstream archives accompany the unmodified `sn-jnl.cls` used by the manuscript sources. They preserve the original Springer Nature template package and a complete official LaTeX base source distribution, including its manifest, copyright notices and licence. They are separate from the original research code and its MIT licence.

The archive contents have not been changed, installed or executed. The manuscript build uses the independently installed TeX environment; this accompaniment does not change the environment or scientific results.

## Archives

| Archive | Upstream version | Bytes | Licence information |
| --- | --- | ---: | --- |
| `latex-base-2026-11-01.zip` | LaTeX base 2026-11-01 | 40,981,416 | LPPL 1.3c or later; see `base/README.md`, `base/legal.txt` and `base/lppl.txt` inside the archive |
| `springer-nature-template-3.1-2024-12.zip` | Springer Nature template 3.1, December 2024 | 901,814 | Original per-file notices retained; `sn-article-template/sn-jnl.cls` states LPPL 1.3c or later |

Sources, retrieved 2 November 2026:

- [LaTeX base official CTAN catalogue](https://www.ctan.org/pkg/latex-base), [canonical archive](https://mirrors.ctan.org/macros/latex/base.zip), and [downloaded archive from the CTAN-listed JAIST mirror](https://ftp.jaist.ac.jp/pub/CTAN/macros/latex/base.zip).
- [Springer Nature author support](https://www.springernature.com/gp/authors/campaigns/latex-author-support) and its [official December 2024 ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12).
- [LPPL 1.3c](https://www.latex-project.org/lppl/lppl-1-3c/).

## Integrity and completeness

`SHA256SUMS` contains the complete archive hashes. `VALIDATION.json` records archive/member counts, checksums and validation details.

Both ZIP archives passed a full CRC check. All member paths are relative, with no parent-directory traversal, backslash/drive paths, duplicate entries, symbolic links or encrypted entries.

The LaTeX ZIP contains all 320 regular files listed by the official mirror's `macros/latex/base/` directory. It includes `manifest.txt`, `legal.txt`, `lppl.txt`, `README.md`, `classes.dtx` and the source version declaration for 2026-11-01. The upstream manifest names three files absent both from that directory and from the delivered ZIP: `latexchanges.tex`, `lttemplate.dtx` and `ltxdoc.cfg`. This discrepancy is preserved; no archive repair or local deletion has been made.

The Springer ZIP's `sn-jnl.cls` has SHA-256 `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc` and matches the manuscript's retained class byte for byte. All original archive contents and notices remain present.
