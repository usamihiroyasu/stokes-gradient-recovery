# External materials

The numerical code requires NumPy, SciPy, and Matplotlib. The optional automatic-differentiation experiment requires PyTorch. Dependencies are installed separately through the pinned requirement files.

The manuscript uses the [official IEEE Access LaTeX template](https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip). Template class files, logos, and font programs are excluded from this source tree. `scripts/prepare_template.py` retrieves the pinned archive and installs the required files locally without modification. The upstream notices and terms remain with those files.

The rational ball map and relevant differential results are attributed in [SOURCES.md](SOURCES.md) and the manuscript bibliography. The saved renderer reference uses the analytic scene described in Supplement S3.

## License scope

Original author-contributed Python source and software documentation are
covered by the root [MIT License](../LICENSE). This code license does not
apply to the manuscript, manuscript figures, author photographs, selected
third-party observations, or the external IEEE template assets. The authors
retain copyright in their manuscript and photographs. This release does not
imply journal acceptance or an IEEE copyright transfer.

## Released observations

The numerical audit uses selected observation vectors derived from
[DeepSfP](https://github.com/UCLA-VMG/DeepSfP/blob/master/data/README.md) and
[PolarNS / PolarBurstSR](https://github.com/KAIST-VCLAB/polarns). Full original
images, masks, maps, model weights, and data archives are not redistributed.
Source versions, archive members, coordinates, and checksums are recorded in
`application/real_data/`.

The [DeepSfP repository license](https://github.com/UCLA-VMG/DeepSfP/blob/master/LICENSE)
covers its software; a separate dataset license for the externally hosted
images was not identified in the linked data documentation. A dataset-specific
license was likewise not identified in the PolarNS documentation inspected for
this release. The selected numerical observations are attributed to their
providers and excluded from our MIT license. This repository does not assert
permission to relicense the source datasets. Consult the original providers
for use of their full data or rights beyond this numerical audit.
