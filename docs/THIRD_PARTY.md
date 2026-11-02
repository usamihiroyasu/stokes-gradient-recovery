# External materials

The numerical code requires NumPy, SciPy, and Matplotlib. The optional automatic-differentiation experiment requires PyTorch. Dependencies are installed separately through the pinned requirement files.

The manuscript sources include the unmodified Springer Nature `sn-jnl.cls` supplied in the official LaTeX author template, version 3.1 (December 2024). Its copyright and licence notice remain in `manuscript/source/sn-jnl.cls`. Template source: https://www.springernature.com/gp/authors/campaigns/latex-author-support .

The rational ball map and relevant differential results are attributed in [SOURCES.md](SOURCES.md) and the manuscript bibliography. The saved renderer reference uses the analytic scene described in Supplement S3.

## License scope

Original author-contributed Python source and software documentation are
covered by the root [MIT License](../LICENSE). This code license does not
apply to the manuscript, manuscript figures, selected
third-party observations, or external template assets. The authors retain copyright in their manuscript. This package does not imply journal acceptance or a copyright transfer.

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
