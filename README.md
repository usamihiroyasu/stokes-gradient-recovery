# Conditional stopping guarantees for a folded parametrization of reduced Stokes vectors

Hiroyasu Usami · Ayato Tsuboi · Keisuke Hara

Private author-review repository. Access is restricted to invited collaborators and reviewers. The manuscript is a working draft.

The study examines how derivatives of a folded reduced-Stokes parametrization can certify a physical objective gap. The model includes partial linear polarization through $(I,Q,U)$; circular polarization is outside its scope. The supplied records retain the direct physical-solver controls and the limits of local fold diagnosis.

## Research overview

[![Play the silent research overview](media/video-preview.png)](media/stokes-conditional-stopping.mp4)

[Open or download the video](media/stokes-conditional-stopping.mp4) · [English captions](media/stokes-conditional-stopping.en.srt)

The video is 126.3 seconds, 1920 × 1080 at 30 fps, and has no audio. It displays the authors' names and affiliations without portrait photographs. Click the preview to open the video on GitHub.

## Start here

1. Read [the main manuscript](paper/main.pdf), [the Supplementary Information](paper/supplement.pdf), or [the combined reader copy](paper/complete.pdf).
2. Run `python3 validate_package.py` from this directory to check supplied file integrity and Python source syntax. This check does not rerun experiments.
3. Use [the result-to-file index](docs/RESULTS_MAP.md) to locate a study and its stored records.
4. Follow [the reproduction guide](docs/REPRODUCING.md) in a fresh working copy with the environment recorded for that study.

## Contents

| Path | Contents |
| --- | --- |
| `paper/` | Main manuscript, supplementary information and combined reader PDF |
| `manuscript/` | Matching standalone TeX sources, figures, class and build instructions |
| `media/` | Silent research video, preview and English captions |
| `examples/` | Small physical-gradient recovery example |
| `repro/` | Formula checks, figure generators, saved network and warm-start records |
| `application/` | Fresnel simulation, selected observations and coordinate-repair studies |
| `theory/` | Stopping-bound code, exact interval arithmetic and frozen results |
| `verification/` | Numerical core required by the application replays |
| `docs/` | Result mapping, environments, numerical conventions and source attribution |
| `latex/images/` | Scientific figures and output location used by figure generators |

## Reproduce the interval certificate

The interval replay requires only the Python standard library and the included numerical inputs. Choose a new or empty output directory:

```sh
python3 theory/validated_stopping/replay.py --output /tmp/stokes-interval-replay
```

A small interval-bound example is also available:

```sh
python3 theory/validated_stopping/cli.py --x 0 1 0 --y 1 .5 0 .5
```

The gradient-recovery example uses the pinned numerical dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r repro/requirements.txt
.venv/bin/python examples/recover_gradient.py
```

Separate studies have separate recorded Python and package versions. The reproduction guide distinguishes these environments and optional training/AD checks.

## Interpretation

- Small raw-coordinate derivatives alone do not certify physical accuracy.
- The uniform stopping theorem requires its stated strong-convexity, intensity, inner-sheet and derivative-error assumptions.
- The interval monitor certifies the stated nominal model; it does not establish unknown instrument calibration or noise-free physical truth.
- Saved observation counts are not population incidence estimates.
- Direct physical methods that were faster than the diagnostic routes remain in the stored comparisons.

## Integrity, licensing and attribution

`MANIFEST.sha256` covers this repository snapshot. `FROZEN_FILES.sha256` identifies unchanged scientific files carried into it. Keep the supplied records for comparison when regenerating outputs. Figure generators use `latex/images/`; manuscript inputs are separate in `manuscript/source/images/`.

Original author-contributed code and software documentation use the [MIT License](LICENSE). Manuscripts, figures and selected third-party observations are outside that code licence. Full original datasets and model weights are not included. Read [the source and rights notes](docs/THIRD_PARTY.md) before using external data.

Use [CITATION.cff](CITATION.cff) for the author and title metadata. Reports of reproduction should identify the snapshot, command, environment and relevant file hashes.
