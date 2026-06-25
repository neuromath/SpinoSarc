# Changelog

All notable changes to SpinoSarc are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-25

### Added
- **Automatic lumbar level detection** via [TotalSpineSeg](https://github.com/neuropoly/totalspineseg): labels intervertebral discs (L1-L2 … L5-S) and the L3 vertebral body, then maps them onto the axial series with gap detection for levels not covered by the axial volume.
- **Automatic dural sac segmentation**: cross-sectional area is computed by resampling the TotalSpineSeg canal mask onto each axial slice in world space, replacing the previous manual polygon ROI.
- **Multi-level analysis** ("Analyze All Levels"): paraspinal muscle metrics + dural sac CSA at every detected level in a single run. Levels falling in axial gaps are analyzed from the nearest slice and flagged with the offset distance.
- **Canal narrowing flags**: neutral comparison of dural sac CSA against published thresholds (absolute <75, relative <100, early <130 mm²), with a caveat at L5-S where the thresholds are less reliable. Reported as a research aid for the radiologist, **not a diagnosis**.
- **L3-based sarcopenia** in the multi-level workflow, with a nearest-slice fallback when the L3 vertebral body falls in an axial gap. PMI is reported only at L3; other levels report muscle CSA / fat fraction.
- **Multi-level PDF and Excel reports**: per-level table, muscle detail, sarcopenia summary, stenosis flags, and disclaimers. The existing report buttons produce the multi-level report when a multi-level analysis has been run.
- **GUI**: "Detect Levels" panel with a clickable level list (jumps to the corresponding axial slice), sagittal level overlay with labels, axial canal overlay, and "Analyze All Levels" action.
- TotalSpineSeg MPS (Apple Silicon GPU) support patch under `patches/`.

### Changed
- Dural sac CSA is now automatic; the manual ROI panel ("Draw ROI" / "Clear") has been removed.
- Analyze now also overlays the segmented dural sac and reports its CSA automatically.
- Sarcopenia notes and batch messages translated to English for consistency.

### Fixed
- Axial canal overlay was rendered at the wrong location due to (1) stale TotalSpineSeg output files from prior runs and (2) a basename match that incorrectly paired `sagittal_s2` with `sagittal_s201`. The output directory is now cleared before each run and the canal NIfTI is matched by exact stem.
- Sagittal level overlay vertical positioning corrected (consistent cranial→caudal mapping in both directions).

### Notes
- Research use only. Stenosis thresholds are from the literature and have not been validated against a local cohort. Supine MRI may underestimate canal narrowing relative to axial-loaded/standing imaging.

## [0.1.1] - 2026-06-24

### Added
- Zenodo DOI badge and citation metadata.
- Application screenshot in the README.

## [0.1.0] - 2026-06-24

### Added
- Initial release of SpinoSarc.
- DICOM ingestion with multi-vendor support (Siemens, Philips, GE), including JPEG Lossless decompression.
- Paraspinal muscle segmentation via the MuscleMap U-Net backbone (8 muscle classes: multifidus, erector spinae, psoas, quadratus lumborum; left and right).
- Quantitative metrics: cross-sectional area (CSA), fat fraction, total psoas area (TPA), psoas muscle index (PMI).
- Manual dural sac CSA via interactive polygon ROI.
- Literature reference values (Hamaguchi 2016, Englesbe 2010, Barz 2010) for context.
- Single-page PDF report and Excel export.
- Native macOS desktop application (Apple Silicon).

[0.2.0]: https://github.com/neuromath/SpinoSarc/releases/tag/v0.2.0
[0.1.1]: https://github.com/neuromath/SpinoSarc/releases/tag/v0.1.1
[0.1.0]: https://github.com/neuromath/SpinoSarc/releases/tag/v0.1.0
