---
title: 'SpinoSarc: An open-source desktop tool for quantitative paraspinal muscle and dural sac analysis on lumbar spine MRI'
tags:
  - Python
  - medical imaging
  - lumbar spine
  - paraspinal muscle
  - sarcopenia
  - MRI
  - radiology
  - segmentation
  - DICOM
authors:
  - name: Berkay Yılmaz
    orcid: 0009-0006-3108-8991
    affiliation: 1
affiliations:
  - name: Department of Radiology, Cerrahpaşa Faculty of Medicine, Istanbul University-Cerrahpaşa, Istanbul, Turkey
    index: 1
date: 19 June 2026
bibliography: paper.bib
---

# Summary

`SpinoSarc` is an open-source desktop application for quantitative analysis of paraspinal musculature and the dural sac on lumbar spine magnetic resonance imaging (MRI). It provides an end-to-end pipeline that ingests DICOM data directly from clinical scanners (Siemens, Philips, GE), performs automated segmentation of eight paraspinal muscle classes via the `MuscleMap` U-Net backbone [@musclemap], computes established quantitative metrics (cross-sectional area, fat fraction, total psoas area, psoas muscle index), supports manual region-of-interest measurement of dural sac cross-sectional area, and generates one-click PDF and Excel reports.

The application is implemented as a native macOS desktop tool with a PyQt6 graphical user interface and runs fully offline on Apple Silicon hardware. No imaging data leaves the local device, an important property for clinical research environments subject to patient privacy regulations. The interface is bilingual (English/Turkish).

`SpinoSarc` is intended for research use only and has not received regulatory approval. Quantitative metrics are reported alongside published literature reference values [@hamaguchi2016; @englesbe2010; @barz2010], but the software deliberately does not perform automated risk classification, leaving clinical interpretation to qualified experts.

# Statement of need

Paraspinal muscle quality is an emerging biomarker in lumbar spine research, with growing evidence linking sarcopenia and muscle fat infiltration to chronic low back pain, surgical outcomes, and degenerative spinal pathology [@shahidi2017; @hyun2016]. Quantitative assessment, however, remains uncommon in clinical practice because existing tools require substantial technical infrastructure: dedicated workstations, command-line proficiency, manual NIfTI conversion of DICOM data, or institutional collaborations with research engineers.

`MuscleMap` [@musclemap] provides a high-quality open-source U-Net for paraspinal muscle segmentation but is implemented as a Python library that consumes NIfTI input and emits segmentation masks. Substantial additional engineering is required to use it in clinical research workflows: parsing multi-vendor DICOM series, aligning axial and sagittal coordinate systems for level identification, computing area and fat-fraction metrics, integrating manual dural sac measurements, and producing structured reports. These gaps create a barrier for clinicians who want to apply quantitative paraspinal analysis to research questions without becoming software engineers.

`SpinoSarc` closes this gap by wrapping `MuscleMap` in a complete clinical research workflow:

- **DICOM ingestion with vendor support.** The tool reads multi-slice axial and sagittal series directly from clinical DICOM folders, including Philips JPEG Lossless compressed data, and uses `dcm2niix` [@li2016dcm2niix] to produce consistent NIfTI input for segmentation. Slices are sorted by `InstanceNumber` to match the behavior of clinical PACS viewers.

- **Anatomical coordinate alignment.** Axial and sagittal series are cross-linked through DICOM world coordinates (`ImagePositionPatient`), allowing the sagittal locator to update with axial navigation and vice versa. This is essential for identifying the lumbar level being analyzed.

- **Quantitative metric extraction.** For each analyzed slice, `SpinoSarc` computes per-muscle cross-sectional area and intensity-based fat fraction, total psoas area, and (when height is provided) psoas muscle index. Reference values from established cohorts [@hamaguchi2016; @englesbe2010] are displayed alongside results without automated classification.

- **Manual dural sac CSA.** A polygon region-of-interest tool allows the user to delineate the dural sac on the axial slice. Cross-sectional area is computed in mm² with literature thresholds for spinal stenosis [@barz2010] displayed as context.

- **Reporting.** Results are exported as a single-page PDF report and a structured Excel workbook containing per-muscle metrics, summary measurements, and patient demographics.

- **Native desktop distribution.** `SpinoSarc` is distributed as a signed-free `.dmg` installer for Apple Silicon Macs. No Python environment, command-line tools, or internet connection is required at runtime.

To our knowledge, `SpinoSarc` is the first open-source desktop application providing end-to-end DICOM-to-report quantitative paraspinal analysis aimed at clinical researchers in radiology and spine medicine.

# Implementation

The application is written in Python 3.11 with PyQt6 for the graphical interface, MONAI [@cardoso2022monai] and PyTorch [@paszke2019pytorch] for inference, `pydicom` [@mason2024pydicom] with `pylibjpeg` for DICOM parsing including compressed transfer syntaxes, `nibabel` [@brett2024nibabel] for NIfTI handling, and `dcm2niix` [@li2016dcm2niix] for DICOM-to-NIfTI conversion. The macOS application is built with PyInstaller and distributed as a self-contained `.app` bundle including the MuscleMap pretrained weights.

A typical analysis on Apple Silicon hardware (M-series, MPS backend) completes in 10–30 seconds per slice.

# Limitations and future work

`SpinoSarc` currently supports macOS Apple Silicon only; Linux and Windows builds are planned. Anatomical vertebral level (L1–S1) is not automatically identified—the user must select the slice of interest, typically guided by the sagittal locator. Future versions will include automated level detection and longitudinal tracking of repeated examinations.

The segmentation backbone (`MuscleMap`) was trained on a specific dataset and may not generalize to all imaging protocols, scanners, or patient populations. Clinical validation in independent cohorts is recommended before use of `SpinoSarc` outputs in any research conclusion.

# Acknowledgements

The author thanks the `MuscleMap` developers for the open-source segmentation backbone, the MONAI and PyTorch communities, and Chris Rorden for `dcm2niix`. Clinical context and feedback were provided by colleagues at the Cerrahpaşa Faculty of Medicine.

# References
