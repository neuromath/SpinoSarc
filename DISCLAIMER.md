# SpinoSarc — Important Disclaimer

## Research Use Only

**SpinoSarc is a research software tool and has NOT received regulatory
approval from any health authority, including but not limited to:**

- U.S. Food and Drug Administration (FDA)
- European Conformity (CE) marking
- Turkish Ministry of Health
- Health Canada
- Any other national or regional medical device regulator

**SpinoSarc is NOT a medical device. It must NOT be used for:**

- Clinical diagnosis
- Treatment planning or decision-making
- Direct patient care
- Any decision that affects patient management

## Intended Use

SpinoSarc is intended **solely** for:

- Academic research
- Methodological exploration
- Educational purposes
- Generation of quantitative measurements that may be analyzed by
  qualified researchers under appropriate institutional oversight

## Limitations

Users should be aware that:

1. **Segmentation accuracy is not clinically validated.** The underlying
   segmentation model (MuscleMap) was trained on a specific dataset and
   may not generalize to all imaging protocols, scanners, or patient
   populations.

2. **Cut-off values shown for reference (e.g., PMI thresholds from
   Hamaguchi 2016, Englesbe 2010; dural sac CSA thresholds from Barz
   2010) are derived from specific populations and may not apply
   universally.** No automated risk classification is performed; raw
   metrics are reported for expert interpretation only.

3. **Anatomical level identification is approximate.** SpinoSarc does
   not currently perform automatic vertebra labeling; the user is
   responsible for confirming the anatomical level being analyzed.

4. **Manual dural sac ROI is operator-dependent.** Measurements vary
   between operators and reflect operator-defined boundaries.

5. **The software has been tested on a limited set of vendor
   configurations** (Siemens, Philips, GE). Other vendors, sequences,
   or acquisition parameters may produce unexpected results.

## Patient Data Privacy

All processing occurs **locally on the user's device**. No imaging data
or patient information is transmitted to external servers. However,
users remain responsible for:

- Compliance with local data protection regulations
  (HIPAA, GDPR, KVKK, etc.)
- Institutional review board (IRB) or ethics committee approval for
  any research use
- Appropriate de-identification of any data shared in derivative
  works, publications, or presentations

## Liability

The authors and contributors of SpinoSarc accept **no liability** for
any consequences arising from the use of this software, including but
not limited to:

- Incorrect measurements
- Clinical decisions made on the basis of SpinoSarc output
- Loss or corruption of data
- Any other direct or indirect damage

Users assume full responsibility for the appropriate use and
interpretation of SpinoSarc results.

## Contact

For questions regarding appropriate use, please contact the corresponding
author through the GitHub repository.

---

By using SpinoSarc, you acknowledge that you have read, understood, and
accept this disclaimer.
