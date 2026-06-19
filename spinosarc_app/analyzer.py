"""
SpinoSarc backend analyzer - in-process MuscleMap kullanir.
"""
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu
from .inference_engine import MuscleMapEngine


# === Data classes ===

@dataclass
class Demographics:
    age: Optional[int] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    @property
    def height_m(self): return self.height_cm/100.0 if self.height_cm else None

    @property
    def bmi(self):
        if self.height_m and self.weight_kg:
            return self.weight_kg / (self.height_m ** 2)
        return None


@dataclass
class MuscleMetrics:
    name: str
    csa_mm2: float
    csa_cm2: float
    fat_fraction: float
    mean_intensity: float
    voxel_count: int

    def to_dict(self): return asdict(self)


@dataclass
class SarcopeniaResult:
    pmi_cm2_per_m2: Optional[float]
    total_psoas_area_mm2: float
    total_psoas_area_cm2: float
    height_m: Optional[float]
    thresholds: dict = field(default_factory=dict)
    risk_category: str = 'Unknown'
    notes: List[str] = field(default_factory=list)


PMI_THRESHOLDS = {
    'Hamaguchi 2016 (HCC, Asian)':    {'M': 6.36, 'F': 3.92},
    'Englesbe 2010 (Transplant, US)': {'M': 5.21, 'F': 3.85},
    'Durand 2014 (Cirrhosis)':        {'M': 5.45, 'F': 3.85},
}


# === Main analyzer ===

class SpinoSarcAnalyzer:
    """Tek 2D slice + demografi -> kas metrikleri + sarkopeni indeksleri."""

    MUSCLE_LABELS = {
        1: 'multifidus_R', 2: 'multifidus_L',
        3: 'erector_R',    4: 'erector_L',
        5: 'psoas_R',      6: 'psoas_L',
        7: 'QL_R',         8: 'QL_L',
    }

    def __init__(self, region='abdomen', use_gpu=True):
        # In-process engine - model bir kere yuklenir
        self.engine = MuscleMapEngine(region=region, use_gpu=use_gpu)

    def analyze(self, slice_path: str, demographics: Optional[Demographics] = None) -> dict:
        slice_path = Path(slice_path)
        if not slice_path.exists():
            raise FileNotFoundError(slice_path)

        img_nii = nib.load(str(slice_path))
        img_arr = img_nii.get_fdata()
        if img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = img_arr[:, :, 0]
        pix = img_nii.header.get_zooms()[:2]
        pixel_area_mm2 = float(pix[0]) * float(pix[1])

        # In-process segmentasyon - ~0.2 sn
        seg = self.engine.segment(str(slice_path))
        if seg.ndim == 3 and seg.shape[2] == 1:
            seg = seg[:, :, 0]

        muscles = self._compute_muscle_metrics(img_arr, seg, pixel_area_mm2)
        asymmetry = self._compute_asymmetry(muscles)
        sarc = self._compute_sarcopenia(muscles, demographics)

        return {
            'slice_path':        str(slice_path),
            'pixel_spacing_mm':  list(map(float, pix)),
            'pixel_area_mm2':    pixel_area_mm2,
            'image_shape':       list(img_arr.shape),
            'demographics':      asdict(demographics) if demographics else None,
            'muscles':           [m.to_dict() for m in muscles],
            'asymmetry':         asymmetry,
            'sarcopenia':        asdict(sarc),
            'image_array':       img_arr,
            'segmentation_mask': seg,
        }

    @staticmethod
    def _otsu(values):
        if len(values) < 10:
            return float('inf')
        try:
            return float(threshold_otsu(values))
        except Exception:
            return float('inf')

    def _compute_muscle_metrics(self, img, seg, pixel_area_mm2) -> List[MuscleMetrics]:
        muscles = []
        for label, name in self.MUSCLE_LABELS.items():
            mask = (seg == label)
            n_vox = int(mask.sum())
            if n_vox < 10:
                continue
            vox = img[mask]
            thr = self._otsu(vox)
            ff = float((vox > thr).sum()) / n_vox if np.isfinite(thr) else 0.0
            muscles.append(MuscleMetrics(
                name=name,
                csa_mm2=n_vox * pixel_area_mm2,
                csa_cm2=n_vox * pixel_area_mm2 / 100.0,
                fat_fraction=ff,
                mean_intensity=float(vox.mean()),
                voxel_count=n_vox,
            ))
        return muscles

    @staticmethod
    def _compute_asymmetry(muscles):
        pairs = [('multifidus', 'multifidus_R', 'multifidus_L'),
                  ('erector', 'erector_R', 'erector_L'),
                  ('psoas', 'psoas_R', 'psoas_L'),
                  ('QL', 'QL_R', 'QL_L')]
        by_name = {m.name: m for m in muscles}
        out = {}
        for label, r, l in pairs:
            mr, ml = by_name.get(r), by_name.get(l)
            if mr and ml:
                mean = (mr.csa_mm2 + ml.csa_mm2) / 2.0
                if mean > 0:
                    out[f'{label}_asymmetry_pct'] = round(
                        100.0 * abs(mr.csa_mm2 - ml.csa_mm2) / mean, 2
                    )
        return out

    def _compute_sarcopenia(self, muscles, demo) -> SarcopeniaResult:
        by_name = {m.name: m for m in muscles}
        psoas_r, psoas_l = by_name.get('psoas_R'), by_name.get('psoas_L')

        tpa = (psoas_r.csa_mm2 if psoas_r else 0.0) + \
              (psoas_l.csa_mm2 if psoas_l else 0.0)
        tpa_cm2 = tpa / 100.0
        notes = []

        if tpa == 0.0:
            notes.append("Psoas segmentasyonu bulunamadi - PMI hesaplanamadi.")
            return SarcopeniaResult(None, 0.0, 0.0, None, notes=notes)

        if not psoas_r or not psoas_l:
            side = 'R' if psoas_r else 'L'
            notes.append(f"Tek tarafli psoas ({side}); PMI sadece bu tarafa dayanir.")

        height_m = demo.height_m if demo else None
        pmi = None
        thresholds = {}
        risk = 'Unknown'

        if height_m and demo and demo.sex in ('M', 'F'):
            pmi = tpa_cm2 / (height_m ** 2)
            for ref, vals in PMI_THRESHOLDS.items():
                thr = vals[demo.sex]
                thresholds[ref] = {
                    'threshold_cm2_per_m2': thr,
                    'patient_value':        round(pmi, 2),
                    'below_threshold':      pmi < thr,
                }
            below = sum(1 for v in thresholds.values() if v['below_threshold'])
            psoas_ff = [m.fat_fraction for m in (psoas_r, psoas_l) if m]
            mean_ff = float(np.mean(psoas_ff)) if psoas_ff else 0.0

            if below >= 2 or (below >= 1 and mean_ff > 0.30):
                risk = 'High'
            elif below >= 1 or mean_ff > 0.25:
                risk = 'Moderate'
            else:
                risk = 'Low'
        else:
            notes.append("Demografi eksik - PMI hesaplanamadi.")

        notes.append("Klinik sarkopeni tanisi degildir. EWGSOP2: kas kuvveti + fonksiyon testi gerekli.")
        notes.append("PMI esikleri CT-bazli (Hamaguchi/Englesbe/Durand); MR'da ~5-10% sapma olasi.")

        return SarcopeniaResult(
            pmi_cm2_per_m2=round(pmi, 2) if pmi else None,
            total_psoas_area_mm2=round(tpa, 1),
            total_psoas_area_cm2=round(tpa_cm2, 2),
            height_m=height_m,
            thresholds=thresholds,
            risk_category=risk,
            notes=notes,
        )
