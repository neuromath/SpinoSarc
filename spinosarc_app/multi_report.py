"""
Coklu-hasta PDF raporu - SpinoSarc.
3+ hasta tek PDF, kapak + her hasta detay + ozet karsilastirma sayfasi.
"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from .analyzer import SpinoSarcAnalyzer, Demographics


MUSCLE_COLORS = {
    1: ('multifidus_R', '#E74C3C'),
    2: ('multifidus_L', '#3498DB'),
    3: ('erector_R',    '#2ECC71'),
    4: ('erector_L',    '#9B59B6'),
    5: ('psoas_R',      '#F39C12'),
    6: ('psoas_L',      '#1ABC9C'),
    7: ('QL_R',         '#E91E63'),
    8: ('QL_L',         '#34495E'),
}

PRIMARY_COLOR = '#1F3864'
ACCENT_COLOR  = '#C0392B'
TEXT_GRAY     = '#555555'

RISK_COLOR = {
    'Low': '#27AE60',
    'Moderate': '#E67E22',
    'High': '#C0392B',
    'Unknown': '#7F8C8D',
}


# ============================================================
def cover_page(pdf, n_patients):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # Üst bant
    fig.add_artist(plt.Rectangle((0, 0.93), 1, 0.07,
                                  transform=fig.transFigure,
                                  facecolor=PRIMARY_COLOR, zorder=1))
    fig.text(0.5, 0.96, 'SpinoSarc', ha='center', fontsize=24,
              weight='bold', color='white', transform=fig.transFigure)

    # Alt başlık
    fig.text(0.5, 0.85, 'Paraspinal Muscle & Sarcopenia Analysis Report',
              ha='center', fontsize=16, weight='bold', color=PRIMARY_COLOR)
    fig.text(0.5, 0.81, 'Automated quantification from lumbar MR axial slices',
              ha='center', fontsize=12, style='italic', color=TEXT_GRAY)

    # Orta panel
    fig.text(0.5, 0.65, f'{n_patients}', ha='center', fontsize=64,
              weight='bold', color=PRIMARY_COLOR)
    fig.text(0.5, 0.60, 'patients analyzed', ha='center', fontsize=14,
              color=TEXT_GRAY)

    # Detay kutusu
    box_y = 0.40
    fig.add_artist(plt.Rectangle((0.10, box_y - 0.05), 0.80, 0.20,
                                  transform=fig.transFigure,
                                  facecolor='#F4F6F8', edgecolor=PRIMARY_COLOR,
                                  linewidth=1.5, zorder=1))
    fig.text(0.5, box_y + 0.12, 'Report contents', ha='center',
              fontsize=12, weight='bold', color=PRIMARY_COLOR)
    contents = [
        '• Per-patient muscle morphometry (CSA, fat fraction, asymmetry)',
        '• Psoas-based sarcopenia screening (PMI vs literature thresholds)',
        '• High-resolution segmentation overlays',
        '• Cross-patient comparison summary',
    ]
    y = box_y + 0.08
    for c in contents:
        fig.text(0.14, y, c, fontsize=10, color='#2C3E50')
        y -= 0.025

    # Footer
    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.5, 0.10, f'Generated: {today}', ha='center', fontsize=10,
              color=TEXT_GRAY)
    fig.text(0.5, 0.07, 'SpinoSarc v0.1.0  ·  For research use only',
              ha='center', fontsize=9, style='italic', color=TEXT_GRAY)
    fig.text(0.5, 0.04,
              'Not a clinical sarcopenia diagnosis (see EWGSOP2 criteria)',
              ha='center', fontsize=8, style='italic', color=ACCENT_COLOR)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ============================================================
def patient_detail_page(pdf, label, result):
    """Hasta bilgileri sayfasi."""
    demo    = result.get('demographics')
    muscles = result['muscles']
    sarc    = result['sarcopenia']
    asym    = result['asymmetry']

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # Header bant
    fig.add_artist(plt.Rectangle((0, 0.94), 1, 0.05,
                                  transform=fig.transFigure,
                                  facecolor=PRIMARY_COLOR))
    fig.text(0.07, 0.96, f'Patient: {label}', fontsize=14, weight='bold', color='white')
    fig.text(0.93, 0.96, 'SpinoSarc', fontsize=10, color='white', ha='right',
              style='italic')

    # === Demografi paneli ===
    y = 0.89
    fig.text(0.07, y, 'Demographics', fontsize=12, weight='bold', color=PRIMARY_COLOR)
    y -= 0.02
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=0.5, transform=fig.transFigure))
    y -= 0.025

    if demo:
        rows = [
            ('Age',    f"{demo['age']} years" if demo.get('age') else '—'),
            ('Sex',    demo.get('sex') or '—'),
            ('Height', f"{demo['height_cm']} cm" if demo.get('height_cm') else '—'),
            ('Weight', f"{demo['weight_kg']} kg" if demo.get('weight_kg') else '—'),
        ]
        bmi = None
        if demo.get('height_cm') and demo.get('weight_kg'):
            bmi = demo['weight_kg'] / ((demo['height_cm']/100)**2)
            rows.append(('BMI', f"{bmi:.1f} kg/m²"))
        col_w = 0.4
        for i, (k, v) in enumerate(rows):
            row, col = i % 3, i // 3
            x = 0.10 + col * col_w
            yy = y - row * 0.025
            fig.text(x, yy, f"{k}:", fontsize=10, color=TEXT_GRAY)
            fig.text(x + 0.10, yy, v, fontsize=10, weight='bold')
        y -= 0.085
    else:
        fig.text(0.10, y, '(no demographics provided)', fontsize=10,
                  style='italic', color=TEXT_GRAY)
        y -= 0.04

    # Slice bilgisi
    y -= 0.01
    fig.text(0.07, y, 'Slice', fontsize=12, weight='bold', color=PRIMARY_COLOR)
    y -= 0.02
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=0.5, transform=fig.transFigure))
    y -= 0.025
    px = result['pixel_spacing_mm']
    fig.text(0.10, y, f"Source: {Path(result['slice_path']).name}", fontsize=9,
              color=TEXT_GRAY)
    y -= 0.022
    fig.text(0.10, y, f"Pixel spacing: {px[0]:.2f} × {px[1]:.2f} mm", fontsize=9,
              color=TEXT_GRAY)
    y -= 0.022
    fig.text(0.10, y, f"Image shape: {result['image_shape'][0]} × {result['image_shape'][1]} px",
              fontsize=9, color=TEXT_GRAY)
    y -= 0.03

    # === Kas metrikleri ===
    fig.text(0.07, y, 'Muscle morphometry', fontsize=12, weight='bold', color=PRIMARY_COLOR)
    y -= 0.02
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=0.5, transform=fig.transFigure))
    y -= 0.025

    headers = ['Muscle', 'CSA (mm²)', 'CSA (cm²)', 'Fat fraction', 'Goutallier']
    col_x   = [0.10, 0.32, 0.46, 0.60, 0.78]
    for h, x in zip(headers, col_x):
        fig.text(x, y, h, fontsize=9, weight='bold', color=PRIMARY_COLOR)
    y -= 0.005
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color='#BDC3C7',
                                linewidth=0.3, transform=fig.transFigure))
    y -= 0.020

    for m in muscles:
        ff = m['fat_fraction']
        if ff < 0.10:   gout = 'G0/1'
        elif ff < 0.25: gout = 'G2'
        elif ff < 0.50: gout = 'G3'
        else:           gout = 'G4'
        # Renkli kas isim hücresi
        color = next((c for lbl, (n, c) in MUSCLE_COLORS.items() if n == m['name']), '#000')
        fig.text(col_x[0], y, '●', fontsize=12, color=color)
        fig.text(col_x[0] + 0.015, y, m['name'], fontsize=9)
        fig.text(col_x[1], y, f"{m['csa_mm2']:.1f}", fontsize=9)
        fig.text(col_x[2], y, f"{m['csa_cm2']:.2f}", fontsize=9)
        # FF renkli
        ff_color = '#27AE60' if ff < 0.10 else ('#E67E22' if ff < 0.25 else '#C0392B')
        fig.text(col_x[3], y, f"{ff*100:.1f}%", fontsize=9, color=ff_color,
                  weight='bold')
        fig.text(col_x[4], y, gout, fontsize=9)
        y -= 0.022

    y -= 0.01

    # === Asimetri ===
    fig.text(0.07, y, 'Bilateral asymmetry', fontsize=12, weight='bold', color=PRIMARY_COLOR)
    y -= 0.02
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=0.5, transform=fig.transFigure))
    y -= 0.025
    for k, v in asym.items():
        label = k.replace('_asymmetry_pct', '')
        # Bar
        bar_x, bar_w_max = 0.30, 0.35
        bar_w = min(v / 30.0, 1.0) * bar_w_max
        color = '#27AE60' if v < 5 else ('#E67E22' if v < 15 else '#C0392B')
        fig.add_artist(plt.Rectangle((bar_x, y - 0.002), bar_w, 0.014,
                                      transform=fig.transFigure,
                                      facecolor=color, alpha=0.7))
        fig.text(0.10, y, f"{label}:", fontsize=10)
        fig.text(0.68, y, f"{v}%", fontsize=10, weight='bold', color=color)
        y -= 0.022

    y -= 0.01

    # === Sarkopeni ===
    fig.text(0.07, y, 'Sarcopenia screening (PMI-based)', fontsize=12,
              weight='bold', color=PRIMARY_COLOR)
    y -= 0.02
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=0.5, transform=fig.transFigure))
    y -= 0.025

    fig.text(0.10, y, 'Total Psoas Area (TPA):', fontsize=10, color=TEXT_GRAY)
    fig.text(0.45, y, f"{sarc['total_psoas_area_cm2']} cm² ({sarc['total_psoas_area_mm2']:.0f} mm²)",
              fontsize=10, weight='bold')
    y -= 0.022

    if sarc.get('pmi_cm2_per_m2') is not None:
        fig.text(0.10, y, 'Psoas Muscle Index (PMI):', fontsize=10, color=TEXT_GRAY)
        fig.text(0.45, y, f"{sarc['pmi_cm2_per_m2']} cm²/m²", fontsize=10,
                  weight='bold', color=PRIMARY_COLOR)
        y -= 0.025

        # Risk gostergesi - buyuk renkli kutu
        risk = sarc['risk_category']
        rc = RISK_COLOR[risk]
        fig.add_artist(plt.Rectangle((0.10, y - 0.030), 0.80, 0.030,
                                      transform=fig.transFigure,
                                      facecolor=rc, alpha=0.15,
                                      edgecolor=rc, linewidth=1.5))
        fig.text(0.50, y - 0.015, f'Risk Category:  {risk.upper()}',
                  ha='center', va='center', fontsize=14, weight='bold', color=rc)
        y -= 0.045

        # Esik tablosu
        fig.text(0.10, y, 'Threshold comparison', fontsize=10, weight='bold',
                  color=PRIMARY_COLOR)
        y -= 0.020
        for ref, info in sarc['thresholds'].items():
            below = info['below_threshold']
            flag_color = '#C0392B' if below else '#27AE60'
            flag_text = '✗ BELOW THRESHOLD' if below else '✓ Normal range'
            fig.text(0.12, y, ref, fontsize=8, color='#2C3E50')
            fig.text(0.62, y, f"thr={info['threshold_cm2_per_m2']}", fontsize=8,
                      color=TEXT_GRAY)
            fig.text(0.74, y, flag_text, fontsize=8, color=flag_color, weight='bold')
            y -= 0.020
    else:
        fig.text(0.10, y, 'Demographics (sex, height) required for PMI calculation',
                  fontsize=10, style='italic', color=TEXT_GRAY)
        y -= 0.025

    # Footer
    fig.text(0.5, 0.04, 'For research use only · Not a clinical diagnosis',
              ha='center', fontsize=8, style='italic', color=TEXT_GRAY)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ============================================================
def patient_overlay_page(pdf, label, result):
    """Yuksek kaliteli overlay sayfasi."""
    img = result['image_array']
    seg = result['segmentation_mask']

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # Header
    fig.add_artist(plt.Rectangle((0, 0.94), 1, 0.05,
                                  transform=fig.transFigure,
                                  facecolor=PRIMARY_COLOR))
    fig.text(0.07, 0.96, f'Patient: {label}  —  Segmentation overlay',
              fontsize=13, weight='bold', color='white')

    # 2-panel: original + overlay
    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.18,
                            left=0.08, right=0.92, hspace=0.05, wspace=0.05)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_over = fig.add_subplot(gs[0, 1])

    # Display rotation/orientation: orient image consistently
    img_disp = np.rot90(img, k=-1) if img.shape[0] == img.shape[1] else img

    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title('Original', fontsize=11, weight='bold', color=PRIMARY_COLOR)
    ax_orig.axis('off')

    seg_disp = np.rot90(seg, k=-1) if seg.shape[0] == seg.shape[1] else seg
    ax_over.imshow(img_disp, cmap='gray')

    overlay = np.zeros((*seg_disp.shape, 4))
    for label_id, (name, color) in MUSCLE_COLORS.items():
        mask = (seg_disp == label_id)
        if mask.sum() == 0:
            continue
        rgba = to_rgba(color, alpha=0.55)
        for c in range(4):
            overlay[..., c][mask] = rgba[c]
    ax_over.imshow(overlay)
    ax_over.set_title('Segmentation overlay', fontsize=11, weight='bold',
                      color=PRIMARY_COLOR)
    ax_over.axis('off')

    # Legend altta
    legend_items = []
    for label_id, (name, color) in MUSCLE_COLORS.items():
        if (seg == label_id).sum() > 0:
            legend_items.append(mpatches.Patch(color=color, label=name))
    if legend_items:
        ax_leg = fig.add_subplot(gs[1, :])
        ax_leg.legend(handles=legend_items, loc='center', ncol=4, fontsize=10,
                       frameon=False)
        ax_leg.axis('off')

    # Footer
    fig.text(0.5, 0.10,
              'Colored regions indicate auto-segmented paraspinal muscles',
              ha='center', fontsize=9, style='italic', color=TEXT_GRAY)
    fig.text(0.5, 0.07, 'Segmentation by MuscleMap contrast-agnostic abdomen model',
              ha='center', fontsize=8, color=TEXT_GRAY)
    fig.text(0.5, 0.04, f"Image: {result['image_shape']} px, pixel {result['pixel_spacing_mm'][0]} mm",
              ha='center', fontsize=8, color=TEXT_GRAY)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ============================================================
def summary_comparison_page(pdf, patients):
    """Tum hastalarin karsilastirma ozeti."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    fig.add_artist(plt.Rectangle((0, 0.94), 1, 0.05,
                                  transform=fig.transFigure,
                                  facecolor=PRIMARY_COLOR))
    fig.text(0.5, 0.96, 'Cross-Patient Summary', ha='center',
              fontsize=14, weight='bold', color='white')

    # Karsilastirma tablosu
    y = 0.89
    fig.text(0.5, y, 'Comparative metrics', ha='center', fontsize=13,
              weight='bold', color=PRIMARY_COLOR)
    y -= 0.04

    headers = ['Patient', 'TPA cm²', 'PMI cm²/m²', 'Multifidus FF%', 'Risk']
    col_x   = [0.10, 0.30, 0.45, 0.60, 0.78]
    for h, x in zip(headers, col_x):
        fig.text(x, y, h, fontsize=10, weight='bold', color=PRIMARY_COLOR)
    y -= 0.005
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=PRIMARY_COLOR,
                                linewidth=1.0, transform=fig.transFigure))
    y -= 0.025

    for label, r in patients:
        s = r['sarcopenia']
        # Multifidus mean FF
        mf = [m['fat_fraction'] for m in r['muscles']
               if 'multifidus' in m['name']]
        mf_mean = np.mean(mf) * 100 if mf else 0.0

        pmi_val = s.get('pmi_cm2_per_m2')
        pmi_str = f"{pmi_val}" if pmi_val else '—'

        risk = s['risk_category']
        rc = RISK_COLOR.get(risk, '#000')

        fig.text(col_x[0], y, label, fontsize=10)
        fig.text(col_x[1], y, f"{s['total_psoas_area_cm2']}", fontsize=10)
        fig.text(col_x[2], y, pmi_str, fontsize=10)
        ff_color = '#27AE60' if mf_mean < 10 else ('#E67E22' if mf_mean < 25 else '#C0392B')
        fig.text(col_x[3], y, f"{mf_mean:.1f}%", fontsize=10, color=ff_color)
        fig.text(col_x[4], y, risk, fontsize=10, weight='bold', color=rc)
        y -= 0.028

    # Bar grafigi: PMI karsilastirma
    has_pmi = [(l, r['sarcopenia']['pmi_cm2_per_m2']) for l, r in patients
                if r['sarcopenia'].get('pmi_cm2_per_m2')]
    if has_pmi:
        ax = fig.add_axes([0.15, 0.40, 0.70, 0.20])
        labels = [l for l, _ in has_pmi]
        vals   = [v for _, v in has_pmi]
        colors = [RISK_COLOR.get(
            next((r['sarcopenia']['risk_category'] for ll, r in patients if ll == l), 'Unknown'),
            '#888'
        ) for l in labels]
        ax.barh(labels, vals, color=colors, edgecolor='black', alpha=0.8)
        # Esik cizgileri
        ax.axvline(6.36, ls='--', color='#C0392B', alpha=0.6, lw=1)
        ax.text(6.36, len(labels)-0.4, 'Hamaguchi M=6.36', fontsize=7, color='#C0392B')
        ax.axvline(3.92, ls='--', color='#E67E22', alpha=0.6, lw=1)
        ax.text(3.92, -0.4, 'Hamaguchi F=3.92', fontsize=7, color='#E67E22')
        ax.set_xlabel('PMI (cm²/m²)', fontsize=10)
        ax.set_title('PMI comparison vs literature thresholds', fontsize=11,
                      color=PRIMARY_COLOR, weight='bold')
        ax.grid(axis='x', alpha=0.3)

    # Footer + disclaimer
    fig.text(0.5, 0.20,
              'PMI thresholds: Hamaguchi 2016 (HCC, Asian) · Englesbe 2010 (US) · Durand 2014',
              ha='center', fontsize=8, color=TEXT_GRAY)
    fig.text(0.5, 0.17,
              'CT-derived thresholds; MR values may differ by ~5-10%',
              ha='center', fontsize=8, style='italic', color=TEXT_GRAY)
    fig.text(0.5, 0.10,
              'Sarcopenia diagnosis requires muscle strength + physical performance testing',
              ha='center', fontsize=9, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.07, '(EWGSOP2 2019)', ha='center', fontsize=8, color=ACCENT_COLOR)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ============================================================
def generate_multi_report(patients_with_demo, output_pdf, musclemap_script):
    """
    patients_with_demo: list of (slice_path, label, Demographics or None)
    """
    an = SpinoSarcAnalyzer(musclemap_script, use_gpu=True)
    results = []

    for slice_path, label, demo in patients_with_demo:
        print(f"[*] Analyzing {label}...")
        r = an.analyze(slice_path, demo)
        results.append((label, r))
        s = r['sarcopenia']
        print(f"    TPA={s['total_psoas_area_cm2']} cm²  PMI={s.get('pmi_cm2_per_m2')}  Risk={s['risk_category']}")

    print(f"\n[*] Generating PDF: {output_pdf}")
    with PdfPages(str(output_pdf)) as pdf:
        cover_page(pdf, len(results))
        for label, r in results:
            patient_detail_page(pdf, label, r)
            patient_overlay_page(pdf, label, r)
        summary_comparison_page(pdf, results)
    print(f"[+] Done: {output_pdf}")


if __name__ == '__main__':
    BASE = Path('/workspace/SpinoSarc/data/l4l5_slices')

    patients = [
        (str(BASE / 'sub-001_L4L5.nii.gz'), 'sub-001 (65y M, 172cm)',
         Demographics(age=65, sex='M', height_cm=172, weight_kg=78)),
        (str(BASE / 'sub-050_L4L5.nii.gz'), 'sub-050 (no demographics)',
         None),
        (str(BASE / 'sub-100_L4L5.nii.gz'), 'sub-100 (72y F, 158cm)',
         Demographics(age=72, sex='F', height_cm=158, weight_kg=65)),
    ]

    out_pdf = '/workspace/SpinoSarc/reports/SpinoSarc_3patient_report.pdf'
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    generate_multi_report(
        patients,
        out_pdf,
        musclemap_script='/workspace/MuscleMap/scripts/mm_segment.py'
    )
