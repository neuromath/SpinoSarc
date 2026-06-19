"""SpinoSarc — Tek sayfa premium klinik rapor (v2)."""
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from .analyzer import SpinoSarcAnalyzer, Demographics

mpl.rcParams['font.family'] = 'DejaVu Sans'

NAVY    = '#0A2540'
PRIMARY = '#1E5AA8'
ACCENT  = '#00B8D4'
SUCCESS = '#0FA958'
WARNING = '#F59E0B'
DANGER  = '#DC2626'
LIGHT   = '#F8FAFC'
GRID    = '#E5E7EB'
TXT_DK  = '#0F172A'
TXT_MD  = '#475569'
TXT_LT  = '#94A3B8'

MUSCLE_COLORS = {
    1: ('Multifidus R', '#EF4444'), 2: ('Multifidus L', '#3B82F6'),
    3: ('Erector R',    '#10B981'), 4: ('Erector L',    '#8B5CF6'),
    5: ('Psoas R',      '#F59E0B'), 6: ('Psoas L',      '#06B6D4'),
    7: ('QL R',         '#EC4899'), 8: ('QL L',         '#64748B'),
}

RISK_STYLE = {
    'Low':      {'color': SUCCESS, 'label': 'LOW RISK',      'desc': 'PMI within normal range, low fat infiltration'},
    'Moderate': {'color': WARNING, 'label': 'MODERATE RISK', 'desc': 'PMI below threshold or elevated fat infiltration'},
    'High':     {'color': DANGER,  'label': 'HIGH RISK',     'desc': 'PMI substantially low + elevated fat infiltration'},
    'Unknown':  {'color': TXT_LT,  'label': 'NOT CALCULATED', 'desc': 'Demographic data required for risk assessment'},
}


def _render_pdf(label, r, output_pdf):
    """Result dict'ten direkt PDF uretir (analiz yapilmis varsayilir)."""
    s = r['sarcopenia']
    muscles = r['muscles']
    asym = r['asymmetry']
    demo_d = r.get('demographics')

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # ============== HEADER ==============
    fig.add_artist(Rectangle((0, 0.985), 1, 0.005, transform=fig.transFigure,
                              facecolor=ACCENT, zorder=2))
    fig.text(0.06, 0.962, 'SpinoSarc', fontsize=18, weight='bold', color=NAVY)
    fig.text(0.06, 0.945, 'Paraspinal Muscle & Sarcopenia Screening Report',
              fontsize=8.5, color=TXT_MD, style='italic')
    fig.text(0.94, 0.962, datetime.now().strftime('%Y-%m-%d  %H:%M'),
              fontsize=8, color=TXT_LT, ha='right')
    fig.text(0.94, 0.945, 'Report v1.0  ·  Research use',
              fontsize=7, color=TXT_LT, ha='right', style='italic')
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.932, 0.932],
                                color=GRID, linewidth=0.5,
                                transform=fig.transFigure))

    # ============== PATIENT INFO BAR (genis tek satir) ==============
    pb_y = 0.880
    pb_h = 0.040
    fig.add_artist(FancyBboxPatch((0.06, pb_y - pb_h), 0.88, pb_h,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    transform=fig.transFigure,
                                    facecolor=NAVY, edgecolor='none'))
    # Sol: PATIENT
    fig.text(0.080, pb_y - 0.011, 'PATIENT', fontsize=7,
              color=ACCENT, weight='bold')
    fig.text(0.080, pb_y - 0.030, label, fontsize=15,
              color='white', weight='bold')
    # Orta: demografi
    if demo_d:
        parts = []
        if demo_d.get('age'):       parts.append(f"{demo_d['age']}y")
        if demo_d.get('sex'):       parts.append('Male' if demo_d['sex']=='M' else 'Female')
        if demo_d.get('height_cm'): parts.append(f"{demo_d['height_cm']} cm")
        if demo_d.get('weight_kg'): parts.append(f"{demo_d['weight_kg']} kg")
        if demo_d.get('height_cm') and demo_d.get('weight_kg'):
            bmi = demo_d['weight_kg']/((demo_d['height_cm']/100)**2)
            parts.append(f"BMI {bmi:.1f}")
        fig.text(0.30, pb_y - 0.022, '  ·  '.join(parts),
                  fontsize=11, color='white', alpha=0.95, va='center')
    # Sag: tarih/zaman (alternatif olarak study no)
    fig.text(0.92, pb_y - 0.011, 'STUDY DATE', fontsize=7,
              color=ACCENT, weight='bold', ha='right')
    fig.text(0.92, pb_y - 0.030, datetime.now().strftime('%Y-%m-%d'),
              fontsize=11, color='white', ha='right')

    # ============== BUYUK RISK PANELI (tam genislik) ==============
    risk = s['risk_category']
    rs = RISK_STYLE[risk]
    risk_y_top = 0.825
    risk_h = 0.085
    fig.add_artist(FancyBboxPatch((0.06, risk_y_top - risk_h), 0.88, risk_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.010",
                                    transform=fig.transFigure,
                                    facecolor=rs['color'], alpha=0.10,
                                    edgecolor=rs['color'], linewidth=1.8))
    # Sol bolum: ETIKET + ACIKLAMA
    fig.text(0.085, risk_y_top - 0.018, 'SARCOPENIA RISK ASSESSMENT',
              fontsize=8, color=TXT_MD, weight='bold')
    fig.text(0.085, risk_y_top - 0.045, rs['label'],
              fontsize=22, weight='bold', color=rs['color'])
    fig.text(0.085, risk_y_top - 0.070, rs['desc'],
              fontsize=9, color=TXT_DK, style='italic')

    # Sag bolum: PMI buyuk rakam
    if s.get('pmi_cm2_per_m2'):
        fig.text(0.92, risk_y_top - 0.018, 'PSOAS MUSCLE INDEX',
                  fontsize=8, color=TXT_MD, weight='bold', ha='right')
        fig.text(0.92, risk_y_top - 0.048, f"{s['pmi_cm2_per_m2']}",
                  fontsize=32, weight='bold', color=rs['color'], ha='right')
        fig.text(0.92, risk_y_top - 0.072, 'cm²/m²',
                  fontsize=9, color=TXT_MD, ha='right', style='italic')

    # ============== 3 KPI STRIP ==============
    kpi_y_top = 0.730
    kpi_h = 0.045
    kpi_w = 0.282
    kpi_gap = 0.013

    mf_ff = [m['fat_fraction'] for m in muscles if 'multifidus' in m['name']]
    mf_mean = np.mean(mf_ff)*100 if mf_ff else 0

    def kpi(x, label_t, val, sub, color):
        fig.add_artist(FancyBboxPatch((x, kpi_y_top - kpi_h), kpi_w, kpi_h,
                                        boxstyle="round,pad=0.002,rounding_size=0.005",
                                        transform=fig.transFigure,
                                        facecolor=LIGHT,
                                        edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.010, kpi_y_top - 0.011, label_t.upper(), fontsize=7,
                  color=TXT_MD, weight='bold')
        fig.text(x + 0.010, kpi_y_top - 0.034, val, fontsize=18,
                  weight='bold', color=color)
        fig.text(x + kpi_w - 0.010, kpi_y_top - 0.034, sub, fontsize=8,
                  color=TXT_LT, ha='right', style='italic', va='center')

    kpi(0.06,                       'Total Psoas Area',
         f"{s['total_psoas_area_cm2']}", 'cm²', PRIMARY)
    kpi(0.06 + kpi_w + kpi_gap,     'Multifidus Fat Fraction',
         f"{mf_mean:.1f}%", 'mean R+L',
         SUCCESS if mf_mean<10 else (WARNING if mf_mean<25 else DANGER))
    kpi(0.06 + 2*(kpi_w + kpi_gap), 'Muscles Segmented',
         f"{len(muscles)}", 'of 8', PRIMARY)

    # ============== OVERLAY (sol) + KAS TABLOSU (sag) ==============
    img = r['image_array']
    seg = r['segmentation_mask']
    img_disp = np.rot90(img, k=-1)
    seg_disp = np.rot90(seg, k=-1)

    sec_y_top = 0.660
    fig.text(0.06, sec_y_top, 'SEGMENTATION', fontsize=8,
              weight='bold', color=PRIMARY)
    fig.add_artist(plt.Line2D([0.06, 0.12], [sec_y_top - 0.005, sec_y_top - 0.005],
                                color=ACCENT, linewidth=2,
                                transform=fig.transFigure))

    fig.text(0.51, sec_y_top, 'MUSCLE METRICS', fontsize=8,
              weight='bold', color=PRIMARY)
    fig.add_artist(plt.Line2D([0.51, 0.58], [sec_y_top - 0.005, sec_y_top - 0.005],
                                color=ACCENT, linewidth=2,
                                transform=fig.transFigure))

    # Overlay - sol
    ax_x, ax_y, ax_w, ax_h = 0.06, 0.395, 0.42, 0.230
    ax_orig = fig.add_axes([ax_x, ax_y, ax_w/2 - 0.005, ax_h])
    ax_orig.imshow(img_disp, cmap='gray', interpolation='bilinear')
    ax_orig.set_title('Original', fontsize=9, weight='bold', color=NAVY, pad=4)
    ax_orig.axis('off')

    ax_seg = fig.add_axes([ax_x + ax_w/2 + 0.005, ax_y, ax_w/2 - 0.005, ax_h])
    ax_seg.imshow(img_disp, cmap='gray', interpolation='bilinear')
    overlay = np.zeros((*seg_disp.shape, 4))
    for lid, (name, color) in MUSCLE_COLORS.items():
        mask = (seg_disp == lid)
        if mask.sum() == 0: continue
        rgba = to_rgba(color, alpha=0.62)
        for c in range(4):
            overlay[..., c][mask] = rgba[c]
    ax_seg.imshow(overlay, interpolation='nearest')
    ax_seg.set_title('Auto-Segmentation', fontsize=9, weight='bold',
                     color=NAVY, pad=4)
    ax_seg.axis('off')

    # Lejant (overlay altinda kompakt 4-col)
    leg_y = 0.380
    visible = [(lid, n, c) for lid, (n, c) in MUSCLE_COLORS.items()
                if (seg == lid).sum() > 0]
    n_per_row = 4
    cell_w = 0.42 / n_per_row
    for i, (lid, name, color) in enumerate(visible):
        row, col = i // n_per_row, i % n_per_row
        cx = ax_x + col * cell_w
        cy = leg_y - row * 0.014
        fig.add_artist(Circle((cx + 0.007, cy), 0.003,
                                facecolor=color, edgecolor='none',
                                transform=fig.transFigure))
        fig.text(cx + 0.014, cy, name, fontsize=7, color=TXT_DK, va='center')

    # Sag: Kas tablosu
    tbl_x = 0.51
    headers = ['Muscle', 'CSA mm²', 'FF %', 'Goutallier']
    col_x = [tbl_x, tbl_x + 0.16, tbl_x + 0.26, tbl_x + 0.33]
    hy = sec_y_top - 0.025
    for h, x in zip(headers, col_x):
        fig.text(x, hy, h.upper(), fontsize=7, weight='bold', color=TXT_MD)
    fig.add_artist(plt.Line2D([tbl_x, 0.94], [hy - 0.005, hy - 0.005],
                                color=NAVY, linewidth=0.8,
                                transform=fig.transFigure))

    row_y = hy - 0.022
    for m in muscles:
        ff = m['fat_fraction']
        if ff < 0.10:   gout, gcol = 'G0/1', SUCCESS
        elif ff < 0.25: gout, gcol = 'G2',   WARNING
        elif ff < 0.50: gout, gcol = 'G3',   DANGER
        else:           gout, gcol = 'G4',   DANGER

        color = next((c for lid, (n, c) in MUSCLE_COLORS.items()
                       if n.lower().replace(' ', '_') == m['name'].lower()), '#000')
        fig.add_artist(Circle((col_x[0] - 0.007, row_y + 0.003), 0.003,
                                facecolor=color, edgecolor='none',
                                transform=fig.transFigure))
        short_name = m['name'].replace('_', ' ').replace('multifidus','Mult.').replace('erector','Erec.').replace('psoas','Psoas').replace('QL', 'QL')
        # Title-case ilk harf
        short_name = ' '.join(w.capitalize() if not w.isupper() else w for w in short_name.split())
        fig.text(col_x[0], row_y, short_name, fontsize=8.5, color=TXT_DK)
        fig.text(col_x[1], row_y, f"{m['csa_mm2']:.0f}", fontsize=8.5, color=TXT_DK)
        ff_color = SUCCESS if ff<0.10 else (WARNING if ff<0.25 else DANGER)
        fig.text(col_x[2], row_y, f"{ff*100:.1f}", fontsize=8.5,
                  weight='bold', color=ff_color)
        fig.text(col_x[3], row_y, gout, fontsize=8.5, weight='bold', color=gcol)
        row_y -= 0.022

    # ============== ASYMMETRY (tam genislik bar grafik) ==============
    asym_y_top = 0.275
    fig.text(0.06, asym_y_top, 'BILATERAL ASYMMETRY',
              fontsize=8, weight='bold', color=PRIMARY)
    fig.add_artist(plt.Line2D([0.06, 0.13], [asym_y_top - 0.005, asym_y_top - 0.005],
                                color=ACCENT, linewidth=2,
                                transform=fig.transFigure))

    y = asym_y_top - 0.028
    for k, v in asym.items():
        muscle_name = k.replace('_asymmetry_pct', '').capitalize()
        a_color = SUCCESS if v < 5 else (WARNING if v < 15 else DANGER)
        # Track
        fig.add_artist(Rectangle((0.20, y - 0.005), 0.66, 0.012,
                                  transform=fig.transFigure,
                                  facecolor=GRID, edgecolor='none'))
        # Bar
        bar_w = min(v / 30.0, 1.0) * 0.66
        fig.add_artist(Rectangle((0.20, y - 0.005), bar_w, 0.012,
                                  transform=fig.transFigure,
                                  facecolor=a_color, edgecolor='none'))
        fig.text(0.06, y, muscle_name, fontsize=9, color=TXT_DK, va='center')
        fig.text(0.88, y, f"{v}%", fontsize=9, weight='bold',
                  color=a_color, ha='left', va='center')
        y -= 0.022

    # Asymmetry severity legend (kucuk altta)
    leg_asym_y = y - 0.005
    fig.text(0.06, leg_asym_y, 'Asymmetry interpretation:',
              fontsize=7, color=TXT_MD, style='italic')
    fig.add_artist(Circle((0.22, leg_asym_y + 0.003), 0.003, facecolor=SUCCESS,
                            transform=fig.transFigure))
    fig.text(0.232, leg_asym_y, '< 5% normal', fontsize=7, color=TXT_MD)
    fig.add_artist(Circle((0.36, leg_asym_y + 0.003), 0.003, facecolor=WARNING,
                            transform=fig.transFigure))
    fig.text(0.372, leg_asym_y, '5–15% mild', fontsize=7, color=TXT_MD)
    fig.add_artist(Circle((0.50, leg_asym_y + 0.003), 0.003, facecolor=DANGER,
                            transform=fig.transFigure))
    fig.text(0.512, leg_asym_y, '> 15% pronounced', fontsize=7, color=TXT_MD)

    # METHODOLOGY ve REFERENCES kaldirildi (end-user icin gereksiz)
    # ============== CLINICAL DISCLAIMER (alttaki sari kutu) ==============
    disc_box_y = 0.08
    disc_box_h = 0.055
    fig.add_artist(FancyBboxPatch((0.06, disc_box_y), 0.88, disc_box_h,
                                    boxstyle="round,pad=0.003,rounding_size=0.005",
                                    transform=fig.transFigure,
                                    facecolor='#FEF3C7', edgecolor=WARNING,
                                    linewidth=0.8))
    fig.text(0.5, disc_box_y + disc_box_h - 0.010, 'CLINICAL DISCLAIMER',
              ha='center', fontsize=7.5, weight='bold', color=DANGER)
    fig.text(0.5, disc_box_y + disc_box_h - 0.024,
              'Supports clinical decision-making — does not constitute a sarcopenia diagnosis.',
              ha='center', fontsize=7, color=TXT_DK)
    fig.text(0.5, disc_box_y + disc_box_h - 0.036,
              'EWGSOP2 (2019) requires muscle strength + physical performance testing. CT-derived thresholds may differ ~5–10% in MR.',
              ha='center', fontsize=6.5, color=TXT_DK, style='italic')

    # Footer
    fig.text(0.06, 0.015, 'SpinoSarc v1.0', fontsize=6.5,
              color=TXT_LT, style='italic')
    fig.text(0.94, 0.015, 'spinosarc.ai', fontsize=6.5,
              color=TXT_LT, ha='right', style='italic')

    with PdfPages(str(output_pdf)) as pdf:
        pdf.savefig(fig, bbox_inches=None, dpi=300)
    plt.close()
    print(f"[+] {output_pdf}")


def generate_single_page_from_result(label, result, output_pdf):
    """GUI'den cagrilir: zaten yapilmis analizden PDF uretir."""
    _render_pdf(label, result, output_pdf)


def generate_single_page(slice_path, label, demo, output_pdf, mm_script=None):
    """Standalone: slice'tan analiz + PDF (CLI/batch icin)."""
    from .analyzer import SpinoSarcAnalyzer
    an = SpinoSarcAnalyzer(use_gpu=True)
    print(f"[*] {label}...")
    r = an.analyze(slice_path, demo)
    _render_pdf(label, r, output_pdf)


if __name__ == '__main__':
    from .analyzer import Demographics
    out = '/tmp/SpinoSarc_OnePage_sub001.pdf'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    generate_single_page(
        slice_path='./data/interim/sub-001/anat/sub-001_axial_t2.nii.gz',  # example path - replace with your own
        label='SUB-001',
        demo=Demographics(age=65, sex='M', height_cm=172, weight_kg=78),
        output_pdf=out,
    )
    print(out)
