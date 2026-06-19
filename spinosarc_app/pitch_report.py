"""
SpinoSarc — Premium tek-hasta klinik rapor (pitch deck icin).
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl

from .analyzer import SpinoSarcAnalyzer, Demographics

# Tipografi
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.weight'] = 'normal'

# Brand palette — koyu mavi + lacivert, modern medikal
NAVY     = '#0A2540'
PRIMARY  = '#1E5AA8'
ACCENT   = '#00B8D4'
GOLD     = '#D4A574'
SUCCESS  = '#0FA958'
WARNING  = '#F59E0B'
DANGER   = '#DC2626'
LIGHT_BG = '#F8FAFC'
GRID     = '#E5E7EB'
TEXT_DK  = '#0F172A'
TEXT_MD  = '#475569'
TEXT_LT  = '#94A3B8'

# Kas renkleri (tutarli, doygun)
MUSCLE_COLORS = {
    1: ('Multifidus R',  '#EF4444'),
    2: ('Multifidus L',  '#3B82F6'),
    3: ('Erector R',     '#10B981'),
    4: ('Erector L',     '#8B5CF6'),
    5: ('Psoas R',       '#F59E0B'),
    6: ('Psoas L',       '#06B6D4'),
    7: ('QL R',          '#EC4899'),
    8: ('QL L',          '#64748B'),
}

RISK_STYLE = {
    'Low':      {'color': SUCCESS, 'label': 'LOW RISK',      'symbol': '✓'},
    'Moderate': {'color': WARNING, 'label': 'MODERATE RISK', 'symbol': '!'},
    'High':     {'color': DANGER,  'label': 'HIGH RISK',     'symbol': '!!'},
    'Unknown':  {'color': TEXT_LT, 'label': 'NOT CALCULATED', 'symbol': '?'},
}


# ============================================================
def _draw_header(fig, subtitle='Paraspinal Muscle & Sarcopenia Analysis'):
    """Tum sayfalarda ortak header."""
    # Top accent bar
    fig.add_artist(plt.Rectangle((0, 0.965), 1, 0.005,
                                  transform=fig.transFigure,
                                  facecolor=ACCENT, zorder=2))
    # Logo + title
    fig.text(0.07, 0.98, 'SpinoSarc', fontsize=18, weight='bold',
              color=NAVY, transform=fig.transFigure)
    fig.text(0.07, 0.955, subtitle, fontsize=8, color=TEXT_MD,
              transform=fig.transFigure)
    # Right side timestamp
    fig.text(0.93, 0.98, datetime.now().strftime('%Y-%m-%d'),
              fontsize=8, color=TEXT_LT, ha='right')
    fig.text(0.93, 0.965, 'Report v1.0', fontsize=7, color=TEXT_LT, ha='right')


def _draw_footer(fig, page_num, total_pages):
    """Tum sayfalarda ortak footer."""
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.045, 0.045],
                                color=GRID, linewidth=0.5,
                                transform=fig.transFigure))
    fig.text(0.07, 0.030, 'SpinoSarc · For research use only',
              fontsize=7, color=TEXT_LT, transform=fig.transFigure)
    fig.text(0.50, 0.030, f'Page {page_num} / {total_pages}',
              fontsize=7, color=TEXT_LT, ha='center')
    fig.text(0.93, 0.030, 'spinosarc.ai',
              fontsize=7, color=TEXT_LT, ha='right')


def _section_header(fig, x, y, text):
    """Bolum basligi - small caps, accent altcizgi."""
    fig.text(x, y, text.upper(), fontsize=9, weight='bold',
              color=PRIMARY, transform=fig.transFigure)
    fig.add_artist(plt.Line2D([x, x + 0.06], [y - 0.008, y - 0.008],
                                color=ACCENT, linewidth=2,
                                transform=fig.transFigure))


def _kpi_card(fig, x, y, w, h, label, value, unit='', color=PRIMARY,
               value_size=22):
    """KPI kart - buyuk rakam + kucuk label."""
    # Background
    fig.add_artist(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    transform=fig.transFigure,
                                    facecolor='white',
                                    edgecolor=GRID, linewidth=0.8))
    # Label (top)
    fig.text(x + w/2, y + h - 0.018, label.upper(),
              ha='center', fontsize=7, color=TEXT_MD, weight='bold')
    # Value (big, centered)
    fig.text(x + w/2, y + h/2 - 0.008, value,
              ha='center', va='center',
              fontsize=value_size, color=color, weight='bold')
    # Unit (bottom)
    if unit:
        fig.text(x + w/2, y + 0.012, unit,
                  ha='center', fontsize=8, color=TEXT_LT, style='italic')


# ============================================================
def cover_page(pdf, label, demo, total_pages):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # Background gradient effect (subtle)
    for i, alpha in enumerate(np.linspace(0.08, 0.0, 80)):
        fig.add_artist(plt.Rectangle((0, 1 - 0.4 + i*0.005), 1, 0.005,
                                      transform=fig.transFigure,
                                      facecolor=NAVY, alpha=alpha, zorder=0))

    # Logo area - üst
    fig.add_artist(plt.Rectangle((0.07, 0.88), 0.06, 0.005,
                                  transform=fig.transFigure,
                                  facecolor=ACCENT))
    fig.text(0.07, 0.91, 'SpinoSarc', fontsize=32, weight='bold', color=NAVY)
    fig.text(0.07, 0.875, 'AI-powered paraspinal muscle quantification',
              fontsize=11, color=TEXT_MD, style='italic')

    # Ana baslik - büyük ortada
    fig.text(0.5, 0.68, 'CLINICAL ANALYSIS', ha='center',
              fontsize=11, color=TEXT_LT, weight='bold')
    fig.text(0.5, 0.62, 'Sarcopenia Screening Report', ha='center',
              fontsize=26, color=NAVY, weight='bold')

    # Decorative line
    fig.add_artist(plt.Line2D([0.40, 0.60], [0.59, 0.59],
                                color=ACCENT, linewidth=2,
                                transform=fig.transFigure))

    # Patient ID card
    fig.add_artist(FancyBboxPatch((0.20, 0.40), 0.60, 0.13,
                                    boxstyle="round,pad=0.01,rounding_size=0.012",
                                    transform=fig.transFigure,
                                    facecolor=NAVY, edgecolor='none'))
    fig.text(0.5, 0.495, 'PATIENT ID', ha='center', fontsize=8,
              color=ACCENT, weight='bold')
    fig.text(0.5, 0.465, label, ha='center', fontsize=18,
              color='white', weight='bold')

    if demo:
        demo_parts = []
        if demo.get('age'):       demo_parts.append(f"{demo['age']} years")
        if demo.get('sex'):       demo_parts.append('Male' if demo['sex']=='M' else 'Female')
        if demo.get('height_cm'): demo_parts.append(f"{demo['height_cm']} cm")
        if demo.get('weight_kg'): demo_parts.append(f"{demo['weight_kg']} kg")
        fig.text(0.5, 0.425, '  ·  '.join(demo_parts),
                  ha='center', fontsize=10, color='white', alpha=0.85)

    # Methodology snapshot
    fig.text(0.5, 0.32, 'METHODOLOGY', ha='center', fontsize=8,
              color=TEXT_LT, weight='bold')

    method_lines = [
        'Single L3-L5 axial MR slice',
        '8 paraspinal muscles (multifidus, erector, psoas, QL — bilateral)',
        'Automated segmentation · CSA + fat fraction quantification',
        'PMI-based sarcopenia risk stratification',
    ]
    y = 0.285
    for line in method_lines:
        fig.text(0.5, y, line, ha='center', fontsize=10, color=TEXT_DK)
        y -= 0.022

    # Generated at
    today = datetime.now().strftime('%B %d, %Y · %H:%M')
    fig.text(0.5, 0.16, today, ha='center', fontsize=10, color=TEXT_MD)
    fig.text(0.5, 0.135, 'Powered by SpinoSarc v1.0',
              ha='center', fontsize=8, color=TEXT_LT, style='italic')

    # Footer disclaimer
    fig.add_artist(plt.Rectangle((0.07, 0.07), 0.86, 0.04,
                                  transform=fig.transFigure,
                                  facecolor=LIGHT_BG, edgecolor='none'))
    fig.text(0.5, 0.092, 'CLINICAL DISCLAIMER',
              ha='center', fontsize=7, color=DANGER, weight='bold')
    fig.text(0.5, 0.077,
              'This report supports clinical decision-making but does not constitute a sarcopenia diagnosis.',
              ha='center', fontsize=8, color=TEXT_MD, style='italic')

    _draw_footer(fig, 1, total_pages)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================
def kpi_dashboard_page(pdf, label, result, total_pages):
    """Sayfa 2: KPI dashboard — buyuk rakamlar."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    _draw_header(fig)

    # Sayfa baslik
    fig.text(0.07, 0.92, 'Executive Summary', fontsize=20,
              weight='bold', color=NAVY)
    fig.text(0.07, 0.905, label, fontsize=10, color=TEXT_MD)

    s = result['sarcopenia']
    muscles = result['muscles']
    risk = s['risk_category']
    rs = RISK_STYLE[risk]

    # === Buyuk risk kutusu ===
    fig.add_artist(FancyBboxPatch((0.07, 0.78), 0.86, 0.09,
                                    boxstyle="round,pad=0.01,rounding_size=0.01",
                                    transform=fig.transFigure,
                                    facecolor=rs['color'], alpha=0.10,
                                    edgecolor=rs['color'], linewidth=2))
    fig.text(0.12, 0.835, 'SARCOPENIA RISK ASSESSMENT',
              fontsize=8, color=TEXT_MD, weight='bold')
    fig.text(0.12, 0.795, rs['label'], fontsize=24, weight='bold',
              color=rs['color'])

    # Risk sembol (sag tarafta)
    fig.text(0.86, 0.82, rs['symbol'], fontsize=44, weight='bold',
              color=rs['color'], ha='center', va='center')

    # === KPI Cards (3 sutun) ===
    _section_header(fig, 0.07, 0.74, 'Key Indices')

    pmi  = s.get('pmi_cm2_per_m2')
    tpa  = s['total_psoas_area_cm2']

    mf = [m['fat_fraction'] for m in muscles if 'multifidus' in m['name']]
    mf_mean = np.mean(mf)*100 if mf else 0
    mf_color = SUCCESS if mf_mean<10 else (WARNING if mf_mean<25 else DANGER)

    pmi_color = PRIMARY
    if pmi is not None:
        # En kotu esige gore renk
        if s.get('thresholds'):
            below = sum(1 for v in s['thresholds'].values() if v['below_threshold'])
            pmi_color = SUCCESS if below==0 else (WARNING if below==1 else DANGER)

    card_w, card_h, gap = 0.27, 0.10, 0.025
    x0 = 0.07
    _kpi_card(fig, x0,                 0.61, card_w, card_h,
                'Psoas Muscle Index', f"{pmi:.2f}" if pmi else '—',
                'cm² / m²', color=pmi_color)
    _kpi_card(fig, x0 + card_w + gap,  0.61, card_w, card_h,
                'Total Psoas Area', f"{tpa:.1f}",
                'cm²', color=PRIMARY)
    _kpi_card(fig, x0 + 2*(card_w + gap), 0.61, card_w, card_h,
                'Multifidus FF', f"{mf_mean:.1f}",
                '% (mean R+L)', color=mf_color)

    # === Threshold comparison panel ===
    _section_header(fig, 0.07, 0.555, 'Literature Threshold Comparison')

    y = 0.51
    if s.get('thresholds'):
        for i, (ref, info) in enumerate(s['thresholds'].items()):
            below = info['below_threshold']
            color = DANGER if below else SUCCESS
            status_text = 'BELOW' if below else 'NORMAL'

            # Row background
            row_color = '#FEF2F2' if below else '#F0FDF4'
            fig.add_artist(plt.Rectangle((0.07, y - 0.005), 0.86, 0.038,
                                          transform=fig.transFigure,
                                          facecolor=row_color,
                                          edgecolor='none'))

            # Reference
            ref_short = ref.split('(')[0].strip()
            ref_paren = ref[ref.find('('):].strip('()') if '(' in ref else ''
            fig.text(0.09, y + 0.018, ref_short, fontsize=10,
                      weight='bold', color=TEXT_DK)
            fig.text(0.09, y + 0.005, ref_paren, fontsize=7, color=TEXT_LT)

            # Threshold value
            fig.text(0.55, y + 0.012,
                      f"Threshold: {info['threshold_cm2_per_m2']}",
                      fontsize=9, color=TEXT_MD)
            fig.text(0.70, y + 0.012,
                      f"Patient: {info['patient_value']}",
                      fontsize=9, weight='bold', color=TEXT_DK)

            # Badge
            fig.add_artist(FancyBboxPatch((0.83, y + 0.005), 0.08, 0.022,
                                            boxstyle="round,pad=0.002,rounding_size=0.005",
                                            transform=fig.transFigure,
                                            facecolor=color, edgecolor='none'))
            fig.text(0.87, y + 0.016, status_text, ha='center',
                      fontsize=7, color='white', weight='bold')

            y -= 0.045
    else:
        fig.text(0.5, 0.46, 'Demographics required for PMI threshold comparison',
                  ha='center', fontsize=10, color=TEXT_LT, style='italic')
        y = 0.40

    # === Asymmetry mini bars ===
    _section_header(fig, 0.07, max(y - 0.01, 0.29), 'Bilateral Asymmetry')
    y2 = max(y - 0.05, 0.25)
    asym = result['asymmetry']
    for k, v in asym.items():
        muscle_name = k.replace('_asymmetry_pct', '').capitalize()
        # Color by severity
        a_color = SUCCESS if v < 5 else (WARNING if v < 15 else DANGER)
        # Background bar
        fig.add_artist(plt.Rectangle((0.30, y2 + 0.005), 0.50, 0.012,
                                      transform=fig.transFigure,
                                      facecolor=GRID, edgecolor='none'))
        # Filled bar
        bar_w = min(v / 30.0, 1.0) * 0.50
        fig.add_artist(plt.Rectangle((0.30, y2 + 0.005), bar_w, 0.012,
                                      transform=fig.transFigure,
                                      facecolor=a_color, edgecolor='none'))
        fig.text(0.09, y2 + 0.008, muscle_name, fontsize=10, color=TEXT_DK)
        fig.text(0.82, y2 + 0.008, f"{v}%", fontsize=10,
                  weight='bold', color=a_color)
        y2 -= 0.028

    _draw_footer(fig, 2, total_pages)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================
def overlay_page(pdf, label, result, total_pages):
    """Sayfa 3: Yuksek kaliteli segmentation overlay."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    _draw_header(fig)

    fig.text(0.07, 0.92, 'Segmentation Overlay', fontsize=20,
              weight='bold', color=NAVY)
    fig.text(0.07, 0.905, label, fontsize=10, color=TEXT_MD)

    img = result['image_array']
    seg = result['segmentation_mask']
    img_disp = np.rot90(img, k=-1)
    seg_disp = np.rot90(seg, k=-1)

    # === Iki goruntu yan yana ===
    gs = fig.add_gridspec(1, 2, top=0.86, bottom=0.40,
                            left=0.07, right=0.93, wspace=0.04)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_disp, cmap='gray', interpolation='bilinear')
    ax1.set_title('Original', fontsize=12, weight='bold',
                   color=NAVY, pad=10)
    ax1.axis('off')
    # Subtle border
    for spine in ['top','right','bottom','left']:
        ax1.spines[spine].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(img_disp, cmap='gray', interpolation='bilinear')
    overlay = np.zeros((*seg_disp.shape, 4))
    for label_id, (name, color) in MUSCLE_COLORS.items():
        mask = (seg_disp == label_id)
        if mask.sum() == 0: continue
        rgba = to_rgba(color, alpha=0.60)
        for c in range(4):
            overlay[..., c][mask] = rgba[c]
    ax2.imshow(overlay, interpolation='nearest')
    ax2.set_title('Auto-Segmentation', fontsize=12, weight='bold',
                   color=NAVY, pad=10)
    ax2.axis('off')

    # === Legend (modern, 4x2 grid altta) ===
    legend_y = 0.34
    fig.text(0.5, legend_y + 0.015, 'SEGMENTED MUSCLES',
              ha='center', fontsize=8, color=TEXT_LT,
              weight='bold')

    visible = [(lid, n, c) for lid, (n, c) in MUSCLE_COLORS.items()
                if (seg == lid).sum() > 0]
    n_cols = 4
    n_rows = (len(visible) + n_cols - 1) // n_cols
    cell_w = 0.18
    grid_w = n_cols * cell_w
    grid_x0 = (1 - grid_w) / 2

    for i, (lid, name, color) in enumerate(visible):
        row, col = i // n_cols, i % n_cols
        cx = grid_x0 + col * cell_w
        cy = legend_y - 0.02 - row * 0.025
        # Color dot
        fig.add_artist(plt.Circle((cx + 0.012, cy + 0.008), 0.005,
                                   facecolor=color, edgecolor='none',
                                   transform=fig.transFigure))
        # Name
        fig.text(cx + 0.025, cy + 0.005, name, fontsize=9,
                  color=TEXT_DK, va='center')

    # === Muscle table - alttaki yarim ===
    _section_header(fig, 0.07, 0.24, 'Per-Muscle Metrics')

    headers = ['Muscle', 'CSA (mm²)', 'Fat Fraction', 'Goutallier']
    col_x = [0.10, 0.35, 0.55, 0.78]
    y = 0.21
    for h, x in zip(headers, col_x):
        fig.text(x, y, h.upper(), fontsize=8, weight='bold',
                  color=TEXT_MD)
    y -= 0.005
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=NAVY,
                                linewidth=1, transform=fig.transFigure))
    y -= 0.018

    muscles = result['muscles']
    for m in muscles:
        ff = m['fat_fraction']
        if ff < 0.10:   gout, gcolor = 'G0/1 — Normal',       SUCCESS
        elif ff < 0.25: gout, gcolor = 'G2 — Mild fat',       WARNING
        elif ff < 0.50: gout, gcolor = 'G3 — Moderate fat',   DANGER
        else:           gout, gcolor = 'G4 — Severe fat',     DANGER

        color = next((c for lid, (n, c) in MUSCLE_COLORS.items()
                       if n.lower().replace(' ', '_') == m['name'].lower()), '#000')

        # Renkli daire
        fig.add_artist(plt.Circle((col_x[0] - 0.012, y + 0.004), 0.004,
                                    facecolor=color, edgecolor='none',
                                    transform=fig.transFigure))
        fig.text(col_x[0], y, m['name'].replace('_', ' '), fontsize=9, color=TEXT_DK)
        fig.text(col_x[1], y, f"{m['csa_mm2']:.1f}", fontsize=9, color=TEXT_DK)

        ff_color = SUCCESS if ff < 0.10 else (WARNING if ff < 0.25 else DANGER)
        fig.text(col_x[2], y, f"{ff*100:.1f}%", fontsize=9,
                  weight='bold', color=ff_color)
        fig.text(col_x[3], y, gout, fontsize=8, color=gcolor)
        y -= 0.020

    _draw_footer(fig, 3, total_pages)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================
def methodology_page(pdf, total_pages):
    """Son sayfa: methodology + disclaimers."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    _draw_header(fig)

    fig.text(0.07, 0.92, 'Methodology & References', fontsize=20,
              weight='bold', color=NAVY)

    # Pipeline
    _section_header(fig, 0.07, 0.86, 'Analysis Pipeline')

    pipeline_steps = [
        ('1. Slice selection',   'User-selected axial slice from lumbar MR series (L3 for sarcopenia screening)'),
        ('2. Segmentation',      'MuscleMap contrast-agnostic abdomen model (deep learning, validated)'),
        ('3. Morphometry',       'Per-muscle cross-sectional area (CSA) in mm² and cm²'),
        ('4. Fat quantification','Otsu thresholding within muscle ROI for intramuscular fat fraction'),
        ('5. PMI calculation',   'Total psoas area ÷ height² (Mourtzakis 2008 method)'),
        ('6. Risk stratification', 'Comparison against published PMI thresholds + fat fraction'),
    ]
    y = 0.81
    for step, desc in pipeline_steps:
        fig.text(0.09, y, step, fontsize=10, weight='bold', color=PRIMARY)
        fig.text(0.32, y, desc, fontsize=9, color=TEXT_DK)
        y -= 0.028

    # References
    _section_header(fig, 0.07, 0.61, 'Key References')

    refs = [
        ('Hamaguchi Y et al., 2016',
         'Proposal for new diagnostic criteria for low skeletal muscle mass based on computed tomography imaging in Asian adults. Nutrition 32(11-12):1200-1205.'),
        ('Englesbe MJ et al., 2010',
         'Sarcopenia and mortality after liver transplantation. Journal of the American College of Surgeons 211(2):271-278.'),
        ('Durand F et al., 2014',
         'Prognostic value of muscle atrophy in cirrhosis using psoas muscle thickness on computed tomography. Journal of Hepatology 60(6):1151-1157.'),
        ('Mourtzakis M et al., 2008',
         'A practical and precise approach to quantification of body composition in cancer patients using computed tomography images. Applied Physiology, Nutrition, and Metabolism 33(5):997-1006.'),
        ('Cruz-Jentoft AJ et al., 2019 (EWGSOP2)',
         'Sarcopenia: revised European consensus on definition and diagnosis. Age and Ageing 48(1):16-31.'),
    ]
    y = 0.56
    for short, full in refs:
        fig.text(0.09, y, short, fontsize=9, weight='bold', color=NAVY)
        # Wrap
        words = full.split()
        line, lines = '', []
        for w in words:
            if len(line) + len(w) > 95:
                lines.append(line); line = w
            else:
                line = (line + ' ' + w).strip()
        lines.append(line)
        for i, ln in enumerate(lines):
            fig.text(0.09, y - 0.014 - i*0.013, ln, fontsize=8, color=TEXT_MD)
        y -= 0.014 * (len(lines) + 1) + 0.010

    # Clinical disclaimer
    fig.add_artist(FancyBboxPatch((0.07, 0.08), 0.86, 0.10,
                                    boxstyle="round,pad=0.008,rounding_size=0.008",
                                    transform=fig.transFigure,
                                    facecolor='#FEF3C7', edgecolor=WARNING,
                                    linewidth=1))
    fig.text(0.5, 0.16, 'CLINICAL DISCLAIMER', ha='center', fontsize=9,
              weight='bold', color=DANGER)
    disc_text = [
        'This report supports clinical decision-making but is not a sarcopenia diagnosis.',
        'EWGSOP2 (2019) criteria require muscle strength (grip test) AND physical performance',
        '(gait speed) testing in addition to muscle mass imaging. PMI thresholds are derived from',
        'CT studies; MR-derived values may differ systematically by 5–10%. Use clinical judgment.',
    ]
    y = 0.143
    for t in disc_text:
        fig.text(0.5, y, t, ha='center', fontsize=8, color=TEXT_DK)
        y -= 0.013

    _draw_footer(fig, total_pages, total_pages)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================
def generate_pitch_report(slice_path, label, demographics, output_pdf,
                           musclemap_script):
    an = SpinoSarcAnalyzer(musclemap_script, use_gpu=True)
    print(f"[*] Analyzing {label}...")
    result = an.analyze(slice_path, demographics)

    s = result['sarcopenia']
    print(f"    TPA={s['total_psoas_area_cm2']} cm²")
    print(f"    PMI={s.get('pmi_cm2_per_m2')} cm²/m²")
    print(f"    Risk={s['risk_category']}")

    print(f"\n[*] Generating premium PDF: {output_pdf}")
    total_pages = 4
    demo_dict = result.get('demographics')
    with PdfPages(str(output_pdf)) as pdf:
        cover_page(pdf, label, demo_dict, total_pages)
        kpi_dashboard_page(pdf, label, result, total_pages)
        overlay_page(pdf, label, result, total_pages)
        methodology_page(pdf, total_pages)
    print(f"[+] Done")


if __name__ == '__main__':
    out = '/workspace/SpinoSarc/reports/SpinoSarc_Premium_sub001.pdf'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    generate_pitch_report(
        slice_path='/workspace/SpinoSarc/data/l4l5_slices/sub-001_L4L5.nii.gz',
        label='SUB-001',
        demographics=Demographics(age=65, sex='M', height_cm=172, weight_kg=78),
        output_pdf=out,
        musclemap_script='/workspace/MuscleMap/scripts/mm_segment.py'
    )
