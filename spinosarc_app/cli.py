"""SpinoSarc CLI — komut satirindan analiz, JSON/PNG/PDF rapor uretir."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from .analyzer import SpinoSarcAnalyzer, Demographics


# 8 kas için renkler
MUSCLE_COLORS = {
    1: ('multifidus_R', '#FF4444'),
    2: ('multifidus_L', '#4477FF'),
    3: ('erector_R',    '#44FF44'),
    4: ('erector_L',    '#AA44FF'),
    5: ('psoas_R',      '#FFAA00'),
    6: ('psoas_L',      '#00DDCC'),
    7: ('QL_R',         '#FF44CC'),
    8: ('QL_L',         '#888888'),
}


def make_overlay(result: dict, out_png: Path):
    """Orijinal + overlay yan yana PNG."""
    img  = result['image_array']
    seg  = result['segmentation_mask']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img.T, cmap='gray', origin='lower')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(img.T, cmap='gray', origin='lower')
    overlay = np.zeros((*seg.shape, 4))
    from matplotlib.colors import to_rgba
    for label, (name, color) in MUSCLE_COLORS.items():
        mask = (seg == label)
        if mask.sum() == 0:
            continue
        rgba = to_rgba(color, alpha=0.5)
        for c in range(4):
            overlay[..., c][mask] = rgba[c]
    axes[1].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[1].set_title('Segmentation Overlay')
    axes[1].axis('off')

    legend_items = [
        mpatches.Patch(color=color, label=name)
        for label, (name, color) in MUSCLE_COLORS.items()
        if (seg == label).sum() > 0
    ]
    axes[1].legend(handles=legend_items, loc='lower center',
                    bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()


def make_pdf_report(result: dict, out_pdf: Path, overlay_png: Path):
    """Klinik PDF raporu."""
    from matplotlib.backends.backend_pdf import PdfPages

    demo = result.get('demographics')
    muscles = result['muscles']
    sarc = result['sarcopenia']
    asym = result['asymmetry']

    with PdfPages(str(out_pdf)) as pdf:
        # === Sayfa 1: Başlık + demografi + tablo ===
        fig = plt.figure(figsize=(8.27, 11.69))  # A4

        # Başlık
        fig.text(0.5, 0.96, 'SpinoSarc Analysis Report',
                  ha='center', fontsize=18, weight='bold')
        fig.text(0.5, 0.93, 'Paraspinal Muscle Quantification',
                  ha='center', fontsize=11, style='italic', color='gray')

        # Demografi
        y = 0.88
        fig.text(0.07, y, 'Patient demographics', fontsize=12, weight='bold')
        y -= 0.025
        if demo:
            for label, val in [('Age', demo.get('age')),
                                ('Sex', demo.get('sex')),
                                ('Height', f"{demo.get('height_cm')} cm" if demo.get('height_cm') else None),
                                ('Weight', f"{demo.get('weight_kg')} kg" if demo.get('weight_kg') else None)]:
                if val is not None:
                    fig.text(0.10, y, f"{label}:", fontsize=10)
                    fig.text(0.25, y, str(val), fontsize=10)
                    y -= 0.022
        else:
            fig.text(0.10, y, '(not provided)', fontsize=10, style='italic')
            y -= 0.022

        # Slice info
        y -= 0.01
        fig.text(0.07, y, 'Slice information', fontsize=12, weight='bold')
        y -= 0.025
        fig.text(0.10, y, f"Source: {Path(result['slice_path']).name}", fontsize=9)
        y -= 0.022
        fig.text(0.10, y, f"Pixel spacing: {result['pixel_spacing_mm'][0]:.2f} × {result['pixel_spacing_mm'][1]:.2f} mm", fontsize=9)
        y -= 0.022
        fig.text(0.10, y, f"Image shape: {result['image_shape']}", fontsize=9)
        y -= 0.03

        # Kas tablosu
        fig.text(0.07, y, 'Muscle metrics', fontsize=12, weight='bold')
        y -= 0.03

        headers = ['Muscle', 'CSA (mm²)', 'CSA (cm²)', 'Fat %', 'Voxels']
        col_x   = [0.10, 0.30, 0.45, 0.58, 0.72]
        for h, x in zip(headers, col_x):
            fig.text(x, y, h, fontsize=10, weight='bold')
        y -= 0.02
        fig.text(0.07, y + 0.015, '─' * 90, fontsize=8, color='gray')

        for m in muscles:
            row = [m['name'], f"{m['csa_mm2']:.1f}", f"{m['csa_cm2']:.2f}",
                    f"{m['fat_fraction']*100:.1f}", f"{m['voxel_count']}"]
            for v, x in zip(row, col_x):
                fig.text(x, y, v, fontsize=9)
            y -= 0.022

        # Asimetri
        y -= 0.02
        fig.text(0.07, y, 'Bilateral asymmetry', fontsize=12, weight='bold')
        y -= 0.025
        for k, v in asym.items():
            label = k.replace('_asymmetry_pct', '')
            fig.text(0.10, y, f"{label}:", fontsize=10)
            fig.text(0.30, y, f"{v}% |R - L| / mean", fontsize=10)
            y -= 0.022

        # Sarkopeni
        y -= 0.02
        fig.text(0.07, y, 'Sarcopenia screening (PMI-based)', fontsize=12, weight='bold')
        y -= 0.03

        fig.text(0.10, y, f"Total Psoas Area (TPA):", fontsize=10)
        fig.text(0.45, y, f"{sarc['total_psoas_area_cm2']} cm² ({sarc['total_psoas_area_mm2']:.0f} mm²)", fontsize=10)
        y -= 0.022

        if sarc.get('pmi_cm2_per_m2'):
            fig.text(0.10, y, f"Psoas Muscle Index (PMI):", fontsize=10)
            fig.text(0.45, y, f"{sarc['pmi_cm2_per_m2']} cm²/m²", fontsize=10, weight='bold')
            y -= 0.025

            # Risk
            risk = sarc['risk_category']
            risk_colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Unknown': 'gray'}
            fig.text(0.10, y, f"Risk category:", fontsize=10)
            fig.text(0.45, y, risk.upper(), fontsize=11, weight='bold',
                      color=risk_colors.get(risk, 'black'))
            y -= 0.03

            # Esik tablosu
            fig.text(0.10, y, 'Literature threshold comparison', fontsize=10, weight='bold')
            y -= 0.022
            for ref, info in sarc['thresholds'].items():
                flag = "BELOW" if info['below_threshold'] else "OK"
                color = 'red' if info['below_threshold'] else 'green'
                fig.text(0.12, y, ref, fontsize=8)
                fig.text(0.62, y, f"thr={info['threshold_cm2_per_m2']}", fontsize=8)
                fig.text(0.78, y, flag, fontsize=8, color=color, weight='bold')
                y -= 0.020
        else:
            fig.text(0.10, y, '(Demographics required for PMI calculation)',
                      fontsize=10, style='italic', color='gray')
            y -= 0.025

        # Notlar
        y -= 0.01
        fig.text(0.07, y, 'Notes', fontsize=10, weight='bold')
        y -= 0.022
        for note in sarc.get('notes', []):
            # Wrap long notes
            wrapped = note
            if len(wrapped) > 100:
                wrapped = wrapped[:100] + '...'
            fig.text(0.10, y, f"• {wrapped}", fontsize=7, style='italic',
                      color='gray', wrap=True)
            y -= 0.022

        # Footer
        fig.text(0.5, 0.04, 'SpinoSarc v0.1.0 — for research use only',
                  ha='center', fontsize=8, style='italic', color='gray')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # === Sayfa 2: Overlay görüntü ===
        import matplotlib.image as mpimg
        img = mpimg.imread(str(overlay_png))
        fig2, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.imshow(img)
        ax.set_title('Segmentation overlay', fontsize=14, pad=20)
        ax.axis('off')
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog='spinosarc',
        description='SpinoSarc - Paraspinal muscle quantification from lumbar MR.'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_an = sub.add_parser('analyze', help='Analyze a single MR slice')
    p_an.add_argument('slice', help='Path to NIfTI (.nii / .nii.gz) slice')
    p_an.add_argument('--age', type=int, default=None)
    p_an.add_argument('--sex', choices=['M', 'F'], default=None)
    p_an.add_argument('--height', type=float, default=None,
                       help='Height in cm')
    p_an.add_argument('--weight', type=float, default=None,
                       help='Weight in kg (optional)')
    p_an.add_argument('--output-dir', default='./spinosarc_report',
                       help='Output directory (default: ./spinosarc_report)')
    p_an.add_argument('--musclemap',
                       default='/workspace/MuscleMap/scripts/mm_segment.py',
                       help='Path to MuscleMap mm_segment.py')
    p_an.add_argument('--cpu', action='store_true', help='Force CPU')
    p_an.add_argument('--no-pdf', action='store_true', help='Skip PDF report')

    args = parser.parse_args()

    if args.cmd == 'analyze':
        demo = None
        if any([args.age, args.sex, args.height, args.weight]):
            demo = Demographics(
                age=args.age, sex=args.sex,
                height_cm=args.height, weight_kg=args.weight
            )

        an = SpinoSarcAnalyzer(args.musclemap, use_gpu=not args.cpu)

        print(f"[*] Analyzing {args.slice} ...")
        result = an.analyze(args.slice, demo)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.slice).name.replace('.nii.gz', '').replace('.nii', '')

        # JSON (numpy array'ler hariç)
        json_data = {k: v for k, v in result.items()
                      if k not in ('image_array', 'segmentation_mask')}
        json_path = out_dir / f'{stem}_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"[+] JSON: {json_path}")

        # Overlay PNG
        png_path = out_dir / f'{stem}_overlay.png'
        make_overlay(result, png_path)
        print(f"[+] PNG:  {png_path}")

        # PDF
        if not args.no_pdf:
            pdf_path = out_dir / f'{stem}_report.pdf'
            make_pdf_report(result, pdf_path, png_path)
            print(f"[+] PDF:  {pdf_path}")

        # Konsol özeti
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for m in result['muscles']:
            print(f"  {m['name']:14s}  CSA={m['csa_mm2']:7.1f} mm²  FF={m['fat_fraction']*100:5.1f}%")
        s = result['sarcopenia']
        print(f"\n  TPA: {s['total_psoas_area_cm2']} cm²")
        if s.get('pmi_cm2_per_m2'):
            print(f"  PMI: {s['pmi_cm2_per_m2']} cm²/m²")
            print(f"  Risk: {s['risk_category']}")
        else:
            print(f"  PMI: (demographics needed)")


if __name__ == '__main__':
    main()
