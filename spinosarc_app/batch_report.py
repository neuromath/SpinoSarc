"""SpinoSarc — Coklu hasta tek-sayfa rapor uretici."""
import pandas as pd
from pathlib import Path
from .analyzer import Demographics
from .single_page_report import generate_single_page


def main(csv_path, slices_dir, output_dir, mm_script):
    df = pd.read_csv(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] {len(df)} hasta icin rapor uretiliyor\n")

    summary = []
    for _, row in df.iterrows():
        sub_id = row['sub_id']
        slice_path = Path(slices_dir) / f'{sub_id}_L4L5.nii.gz'
        if not slice_path.exists():
            print(f"  SKIP {sub_id}: slice not found")
            continue

        demo = Demographics(
            age=int(row['age']) if pd.notna(row.get('age')) else None,
            sex=row['sex'] if pd.notna(row.get('sex')) else None,
            height_cm=float(row['height_cm']) if pd.notna(row.get('height_cm')) else None,
            weight_kg=float(row['weight_kg']) if pd.notna(row.get('weight_kg')) else None,
        )

        output_pdf = out_dir / f'SpinoSarc_OnePage_{sub_id}.pdf'

        try:
            generate_single_page(
                slice_path=str(slice_path),
                label=row.get('label', sub_id),
                demo=demo,
                output_pdf=str(output_pdf),
                mm_script=mm_script,
            )
            summary.append((sub_id, 'OK', str(output_pdf)))
        except Exception as e:
            print(f"  HATA {sub_id}: {e}")
            summary.append((sub_id, f'FAIL: {e}', ''))

    print("\n" + "="*60)
    print("OZET")
    print("="*60)
    for sub_id, status, pdf_path in summary:
        marker = '✓' if status == 'OK' else '✗'
        print(f"  {marker} {sub_id:10s}  {status:20s}  {Path(pdf_path).name if pdf_path else ''}")


if __name__ == '__main__':
    main(
        csv_path='/workspace/SpinoSarc/reports/demographics.csv',
        slices_dir='/workspace/SpinoSarc/data/l4l5_slices',
        output_dir='/workspace/SpinoSarc/reports/onepage_batch',
        mm_script='/workspace/MuscleMap/scripts/mm_segment.py',
    )
