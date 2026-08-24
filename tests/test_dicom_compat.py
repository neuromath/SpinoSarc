import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import MRImageStorage, generate_uid

from spinosarc_app import dicom_loader


def _base_dataset(rows=4, columns=5):
    ds = Dataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = 'MR'
    ds.SeriesNumber = 7
    ds.SeriesDescription = 'T2 AX'
    ds.ProtocolName = 'T2 AX'
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.7, 0.8]
    ds.SliceThickness = 4.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.InstanceNumber = 1
    ds.EchoTime = 90
    ds.RepetitionTime = 4000
    return ds


def _write_non_part10(path, ds):
    pydicom.dcmwrite(
        str(path), ds, implicit_vr=True, little_endian=True,
        enforce_file_format=False,
    )


class DicomCompatibilityTests(unittest.TestCase):
    def test_non_part10_pacs_file_is_rewritten_for_dcm2niix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            target = root / 'target.dcm'
            ds = _base_dataset()
            ds.PixelData = np.arange(20, dtype=np.uint16).tobytes()
            _write_non_part10(source, ds)

            self.assertFalse(dicom_loader._is_part10_dicom(source))
            dicom_loader._write_part10_copy(source, target)
            self.assertTrue(dicom_loader._is_part10_dicom(target))
            staged = pydicom.dcmread(str(target))
            self.assertEqual(staged.pixel_array.shape, (4, 5))

    def test_enhanced_mr_frames_count_as_slices_and_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'enhanced'
            ds = _base_dataset(rows=3, columns=4)
            ds.NumberOfFrames = 3
            ds.PixelData = np.arange(36, dtype=np.uint16).reshape(3, 3, 4).tobytes()

            shared = Dataset()
            orientation = Dataset()
            orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            shared.PlaneOrientationSequence = Sequence([orientation])
            measures = Dataset()
            measures.PixelSpacing = [0.7, 0.8]
            measures.SliceThickness = 4.0
            shared.PixelMeasuresSequence = Sequence([measures])
            ds.SharedFunctionalGroupsSequence = Sequence([shared])

            per_frame = []
            for index in range(3):
                group = Dataset()
                position = Dataset()
                position.ImagePositionPatient = [0, 0, index * 4.0]
                group.PlanePositionSequence = Sequence([position])
                content = Dataset()
                content.InStackPositionNumber = index + 1
                group.FrameContentSequence = Sequence([content])
                per_frame.append(group)
            ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
            _write_non_part10(source, ds)

            rows = dicom_loader.scan_dicom_folder(root)
            self.assertEqual(rows[0]['n_slices'], 3)
            frames = dicom_loader._read_dicom_frames(rows[0])
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[2]['source_frame_index'], 2)
            self.assertEqual(frames[2]['z_world_mm'], 8.0)

    def test_decoded_frame_writes_single_slice_nifti(self):
        slice_data = {
            'pixel_array': np.arange(20, dtype=np.float32).reshape(4, 5),
            'pixel_spacing': (0.7, 0.8),
            'image_position': (0.0, 0.0, 12.0),
            'image_orientation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'slice_thickness': 4.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'slice.nii.gz'
            dicom_loader.write_axial_slice_nifti(slice_data, output)
            image = nib.load(str(output))
            self.assertEqual(image.shape, (4, 5, 1))
            self.assertAlmostEqual(image.header.get_zooms()[0], 0.7, places=5)
            self.assertAlmostEqual(image.header.get_zooms()[1], 0.8, places=5)

    def test_orientation_fallback_keeps_unlabelled_axial_series(self):
        rows = [{
            'orientation': 'axial', 'n_slices': 18, 'matrix': '320x320',
            'is_t2': False, 'is_stir': False, 'has_contrast': False,
        }]
        self.assertEqual(
            dicom_loader.pick_orientation_candidates(rows, 'axial'), rows)


if __name__ == '__main__':
    unittest.main()
