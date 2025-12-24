from pathlib import Path
import numpy as np
from PIL import Image
import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, SecondaryCaptureImageStorage, ExplicitVRLittleEndian


def cxr14_png_to_dicom_floatpixeldata(png_path: str, dcm_path: str):
    png_path = Path(png_path)
    dcm_path = Path(dcm_path)

    # Load CXR14 PNG (typically 8-bit grayscale)
    im = Image.open(png_path).convert("L")
    arr8 = np.asarray(im, dtype=np.uint8)

    # Convert to FP16 pixel array (your requested type)
    # Normalize to [0, 1] (common); adjust if you want [-1,1] or keep 0..255
    arr_fp16 = (arr8.astype(np.float16) / np.float16(255.0))

    # DICOM FloatPixelData is float32, so store float32 (standard)
    arr_f32 = arr_fp16.astype(np.float32)

    # --- File meta ---
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # --- IDs / minimal attributes ---
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.PatientName = "Anonymous"
    ds.PatientID = "CXR14"
    ds.Modality = "OT"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # --- Image module ---
    ds.Rows, ds.Columns = arr_f32.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # For FloatPixelData, these integer pixel fields are not the “truth”,
    # but many tools expect them present; keep them consistent-ish.
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0

    # Store float pixels
    ds.FloatPixelData = arr_f32.tobytes()

    ds.save_as(str(dcm_path), write_like_original=False)
    return arr_fp16  # returning FP16 array if you want it


# Example:
home_path = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/data/versions/3'
for folder in os.listdir(home_path):
    # print(folder)
    for img in os.listdir(f'{home_path}/{folder}/images/'):
        # print(img)
        fp16 = cxr14_png_to_dicom_floatpixeldata(f'{home_path}/{folder}/images/{img}/', f"/gpfs/data/oermannlab/public_data/nih-chest-xrays/cxr14_nyu_mimic/{img}.dcm")
