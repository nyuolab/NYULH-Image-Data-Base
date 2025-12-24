import os
import pydicom
import h5py, hdf5plugin
import numpy as np
import time  # <--- Added this
from tqdm import tqdm

# CONFIGURATION
dicom_dir = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/cxr14_nyu_mimic'
h5_filename = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/hdf5/cxr14_nyu_mimic_blosc.hdf5'

def save_hierarchical_pixels_timed(dicom_dir, h5_filename):
    files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    total_files = len(files)
    
    with h5py.File(h5_filename, 'a') as h5f:
        print(f"Starting conversion of {total_files} images...")
        
        # 1. Start the Timer
        start_time = time.time()
        
        for f in tqdm(files):
            file_path = os.path.join(dicom_dir, f)
            
            try:
                ds = pydicom.dcmread(file_path)
                # Logic to create Patient Group -> Image Dataset
                pat_id = str(f.split('_')[0])
                print(pat_id)
                img_suffix = f.split('_')[-1].replace('.png.dcm', '')
                img_name = f"image_{img_suffix}"
                
                group_name = f"Patient_{pat_id}"
                
                # Get or create group
                if group_name in h5f:
                    grp = h5f[group_name]
                else:
                    grp = h5f.create_group(group_name)
                
                # Process Pixels
                pixels = ds.pixel_array
                
                grp.create_dataset(img_name
                                   , data=pixels
                                   , compression=hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE))
                            
                
                
            except Exception as e:
                print(f"Error on {f}: {e}")

        # 2. Stop the Timer
        end_time = time.time()

    # 3. Calculate Results
    elapsed_time = end_time - start_time
    avg_per_img = elapsed_time / total_files if total_files > 0 else 0

    print("-" * 30)
    print(f"Conversion Complete.")
    print(f"Total Time:      {elapsed_time:.2f} seconds")
    print(f"Average per Img: {avg_per_img:.4f} seconds")
    print("-" * 30)


save_hierarchical_pixels_timed(dicom_dir, h5_filename)




