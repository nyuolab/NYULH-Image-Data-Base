import os
import pydicom
import h5py, hdf5plugin
import numpy as np
import time  
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training for NIH CXR DenseNet")

    parser.add_argument("--compression"
                        , type=str
                        , default='no_compress'
                        , help="Compression? choose between no_compress and blosc")
    
    parser.add_argument("--index"
                        , type=str
                        , default='image'
                        , help="how do you want to index your hdf5? patient or image? \
                                by patient the file structure will be: \
                                    --patient \
                                        |__img 1 \
                                        |__img 2 \
                                        |__img N \
                                by image, each hdf5 file is a single radiology image.")
 

    return parser.parse_args()

def save_hierarchical_pixels_timed():
    args = parse_args()


    dicom_dir = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/cxr14_nyu_mimic'

    os.makedirs(f'/gpfs/data/oermannlab/public_data/nih-chest-xrays/hdf5_single/{args.index}/{args.compression}/'
                            , exist_ok=True)
    h5_dir =  f'/gpfs/data/oermannlab/public_data/nih-chest-xrays/hdf5_single/{args.index}/{args.compression}/'
    
    files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    total_files = len(files)
    
    
    print(f"Starting conversion of {total_files} images, indexed by {args.index}, with compression method {args.compression}...")
    
    if args.index == 'patient':
        #create a hashmap for patient and images pairing
        pat_img_map = {}

        for f in tqdm(files):
            pat_id = str(f.split('_')[0])

            if pat_id not in pat_img_map:
                pat_img_map[pat_id] = []

            pat_img_map[pat_id].append(f)

        for pat_id in pat_img_map:
            hdf5_file_path = os.path.join(h5_dir, pat_id) + '.hdf5'

            with h5py.File(hdf5_file_path, 'w') as h5f:
                grp = h5f.create_group(pat_id)

                for dicom in pat_img_map[pat_id]:

                    img_suffix = dicom.split('_')[-1].replace('.png.dcm', '')
                    img_name = f"image_{img_suffix}"

                    dicom_file_path = os.path.join(dicom_dir, dicom)
                    
                    try:
                        ds = pydicom.dcmread(dicom_file_path)
                        # Logic to create Patient Group -> Image Dataset
 
                        # Process Pixels
                        pixels = ds.pixel_array
                        if pat_id in h5f:
                            grp = h5f[pat_id]
                        else:
                            grp = h5f.create_group(pat_id)

                    except Exception as e:
                        print(f"Error on {f}: {e}")

                    if args.compression == 'no_compress':
                        grp.create_dataset(img_name
                                            , data=pixels
                                            , compression=None)
                            
                    else:
                        grp.create_dataset(img_name
                                            , data=pixels
                                            , compression=hdf5plugin.Blosc(cname='blosclz', clevel=9
                                            , shuffle=hdf5plugin.Blosc.SHUFFLE))
                
    else:
        for f in tqdm(files):
            pat_id = str(f.split('_')[0])

            img_suffix = f.replace('.png.dcm', '')
            img_name = img_suffix
        
            hdf5_file_path = os.path.join(h5_dir, img_name) + '.hdf5'

            with h5py.File(hdf5_file_path, 'w') as h5f:
                dicom_file_path = os.path.join(dicom_dir, f)
                ds = pydicom.dcmread(dicom_file_path)
               
                pixels = ds.pixel_array

                if args.compression == 'no_compress':
                    h5f.create_dataset(img_name
                                        , data=pixels
                                        , compression=None)
                                
                else:
                    h5f.create_dataset(img_name
                                        , data=pixels
                                        , compression=hdf5plugin.Blosc(cname='blosclz', clevel=9
                                        , shuffle=hdf5plugin.Blosc.SHUFFLE))

            
        
def main():
    start_time = time.time()
    save_hierarchical_pixels_timed()
     # 2. Stop the Timer
    end_time = time.time()

    # 3. Calculate Results
    elapsed_time = end_time - start_time
    avg_per_img = elapsed_time / 112120  #--> change the number of files if you are using a different dataset

    print("-" * 30)
    print(f"Conversion Complete.")
    print(f"Total Time:      {elapsed_time:.2f} seconds")
    print(f"Average per Img: {avg_per_img:.4f} seconds")
    print("-" * 30)


if __name__ == "__main__":
    main()



