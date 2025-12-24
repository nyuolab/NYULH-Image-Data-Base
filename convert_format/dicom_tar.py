import tarfile
import os
import time
from tqdm import tqdm

def create_dicom_tar(source_dir, output_dir):

    for root, dirs, files in os.walk(source_dir):
        print(root, len(files))
        
        total_tar, total_tar_compress = 0,0
        
        for file in tqdm(files):
            out_file_name= file.replace('.png.dcm','')
            with tarfile.open(f"{output_dir}/{out_file_name}.tar", "w:") as tar:
                file_path = os.path.join(root, file)
                if file.endswith('.dcm') or os.path.isfile(file_path):
                    # Add the file to the tar archive
                    t0 = time.perf_counter()
                    tar.add(file_path, arcname=os.path.basename(file_path))
                    t_tar = time.perf_counter()
            

                    delta_tar = t_tar - t0
                 

                    total_tar += delta_tar
        break
    print(total_tar, total_tar_compress)


source_directory = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/cxr14_nyu_mimic'

output_tar_file = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/tar'

if os.path.isdir(source_directory):
    create_dicom_tar(source_directory, output_tar_file)
else:
    print(f"Error: Directory not found at {source_directory}")

