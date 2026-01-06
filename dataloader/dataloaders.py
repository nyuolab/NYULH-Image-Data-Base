import os
import tarfile
import io

import numpy as np
import pandas as pd
import torch
import pydicom
from tqdm import tqdm


import hdf5plugin  
import h5py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TarDicomDataset(Dataset):
    """
    Each item is one .tar or one .tar.gz file that contains a single DICOM file.

    tar_dir: Path to your tar file folder
    image_list_file: .txt file that contains image names in train, val and test split
    label_dir: .csv file contains pairs of image names and label

    """
    def __init__(self, tar_dir, image_list_file, label_dir, transform=None):
        #Collect all image names in the split
        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                image_name = line.split()[0]
                image_names.append(image_name)

        #Collect all tar file paths in the folder
        self.tar_files = sorted(
            os.path.join(tar_dir, f)
            for f in os.listdir(tar_dir)
            if f.endswith(".tar") or f.endswith(".tar.gz")
        )
        if not self.tar_files:
            raise RuntimeError(f"No .tar files found in: {tar_dir}") #Sanity check; return error if no tar file is found

        #Collect label corresponding to each image 
        self.data = pd.read_csv(label_dir) 
        
        #Transform, if needed
        self.transform = transform

        self.labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion',
                        'Pneumonia','Pleural_Thickening','Cardiomegaly','Mass','Nodule','Hernia', 'No Finding']
        
        self.sentence_idx = np.linspace(0,len(self.labels), len(self.labels), False)

    def __len__(self):
        return len(self.tar_files)

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        
        tar_path = self.tar_files[idx]

        #Read the .tar or .tar.gz
        with tarfile.open(tar_path, "r:*") as tar:
            dicoms = [m for m in tar.getmembers()]

            dcm_member = next((m for m in dicoms if m.name.lower().endswith(".dcm")), dicoms[0])
            # print(f"DEBUG: Loaded {member_name} with shape {image.shape}")

            f = tar.extractfile(dcm_member)

            dcm_bytes = f.read()

        # Read DICOM from bytes
        ds = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)

        # Pixel array -> float32 tensor
        arr = ds.pixel_array 
        x = np.asarray(arr)  # numpy array

        #Convert to torch and ensure CHW
        x = torch.from_numpy(x).float()

        x = x.unsqueeze(0)
        
        if self.transform is not None:
            x = self.transform(x)

        #Find the label by matching image name
        label = self.data[self.data['Image Index'] == dcm_member.name.replace('.dcm', '')]['labels'].values[0]

        label_num = self.labels.index(label)


        return x, torch.tensor([label_num]).squeeze()
    

class HDF5DicomDataset(Dataset):
    """
    Each item is one .hdf5 file that contains **MULTIPLE** DICOM file.

    data_dir: Path to your hdf5 file 
    image_list_file: .txt file that contains image names in train, val and test split
    label_dir: .csv file contains pairs of image names and label

    """
    def __init__(self, data_dir, image_list_file, label_dir, transform=None):

        image_names_set = set()
        with open(image_list_file, "r") as f:
            for line in f:
                image_names_set.add(line.strip().split()[0])

        self.h5_path = data_dir

        self.pat_img_pair = []

        with h5py.File(self.h5_path, "r") as f:
            for pid in f.keys():
                # Each 'pid' (Patient ID) is a group
                for image_key in f[pid].keys():
                    # Reconstruct the filename
                    # Example: Patient_00000001 + image_000 -> 00000001_000.png
                    clean_pid = pid.replace('Patient_', '')
                    clean_img = image_key.replace('image', '')
                    array_png_name = f"{clean_pid}_{clean_img}.png"

                    if array_png_name in image_names_set:
                        self.pat_img_pair.append((pid, image_key, array_png_name))

        self.data = pd.read_csv(label_dir) 

        self.transform = transform

        self.labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion',
                        'Pneumonia','Pleural_Thickening','Cardiomegaly','Mass','Nodule','Hernia', 'No Finding']
        
        self.sentence_idx = np.linspace(0,len(self.labels), len(self.labels), False)

    def __len__(self):
        return len(self.pat_img_pair)

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        self._h5 = h5py.File(self.h5_path, "r") #open the file here

        pid, image_key, array_png_name = self.pat_img_pair[idx]

        arr = self._h5[pid][image_key][()] #read the pixel array

        x = np.asarray(arr)

        #Convert to torch and ensure CHW
        x = torch.from_numpy(x).float()
        
        x = x.unsqueeze(0)
        
        if self.transform is not None:
            x = self.transform(x)
 
        array_png_name = pid.replace('Patient_', '') + image_key.replace('image', '') + '.png'
    

        label = self.data[self.data['Image Index'] == array_png_name]['labels'].values[0]


        label_num = self.labels.index(label)


        return x, torch.tensor([label_num]).squeeze()
    

class HDF5DicomSingleDataset(Dataset):
    """
    Each item is one .hdf5 file that contains **SINGLE** DICOM file.

    data_dir: Path to your hdf5 files directory
    image_list_file: .txt file that contains image names in train, val and test split
    label_dir: .csv file contains pairs of image names and label

    """
    def __init__(self, data_dir, image_list_file, label_dir, transform=None):

        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                image_name = line.split()[0]
                image_names.append(image_name)

        self.hdf5_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".hdf5") 
        )

        self.data_name = sorted([f.replace(".hdf5", "") for f in os.listdir(data_dir)])

        if not self.hdf5_files:
            raise RuntimeError(f"No .hdf5 files found in: {data_dir}")


        self.data = pd.read_csv(label_dir) 

        self.transform = transform

        self.labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion',
                        'Pneumonia','Pleural_Thickening','Cardiomegaly','Mass','Nodule','Hernia', 'No Finding']
        
        self.sentence_idx = np.linspace(0,len(self.labels), len(self.labels), False)

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """

        
        with h5py.File(self.hdf5_files[idx], "r") as f: #open the file here
            arr = f[self.data_name[idx]][()]
     

        x = np.asarray(arr)
        x = torch.from_numpy(x).float()

        
        x = x.unsqueeze(0)
        
        if self.transform is not None:
            x = self.transform(x)

        array_png_name = self.data_name[idx] + '.png'
    

        label = self.data[self.data['Image Index'] == array_png_name]['labels'].values[0]

        label_num = self.labels.index(label)


        return x, torch.tensor([label_num]).squeeze()
    

#Use the following code for sanity check - remember to comment them out once you call the classes in training scripts

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

ts = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    normalize,
])

#TODO: Put your own paths here -

ds = TarDicomDataset()

ds = HDF5DicomDataset()

ds = HDF5DicomSingleDataset()


for d in tqdm(ds):
    x,y = d
    print(x.shape)
    print(x,y)
    print("type:", type(x))
    print("dtype:", x.dtype)
    print("shape:", x.shape)
    print("dtype:", y.dtype)
    print("shape:", y.shape)
    print("min / max:", x.min().item(), x.max().item())
    break