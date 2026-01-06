# NYULH-Image-Data-Base


This code base is dedicated to experiments and benchmark for NYULH Image Data Base.


[convert format](https://github.com/nyuolab/NYULH-Image-Data-Base/tree/main/convert_format) includes code for converting Dicom files to tar, tar.gz, HDF5 and HDF5 with Blosc.

[dataloader](https://github.com/nyuolab/NYULH-Image-Data-Base/tree/main/dataloader) includes Dataset Class loads data in each type of format into a dataset as inputs of DataLoader.


## Method

- Convert CXR14 into FP16 pixel array and save them as DICOM to create a mimic NYULH radiology image dataset

- Convert CXR14 Dicom into
  - tar
  - tar.gz
  - HDF5 
  - HDF5 with Blosc
  - HDF5, single file (all studies in one single HDF5)
  - HDF5 with Blosc, single file (all studies in one single HDF5)

- For DICOM with metadata, write the headers from dicom and save them into a csv for data ingress to databricks

- Write a data loader for each format of the data, train Densenet with
  - batch size = 256 and
  - 8 H100 GPUs
    
- Experiment with combinations of different parameters 
  - worker = 1, worker = 20
  - GPU = 1, GPU = 8

- Regression test by HPC team



## Metrics
- Convert time from DICOM to each file format
- Data loader I/O time
- Avg. model train time per epoch


## Benchmarks

### Convert time

| | tar | tar.gz | HDF5 | HDF5 (Blosc) | HDF5, single file | HDF5, single file (Blosc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **File Count** | 112,120 | 112,120 | 112,120 | 112,120 | 1 | 1 |
| **Disk Usage** | $445\text{ G}$ | **$90\text{ G}$** | $445\text{ G}$ | $135\text{ G}$ | $439\text{ G}$ | $135\text{ G}$ |
| **% of Original Size** | $100\\%$ | $20\\%$ | $100\\%$ | $30\\%$ | $98\\%$ | $30\\%$ |
| **Convert Time** | $4380\text{s}$ | $1.5\text{ Days}$ | **$3200\text{s}$** | **$4215\text{s}$** | $4804\text{s}$ | $8891\text{s}$ |


### Data Loading and training
####  1 GPU 
| Configuration | Metric | tar | tar.gz | HDF5 | HDF5 (Blosc) | HDF5, single file | HDF5, single file (Blosc) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Workers 1 GPU** | Loading Time | $10.3\text{ s}$ | $12.6\text{ s}$ | $8.55\text{ s}$ | **$7.86\text{ s}$** | $23.5\text{ s}$ | $17.6\text{ s}$ |
| **1 Workers 1 GPU** | Avg. Training Time /epoch | $2.4\text{ Hrs}$ | $3\text{ Hrs}$ | $2.13\text{ Hrs}$ | **$1.88\text{ Hrs}$** | $4\text{ Hrs}$ | $4\text{ Hrs}$ |
| **20 Workers 1 GPU** | Loading Time | $11.21\text{ s}$ | $15.87\text{ s}$ | $10.08\text{ s}$ | **$7.40\text{ s}$** | $12.03\text{ s}$ | $14.92\text{ s}$ |
| **20 Workers 1 GPU** | Avg. Training Time /epoch | $506.58\text{ s}$ | $728.14\text{ s}$ | $397.36\text{ s}$ | **$376.37\text{ s}$** | $538.76\text{ s}$ | $655.06\text{ s}$ |


####  8 GPUs
| Configuration | Metric | tar | tar.gz | HDF5 | HDF5 (Blosc) | HDF5, single file | HDF5, single file (Blosc) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Workers 8 GPUs** | Loading Time | $11.1\text{ s}$ | $12.6\text{ s}$ | $13.37\text{ s}$ | $8.23\text{ s}$ | $15.8\text{ s}$ | **$5.9\text{ s}$** |
| **1 Workers 8 GPUs** | Avg. Training Time /epoch | $1,213.7\text{ s}$ | $1,390\text{ s}$ | $433.54\text{ s}$ | $902.73\text{ s}$ | $1,685.8\text{ s}$ | $631.2\text{ s}$ |
| **20 Workers 8 GPUs** | Loading Time | $12.8\text{ s}$ | $15.1\text{ s}$ | $11.69\text{ s}$ | **$6.97\text{ s}$** | $9.9\text{ s}$ | $113.8\text{ s}$ |
| **20 Workers 8 GPUs** | Avg. Training Time /epoch | $86\text{ s}$ | $95.8\text{ s}$ | $78.79\text{ s}$ | **$48.54\text{ s}$** | $52\text{ s}$ | $620.6\text{ s}$ |


#### 1 GPU vs. 8 GPUs, Workers = 20 
| Format | 1 GPU Time | 8 GPU Time | Speedup Factor | Comment |
| :--- | :---: | :---: | :---: | :--- |
| **HDF5, single file** | $538.8\text{s}$ | $52.0\text{s}$ | $10.3\text{x}$ | |
| **HDF5 (Blosc)** | **$376.37\text{s}$** | **$48.54\text{s}$** | $7.75\text{x}$ | |
| **tar.gz** | $728.1\text{s}$ | $95.8\text{s}$ | $7.6\text{x}$ | |
| **tar** | $506.6\text{s}$ | $86.0\text{s}$ | $5.9\text{x}$ | |
| **HDF5** | $397.36\text{s}$ | $78.79\text{s}$ | $5.04\text{x}$ | |
| **HDF5, single file (Blosc)** | $655.1\text{s}$ | $620.6\text{s}$ | $1.05\text{x}$ | GIL challenge |


#### 1 Worker vs. 20 Workers, GPU = 8
| Format | 1 worker | 20 workers | Speedup Factor | Comment |
| :--- | :---: | :---: | :---: | :--- |
| **HDF5, single file** | $1685.8\text{ s}$ | $52.0\text{ s}$ | $32.4\text{ x}$ | |
| **HDF5 (Blosc)** | $902.73\text{ s}$ | **$48.54\text{ s}$** | $18.6\text{ x}$ | Training time is longer than non-compressed HDF5 because decompression takes time |
| **tar.gz** | $1390.0\text{ s}$ | $95.8\text{ s}$ | $14.5\text{ x}$ | |
| **tar** | $1213.7\text{ s}$ | $86.0\text{ s}$ | $14.1\text{ x}$ | |
| **HDF5** | **$433.54\text{ s}$** | $78.79\text{ s}$ | $5.5\text{ x}$ | |
| **HDF5, single file (Blosc)** | $631.2\text{ s}$ | $620.6\text{ s}$ | $1.01\text{ x}$ | Lock Contention (GIL bottleneck) |

### Takeaway
| Format | Pros | Cons |
| :--- | :--- | :--- |
| **tar.gz** | “Linear” performance<br>Saves the most space | Long conversion time |
| **HDF5** | Fast data loading and training<br>Simple | Doesn’t save much space |
| **HDF5(Blosc)** | **Fastest data loading and training** <br> **Saves a lot of space** | Can be slower when computational resources are limited |
| **HDF5, single file (Blosc)** | Saves space<br>**Takes advantage of HDF5 file structure**<br>Fairly shorter conversion time | Requires users to write a custom `DataLoader` to avoid GIL bottleneck |

 **GPU Utilization**  Heavily dependent on optimal dataloading.Users *need to pay attention*, particularly when requesting multiple GPUs. 


