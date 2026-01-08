import ssl
import urllib.request

# Global switch to disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import os, time, copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler

from model import DenseNet121_SingleLab
import torch.distributed as dist
from pytorchtools import EarlyStopping

import hdf5plugin
import h5py
from dataloaders import TarDicomDataset, HDF5DicomDataset, HDF5DicomSingleDataset


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training for NIH CXR DenseNet")

    parser.add_argument("--num_workers", type=int, default=20,
                        help="DataLoader workers per GPU")
    parser.add_argument("--data_dir", type=str,
                        default="/gpfs/data/oermannlab/public_data/nih-chest-xrays/tar",
                        help="Directory with tar DICOMs")
    parser.add_argument("--data_format", type=str,
                        default='tar',
                        help="data format you want to make benchmark with")
    parser.add_argument("--small_set", type=bool,
                        default=False,
                        help="debug mode")

    return parser.parse_args()


def train_model(model, criterion, optimizer, scheduler, dls, dataset_sizes, early_stopping, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_train_loss = []
    avg_val_loss = []
    train_acc = []
    val_acc = []

     # ---- timing containers ----
    data_times = []
    compute_times = []
    
    for epochID in range(num_epochs):
        if is_main_process():
            print('Epoch {}/{}'.format(epochID, num_epochs - 1))
            print('-' * 10)
        start_time = time.time()
        tsTime = time.strftime("%H%M%S")
        tsDate = time.strftime("%d%m%Y")
        tsStart = tsDate + '-' + tsTime
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            end = time.time()


            for batchID, (X, Y) in enumerate(dls[phase]):

                # -------- measure data loading time ----------
                data_time = time.time() - end
        
                X = X.cuda()
                Y = Y.cuda()

                optimizer.zero_grad()
                
                # -------- compute (forward/backward) ----------
                t_compute_start = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(X)
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, Y.long())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                torch.cuda.synchronize()

                compute_time = time.time() - t_compute_start

                if phase == 'train' and batchID == 0:
                    data_times.append(data_time) #one batch 0 per epoch, len(data_times) == 50

                running_loss += loss.item() * X.size(0)
                running_corrects += torch.sum(preds == Y.data)

                # reset end for next iter
                end = time.time()

                if is_main_process() and batchID % 50 == 0:
                    print(
                        f"[{phase}] batch {batchID} | "
                        f"data={data_time:.3f}s | "
                        f"compute={compute_time:.3f}s"
                    )
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val':
                scheduler.step(epoch_loss)  
            if is_main_process():
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        time_elapsed = time.time() - start_time
        
        compute_times.append(time_elapsed)

        if is_main_process():
            print(f'Time Taken for Epoch{epochID}: %.2fs' % (time.time() - start_time))
            # print('Avg. Time Taken for loading data: %.2fs' % (sum(data_times)/num_epochs))
            # print('Avg. Time Taken for training model: %.2fs' % (sum(compute_times)/num_epochs))
            
        
        # Check early stopping
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop and is_main_process():
            print("Early stopping")
            break
    if is_main_process():
        print('Best val Acc: {:4f}'.format(best_acc))
        print(len(data_times), len(compute_times))
        print('Avg. Time Taken for loading data: %.2fs' % (sum(data_times)/num_epochs))
        print('Avg. Time Taken for training model: %.2fs' % (sum(compute_times)/num_epochs))

    model.load_state_dict(best_model_wts)
    return model, avg_train_loss, avg_val_loss, train_acc, val_acc


def main():
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


    small_set = args.small_set

    # Define variables
    num_classes = 15
    class_names  = ['Atelectasis'
                    ,'Consolidation'
                    ,'Infiltration'
                    ,'Pneumothorax'
                    ,'Edema'
                    ,'Emphysema'
                    ,'Fibrosis'
                    ,'Effusion'
                    ,'Pneumonia'
                    ,'Pleural_Thickening'
                    ,'Cardiomegaly'
                    ,'Mass'
                    ,'Nodule'
                    ,'Hernia'
                    ,'No Finding']

    data_dir = args.data_dir

    batch_size = 256

    # Create a DenseNet Model
    if is_main_process():
        print('Loading Model')


    device =  torch.device("cuda", local_rank)
    model = DenseNet121_SingleLab(num_classes).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )


    ts = transforms.Compose([                 
        transforms.Resize((224, 224)),                  # (1, 224, 224)
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # (3, 224, 224)
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225],
                            ),
    ])

    data_list_path = '/gpfs/data/oermannlab/public_data/nih-chest-xrays/data/versions/3/'

    # Try small set if set to True, otherwise run on whole dataset
    if small_set == True:
        train_list = data_list_path + 'train_images_single_small.txt'
        val_list = data_list_path + 'val_images_single_small.txt'
    else:
        train_list =  data_list_path + 'train_val_list.txt'
        val_list =  data_list_path + 'test_list.txt'

    # Create Datasets and Dataloaders
    if is_main_process():
        print("Loading Dataset + DataLoader")

    if args.data_format == 'tar' or args.data_format == 'tar.gz':
        Dataset = TarDicomDataset

    elif args.data_format == 'hdf5':
        Dataset = HDF5DicomDataset

    else:
        Dataset = HDF5DicomSingleDataset

    time_0 = time.perf_counter()
    train_ds =  Dataset(data_dir,
                                image_list_file=train_list,
                                label_dir = f'{data_list_path}Data_Entry_2017_single_label.csv',
                                transform=ts, 
                                )

    train_sampler = DistributedSampler(train_ds, shuffle=True)


    train_loader = DataLoader(dataset=train_ds
                            , batch_size=batch_size
                            , num_workers = args.num_workers
                            , shuffle=False
                            , sampler=train_sampler
                            , persistent_workers=True)



    val_ds =  Dataset(data_dir,
                            image_list_file=val_list,
                            label_dir = f'{data_list_path}Data_Entry_2017_single_label.csv',
                            transform=ts,
                                )

    val_sampler = DistributedSampler(val_ds, shuffle=True)


    val_loader = DataLoader(dataset=val_ds
                            , batch_size=batch_size
                            , num_workers = args.num_workers
                            , sampler=val_sampler
                            , shuffle=False
                            , persistent_workers=True)



    time_1 = time.perf_counter()

    if is_main_process():
        print('Data Loader takes' + str(time_1 - time_0) + f's for num_workers = {args.num_workers}') 

    dls = {'train': train_loader, 'val': val_loader}

    dataset_sizes = {'train': len(train_ds), 'val': len(val_ds)}

    # Create optimizer, LR scheduler, and loss function
    opt = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 5, mode = 'min')
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create early stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)



    start_training_time = time.perf_counter()
    model_ft, avg_train_loss, avg_val_loss, train_acc, val_acc = \
                    train_model(model, loss_fn, opt, scheduler, dls, dataset_sizes, early_stopping, num_epochs=20)
    end_training_time = time.perf_counter()

    if is_main_process():
        print("Done Training Model")      
        print("Training Time:", str(end_training_time-start_training_time))  
        print("Train Acc:", train_acc)
        print("Train Loss:",avg_train_loss)
        print("Val Acc:",val_acc)
        print("Val Loss:",avg_val_loss)

    tsTime = time.strftime("%H-%M-%S")
    tsDate = time.strftime("%m-%d-%Y")
    tsEnd = tsDate + '_' + tsTime
    outputdir = '/gpfs/data/oermannlab/users/xh852/image_db_experiment' + f'/{tsEnd}/'

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    torch.save(model.state_dict(), outputdir + f'model_dict_single_{tsEnd}.pt')
    np.save(outputdir +  'train_loss_epoch_single_' + tsEnd, avg_train_loss)
    np.save(outputdir +  'val_loss_epoch_single_' + tsEnd, val_acc)
    np.save(outputdir +  'train_acc_epoch_single_' + tsEnd, train_acc)
    np.save(outputdir +  'val_acc_epoch_single_' + tsEnd, val_acc)
    print("Saved!")

if __name__ == "__main__":
    main()

