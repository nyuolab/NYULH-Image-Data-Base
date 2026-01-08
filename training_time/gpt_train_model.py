import ssl
import urllib.request

# Global switch to disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import DenseNet121_SingleLab
from pytorchtools import EarlyStopping
from dataloaders import TarDicomDataset


# -----------------------------
# Distributed setup utilities
# -----------------------------
def setup_distributed():
    """
    Initialize torch.distributed if we are in a distributed context.
    Returns:
        device: torch.device
        rank: int
        world_size: int
        local_rank: int
        is_distributed: bool
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = True

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Fallback: single-GPU training
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size, local_rank, is_distributed


def is_main_process(rank: int) -> bool:
    return rank == 0


# -----------------------------
# Training function
# -----------------------------
def train_model(
    model,
    dataloaders,
    dataset_sizes,
    train_sampler,
    val_sampler,
    criterion,
    optimizer,
    scheduler,
    early_stopping,
    device,
    rank,
    world_size,
    num_epochs=25,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_train_loss = []
    avg_val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        # Tell DistributedSamplers which epoch we are in
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 20)

        start_time = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                sampler = train_sampler
            else:
                model.eval()
                sampler = val_sampler

            running_loss = 0.0
            running_corrects = 0.0

            dataloader = dataloaders[phase]

            for batch_idx, (X, Y) in enumerate(dataloader):
                # Move data to GPU
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(X)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, Y.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Accumulate stats (per rank)
                batch_size_curr = X.size(0)
                running_loss += loss.item() * batch_size_curr
                running_corrects += (preds == Y.data).sum().item()

            # ---------------------------------------------
            # Reduce metrics across all ranks (if DDP)
            # ---------------------------------------------
            loss_tensor = torch.tensor(running_loss, device=device)
            corrects_tensor = torch.tensor(running_corrects, device=device)

            if dist.is_initialized():
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(corrects_tensor, op=dist.ReduceOp.SUM)

            epoch_loss = loss_tensor.item() / dataset_sizes[phase]
            epoch_acc = corrects_tensor.item() / dataset_sizes[phase]

            # Only main rank prints/logs/schedules
            if is_main_process(rank):
                if phase == "val":
                    scheduler.step(epoch_loss)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "train":
                    avg_train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                else:
                    avg_val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)

                # Track best model on validation
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - start_time
        if is_main_process(rank):
            print(f"Time taken for epoch: {time_elapsed:.2f}s")
            print()

            # Early stopping (using val loss from this epoch)
            # Use last val loss we logged
            if len(avg_val_loss) > 0:
                current_val_loss = avg_val_loss[-1]
                early_stopping(current_val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

    # Broadcast best weights from rank 0 to all ranks (so state is consistent)
    if dist.is_initialized():
        # Rank 0 already has best_model_wts; others just get state_dict from DDP model
        # Simplest: load best on rank 0, then broadcast model params via DDP sync
        if is_main_process(rank):
            model.load_state_dict(best_model_wts)
        # Ensure all ranks have the same final params
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    else:
        # Single process
        model.load_state_dict(best_model_wts)

    if is_main_process(rank):
        print(f"Best val Acc: {best_acc:.4f}")

    return model, avg_train_loss, avg_val_loss, train_acc, val_acc


# -----------------------------
# Main script
# -----------------------------
def main():
    device, rank, world_size, local_rank, is_distributed = setup_distributed()

    small_set = False

    # Define variables
    num_classes = 15
    class_names  = [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Mass",
        "Nodule",
        "Hernia",
        "No Finding",
    ]

    data_dir = "/gpfs/data/oermannlab/public_data/nih-chest-xrays/tar"

    batch_size = 256
    num_workers = 4   # reasonable starting point for your node
    max_epoch = 50

    if is_main_process(rank):
        print("Loading model...")

    model = DenseNet121_SingleLab(num_classes)
    model = model.to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Transforms: assume TarDicomDataset returns a tensor of shape [1, H, W]
    ts = transforms.Compose([
        transforms.Resize((224, 224)),                       # [1, 224, 224]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)        # [3, 224, 224] if single channel
                           if x.shape[0] == 1 else x),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    data_list_path = "/gpfs/data/oermannlab/public_data/nih-chest-xrays/data/versions/3/"

    # Small subset or full dataset
    if small_set:
        train_list = data_list_path + "train_images_single_small.txt"
        val_list = data_list_path + "val_images_single_small.txt"
    else:
        train_list = data_list_path + "train_val_list.txt"
        val_list = data_list_path + "test_list.txt"

    # Datasets
    if is_main_process(rank):
        print("Loading Dataset + DataLoader...")

    t0 = time.perf_counter()

    train_ds = TarDicomDataset(
        tar_dir=data_dir,
        image_list_file=train_list,
        label_dir=f"{data_list_path}Data_Entry_2017_single_label.csv",
        transform=ts,
    )

    val_ds = TarDicomDataset(
        tar_dir=data_dir,
        image_list_file=val_list,
        label_dir=f"{data_list_path}Data_Entry_2017_single_label.csv",
        transform=ts,
    )

    # Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        shuffle=False,                # sampler controls order
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    t1 = time.perf_counter()
    if is_main_process(rank):
        print(f"Data Loader took {t1 - t0:.2f}s for num_workers = {num_workers}")

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}

    # Optimizer, scheduler, loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode="min")
    criterion = nn.CrossEntropyLoss()

    # Early stopping (only really matters on rank 0, but we pass model for compatibility)
    early_stopping = EarlyStopping(patience=10, verbose=is_main_process(rank))

    # Train
    model_ft, avg_train_loss, avg_val_loss, train_acc, val_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        rank=rank,
        world_size=world_size,
        num_epochs=max_epoch,
    )

    # Save only on rank 0
    if is_main_process(rank):
        print("Done Training Model")
        print("Train Acc:", train_acc)
        print("Train Loss:", avg_train_loss)
        print("Val Acc:", val_acc)
        print("Val Loss:", avg_val_loss)

        tsTime = time.strftime("%H-%M-%S")
        tsDate = time.strftime("%m-%d-%Y")
        tsEnd = tsDate + "_" + tsTime
        outputdir = f"/gpfs/data/oermannlab/users/xh852/image_db_experiment/{tsEnd}/"

        os.makedirs(outputdir, exist_ok=True)

        # If model is DDP, get underlying module
        to_save = model_ft.module if isinstance(model_ft, torch.nn.parallel.DistributedDataParallel) else model_ft

        torch.save(to_save.state_dict(), os.path.join(outputdir, f"model_dict_single_{tsEnd}.pt"))
        print("Saved model to:", outputdir)
        print(outputdir + " train_loss_epoch_single_" + tsEnd, avg_train_loss)
        print(outputdir + " val_loss_epoch_single_" + tsEnd, avg_val_loss)
        print(outputdir + " train_acc_epoch_single_" + tsEnd, train_acc)
        print(outputdir + " val_acc_epoch_single_" + tsEnd, val_acc)


if __name__ == "__main__":
    main()
