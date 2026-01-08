import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import DenseNet121_SingleLab

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, partial=False):

        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                image_name = line.split()[0]
                image_names.append(image_name)

        self.image_names = image_names
        self.data = pd.read_csv(data_dir)
        if partial==True:
            self.image_names = image_names[:500]
        self.transform = transform
        self.labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion',
                        'Pneumonia','Pleural_Thickening','Cardiomegaly','Mass','Nodule','Hernia', 'No Finding']
        self.sentence_idx = np.linspace(0,len(self.labels), len(self.labels), False)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image_path = 'images/' + image_name
        image = Image.open(image_path).convert('RGB')
        label = self.data[self.data['Image Index'] == image_name]['Finding Labels'].values
        label_num = self.labels.index(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor([label_num]).squeeze()

    def __len__(self):
        return len(self.image_names)

if __name__ == "__main__":
    data_dir = 'single_finding/data_entry_single_finding.csv'
    img_dir = '/gpfs/data/oermannlab/data/nih_cxr/single_finding/data_list/train_images_single_small.txt'

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])

    ts = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = ChestXrayDataSet(data_dir, img_dir, ts)

    train_loader = DataLoader(dataset=train_ds, batch_size=16,
                                shuffle=False)

    model = DenseNet121_SingleLab(15).cuda()

    for x,y in train_loader:
        x = x.cuda()
        y = y.cuda()
        out = model(x)
        lsm = torch.nn.Softmax(dim=1)
        pred = torch.argmax(lsm(out), dim=1) 
        print(pred)
        print(pred.shape)
        print(y.shape)
        break

